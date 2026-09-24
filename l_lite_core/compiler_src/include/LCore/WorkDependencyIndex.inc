// Snapshot-local SSA links. IDs bind facts only; never model coordinates.
struct WorkDependencyIndex {
  DenseMap<Operation *, int64_t> operations;
  DenseMap<Value, int64_t> values;
  SmallVector<Operation *> ordered;
  bool complete = true;

  explicit WorkDependencyIndex(FuncOp entry) {
    entry.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
      if (ordered.size() >= 4096 || values.size() > 16384) {
        complete = false;
        return WalkResult::interrupt();
      }
      operations[op] = ordered.size();
      ordered.push_back(op);
      for (Value result : op->getResults())
        values.try_emplace(result, values.size());
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (BlockArgument arg : block.getArguments())
            values.try_emplace(arg, values.size());
      return WalkResult::advance();
    });
    if (values.size() > 16384) complete = false;
  }

  llvm::json::Value valueId(Value value) const {
    auto it = values.find(value);
    return it == values.end() ? llvm::json::Value(nullptr)
                             : llvm::json::Value(it->second);
  }

  llvm::json::Object json() const {
    if (!complete)
      return llvm::json::Object{{"complete", false}, {"reason", "snapshot_dependency_budget"}};
    auto shapeOf = [](Type type) -> llvm::json::Value {
      auto ranked = dyn_cast<RankedTensorType>(type);
      if (!ranked) return nullptr;
      llvm::json::Array shape;
      for (int64_t extent : ranked.getShape()) shape.push_back(extent);
      return shape;
    };
    llvm::json::Array nodes, arguments, rankedTensorValues;
    for (Operation *op : ordered) {
      llvm::json::Array operands, results, resultShapes;
      for (Value operand : op->getOperands()) operands.push_back(valueId(operand));
      for (Value result : op->getResults()) {
        results.push_back(valueId(result));
        resultShapes.push_back(shapeOf(result.getType()));
        if (isa<RankedTensorType>(result.getType()))
          rankedTensorValues.push_back(valueId(result));
      }
      llvm::json::Object node{
          {"operation_id", operations.lookup(op)},
          {"operation", op->getName().getStringRef().str()},
          {"operands", std::move(operands)}, {"results", std::move(results)},
          {"result_shapes", std::move(resultShapes)},
          {"has_regions", op->getNumRegions() != 0},
          {"exact_elementwise_packable", isGenericExactPackableOperation(op)},
          {"exact_memory_packable", isExactPackableMemoryOperation(op)},
          {"memory_effect_free", isMemoryEffectFree(op)}};
      auto parent = operations.find(op->getParentOp());
      node["parent_operation_id"] = parent == operations.end()
          ? llvm::json::Value(nullptr) : llvm::json::Value(parent->second);
      // Raw IR attributes for absolute interval observation. SSA topology and
      // result sizes alone cannot reconstruct literal values or range origins.
      // Do not reuse the legality proof's translation-cancelling intervals.
      if (op->getNumResults() == 1) {
        if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
          if (auto literal = splatInteger(constant.getResult()))
            node["integer_constant"] = *literal;
        }
      }
      if (auto range = dyn_cast<MakeRangeOp>(op)) {
        node["range_start"] = range.getStartAttr().getInt();
        node["range_end"] = range.getEndAttr().getInt();
      }
      if (auto expand = dyn_cast<ExpandDimsOp>(op))
        node["expand_axis"] = int64_t(expand.getAxis());
      if (auto pid = dyn_cast<GetProgramIdOp>(op))
        node["program_axis"] = int64_t(pid.getAxisAsInt());
      if (auto compare = dyn_cast<arith::CmpIOp>(op))
        node["integer_predicate"] = arith::stringifyCmpIPredicate(compare.getPredicate()).str();
      if (auto load = dyn_cast<LoadOp>(op)) {
        node["address_value"] = valueId(load.getPtr());
        if (load.getMask()) node["mask_value"] = valueId(load.getMask());
        // Effect independence must distinguish ordinary reads from volatile
        // reads.  Do not infer this from operation spelling or missing attrs.
        node["is_volatile"] = load.getIsVolatile();
      }
      if (auto store = dyn_cast<StoreOp>(op)) {
        node["address_value"] = valueId(store.getPtr());
        if (store.getMask()) node["mask_value"] = valueId(store.getMask());
      }
      nodes.push_back(std::move(node));
      int64_t regionIndex = 0;
      for (Region &region : op->getRegions()) {
        int64_t blockIndex = 0;
        for (Block &block : region) {
          for (BlockArgument arg : block.getArguments()) {
            if (isa<RankedTensorType>(arg.getType()))
              rankedTensorValues.push_back(valueId(arg));
            llvm::json::Object a{{"value_id", values.lookup(arg)},
                {"owner_operation_id", operations.lookup(op)},
                {"region", regionIndex}, {"block", blockIndex},
                {"argument", int64_t(arg.getArgNumber())},
                {"shape", shapeOf(arg.getType())}};
            if (auto loop = dyn_cast<scf::ForOp>(op)) {
              if (arg.getArgNumber() >= loop.getNumInductionVars()) {
                a["initial_value"] = valueId(loop.getTiedLoopInit(arg)->get());
                a["yield_value"] = valueId(loop.getBody()->getTerminator()->getOperand(
                    arg.getArgNumber() - loop.getNumInductionVars()));
              }
            }
            arguments.push_back(std::move(a));
          }
          ++blockIndex;
        }
        ++regionIndex;
      }
    }
    return llvm::json::Object{{"complete", true},
            {"scope", "entry_ssa_snapshot_not_execution_or_backend_layout"},
            {"operations", std::move(nodes)}, {"block_arguments", std::move(arguments)},
            {"ranked_tensor_values", std::move(rankedTensorValues)}};
  }
};
