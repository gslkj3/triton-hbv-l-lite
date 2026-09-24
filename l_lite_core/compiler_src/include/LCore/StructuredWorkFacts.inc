int64_t staticElementCount(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped)
    return 1;
  if (!shaped.hasStaticShape())
    return 0;
  return shaped.getNumElements();
}

int64_t staticValueBytes(Value value) {
  Type type = value.getType();
  int64_t elements = staticElementCount(type);
  Type element = getElementTypeOrSelf(type);
  if (elements < 1 || !element.isIntOrFloat())
    return 0;
  unsigned bits = element.getIntOrFloatBitWidth();
  return elements * ((bits + 7) / 8);
}

// Structured issued-container work, never a latency or DRAM-byte claim.
// Only globally proven launch conditions: no measured branch frequencies and
// no assumption that different SSA values vary independently in lockstep.
using WorkIntegerRanges = llvm::DenseMap<Value, std::pair<int64_t, int64_t>>;
std::optional<std::pair<int64_t, int64_t>>
launchWorkIntegerRange(Value value, unsigned &budget,
                       const WorkIntegerRanges *bindings = nullptr) {
  if (!budget)
    return std::nullopt;
  --budget;
  auto type = dyn_cast<IntegerType>(value.getType());
  if (!type || type.getWidth() > 64)
    return std::nullopt;
  if (bindings) {
    auto found = bindings->find(value);
    if (found != bindings->end())
      return found->second;
  }
  unsigned pointBudget = 128;
  if (auto point = launchIntegerObservation(value, pointBudget))
    return std::make_pair(*point, *point);
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    auto loop = dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp());
    if (!loop || value != loop.getInductionVar())
      return std::nullopt;
    auto lower = launchWorkIntegerRange(loop.getLowerBound(), budget, bindings);
    auto upper = launchWorkIntegerRange(loop.getUpperBound(), budget, bindings);
    auto step = launchWorkIntegerRange(loop.getStep(), budget, bindings);
    if (!lower || !upper || !step || lower->first != lower->second ||
        upper->first != upper->second || step->first != step->second ||
        step->first <= 0 || upper->first <= lower->first)
      return std::nullopt;
    __int128 distance = static_cast<__int128>(upper->first) - lower->first - 1;
    auto last = checkedI64(static_cast<__int128>(lower->first) +
                          (distance / step->first) * step->first);
    auto next = last ? checkedI64(static_cast<__int128>(*last) + step->first)
                     : std::nullopt;
    if (!last || !next || !llvm::isIntN(type.getWidth(), *last) ||
        !llvm::isIntN(type.getWidth(), *next))
      return std::nullopt;
    return std::make_pair(lower->first, *last);
  }
  if (auto pid = value.getDefiningOp<GetProgramIdOp>()) {
    auto extent = pid->getAttrOfType<IntegerAttr>(kBridgeAxisExtentAttr);
    if (extent && extent.getInt() > 0 &&
        llvm::isIntN(type.getWidth(), extent.getInt() - 1))
      return std::make_pair(int64_t(0), extent.getInt() - 1);
    return std::nullopt;
  }
  Operation *op = value.getDefiningOp();
  if (!op || op->getNumOperands() != 2)
    return std::nullopt;
  StringRef name = op->getName().getStringRef();
  if (name != "arith.addi" && name != "arith.subi" && name != "arith.muli")
    return std::nullopt;
  if (auto flags = op->getAttrOfType<arith::IntegerOverflowFlagsAttr>("overflowFlags"))
    if (flags.getValue() != arith::IntegerOverflowFlags::none)
      return std::nullopt;
  auto a = launchWorkIntegerRange(op->getOperand(0), budget, bindings);
  auto b = launchWorkIntegerRange(op->getOperand(1), budget, bindings);
  if (!a || !b)
    return std::nullopt;
  // Cartesian endpoint bounds remain sound for correlated SSA operands.
  // Reject wraparound instead of silently applying mathematical integers.
  __int128 lo = 0, hi = 0;
  if (name == "arith.addi") {
    lo = static_cast<__int128>(a->first) + b->first;
    hi = static_cast<__int128>(a->second) + b->second;
  } else if (name == "arith.subi") {
    lo = static_cast<__int128>(a->first) - b->second;
    hi = static_cast<__int128>(a->second) - b->first;
  } else {
    __int128 products[] = {
        static_cast<__int128>(a->first) * b->first,
        static_cast<__int128>(a->first) * b->second,
        static_cast<__int128>(a->second) * b->first,
        static_cast<__int128>(a->second) * b->second};
    lo = *std::min_element(std::begin(products), std::end(products));
    hi = *std::max_element(std::begin(products), std::end(products));
  }
  auto lower = checkedI64(lo), upper = checkedI64(hi);
  if (!lower || !upper || !llvm::isIntN(type.getWidth(), *lower) ||
      !llvm::isIntN(type.getWidth(), *upper))
    return std::nullopt;
  return std::make_pair(*lower, *upper);
}

std::optional<bool> launchWorkCondition(Value condition,
                                      const WorkIntegerRanges *bindings = nullptr) {
  auto cmp = condition.getDefiningOp<arith::CmpIOp>();
  if (!cmp)
    return std::nullopt;
  auto range = [&](Value value)
      -> std::optional<std::pair<int64_t, int64_t>> {
    unsigned budget = 128;
    return launchWorkIntegerRange(value, budget, bindings);
  };
  auto a = range(cmp.getLhs()), b = range(cmp.getRhs());
  if (!a || !b)
    return std::nullopt;
  bool disjoint = a->second < b->first || b->second < a->first;
  bool equalPoints = a->first == a->second && b->first == b->second &&
                     a->first == b->first;
  switch (cmp.getPredicate()) {
  case arith::CmpIPredicate::eq:
    if (disjoint) return false;
    if (equalPoints) return true;
    break;
  case arith::CmpIPredicate::ne:
    if (disjoint) return true;
    if (equalPoints) return false;
    break;
  case arith::CmpIPredicate::slt:
    if (a->second < b->first) return true;
    if (a->first >= b->second) return false;
    break;
  case arith::CmpIPredicate::sle:
    if (a->second <= b->first) return true;
    if (a->first > b->second) return false;
    break;
  case arith::CmpIPredicate::sgt:
    if (a->first > b->second) return true;
    if (a->second <= b->first) return false;
    break;
  case arith::CmpIPredicate::sge:
    if (a->first >= b->second) return true;
    if (a->second < b->first) return false;
    break;
  default: break; // Unsigned reinterpretation requires its own range proof.
  }
  return std::nullopt;
}

llvm::json::Object structuredContainerCFG(
    Block &block, unsigned &budget, unsigned depth,
    llvm::SmallPtrSetImpl<Block *> &active,
    const WorkIntegerRanges *bindings = nullptr);

// Region ownership supplies exclusive/exhaustive if/else semantics. Unknown
// regions/calls remain explicit rather than being flattened or counted twice.
llvm::json::Object structuredContainerBlock(Block &block, unsigned &budget,
                                           unsigned depth,
                                           bool typedArithmetic = false,
                                           const WorkDependencyIndex *dependencies = nullptr,
                                           const WorkIntegerRanges *bindings = nullptr) {
  if (depth > 128 || !budget)
    return llvm::json::Object{{"kind", "unknown"}};
  llvm::json::Array children;
  for (Operation &op : block.without_terminator()) {
    if (!budget) {
      children.push_back(llvm::json::Object{{"kind", "unknown"}});
      break;
    }
    --budget;
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      llvm::json::Array body;
      body.push_back(structuredContainerBlock(*loop.getBody(), budget, depth+1,
                                             typedArithmetic, dependencies, bindings));
      unsigned scalarBudget = 128;
      auto lower = launchIntegerObservation(loop.getLowerBound(), scalarBudget);
      scalarBudget = 128;
      auto upper = launchIntegerObservation(loop.getUpperBound(), scalarBudget);
      scalarBudget = 128;
      auto step = launchIntegerObservation(loop.getStep(), scalarBudget);
      // The ordinary census already proves constant index/iN bounds. Do not
      // discard that fact because runtime integer projection is narrower.
      std::optional<int64_t> trip = exactStaticTripCount(loop);
      if (!trip && lower && upper && step && *step > 0) {
        __int128 distance = static_cast<__int128>(*upper) - *lower;
        trip = distance <= 0 ? std::optional<int64_t>(0)
                             : checkedI64((distance + *step - 1) / *step);
      }
      llvm::json::Object node{{"kind", "loop"}, {"children", std::move(body)}};
      node["trip_count"] = trip ? llvm::json::Value(*trip) : llvm::json::Value(nullptr);
      children.push_back(std::move(node));
    } else if (auto branch = dyn_cast<scf::IfOp>(op)) {
      if (auto known = launchWorkCondition(branch.getCondition(), bindings)) {
        if (*known)
          children.push_back(structuredContainerBlock(*branch.thenBlock(), budget,
                                                      depth+1, typedArithmetic, dependencies, bindings));
        else if (!branch.getElseRegion().empty())
          children.push_back(structuredContainerBlock(*branch.elseBlock(), budget,
                                                      depth+1, typedArithmetic, dependencies, bindings));
        continue;
      }
      llvm::json::Array arms;
      arms.push_back(structuredContainerBlock(*branch.thenBlock(), budget, depth+1,
                                             typedArithmetic, dependencies, bindings));
      if (!branch.getElseRegion().empty())
        arms.push_back(structuredContainerBlock(*branch.elseBlock(), budget, depth+1,
                                               typedArithmetic, dependencies, bindings));
      else
        arms.push_back(llvm::json::Object{{"kind", "sequence"},
                                        {"children", llvm::json::Array()}});
      children.push_back(llvm::json::Object{{"kind", "if_else"},
                                           {"children", std::move(arms)}});
    } else if (auto call = dyn_cast<CallOp>(op); call && !typedArithmetic) {
      // A visible callee contributes work even when native inlining cannot
      // move a multi-block body into an SCF region. This is a work summary,
      // not permission to reorder memory or specialize the call. Never reuse
      // a callee's launch-bound scalar assumption for a different actual.
      auto callee = SymbolTable::lookupNearestSymbolFrom<FuncOp>(
          call, call.getCalleeAttr());
      bool bound = callee && !callee.getBody().empty() &&
                   callee.getNumArguments() == call.getNumOperands();
      WorkIntegerRanges callBindings;
      if (bound) {
        for (unsigned i = 0; i < callee.getNumArguments(); ++i) {
          unsigned rangeBudget = 128;
          auto range = launchWorkIntegerRange(call.getOperand(i), rangeBudget, bindings);
          if (range)
            callBindings[callee.getArgument(i)] = *range;
          auto expected = callee.getArgAttrOfType<IntegerAttr>(i, kBridgeBoundScalarAttr);
          if (!expected)
            continue;
          if (!range || range->first != expected.getInt() || range->second != expected.getInt()) {
            bound = false;
            break;
          }
        }
      }
      if (!bound) {
        children.push_back(llvm::json::Object{{"kind", "unknown"}});
      } else if (callee.getBody().hasOneBlock()) {
        children.push_back(structuredContainerBlock(callee.getBody().front(),
                                                    budget, depth + 1, false, nullptr, &callBindings));
      } else {
        llvm::SmallPtrSet<Block *, 32> active;
        children.push_back(structuredContainerCFG(callee.getBody().front(),
                                                  budget, depth + 1, active, &callBindings));
      }
      // Shared depth/node budgets also bound recursive calls. Unknown work
      // stays unknown, including external callees and unresolved arguments.
    } else if (auto load = dyn_cast<LoadOp>(op)) {
      int64_t bytes = staticValueBytes(load.getResult());
      children.push_back(bytes > 0
          ? llvm::json::Object{{"kind", "leaf"}, {"load_bytes", bytes}}
          : llvm::json::Object{{"kind", "unknown"}});
    } else if (auto store = dyn_cast<StoreOp>(op)) {
      int64_t bytes = staticValueBytes(store.getValue());
      children.push_back(bytes > 0
          ? llvm::json::Object{{"kind", "leaf"}, {"store_bytes", bytes}}
          : llvm::json::Object{{"kind", "unknown"}});
    } else {
      // A workgroup barrier orders memory effects but has no explicit load,
      // store, or arithmetic result container of its own (GPUOps.td).
      // This is only a work-unit projection: do not weaken its effects in
      // legality/scheduling or infer zero synchronization latency. The loop
      // census independently retains explicit_synchronization_barrier_count.
      if (op.getName().getStringRef() == "gpu.barrier")
        continue;
      if (typedArithmetic && op.getNumRegions()) {
        // A pure region (e.g. a reduction combiner) has no memory work,
        // but its arithmetic multiplicity needs a separate transfer rule.
        // Do not treat it as zero or flatten its region as one invocation.
        // Inventory bounds *which explicit IR operations* may occur, not
        // their multiplicity. Unrecognized leaves/calls keep it incomplete.
        bool inventoryComplete =
            isMemoryEffectFree(&op) ||
            op.hasTrait<OpTrait::HasRecursiveMemoryEffects>();
        unsigned inventoryBudget = 4096;
        std::set<std::pair<std::string, std::string>> inventory;
        op.walk([&](Operation *nested) -> WalkResult {
          if (!inventoryBudget--) {
            inventoryComplete = false;
            return WalkResult::interrupt();
          }
          if (nested == &op || nested->hasTrait<OpTrait::IsTerminator>())
            return WalkResult::advance();
          StringRef ns = nested->getName().getDialectNamespace();
          if ((ns == "arith" || ns == "math") &&
              !isa<arith::ConstantOp>(nested)) {
            inventoryComplete &= isMemoryEffectFree(nested) &&
                                 nested->getNumResults() > 0;
            for (Type type : nested->getResultTypes()) {
              Type element = getElementTypeOrSelf(type);
              inventoryComplete &= element.isIntOrFloat();
              std::string spelling;
              llvm::raw_string_ostream stream(spelling);
              element.print(stream);
              stream.flush();
              inventory.emplace(nested->getName().getStringRef().str(),
                                spelling);
            }
          } else if (!isa<arith::ConstantOp>(nested)) {
            inventoryComplete &=
                nested->getNumRegions() > 0 &&
                (isMemoryEffectFree(nested) ||
                 nested->hasTrait<OpTrait::HasRecursiveMemoryEffects>());
          }
          return WalkResult::advance();
        });
        llvm::json::Array inventoryRows;
        for (const auto &[operation, elementType] : inventory)
          inventoryRows.push_back(llvm::json::Object{
              {"operation", operation}, {"element_type", elementType}});
        children.push_back(llvm::json::Object{
            {"kind", "unknown"},
            {"operation", op.getName().getStringRef().str()},
            {"reason", "region_arithmetic_multiplicity_not_projected"},
            {"explicit_typed_inventory_complete", inventoryComplete},
            {"explicit_typed_inventory", std::move(inventoryRows)}});
        continue;
      }
      if (typedArithmetic && isa<DotOp>(op)) {
        llvm::json::Array shapes, elementTypes;
        bool known = op.getNumOperands() == 3 && op.getNumResults() == 1;
        for (Type type : op.getOperandTypes()) {
          auto tensor = dyn_cast<RankedTensorType>(type);
          known &= tensor && tensor.hasStaticShape();
          llvm::json::Array shape;
          if (tensor)
            for (int64_t dim : tensor.getShape())
              shape.push_back(dim);
          shapes.push_back(std::move(shape));
          std::string spelling;
          llvm::raw_string_ostream stream(spelling);
          getElementTypeOrSelf(type).print(stream);
          stream.flush();
          elementTypes.push_back(spelling);
        }
        llvm::json::Object node{
            {"kind", known ? "tensor_contraction" : "unknown"},
            {"operation", "tt.dot"},
            {"operand_shapes", std::move(shapes)},
            {"operand_element_types", std::move(elementTypes)},
            {"scope", "logical_tt_dot_per_invocation_not_machine_instructions"}};
        if (!known)
          node["reason"] = "contraction_shape_unavailable";
        if (dependencies && dependencies->complete)
          node["operation_id"] = dependencies->operations.lookup(&op);
        children.push_back(std::move(node));
        continue;
      }
      StringRef dialect = op.getName().getDialectNamespace();
      if (typedArithmetic && (dialect == "arith" || dialect == "math") &&
          !isa<arith::ConstantOp>(op) && isMemoryEffectFree(&op)) {
        llvm::json::Array results, operandElementTypes;
        bool known = op.getNumResults() > 0;
        for (Type type : op.getOperandTypes()) {
          std::string spelling;
          llvm::raw_string_ostream stream(spelling);
          getElementTypeOrSelf(type).print(stream);
          stream.flush();
          operandElementTypes.push_back(spelling);
        }
        for (Type type : op.getResultTypes()) {
          int64_t elements = staticElementCount(type);
          Type element = getElementTypeOrSelf(type);
          known &= elements > 0 && element.isIntOrFloat();
          std::string spelling;
          llvm::raw_string_ostream stream(spelling);
          element.print(stream);
          stream.flush();
          results.push_back(llvm::json::Object{
              {"element_type", spelling}, {"elements", elements}});
        }
        if (known) {
          llvm::json::Object node{
              {"kind", "typed_arithmetic"},
              {"operation", op.getName().getStringRef().str()},
              {"operand_element_types", std::move(operandElementTypes)},
              {"results", std::move(results)},
              {"scope", "ir_result_container_per_operation_invocation"}};
          if (dependencies && dependencies->complete)
            node["operation_id"] = dependencies->operations.lookup(&op);
          children.push_back(std::move(node));
        } else
          children.push_back(llvm::json::Object{
              {"kind", "unknown"},
              {"reason", "arithmetic_result_shape_or_type_unavailable"}});
        continue;
      }
      // Pure region operations can carry arithmetic without issuing memory
      // containers. Check every nested operation, not an operation-name list.
      bool noMemoryEffects = true;
      op.walk([&](Operation *nested) {
        noMemoryEffects &= isMemoryEffectFree(nested);
      });
      if (!noMemoryEffects)
        children.push_back(llvm::json::Object{{"kind", "unknown"}});
    }
  }
  return llvm::json::Object{{"kind", "sequence"}, {"children", std::move(children)}};
}

// Project acyclic CFG paths without mutating IR. A shared successor appears in
// each exclusive branch, never twice on one path. Backedges and unsupported
// terminators retain unknown suffix work; no visit-count or probability guess.
llvm::json::Object structuredContainerCFG(
    Block &block, unsigned &budget, unsigned depth,
    llvm::SmallPtrSetImpl<Block *> &active,
    const WorkIntegerRanges *bindings) {
  if (depth > 32 || !budget || !active.insert(&block).second)
    return llvm::json::Object{{"kind", "unknown"}};
  auto cleanup = llvm::make_scope_exit([&] { active.erase(&block); });
  --budget;
  llvm::json::Array sequence;
  sequence.push_back(structuredContainerBlock(block, budget, depth * 3 + 1,
                                               false, nullptr, bindings));
  Operation *terminator = block.getTerminator();
  if (auto branch = dyn_cast<cf::BranchOp>(terminator)) {
    sequence.push_back(structuredContainerCFG(*branch.getDest(), budget,
                                              depth + 1, active, bindings));
  } else if (auto branch = dyn_cast<cf::CondBranchOp>(terminator)) {
    if (auto known = launchWorkCondition(branch.getCondition(), bindings)) {
      Block *destination = *known ? branch.getTrueDest() : branch.getFalseDest();
      sequence.push_back(structuredContainerCFG(*destination, budget, depth + 1,
                                                active, bindings));
      return llvm::json::Object{{"kind", "sequence"},
                                {"children", std::move(sequence)}};
    }
    llvm::json::Array arms;
    arms.push_back(structuredContainerCFG(*branch.getTrueDest(), budget,
                                          depth + 1, active, bindings));
    arms.push_back(structuredContainerCFG(*branch.getFalseDest(), budget,
                                          depth + 1, active, bindings));
    sequence.push_back(llvm::json::Object{{"kind", "if_else"},
                                         {"children", std::move(arms)}});
  } else if (!isa<ReturnOp>(terminator)) {
    sequence.push_back(llvm::json::Object{{"kind", "unknown"}});
  }
  return llvm::json::Object{{"kind", "sequence"},
                            {"children", std::move(sequence)}};
}

// Match the downstream structured-work decoder budget. CFG path expansion can
// emit more tree nodes than visited operations; fail explicitly to unknown
// rather than emitting an undecodable fact or truncating a branch to zero.
bool boundedContainerTree(const llvm::json::Object &node, unsigned &remaining,
                          unsigned depth = 0) {
  if (!remaining || depth > 128)
    return false;
  --remaining;
  if (auto *children = node.getArray("children"))
    for (const auto &child : *children) {
      auto *object = child.getAsObject();
      if (!object || !boundedContainerTree(*object, remaining, depth + 1))
        return false;
    }
  return true;
}

// Preserve control-flow ownership; a bounded tree is not an execution profile.
void appendStructuredWorkFacts(FuncOp entry, llvm::json::Object &facts) {
  WorkDependencyIndex dependencies(entry);
  facts["work_dependency_graph"] = dependencies.json();
  unsigned structuredWorkBudget = 4096;
  llvm::SmallPtrSet<Block *, 32> activeWorkBlocks;
  facts["structured_container_work"] =
      entry.getBody().hasOneBlock()
          ? structuredContainerBlock(entry.getBody().front(), structuredWorkBudget, 0)
          : structuredContainerCFG(entry.getBody().front(), structuredWorkBudget,
                                   0, activeWorkBlocks);
  unsigned typedWorkBudget = 4096;
  facts["structured_typed_container_work"] =
      entry.getBody().hasOneBlock()
          ? structuredContainerBlock(entry.getBody().front(), typedWorkBudget,
                                     0, true, &dependencies)
          : llvm::json::Object{{"kind", "unknown"}};
  facts["structured_typed_container_work_schema"] = "tt_dot_geometry_v1";
  unsigned decodedWorkBudget = 4096;
  if (!boundedContainerTree(*facts.getObject("structured_container_work"),
                            decodedWorkBudget))
    facts["structured_container_work"] = llvm::json::Object{{"kind", "unknown"}};
}
