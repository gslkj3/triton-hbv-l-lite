// Raw static IR inventory; execution multiplicity is a separate observation.
struct StaticWorkInventory {
  int64_t loadCount = 0;
  int64_t storeCount = 0;
  int64_t atomicCount = 0;
  int64_t loopCount = 0;
  int64_t reduceCount = 0;
  int64_t maxLoadWidth = 1;
  int64_t inputBytes = 0;
  int64_t outputBytes = 0;
  int64_t arithmeticElements = 0;
  llvm::json::Array reductionGeometry;
  int64_t runtimeActiveExtentArgumentIndex = -1;
  int64_t runtimeActiveExtentContainerWidth = 0;
  int64_t activeInputBytesPerElement = 0;
  int64_t activeOutputBytesPerElement = 0;
  int64_t activeArithmeticElementsPerElement = 0;
  int64_t activeInputFixedBytes = 0;
  int64_t activeOutputFixedBytes = 0;
  int64_t activeArithmeticFixedElements = 0;
  std::set<std::string> bodyTypeSignature;
  std::map<std::string, int64_t> bodySpecialFunctionCounts;
  SmallVector<GetProgramIdOp> xPids;

  void observe(FuncOp entry, ArrayRef<FuncOp> reachableFunctions,
               const llvm::DenseMap<Operation *, std::string> &planningCutLoopLocators) {
    // Establish one common dynamic active extent from masked entry loads.
    // Multiple masked loads must agree on both the entry argument and the
    // vector container.  Ambiguity leaves the legacy exact-static facts in
    // force and publishes no runtime fact.
    bool activeExtentAmbiguous = false;
    SmallVector<Value> activeLoadResults;
    entry.walk([&](LoadOp load) {
      auto fact = runtimeActiveExtentFromMask(load.getMask(), entry);
      if (!fact)
        return;
      if (runtimeActiveExtentArgumentIndex < 0) {
        runtimeActiveExtentArgumentIndex = fact->argumentIndex;
        runtimeActiveExtentContainerWidth = fact->containerWidth;
      } else if (runtimeActiveExtentArgumentIndex != fact->argumentIndex ||
                 runtimeActiveExtentContainerWidth != fact->containerWidth) {
        activeExtentAmbiguous = true;
        return;
      }
      activeLoadResults.push_back(load.getResult());
    });
    if (activeExtentAmbiguous || activeLoadResults.empty()) {
      runtimeActiveExtentArgumentIndex = -1;
      runtimeActiveExtentContainerWidth = 0;
      activeLoadResults.clear();
    }

    auto dependsOnActiveLoad = [&](Value value) {
      return llvm::any_of(activeLoadResults, [&](Value source) {
        return valueDependsOn(value, source);
      });
    };
    for (FuncOp function : reachableFunctions)
      function.walk([&](Operation *op) {
      if (!op->hasTrait<OpTrait::IsTerminator>()) {
        auto recordType = [&](Type type) {
          Type scalar = getElementTypeOrSelf(type);
          if (auto pointer = dyn_cast<PointerType>(scalar))
            scalar = getElementTypeOrSelf(pointer.getPointeeType());
          std::string text;
          llvm::raw_string_ostream stream(text);
          scalar.print(stream);
          stream.flush();
          bodyTypeSignature.insert(std::move(text));
        };
        for (Value operand : op->getOperands())
          recordType(operand.getType());
        for (Value result : op->getResults())
          recordType(result.getType());
        StringRef operationName = op->getName().getStringRef();
        if (operationName.starts_with("math."))
          ++bodySpecialFunctionCounts[std::string(operationName)];
      }
      if (function == entry)
        if (auto pid = dyn_cast<GetProgramIdOp>(op);
            pid && pid.getAxisAsInt() == 0)
          xPids.push_back(pid);
      if (auto load = dyn_cast<LoadOp>(op)) {
        ++loadCount;
        int64_t width = staticElementCount(load.getResult().getType());
        maxLoadWidth = std::max(maxLoadWidth, width);
        inputBytes += staticValueBytes(load.getResult());
        if (runtimeActiveExtentArgumentIndex >= 0) {
          auto fact = runtimeActiveExtentFromMask(load.getMask(), entry);
          if (fact && fact->argumentIndex == runtimeActiveExtentArgumentIndex &&
              fact->containerWidth == runtimeActiveExtentContainerWidth) {
            Type element = getElementTypeOrSelf(load.getResult().getType());
            activeInputBytesPerElement +=
                (element.getIntOrFloatBitWidth() + 7) / 8;
          }
        }
      } else if (auto store = dyn_cast<StoreOp>(op)) {
        ++storeCount;
        outputBytes += staticValueBytes(store.getValue());
        if (runtimeActiveExtentArgumentIndex >= 0) {
          auto fact = runtimeActiveExtentFromMask(store.getMask(), entry);
          if (fact && fact->argumentIndex == runtimeActiveExtentArgumentIndex &&
              fact->containerWidth == runtimeActiveExtentContainerWidth) {
            Type element = getElementTypeOrSelf(store.getValue().getType());
            activeOutputBytesPerElement +=
                (element.getIntOrFloatBitWidth() + 7) / 8;
          }
        }
      } else if (isa<AtomicRMWOp, AtomicCASOp>(op)) {
        ++atomicCount;
      } else if (isa<scf::ForOp>(op)) {
        ++loopCount;
      } else if (auto reduce = dyn_cast<ReduceOp>(op)) {
        ++reduceCount;
        for (auto [operandIndex, operand] : llvm::enumerate(reduce.getSrcs())) {
          auto tensor = dyn_cast<RankedTensorType>(operand.getType());
          if (!tensor || !tensor.hasStaticShape())
            continue;
          llvm::json::Object geometry;
          llvm::json::Array shape;
          for (int64_t extent : tensor.getShape())
            shape.push_back(extent);
          geometry["input_shape"] = std::move(shape);
          geometry["axis"] = reduce.getAxis();
          geometry["operand_index"] = static_cast<int64_t>(operandIndex);
          geometry["reduction_ordinal"] = reduceCount - 1;
          geometry["entry_owned"] = function == entry;
          bool unconditionalLoopAncestry = function == entry;
          for (Operation *parent = reduce->getParentOp();
               parent && parent != function.getOperation();
               parent = parent->getParentOp())
            if (!isa<scf::ForOp>(parent))
              unconditionalLoopAncestry = false;
          geometry["unconditional_loop_ancestry"] = unconditionalLoopAncestry;
          auto enclosingLoop = reduce->getParentOfType<scf::ForOp>();
          geometry["enclosing_loop_locator"] = enclosingLoop
              ? planningCutLoopLocators.lookup(enclosingLoop.getOperation()) : "";
          geometry["scope"] = "static_ir_container_not_execution_count";
          std::string elementType;
          llvm::raw_string_ostream typeStream(elementType);
          tensor.getElementType().print(typeStream);
          typeStream.flush();
          geometry["element_type"] = elementType;
          reductionGeometry.push_back(std::move(geometry));
        }
        for (Value operand : reduce.getSrcs()) {
          arithmeticElements += std::max<int64_t>(1, staticElementCount(operand.getType()) - 1);
          if (runtimeActiveExtentArgumentIndex >= 0 &&
              staticElementCount(operand.getType()) ==
                  runtimeActiveExtentContainerWidth &&
              dependsOnActiveLoad(operand))
            ++activeArithmeticElementsPerElement;
        }
      }
      StringRef name = op->getName().getStringRef();
      if (!isa<arith::ConstantOp>(op) &&
          (name.starts_with("arith.") || name.starts_with("math.")) &&
          op->getNumResults() == 1) {
        Type element = getElementTypeOrSelf(op->getResult(0).getType());
        if (isa<FloatType>(element))
          arithmeticElements += std::max<int64_t>(1, staticElementCount(op->getResult(0).getType()));
        if (runtimeActiveExtentArgumentIndex >= 0 && isa<FloatType>(element) &&
            staticElementCount(op->getResult(0).getType()) ==
                runtimeActiveExtentContainerWidth &&
            dependsOnActiveLoad(op->getResult(0)))
          ++activeArithmeticElementsPerElement;
      }
      });


    if (runtimeActiveExtentArgumentIndex >= 0) {
      activeInputFixedBytes = inputBytes -
          activeInputBytesPerElement * runtimeActiveExtentContainerWidth;
      activeOutputFixedBytes = outputBytes -
          activeOutputBytesPerElement * runtimeActiveExtentContainerWidth;
      activeArithmeticFixedElements = arithmeticElements -
          activeArithmeticElementsPerElement *
              runtimeActiveExtentContainerWidth;
      // The legacy single active-extent tuple describes the complete logical
      // group, whereas operation-local runtime_mask_scalars can describe a
      // masked inner dimension repeated inside a wider rank-N group.  Do not
      // publish the legacy tuple when those two containers differ or when the
      // resulting affine projection does not conserve the full group facts.
      // The operation-local mask facts remain available; downstream generic
      // loop routes retain the conservative full-container accounting.
      bool hasPhysicalSlope = activeInputBytesPerElement > 0 ||
                              activeOutputBytesPerElement > 0 ||
                              activeArithmeticElementsPerElement > 0;
      bool affineProjectionComplete =
          runtimeActiveExtentContainerWidth > 0 && hasPhysicalSlope &&
          maxLoadWidth <= runtimeActiveExtentContainerWidth &&
          activeInputFixedBytes >= 0 && activeOutputFixedBytes >= 0 &&
          activeArithmeticFixedElements >= 0 &&
          inputBytes == activeInputFixedBytes +
                            activeInputBytesPerElement * maxLoadWidth &&
          outputBytes == activeOutputFixedBytes +
                             activeOutputBytesPerElement * maxLoadWidth &&
          arithmeticElements == activeArithmeticFixedElements +
                                    activeArithmeticElementsPerElement *
                                        maxLoadWidth;
      if (!affineProjectionComplete) {
        runtimeActiveExtentArgumentIndex = -1;
        runtimeActiveExtentContainerWidth = 0;
        activeInputBytesPerElement = 0;
        activeOutputBytesPerElement = 0;
        activeArithmeticElementsPerElement = 0;
        activeInputFixedBytes = 0;
        activeOutputFixedBytes = 0;
        activeArithmeticFixedElements = 0;
      }
    }
  }

  void append(llvm::json::Object &facts) {
    facts["axis_zero_program_id_count"] = static_cast<int64_t>(xPids.size());
    facts["load_count"] = loadCount;
    facts["static_work_inventory_scope"] =
        "reachable_ir_once_not_execution_multiplicity";
    facts["store_count"] = storeCount;
    facts["atomic_count"] = atomicCount;
    facts["native_loop_count"] = loopCount;
    facts["reduction_count"] = reduceCount;
    facts["reduction_geometry"] = std::move(reductionGeometry);
    facts["group_width"] = maxLoadWidth;
    facts["input_bytes_per_logical_program"] = inputBytes;
    facts["output_bytes_per_logical_program"] = outputBytes;
    facts["arithmetic_elements_per_logical_program"] = arithmeticElements;
    llvm::json::Array typeSignature;
    for (const std::string &type : bodyTypeSignature)
      typeSignature.push_back(type);
    facts["body_type_signature"] = std::move(typeSignature);
    llvm::json::Object specialFunctionSignature;
    for (const auto &[name, count] : bodySpecialFunctionCounts)
      specialFunctionSignature[name] = count;
    facts["body_special_function_counts"] =
        std::move(specialFunctionSignature);
    facts["runtime_active_extent_argument_index"] =
        runtimeActiveExtentArgumentIndex;
    facts["runtime_active_extent_container_width"] =
        runtimeActiveExtentContainerWidth;
    facts["active_input_bytes_per_element"] = activeInputBytesPerElement;
    facts["active_input_fixed_bytes"] = activeInputFixedBytes;
    facts["active_output_bytes_per_element"] = activeOutputBytesPerElement;
    facts["active_output_fixed_bytes"] = activeOutputFixedBytes;
    facts["active_arithmetic_elements_per_element"] =
        activeArithmeticElementsPerElement;
    facts["active_arithmetic_fixed_elements"] =
        activeArithmeticFixedElements;
  }
};
