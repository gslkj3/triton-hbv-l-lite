// Read-only address and mask observations; no route or profitability authority.
bool nativeAxisCallGraphHasCycle(ModuleOp module) {
  // Native AxisInfo traverses the module call graph, not only the entry's
  // reachable functions. Kahn traversal avoids recursion in this preflight.
  llvm::DenseMap<Operation *, unsigned> incoming;
  llvm::DenseMap<Operation *, SmallVector<Operation *>> edges;
  module.walk([&](FuncOp function) { incoming[function.getOperation()] = 0; });
  module.walk([&](FuncOp function) {
    function.walk([&](CallOp call) {
      auto callee = dyn_cast_or_null<FuncOp>(call.resolveCallable());
      if (callee && incoming.count(callee.getOperation())) {
        edges[function.getOperation()].push_back(callee.getOperation());
        ++incoming[callee.getOperation()];
      }
    });
  });
  SmallVector<Operation *> ready;
  for (auto &item : incoming)
    if (item.second == 0)
      ready.push_back(item.first);
  for (size_t index = 0; index < ready.size(); ++index)
    for (Operation *callee : edges[ready[index]])
      if (--incoming[callee] == 0)
        ready.push_back(callee);
  return ready.size() != incoming.size();
}

void appendNativePointerAxisFacts(
    FuncOp entry, llvm::json::Object &facts,
    const llvm::DenseMap<Operation *, std::string> &loopLocators) {
  auto module = entry->getParentOfType<ModuleOp>();
  if (nativeAxisCallGraphHasCycle(module)) {
    facts["native_pointer_axis_facts"] = nullptr;
    facts["native_pointer_axis_scope"] = "unavailable_recursive_module_call_graph";
    return;
  }
  ModuleAxisInfoAnalysis analysis(module);
  llvm::json::Array rows;
  int64_t ordinal = 0;
  entry.walk([&](Operation *op) {
    Value ptr, mask;
    if (auto load = dyn_cast<LoadOp>(op)) {
      ptr = load.getPtr(); mask = load.getMask();
    } else if (auto store = dyn_cast<StoreOp>(op)) {
      ptr = store.getPtr(); mask = store.getMask();
    }
    else return;
    llvm::json::Object row;
    row["memory_ordinal"] = ordinal++;
    row["operation"] = op->getName().getStringRef().str();
    llvm::json::Array ancestors;
    for (Operation *parent = op->getParentOp(); parent && parent != entry;
         parent = parent->getParentOp()) {
      auto found = loopLocators.find(parent);
      if (found != loopLocators.end()) ancestors.push_back(found->second);
    }
    // Innermost first; ordinary structural locators, never source provenance.
    row["loop_ancestry_innermost_first"] = std::move(ancestors);
    row["direct_for_body"] = isa<scf::ForOp>(op->getParentOp());
    row["mask_present"] = static_cast<bool>(mask);
    // Retain axis-wise source constancy, not the final layout-dependent
    // getMaskAlignment result. Activity bounds cannot reconstruct constancy.
    if (mask) {
      auto maskType = dyn_cast<RankedTensorType>(mask.getType());
      auto *maskInfo = analysis.getAxisInfo(mask);
      bool maskAvailable = maskType && maskInfo &&
          maskInfo->getConstancy().size() ==
              static_cast<size_t>(maskType.getRank());
      row["mask_axis_available"] = maskAvailable;
      if (maskAvailable) {
        llvm::json::Array constancy;
        for (int64_t n : maskInfo->getConstancy()) constancy.push_back(n);
        row["mask_constancy_elements"] = std::move(constancy);
      }
    }
    auto type = dyn_cast<RankedTensorType>(ptr.getType());
    row["scalar_pointer"] = isa<PointerType>(ptr.getType());
    Type scalar = type ? type.getElementType() : ptr.getType();
    if (auto pointer = dyn_cast<PointerType>(scalar)) {
      Type element = pointer.getPointeeType();
      if (element.isIntOrFloat())
        row["element_bits"] = element.getIntOrFloatBitWidth();
    }
    auto *info = analysis.getAxisInfo(ptr);
    bool available = type && info &&
        info->getContiguity().size() == static_cast<size_t>(type.getRank()) &&
        info->getDivisibility().size() == static_cast<size_t>(type.getRank());
    row["available"] = available;
    if (available) {
      llvm::json::Array shape, contiguity, divisibility;
      for (int64_t n : type.getShape()) shape.push_back(n);
      for (int64_t n : info->getContiguity()) contiguity.push_back(n);
      for (int64_t n : info->getDivisibility()) divisibility.push_back(n);
      row["pointer_shape"] = std::move(shape);
      row["contiguity_elements"] = std::move(contiguity);
      row["divisibility_bytes"] = std::move(divisibility);
    }
    rows.push_back(std::move(row));
  });
  facts["native_pointer_axis_facts"] = std::move(rows);
  facts["native_pointer_axis_scope"] =
      "entry_load_store_at_planning_cut_not_final_coalescing_or_mask_eligibility";
}

void appendMemoryAccessFacts(FuncOp entry, llvm::json::Object &facts) {
  auto programAffineAccesses = programAffineAccessFacts(entry);
  auto programMixedRadixAccesses = programMixedRadixAccessFacts(entry);
  SmallVector<RuntimeMaskScalarFact> runtimeMaskScalars;
  entry.walk([&](Operation *operation) {
    if (auto load = dyn_cast<LoadOp>(operation)) {
      if (auto fact = runtimeMaskScalarFromMask(load.getMask(), entry, false);
          fact && fact->complete)
        runtimeMaskScalars.push_back(*fact);
    } else if (auto store = dyn_cast<StoreOp>(operation)) {
      if (auto fact = runtimeMaskScalarFromMask(store.getMask(), entry, true);
          fact && fact->complete)
        runtimeMaskScalars.push_back(*fact);
    }
  });
  llvm::json::Array programAccesses;
  for (const IndependentAffineAccessFact &access : programAffineAccesses) {
    llvm::json::Object item;
    item["effect"] = access.write ? "write" : "read";
    item["root_argument"] = access.rootArgument;
    item["element_bytes"] = access.elementBytes;
    item["lane_count"] = access.laneCount;
    llvm::json::Array programStrides;
    llvm::json::Array programDependencies;
    llvm::json::Array programAffineComplete;
    for (int64_t stride : access.programAxisStrides)
      programStrides.push_back(stride);
    for (bool value : access.programAxisDependencies)
      programDependencies.push_back(value);
    for (bool value : access.programAxisAffineComplete)
      programAffineComplete.push_back(value);
    item["program_axis_strides"] = std::move(programStrides);
    item["program_axis_dependencies"] =
        std::move(programDependencies);
    item["program_axis_affine_complete"] =
        std::move(programAffineComplete);
    item["local_minimum"] = access.localMinimum;
    item["local_maximum"] = access.localMaximum;
    item["masked"] = access.masked;
    item["complete"] = access.complete;
    programAccesses.push_back(std::move(item));
  }
  facts["program_affine_accesses"] = std::move(programAccesses);
  llvm::json::Array runtimeMaskScalarFacts;
  for (const RuntimeMaskScalarFact &fact : runtimeMaskScalars) {
    llvm::json::Object item;
    item["effect"] = fact.write ? "write" : "read";
    item["argument_index"] = fact.argumentIndex;
    item["container_width"] = fact.containerWidth;
    item["mask_element_count"] = fact.maskElementCount;
    item["replication_factor"] = fact.replicationFactor;
    item["bound_value"] = fact.boundValue;
    item["divisibility"] = fact.divisibility;
    item["bound_value_known"] = fact.boundValueKnown;
    item["complete"] = fact.complete;
    runtimeMaskScalarFacts.push_back(std::move(item));
  }
  facts["runtime_mask_scalars"] = std::move(runtimeMaskScalarFacts);
  llvm::json::Array maskUpperFacts;
  int64_t memoryOrdinal = 0;
  entry.walk([&](Operation *op) {
    Value mask;
    bool write = false;
    if (auto load = dyn_cast<LoadOp>(op)) mask = load.getMask();
    else if (auto store = dyn_cast<StoreOp>(op)) {
      mask = store.getMask(); write = true;
    } else return;
    int64_t ordinal = memoryOrdinal++;
    if (!mask) return;
    int64_t elements = staticElementCount(mask.getType());
    if (elements < 1) return;
    unsigned budget = 256;
    auto upper = necessaryMaskActiveUpper(mask, elements, budget);
    llvm::json::Object item;
    item["memory_ordinal"] = ordinal;
    item["effect"] = write ? "write" : "read";
    item["container_elements"] = elements;
    item["active_elements_upper"] = upper.value_or(elements);
    item["necessary_condition_observed"] = upper.has_value();
    item["exact_activity_proven"] = false;
    item["scope"] = "per_memory_invocation_mask_upper_not_execution_count";
    maskUpperFacts.push_back(std::move(item));
  });
  facts["memory_mask_activity_upper_bounds"] = std::move(maskUpperFacts);
  llvm::json::Array mixedRadixAccessFacts;
  for (const ProgramMixedRadixAccessFact &fact :
       programMixedRadixAccesses) {
    llvm::json::Object item;
    item["effect"] = fact.write ? "write" : "read";
    item["root_argument"] = fact.rootArgument;
    item["element_bytes"] = fact.elementBytes;
    item["lane_count"] = fact.laneCount;
    item["program_axis"] = fact.programAxis;
    item["divisor"] = fact.divisor;
    item["quotient_stride"] = fact.quotientStride;
    item["remainder_stride"] = fact.remainderStride;
    item["local_minimum"] = fact.localMinimum;
    item["local_maximum"] = fact.localMaximum;
    item["masked"] = fact.masked;
    item["complete"] = fact.complete;
    mixedRadixAccessFacts.push_back(std::move(item));
  }
  facts["program_mixed_radix_accesses"] =
      std::move(mixedRadixAccessFacts);
}
