// Ordinary loop-local observations, independent of Bridge origin or route.
void appendOrdinaryLoopWorkFacts(scf::ForOp loop, FuncOp entry,
                                llvm::json::Object &census) {
  auto function = loop->getParentOfType<FuncOp>();
  // Pass-service opportunity facts are counted on the exact source
  // subject that the route will mutate.  They are operation-neutral:
  // no kernel, operator family or workload identity enters the census.
  // The later route attestation separately records what was actually
  // materialized, so an opportunity count is never treated as proof of
  // a realized effect.
  int64_t sourceOperationGroupCount = 0;
  int64_t phaseBarrierOperationCount = 0;
  int64_t logicalElementwisePackableCount = 0;
  int64_t logicalLoadAdapterOpportunityCount = 0;
  int64_t logicalStoreAdapterOpportunityCount = 0;
  int64_t iterationInvariantStoreAddressCount = 0;
  int64_t explicitSynchronizationBarrierCount = 0;
  int64_t directLoadContainerBytes = 0;
  int64_t directStoreContainerBytes = 0;
  int64_t opaqueBodyRegionCount = 0;
  int64_t memoryOpaqueBodyRegionCount = 0;
  bool unconditionalEntryAncestry = function == entry;
  for (Operation *parent = loop->getParentOp();
       parent && parent != function.getOperation();
       parent = parent->getParentOp())
    unconditionalEntryAncestry &= isa<scf::ForOp>(parent);
  for (Operation &operation : loop.getBody()->without_terminator()) {
    // Direct body only: nested loops have their own census row, avoiding
    // double counting. These are issued container bytes, not cache/DRAM
    // traffic or mask-active byte counts.
    if (auto load = dyn_cast<LoadOp>(operation))
      directLoadContainerBytes += staticValueBytes(load.getResult());
    if (auto store = dyn_cast<StoreOp>(operation))
      directStoreContainerBytes += staticValueBytes(store.getValue());
    if (auto store = dyn_cast<StoreOp>(operation)) {
      // A carried pointer or an indirect memory read may change without
      // referring to the IV. Only a regionless, effect-free backward
      // slice rooted outside the loop proves address invariance.
      // Failure is unknown, not a proof of varying/disjoint addresses.
      SmallVector<Value> pending{store.getPtr()};
      llvm::SmallPtrSet<Value, 32> visited;
      bool invariant = true;
      unsigned budget = 512;
      while (!pending.empty() && invariant) {
        Value value = pending.pop_back_val();
        if (!visited.insert(value).second ||
            loop.isDefinedOutsideOfLoop(value))
          continue;
        if (!budget--) { invariant = false; break; }
        Operation *definition = value.getDefiningOp();
        if (!definition || definition->getNumRegions() ||
            !isMemoryEffectFree(definition)) {
          invariant = false;
          break;
        }
        llvm::append_range(pending, definition->getOperands());
      }
      iterationInvariantStoreAddressCount += invariant;
    }
    explicitSynchronizationBarrierCount +=
        operation.getName().getStringRef() == "gpu.barrier";
    opaqueBodyRegionCount += operation.getNumRegions() != 0 &&
                             !isa<scf::ForOp>(operation);
    if (operation.getNumRegions() && !isa<scf::ForOp>(operation)) {
      // Region arithmetic is not unknown memory work. Prove absence of
      // memory effects recursively; do not whitelist reduction/scan names.
      bool memoryFree = true;
      unsigned remaining = 4096;
      operation.walk([&](Operation *nested) -> WalkResult {
        if (!remaining--) {
          memoryFree = false;
          return WalkResult::interrupt();
        }
        memoryFree &= isMemoryEffectFree(nested);
        return WalkResult::advance();
      });
      memoryOpaqueBodyRegionCount += !memoryFree;
    } else if (!operation.getNumRegions() && !isa<LoadOp, StoreOp>(operation) &&
               operation.getName().getStringRef() != "gpu.barrier" &&
               !isMemoryEffectFree(&operation)) {
      // Calls and other unclassified effects are not zero memory work merely
      // because they own no region. Direct counts above remain direct; until
      // an inclusive body summary is bound, mark their work unresolved.
      ++memoryOpaqueBodyRegionCount;
    }
    ++sourceOperationGroupCount;
    phaseBarrierOperationCount +=
        !isMemoryEffectFree(&operation) || !isSpeculatable(&operation);
    logicalElementwisePackableCount +=
        isGenericExactPackableOperation(&operation);
    if (auto load = dyn_cast<LoadOp>(operation))
      logicalLoadAdapterOpportunityCount +=
          isa<RankedTensorType>(load.getPtr().getType()) &&
          load.getBoundaryCheck().empty() && !load.getPadding();
    if (auto store = dyn_cast<StoreOp>(operation))
      logicalStoreAdapterOpportunityCount +=
          isa<RankedTensorType>(store.getPtr().getType()) &&
          store.getBoundaryCheck().empty();
  }
  census["source_operation_group_count"] = sourceOperationGroupCount;
  census["direct_load_container_bytes"] = directLoadContainerBytes;
  census["direct_store_container_bytes"] = directStoreContainerBytes;
  census["opaque_body_region_count"] = opaqueBodyRegionCount;
  census["memory_opaque_body_region_count"] = memoryOpaqueBodyRegionCount;
  census["unconditional_entry_loop_ancestry"] = unconditionalEntryAncestry;
  census["direct_container_work_observed"] = true;
  // Inclusive per-invocation body work is a different observation from direct
  // load/store counts. Preserve control and call boundaries in the summary.
  unsigned bodyBudget = 4096;
  auto bodyWork = structuredContainerBlock(*loop.getBody(), bodyBudget, 0);
  unsigned treeBudget = 4096;
  if (!boundedContainerTree(bodyWork, treeBudget))
    bodyWork = llvm::json::Object{{"kind", "unknown"}};
  census["structured_body_container_work"] = std::move(bodyWork);
  census["phase_barrier_operation_count"] =
      phaseBarrierOperationCount;
  census["native_visible_async_load_count"] =
  nativeVisibleAsyncLoadServiceCount(loop);
  census["logical_elementwise_packable_operation_count"] =
      logicalElementwisePackableCount;
  census["logical_load_adapter_opportunity_count"] =
      logicalLoadAdapterOpportunityCount;
  census["logical_store_adapter_opportunity_count"] =
      logicalStoreAdapterOpportunityCount;
  census["direct_memory_order_constraints_observed"] = true;
  census["iteration_invariant_store_address_count"] =
      iterationInvariantStoreAddressCount;
  census["explicit_synchronization_barrier_count"] =
      explicitSynchronizationBarrierCount;
}
