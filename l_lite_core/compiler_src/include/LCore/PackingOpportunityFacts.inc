// Structural packing opportunities, not whole-route legality or backend realization.
void appendPackingOpportunityFacts(FuncOp entry, llvm::json::Object &facts) {
  int64_t exactGroupedLoadAdapterOpportunityCount = 0;
  int64_t exactGroupedStoreAdapterOpportunityCount = 0;
  int64_t exactElementwiseOperationPackableCount = 0;
  int64_t exactGroupedValueRoundTripCount = 0;
  entry.walk([&](LoadOp load) {
    exactGroupedLoadAdapterOpportunityCount +=
        isa<RankedTensorType>(load.getPtr().getType()) &&
        load.getBoundaryCheck().empty() && !load.getPadding();
    // This is the exact pre-materialization counterpart of the
    // split-tree recovery in materializeBridgeLogical: one tensor load
    // result is consumed directly and exclusively as a tensor store value.
    // Cloning independent virtual programs therefore creates a split tree
    // whose complete ordered leaves meet at the same grouped terminal.
    // No operator, shape, factor or performance identity is involved.
    if (!isa<RankedTensorType>(load.getResult().getType()) ||
        !load.getBoundaryCheck().empty() || load.getPadding() ||
        !load.getResult().hasOneUse())
      return;
    Operation *user = *load.getResult().getUsers().begin();
    auto store = dyn_cast<StoreOp>(user);
    if (store && store.getValue() == load.getResult() &&
        isa<RankedTensorType>(store.getPtr().getType()) &&
        store.getBoundaryCheck().empty())
      ++exactGroupedValueRoundTripCount;
  });
  entry.walk([&](StoreOp store) {
    exactGroupedStoreAdapterOpportunityCount +=
        isa<RankedTensorType>(store.getPtr().getType()) &&
        store.getBoundaryCheck().empty();
  });
  entry.walk([&](Operation *operation) {
    exactElementwiseOperationPackableCount +=
        isGenericExactPackableOperation(operation);
  });
  bool exactGroupedLoadEligible =
      exactGroupedLoadAdapterOpportunityCount > 0;
  bool exactGroupedStoreEligible =
      exactGroupedStoreAdapterOpportunityCount > 0;
  bool logicalExactOperationPackingEligible =
      exactGroupedLoadEligible || exactGroupedStoreEligible ||
      exactElementwiseOperationPackableCount > 0;
  // Historical field retained as a compatibility projection.  Its current
  // authority is the operation-capability union above, not load presence.
  bool tensorLaneFusionEligible =
      logicalExactOperationPackingEligible;
  facts["tensor_lane_fusion_eligible"] = tensorLaneFusionEligible;
  facts["logical_exact_operation_packing_eligible"] =
      logicalExactOperationPackingEligible;
  facts["exact_grouped_load_adapter_eligible"] =
      exactGroupedLoadEligible;
  facts["exact_grouped_store_adapter_eligible"] =
      exactGroupedStoreEligible;
  facts["exact_grouped_load_adapter_opportunity_count"] =
      exactGroupedLoadAdapterOpportunityCount;
  facts["exact_grouped_store_adapter_opportunity_count"] =
      exactGroupedStoreAdapterOpportunityCount;
  facts["exact_elementwise_operation_packable_count"] =
      exactElementwiseOperationPackableCount;
  facts["exact_grouped_value_round_trip_count"] =
      exactGroupedValueRoundTripCount;
}
