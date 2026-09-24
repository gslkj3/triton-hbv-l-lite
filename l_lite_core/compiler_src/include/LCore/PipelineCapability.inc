int64_t nativeVisibleAsyncLoadServiceCount(scf::ForOp loop) {
  int64_t count = 0;
  for (Operation &operation : loop.getBody()->without_terminator()) {
    if (isa<DescriptorLoadOp, DescriptorGatherOp>(operation)) {
      ++count;
      continue;
    }
    auto load = dyn_cast<LoadOp>(operation);
    if (!load)
      continue;
    // This is TTIR candidate admission only.  Distributed layout and the
    // final async-copy conversion are not available at this stage; require
    // one 32-bit payload unit without pretending this proves realization.
    Value pointer = load.getPtr();
    Type pointee;
    int64_t lanes = 1;
    if (auto tensor = dyn_cast<RankedTensorType>(pointer.getType())) {
      auto pointerType = dyn_cast<PointerType>(tensor.getElementType());
      if (!pointerType)
        continue;
      pointee = pointerType.getPointeeType();
      lanes = tensor.getNumElements();
    } else {
      auto pointerType = dyn_cast<PointerType>(pointer.getType());
      if (!pointerType)
        continue;
      pointee = pointerType.getPointeeType();
    }
    if (lanes > 0 &&
        lanes * static_cast<int64_t>(pointee.getIntOrFloatBitWidth()) >= 32)
      ++count;
  }
  return count;
}

bool hasNativeVisibleAsyncLoadService(scf::ForOp loop) {
  return nativeVisibleAsyncLoadServiceCount(loop) > 0;
}

// Candidate-side counterpart of the native GPU pipeliner's dynamic-loop
// contract.  The native expander runs with supportDynamicLoops=true and
// predicates its prologue, steady kernel and epilogue against the runtime
// upper bound.  Therefore an unknown trip count is not a rejection reason.
// This planning-cut certificate intentionally proves only the structural
// conditions visible before TTGIR scheduling; final async realization remains
// owned by the native pipeline and post-codegen attestation.
LoopDependenceCertificate certifyNativeDynamicPipelineSubject(
    scf::ForOp loop) {
  bool hasNestedLoop = false;
  bool hasTopLevelLoad = hasNativeVisibleAsyncLoadService(loop);
  bool hasAnyTopLevelLoad = llvm::any_of(
      loop.getBody()->without_terminator(), [](Operation &operation) {
        return isa<LoadOp, DescriptorLoadOp, DescriptorGatherOp>(operation);
      });
  loop.walk([&](Operation *operation) {
    // Native isOuterLoop checks direct for/while children. Our innermost-only
    // project boundary additionally excludes loops nested inside regions.
    // This is a conservative structural restriction, not an exact simulation
    // of the complete native scheduling decision.
    if (operation != loop.getOperation() &&
        isa<scf::ForOp, scf::WhileOp>(operation))
      hasNestedLoop = true;
  });
  // AssignLoadLatencies schedules only a candidate load service rooted
  // directly in the loop body.  This planning-cut fact does not claim a final
  // per-thread copy width; final async artifact attestation owns that fact.
  if (hasNestedLoop)
    return {false, "", "nested_loop_requires_independent_pipeline_subject"};
  if (!hasTopLevelLoad)
    return {
        false, "",
        hasAnyTopLevelLoad
            ? "pipeline_source_payload_below_minimum_async_transfer_unit"
            : "pipeline_subject_has_no_native_visible_async_load_service"};
  bool hasUnsupportedDistance = llvm::any_of(
      loop.getBody()->getTerminator()->getOperands(),
      [](Value operand) { return !operand.getDefiningOp(); });
  if (hasUnsupportedDistance)
    return {false, "", "pipeline_loop_carried_distance_exceeds_one"};
  return {true,
          "native_dynamic_pipeline_predicated_prologue_epilogue_v1", ""};
}
