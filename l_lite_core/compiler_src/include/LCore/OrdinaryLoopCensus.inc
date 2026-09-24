// Ordinary route census. Source origin and legacy pattern categories confer no authority.
llvm::json::Object ordinaryLoopCensus(
    scf::ForOp loop, FuncOp entry,
    const llvm::DenseMap<Operation *, std::string> &planningCutLoopLocators,
    int64_t compilerDefaultStages) {
  LoopDependenceCertificate pipelineCapability =
      certifyNativeDynamicPipelineSubject(loop);
  auto reorderAnalysis = analyzeOrdinaryUnroll(loop, kPhaseRoute);
  auto vectorAnalysis = analyzeOrdinaryUnroll(loop, kLogicalRoute);
  const auto &reorderCapability = reorderAnalysis.certificate;
  bool vectorizationCapability = vectorAnalysis.certificate.safe;
  std::string reorderReason = reorderCapability.safe ? "" : reorderCapability.reason;
  std::string vectorizationReason = vectorizationCapability
      ? "" : vectorAnalysis.certificate.reason;
  llvm::json::Object census;
  census["locator"] = planningCutLoopLocators.lookup(
      loop.getOperation());
  census["entry_owned"] = loop->getParentOfType<FuncOp>() == entry;
  std::string location;
  llvm::raw_string_ostream locationStream(location);
  loop.getLoc().print(locationStream);
  locationStream.flush();
  census["source_location"] = location;
  int64_t nestingDepth = 0;
  for (Operation *parent = loop->getParentOp(); parent;
       parent = parent->getParentOp())
    nestingDepth += isa<scf::ForOp>(parent);
  census["nesting_depth"] = nestingDepth;
  auto parentLoop = loop->getParentOfType<scf::ForOp>();
  census["parent_locator"] =
      parentLoop
          ? planningCutLoopLocators.lookup(parentLoop.getOperation())
          : "";
  census["lower_kind"] =
      matchPattern(loop.getLowerBound(), m_Constant())
          ? "static"
          : "runtime";
  census["upper_kind"] =
      matchPattern(loop.getUpperBound(), m_Constant())
          ? "static"
          : "runtime";
  census["step_kind"] =
      matchPattern(loop.getStep(), m_Constant())
          ? "static"
          : "runtime";
  auto censusTripCount = exactStaticTripCount(loop);
  census["exact_static_trip_count_known"] = censusTripCount.has_value();
  census["exact_static_trip_count"] =
      censusTripCount ? *censusTripCount : 0;
  // Launch-bound trip is an observation for service modelling, not a
  // static constant or permission to specialize/unroll this loop. Keep
  // it separate from exact_static_trip_count and route legality.
  auto launchScalar = [](Value value) -> std::optional<int64_t> {
    unsigned budget = 128;
    return launchIntegerObservation(value, budget);
  };
  auto launchLower = launchScalar(loop.getLowerBound());
  auto launchUpper = launchScalar(loop.getUpperBound());
  auto launchStep = launchScalar(loop.getStep());
  std::optional<int64_t> launchTrip = censusTripCount;
  if (!launchTrip && launchLower && launchUpper && launchStep && *launchStep > 0) {
    __int128 distance = static_cast<__int128>(*launchUpper) - *launchLower;
    launchTrip = distance <= 0
                     ? std::optional<int64_t>(0)
                     : checkedI64((distance + *launchStep - 1) / *launchStep);
  }
  census["launch_trip_count_known"] = launchTrip.has_value();
  census["launch_trip_count"] = launchTrip.value_or(0);
  census["launch_trip_count_scope"] = "per_loop_invocation_if_reached";
  census["launch_trip_count_source"] = censusTripCount
      ? "exact_static_ir_bounds"
      : launchTrip
      ? "finite_width_ir_arithmetic_and_bound_entry_scalars"
      : "unresolved_bound_step_or_integer_range";
  census["carried_value_count"] =
      static_cast<int64_t>(loop.getInitArgs().size());
  appendOrdinaryLoopWorkFacts(loop, entry, census);
  census["native_pipeline_capable"] = pipelineCapability.safe;
  census["native_pipeline_capability_certificate"] =
      pipelineCapability.kind;
  census["native_pipeline_capability_reason"] =
      pipelineCapability.reason;
  // This certificate is deliberately scoped to the pre-TTGIR planning
  // cut.  At TTIR we can see a tensor-shaped load and a coarse payload
  // width, but not the distributed layout, per-thread contiguity, mask
  // alignment, or the backend copy-vector decision that determines
  // whether LowerLoops can emit async.copy/commit/wait.  Keeping the
  // scope explicit prevents a structural admission from being reported
  // as a realized backend pipeline capability.
  census["native_pipeline_capability_scope"] =
      "ttir_structural_pre_ttgir_copy_width_unknown_v1";
  census["native_pipeline_copy_width_observable"] = false;
  census["native_pipeline_copy_width_source"] =
      "backend_lowerloops_attestation";
  auto existingPipelineStages =
      loop->getAttrOfType<IntegerAttr>("tt.num_stages");
  census["preexisting_pipeline_stage_count"] =
      existingPipelineStages ? existingPipelineStages.getInt() : 0;
  census["full_unroll_reorder_capable"] =
      reorderCapability.safe;
  census["full_unroll_reorder_capability_certificate"] =
      reorderCapability.kind;
  census["full_unroll_reorder_capability_reason"] = reorderReason;
  census["full_unroll_vectorization_capable"] =
      vectorizationCapability;
  census["full_unroll_vectorization_capability_certificate"] =
      vectorAnalysis.providerCapabilityCertificate;
  census["full_unroll_vectorization_capability_reason"] =
      vectorizationReason;
  LoopDependenceCertificate nestedCapability =
      certifyNestedInnerDimension(loop);
  census["nested_inner_dimension_capable"] =
      nestedCapability.safe;
  census["nested_inner_dimension_certificate"] =
      nestedCapability.kind;
  census["nested_inner_dimension_reason"] =
      nestedCapability.reason;
  census["runtime_main_tail_certificate"] =
      !censusTripCount &&
              (reorderCapability.safe || vectorizationCapability)
          ? "native_dynamic_unroll_main_ordered_remainder_v1"
          : "";
  auto explicitStages =
      loop->getAttrOfType<IntegerAttr>("tt.num_stages");
  census["resolved_num_stages"] =
      explicitStages ? explicitStages.getInt() : compilerDefaultStages;
  return census;
}
