// Optional unique-subject summary. Per-loop census remains authoritative.
struct PipelineSummary {
  int64_t count = 0;
  int64_t stages = 0;
  std::optional<int64_t> trip;
  std::string certificate;

  void observe(scf::ForOp loop, int64_t compilerDefaultStages) {
    auto proof = certifyNativeDynamicPipelineSubject(loop);
    if (!proof.safe)
      return;
    auto explicitStages = loop->getAttrOfType<IntegerAttr>("tt.num_stages");
    int64_t effectiveStages = explicitStages ? explicitStages.getInt()
                                           : compilerDefaultStages;
    if (effectiveStages <= 1)
      return;
    ++count;
    if (count == 1) {
      stages = effectiveStages;
      trip = exactStaticTripCount(loop);
      certificate = proof.kind;
    } else {
      stages = 0;
      trip.reset();
      certificate.clear();
    }
  }

  void append(llvm::json::Object &facts) const {
    facts["native_pipeline_scope_count"] = count;
    facts["native_pipeline_candidate_count"] = count;
    // Historical field spelling is not full-unroll or backend authorization.
    facts["native_pipeline_unroll_legal"] = count == 1;
    facts["native_pipeline_certificate"] = certificate;
    facts["native_trip_count_known"] = trip.has_value();
    facts["native_trip_count"] = trip.value_or(0);
    facts["native_default_stages"] = stages;
    facts["native_pipeline_summary_scope"] =
        "unique_ttir_structural_subject_not_backend_attestation";
  }
};
