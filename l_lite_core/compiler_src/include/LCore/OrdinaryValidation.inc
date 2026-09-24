// TTIR postcondition validation, not numerical or backend performance evidence.
LogicalResult validateOrdinaryRoute(FuncOp function, const lcore::Plan &plan,
                                   std::string &reason) {
  auto reject = [&](StringRef message) {
    reason = message.str();
    return failure();
  };
  auto equals = [&](StringRef attribute, StringRef expected) {
    auto actual = function->getAttrOfType<StringAttr>(attribute);
    return actual && actual.getValue() == expected;
  };
  auto memoryGuard = function->getAttrOfType<StringAttr>("tt.hbv.l.runtime_memory_guard");
  if (plan.memoryGuardRef.empty() ? static_cast<bool>(memoryGuard)
      : (!memoryGuard || memoryGuard.getValue()!=plan.memoryGuardRef))
    return reject("conditional memory runtime obligation changed");
  if (!equals(kRouteAttr, plan.route) || !equals(kMechanismRouteAttr, plan.route) ||
      !equals(kArtifactRouteAttr, plan.route) || !equals(kRouteSubtypeAttr, "base") ||
      !equals(kSubjectRefAttr, plan.subject) || !equals(kPostconditionAttr, "pass") ||
      !equals(kProviderBoundMembersAttr, plan.memberSignature))
    return reject("ordinary route identity or member postcondition changed");
  bool transient = false;
  function.walk([&](Operation *op) {
    transient |= op->hasAttr(kRoleAttr) || op->hasAttr(kRoleSubjectAttr) ||
                 op->hasAttr(kRoleIndexAttr) || op->hasAttr(kOperationGroupRefAttr) ||
                 op->hasAttr(kUnrollPartitionLineageAttr);
  });
  if (transient)
    return reject("ordinary route retains transient operation lineage");
  if (plan.pipeline) {
    if (!equals(kRealizedAttr, "software_pipeline") ||
        !equals(kDependenceAttr, "provider_bound_pipeline_subject_set_v1"))
      return reject("ordinary pipeline realization or dependence changed");
    std::set<std::string> members;
    bool invalid = false;
    function.walk([&](scf::ForOp loop) {
      auto ref = loop->getAttrOfType<StringAttr>(kPipelineMemberRefAttr);
      if (!ref && !loop->hasAttr(kSubjectAttr)) return;
      auto stages = loop->getAttrOfType<IntegerAttr>("tt.num_stages");
      invalid |= !ref || !stages || stages.getInt() != plan.stageCount ||
                 !loop->hasAttr(kSubjectAttr) || loop->hasAttr("tt.loop_unroll_factor");
      if (ref) invalid |= !members.insert(ref.getValue().str()).second;
    });
    if (invalid || members.size() != plan.members.size())
      return reject("ordinary pipeline member stage or cardinality changed");
    for (const auto &member : plan.members)
      if (!members.count(member.reference))
        return reject("ordinary pipeline selected member missing");
    return success();
  }
  auto dependence = function->getAttrOfType<StringAttr>(kDependenceAttr);
  if (!dependence || !lcore::isOrdinaryUnrollCertificate(dependence.getValue().str()))
    return reject("ordinary unroll dependence certificate missing");
  auto positiveCount = [&](StringRef attribute) {
    auto count = function->getAttrOfType<IntegerAttr>(attribute);
    return count && count.getInt() > 0;
  };
  if (plan.route == kPhaseRoute) {
    if ((!equals(kRealizedAttr, "phase_major") &&
         !equals(kRealizedAttr, "phase_major_factor_main_tail")) ||
        !positiveCount(kPhaseOperationGroupCountAttr))
      return reject("ordinary reorder lacks materialized source groups");
  } else if (plan.route == kLogicalRoute) {
    if ((!equals(kRealizedAttr, "logical_group") &&
         !equals(kRealizedAttr, "logical_group_factor_main_tail")) ||
        (!positiveCount(kLogicalReductionGroupCountAttr) &&
         !positiveCount(kVectorizedLoadGroupCountAttr) &&
         !positiveCount(kVectorizedOperationGroupCountAttr)))
      return reject("ordinary vectorization lacks materialized groups");
  } else {
    return reject("ordinary postcondition has an unknown route");
  }
  return success();
}
