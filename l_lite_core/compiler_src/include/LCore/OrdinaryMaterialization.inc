// Three peer routes consume ordinary loop members. No Bridge origin, combined
// route, benchmark identity, prediction, or profitability enters this function.
// IR helper implementations are migrated separately.
LogicalResult materializeOrdinaryRoute(FuncOp function, const lcore::Plan &plan,
                                      std::string &reason) {
  auto reject = [&](StringRef message) {
    reason = message.str();
    return failure();
  };
  Builder builder(function.getContext());
  for (auto [a,b] : plan.guardedDisjointPairs)
    if (b>=function.getNumArguments() ||
        !isa<PointerType>(function.getArgument(a).getType()) ||
        !isa<PointerType>(function.getArgument(b).getType()))
      return reject("conditional memory pair does not bind pointer arguments");
  if (plan.members.empty()) return reject("ordinary route has no bound members");
  if (plan.pipeline) {
    SmallVector<scf::ForOp> subjects;
    function.walk([&](scf::ForOp loop) {
      if (loop->hasAttr(kSubjectAttr)) subjects.push_back(loop);
    });
    std::set<std::string> realized;
    if (subjects.size() != plan.members.size())
      return reject("native pipeline subject count changed");
    for (scf::ForOp subject : subjects) {
      auto stages = subject->getAttrOfType<IntegerAttr>("tt.num_stages");
      auto member = subject->getAttrOfType<StringAttr>(kPipelineMemberRefAttr);
      if (!stages || stages.getInt() != plan.stageCount || !member ||
          !realized.insert(member.getValue().str()).second)
        return reject("native pipeline member or shared stage request changed");
    }
    for (const auto &member : plan.members)
      if (!realized.count(member.reference))
        return reject("native pipeline lost a selected member");
    // This is the TTIR request attestation, not proof of backend async copies.
    function->setAttr(kRealizedAttr, builder.getStringAttr("software_pipeline"));
  } else {
    if (plan.route != kPhaseRoute && plan.route != kLogicalRoute)
      return reject("ordinary unroll route is not reorder or vectorization");
    auto dependence = function->getAttrOfType<StringAttr>(kDependenceAttr);
    if (!dependence ||
        !lcore::isOrdinaryUnrollCertificate(dependence.getValue().str()))
      return reject("ordinary unroll lost its dependence certificate");
    auto subjects = collectRoleSubjects(function);
    std::set<std::string> actual(subjects.begin(), subjects.end());
    std::set<std::string> expected;
    for (const auto &member : plan.members) expected.insert(member.reference);
    // Validate the entire binding before touching any member. A partial set
    // must never succeed merely because the remaining members can materialize.
    if (actual != expected || expected.size() != plan.members.size())
      return reject("ordinary unroll member set changed before materialization");
    bool hadMain = false, hadTail = false;
    function.walk([&](Operation *op) {
      auto partition = op->getAttrOfType<StringAttr>(kUnrollPartitionLineageAttr);
      hadMain |= partition && partition.getValue() == "main";
      hadTail |= partition && partition.getValue() == "tail";
    });
    for (const auto &member : plan.members) {
      std::optional<StringRef> filter = StringRef(member.reference);
      LogicalResult result = plan.route == kPhaseRoute
          ? materializeOperationNeutralPhase(function, filter, true, plan.guardedDisjointPairs)
          : materializeLogical(function, filter, member.factor, nullptr, &reason);
      if (failed(result)) {
        if (reason.empty()) reason = "ordinary operation groups cannot realize selected route";
        return failure();
      }
    }
    // Only after every member succeeds close transient unroll provenance.
    function.walk([&](Operation *op) {
      op->removeAttr(kRoleAttr);
      op->removeAttr(kRoleSubjectAttr);
      op->removeAttr(kRoleIndexAttr);
      op->removeAttr(kOperationGroupRefAttr);
      op->removeAttr(kUnrollPartitionLineageAttr);
    });
    bool livePartition = false;
    function.walk([&](scf::ForOp loop) {
      auto role = loop->getAttrOfType<StringAttr>(kMainTailAttr);
      livePartition |= role && (role.getValue() == "main" || role.getValue() == "tail");
    });
    bool mainTail = livePartition || (hadMain && hadTail);
    StringRef realized = mainTail
        ? (plan.route == kPhaseRoute ? "phase_major_factor_main_tail" : "logical_group_factor_main_tail")
        : (plan.route == kPhaseRoute ? "phase_major" : "logical_group");
    function->setAttr(kRealizedAttr, builder.getStringAttr(realized));
    function->setAttr(kProviderBoundMembersAttr, builder.getStringAttr(plan.memberSignature));
  }
  function->setAttr(kPostconditionAttr, builder.getStringAttr("pass"));
  if (!plan.memoryGuardRef.empty())
    function->setAttr("tt.hbv.l.runtime_memory_guard", builder.getStringAttr(plan.memoryGuardRef));
  return success();
}
