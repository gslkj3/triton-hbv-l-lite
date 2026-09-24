// Bind a selected ordinary-loop plan to native pipeline guidance. This is not
// candidate ranking and has no access to Bridge origin or prediction models.
LogicalResult closeOrdinaryPipeline(FuncOp entry, const lcore::Plan &plan,
                                    std::string &reason) {
  auto reject = [&](const Twine &message) {
    reason = message.str();
    return failure();
  };
  if (!plan.pipeline || plan.route != kPipelineRoute || plan.stageCount < 2)
    return reject("ordinary pipeline plan is inconsistent");
  // Decisions bind entry-owned loops only. Reachable helper observations do
  // not authorize rewriting callees before ordinary inlining has exposed them.
  auto locators = planningLoopLocators(ArrayRef<FuncOp>(entry));
  SmallVector<std::pair<scf::ForOp, LoopDependenceCertificate>> eligible;
  entry.walk([&](scf::ForOp loop) {
    auto certificate = certifyNativeDynamicPipelineSubject(loop);
    if (certificate.safe) eligible.emplace_back(loop, std::move(certificate));
  });
  if (eligible.size() != plan.members.size() || eligible.empty())
    return reject("ordinary pipeline plan does not bind the complete capable set");
  SmallVector<scf::ForOp> selected;
  std::set<Operation *> seen;
  for (const lcore::Member &member : plan.members) {
    auto found = llvm::find_if(eligible, [&](const auto &candidate) {
      scf::ForOp loop = candidate.first;
      return locators.lookup(loop.getOperation()) == member.locator;
    });
    if (found == eligible.end())
      return reject(Twine("ordinary pipeline member absent: ") + member.locator);
    scf::ForOp loop = found->first;
    if (!seen.insert(loop.getOperation()).second)
      return reject("ordinary pipeline member duplicated");
    auto trip = exactStaticTripCount(loop);
    bool tripMatches = member.trip > 0 ? trip && *trip == member.trip
        : !trip && member.runtimeCertificate == "native_dynamic_unroll_main_ordered_remainder_v1";
    int64_t depth = 0;
    for (Operation *parent = loop->getParentOp(); parent; parent = parent->getParentOp())
      depth += isa<scf::ForOp>(parent);
    auto parent = loop->getParentOfType<scf::ForOp>();
    std::string parentLocator = parent ? locators.lookup(parent.getOperation()) : "";
    if (!tripMatches || found->second.kind != member.capability ||
        member.depth != depth || member.parent != parentLocator ||
        member.factor != plan.stageCount)
      return reject(Twine("ordinary pipeline facts changed: ") + member.locator);
    if (loop->hasAttr(kSubjectAttr))
      return reject("ordinary pipeline member has stale or conflicting guidance");
    selected.push_back(loop);
  }
  // No mutation occurs until every member has passed the same checks.
  Builder builder(entry.getContext());
  auto locator = builder.getDictionaryAttr({
      builder.getNamedAttr("adapter_version", builder.getI32IntegerAttr(plan.adapterVersion)),
      builder.getNamedAttr("decision_ref", builder.getStringAttr(plan.decision)),
      builder.getNamedAttr("route_ref", builder.getStringAttr(plan.route)),
      builder.getNamedAttr("subject_ref", builder.getStringAttr(plan.subject))});
  for (auto [loop, member] : llvm::zip(selected, plan.members)) {
    recordReplacedLoopGuidance(entry, loop, member.locator);
    loop->removeAttr("tt.loop_unroll_factor");
    loop->setAttr(kSubjectAttr, locator);
    loop->setAttr("tt.num_stages", builder.getI32IntegerAttr(plan.stageCount));
    loop->setAttr(kPipelineMemberRefAttr, builder.getStringAttr(member.reference));
  }
  entry->setAttr(kRouteAttr, builder.getStringAttr(plan.route));
  entry->setAttr(kMechanismRouteAttr, builder.getStringAttr(plan.route));
  entry->setAttr(kArtifactRouteAttr, builder.getStringAttr(plan.route));
  entry->setAttr(kRouteSubtypeAttr, builder.getStringAttr("base"));
  entry->setAttr(kSubjectRefAttr, builder.getStringAttr(plan.subject));
  entry->setAttr(kDependenceAttr, builder.getStringAttr("provider_bound_pipeline_subject_set_v1"));
  entry->setAttr(kProviderBoundMembersAttr, builder.getStringAttr(plan.memberSignature));
  return success();
}
