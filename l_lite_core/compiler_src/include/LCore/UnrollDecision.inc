// Close a current ordinary-loop plan before native factor unrolling.
LogicalResult closeOrdinaryUnroll(FuncOp entry, const lcore::Plan &plan,
                                  std::string &reason) {
  auto reject = [&](const Twine &message) {
    reason = message.str();
    return failure();
  };
  if (plan.pipeline || (plan.route != kPhaseRoute && plan.route != kLogicalRoute) ||
      plan.members.empty())
    return reject("ordinary unroll plan is inconsistent");
  auto capable = collectOrdinaryUnrollSubjects(entry, plan.route);
  SmallVector<ExistingLoopRouteSubject, 2> selected;
  std::set<Operation *> seen;
  for (const auto &member : plan.members) {
    auto found = llvm::find_if(capable, [&](const auto &subject) {
      return subject.providerLoopLocator == member.locator;
    });
    if (found == capable.end())
      return reject(Twine("ordinary unroll member is not route-capable: ") + member.locator);
    if (!seen.insert(found->loop.getOperation()).second)
      return reject("ordinary unroll member duplicated");
    auto trip = exactStaticTripCount(found->loop);
    bool tripMatches = member.trip > 0 ? trip && *trip == member.trip
        : !trip && member.runtimeCertificate == "native_dynamic_unroll_main_ordered_remainder_v1";
    if (!tripMatches || found->providerCapabilityCertificate != member.capability ||
        found->nestingDepth != member.depth || found->parentLoopLocator != member.parent)
      return reject(Twine("ordinary unroll facts changed: ") + member.locator);
    if (found->loop->hasAttr(kSubjectAttr))
      return reject("ordinary unroll member has stale or conflicting guidance");
    selected.push_back(*found);
    selected.back().boundMemberRef = member.reference;
    selected.back().boundRouteFactor = member.factor;
  }
  LoopDependenceCertificate nested;
  if (plan.nested) {
    if (selected.size() != 1)
      return reject("nested unroll requires one certified inner subject");
    nested = certifyNestedInnerDimension(selected.front().loop);
    if (!nested.safe || nested.kind != plan.members.front().context)
      return reject("ordinary unroll nested context changed");
  }
  // All guards above precede mutation, including guards on later members.
  Builder builder(entry.getContext());
  auto locator = builder.getDictionaryAttr({
      builder.getNamedAttr("adapter_version", builder.getI32IntegerAttr(plan.adapterVersion)),
      builder.getNamedAttr("decision_ref", builder.getStringAttr(plan.decision)),
      builder.getNamedAttr("route_ref", builder.getStringAttr(plan.route)),
      builder.getNamedAttr("subject_ref", builder.getStringAttr(plan.subject))});
  entry->setAttr(kRouteAttr, builder.getStringAttr(plan.route));
  entry->setAttr(kMechanismRouteAttr, builder.getStringAttr(plan.route));
  entry->setAttr(kArtifactRouteAttr, builder.getStringAttr(plan.route));
  entry->setAttr(kRouteSubtypeAttr, builder.getStringAttr("base"));
  entry->setAttr(kSubjectRefAttr, builder.getStringAttr(plan.subject));
  for (auto &subject : selected) {
    auto loop = subject.loop;
    recordReplacedLoopGuidance(entry, loop, subject.providerLoopLocator);
    // An unrolled main/tail may still contain loops: do not accidentally apply
    // the native default pipeline to the same selected route as well.
    loop->setAttr("tt.num_stages", builder.getI32IntegerAttr(1));
    auto member = builder.getStringAttr(subject.boundMemberRef);
    loop->setAttr(kSubjectAttr, locator);
    loop->setAttr("tt.loop_unroll_factor", builder.getI32IntegerAttr(subject.boundRouteFactor));
    auto trip = exactStaticTripCount(loop);
    if (trip)
      loop->setAttr(kSourceExactTripCountAttr, builder.getI64IntegerAttr(*trip));
    if (!trip || *trip != subject.boundRouteFactor)
      loop->setAttr(kMainTailAttr, builder.getStringAttr("native_factor_requested"));
    int64_t ordinal = 0;
    for (Operation &operation : loop.getBody()->without_terminator()) {
      operation.setAttr(kOperationGroupRefAttr, builder.getStringAttr(
          (Twine(subject.boundMemberRef) + ":op:" + Twine(ordinal++)).str()));
      operation.setAttr(kRoleSubjectAttr, member);
    }
    int64_t loadIndex = 0;
    for (Operation *operation : subject.loads) {
      operation->setAttr(kRoleAttr, builder.getStringAttr("load"));
      operation->setAttr(kRoleSubjectAttr, member);
      if (isa<LoadOp>(operation))
        operation->setAttr(kRoleIndexAttr, builder.getI32IntegerAttr(loadIndex++));
    }
    for (Operation *operation : subject.computes) {
      operation->setAttr(kRoleAttr, builder.getStringAttr("compute"));
      operation->setAttr(kRoleSubjectAttr, member);
    }
    for (Operation *operation : subject.states) {
      operation->setAttr(kRoleAttr, builder.getStringAttr("reduce"));
      operation->setAttr(kRoleSubjectAttr, member);
    }
  }
  StringRef aggregate = plan.route == kLogicalRoute
      ? "per_loop_exact_operation_vectorization_v3"
      : "per_loop_operation_neutral_topological_reorder_v1";
  entry->setAttr(kDependenceAttr, builder.getStringAttr(plan.nested ? nested.kind
      : selected.size() == 1 ? StringRef(selected.front().certificate.kind) : aggregate));
  return success();
}
