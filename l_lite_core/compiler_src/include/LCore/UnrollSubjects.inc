// Ordinary-loop route analysis. Capability comes from dependency/type/effect
// rules, never kernel names, Bridge origin, or training coverage.
struct ExistingLoopRouteSubject {
  scf::ForOp loop;
  std::string providerLoopLocator;
  std::string parentLoopLocator;
  std::string providerCapabilityCertificate;
  std::string boundMemberRef;
  int64_t nestingDepth = 0;
  int64_t boundRouteFactor = 0;
  SmallVector<Operation *> loads;
  SmallVector<Operation *> computes;
  SmallVector<Operation *> states;
  LoopDependenceCertificate certificate;
  bool coarseOrderPreserving = false;
};

// One analysis owner for census and decision revalidation.
ExistingLoopRouteSubject analyzeOrdinaryUnroll(scf::ForOp loop, StringRef route) {
  ExistingLoopRouteSubject result;
  result.loop = loop;
  if (route != kPhaseRoute && route != kLogicalRoute) {
    result.certificate = {false, "", "not_an_ordinary_unroll_route"};
    return result;
  }
  SmallVector<Operation *> loads;
  Operation *compute = nullptr, *reduce = nullptr;
  bool narrow = matchesOrderedAddShape(loop, loads, compute, reduce,
                                      /*requireStaticTripCount=*/false);
  LoopDependenceCertificate certificate;
  if (narrow && !loads.empty())
    if (auto load = dyn_cast<LoadOp>(loads.back()))
      certificate = certifyExistingUnrollReorder(loop, load, compute, reduce);
  bool narrowLogical = route != kLogicalRoute ||
      (certificate.safe && compute && compute->getNumResults() == 1 &&
       isRankOneIntegerVector(compute->getResult(0)));
  if (certificate.safe && narrowLogical) {
    result.providerCapabilityCertificate = route == kLogicalRoute
        ? "existing_rank1_integer_exact_grouping_v2" : certificate.kind;
    result.certificate = std::move(certificate);
    result.loads = std::move(loads);
    result.computes = {compute};
    result.states = {reduce};
    return result;
  }
  certificate = certifyOrderPreservingReadExposure(
      loop, result.loads, result.computes, result.states,
      /*requireVectorizableTensorLoad=*/route == kLogicalRoute);
  result.coarseOrderPreserving = true;
  if (!certificate.safe) {
    // Failed proof attempts must not leak operation roles into another proof.
    result.loads.clear();
    result.computes.clear();
    result.states.clear();
    certificate = route == kLogicalRoute
        ? certifyOperationNeutralExactPacking(loop)
        : certifyOperationNeutralFullUnrollReorder(loop);
    if (certificate.safe)
      for (Operation &op : loop.getBody()->without_terminator())
        result.computes.push_back(&op);
  }
  result.providerCapabilityCertificate = certificate.safe ? certificate.kind : "";
  result.certificate = std::move(certificate);
  return result;
}

SmallVector<ExistingLoopRouteSubject, 2>
collectOrdinaryUnrollSubjects(FuncOp entry, StringRef route) {
  // Decisions bind entry-owned loops only. Reachable helper observations do
  // not authorize rewriting callees before ordinary inlining has exposed them.
  auto locators = planningLoopLocators(ArrayRef<FuncOp>(entry));
  SmallVector<ExistingLoopRouteSubject, 2> eligible;
  entry.walk([&](scf::ForOp loop) {
    auto result = analyzeOrdinaryUnroll(loop, route);
    if (!result.certificate.safe)
      return;
    result.providerLoopLocator = locators.lookup(loop.getOperation());
    auto parentLoop = loop->getParentOfType<scf::ForOp>();
    result.parentLoopLocator = parentLoop ? locators.lookup(parentLoop.getOperation()) : "";
    for (Operation *parent = loop->getParentOp(); parent; parent = parent->getParentOp())
      result.nestingDepth += isa<scf::ForOp>(parent);
    eligible.push_back(std::move(result));
  });
  return eligible;
}
