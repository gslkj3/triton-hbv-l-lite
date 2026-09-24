// One exact additive recurrence analysis, not general route admission.
// Callers must retain the independent read-exposure and neutral rules.
bool matchesOrderedAddShape(scf::ForOp loop, SmallVectorImpl<Operation *> &loadSlice,
                    Operation *&compute, Operation *&reduce,
                    bool requireStaticTripCount) {
  auto trip = exactStaticTripCount(loop);
  auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
  if ((requireStaticTripCount && !trip) || (trip && *trip < 2) ||
      loop.getInitArgs().size() != 1 ||
      !yield || yield.getNumOperands() != 1)
    return false;
  Value accumulator = loop.getRegionIterArgs().front();
  Operation *combiner = yield.getOperand(0).getDefiningOp();
  if (!combiner || !isa<arith::AddIOp, arith::AddFOp>(combiner) ||
      combiner->getNumOperands() != 2 || combiner->getNumResults() != 1)
    return false;
  Value contribution;
  if (combiner->getOperand(0) == accumulator)
    contribution = combiner->getOperand(1);
  else if (combiner->getOperand(1) == accumulator)
    contribution = combiner->getOperand(0);
  else
    return false;
  compute = contribution.getDefiningOp();
  reduce = combiner;
  if (!compute || compute->getBlock() != loop.getBody() ||
      valueDependsOn(contribution, accumulator))
    return false;

  SmallVector<LoadOp> loads;
  bool invalid = false;
  loop.walk([&](Operation *operation) {
    if (operation != loop.getOperation() && isa<scf::ForOp>(operation)) {
      invalid = true;
      return;
    }
    if (auto load = dyn_cast<LoadOp>(operation)) {
      invalid |= load.getIsVolatile() ||
                 valueDependsOn(load.getPtr(), accumulator) ||
                 (load.getMask() &&
                  valueDependsOn(load.getMask(), accumulator)) ||
                 (load.getOther() &&
                  valueDependsOn(load.getOther(), accumulator));
      loads.push_back(load);
      return;
    }
    if (isa<StoreOp, AtomicRMWOp, AtomicCASOp>(operation)) {
      invalid = true;
      return;
    }
    if (operation != loop.getOperation() &&
        !operation->hasTrait<OpTrait::IsTerminator>() &&
        !isMemoryEffectFree(operation))
      invalid = true;
  });
  if (invalid || loads.empty() ||
      llvm::none_of(loads, [&](LoadOp load) {
        return valueDependsOn(contribution, load.getResult());
      }))
    return false;

  llvm::SmallPtrSet<Operation *, 32> slice;
  SmallVector<Operation *> worklist;
  for (LoadOp load : loads) {
    slice.insert(load.getOperation());
    worklist.push_back(load.getOperation());
  }
  while (!worklist.empty()) {
    Operation *operation = worklist.pop_back_val();
    for (Value operand : operation->getOperands()) {
      Operation *definition = operand.getDefiningOp();
      if (!definition || definition->getBlock() != loop.getBody() ||
          slice.contains(definition))
        continue;
      if (!isMemoryEffectFree(definition))
        return false;
      slice.insert(definition);
      worklist.push_back(definition);
    }
  }
  for (Operation &operation : loop.getBody()->without_terminator())
    if (slice.contains(&operation))
      loadSlice.push_back(&operation);
  return !loadSlice.empty();
}
