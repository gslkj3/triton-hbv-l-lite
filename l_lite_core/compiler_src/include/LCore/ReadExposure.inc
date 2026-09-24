bool dependsOnAny(Value value, ValueRange targets) {
  return llvm::any_of(targets,
                      [&](Value target) { return valueDependsOn(value, target); });
}

// A pointer induction is semantic state, but not data-dependent state.  Moving
// the cloned address/read slice after unrolling is legal when the next pointer
// is exactly ``addptr(carried, invariant_step)`` and the pointer lineage ends
// only at reads or that matching yield.  This covers the ordinary Triton
// matmul idiom without naming dot, matmul, a benchmark, or a backend schedule.
bool isAffineReadOnlyPointerInduction(scf::ForOp loop, unsigned index) {
  if (index >= loop.getRegionIterArgs().size())
    return false;
  auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
  if (!yield || yield.getNumOperands() != loop.getRegionIterArgs().size())
    return false;
  Value carried = loop.getRegionIterArg(index);
  Value next = yield.getOperand(index);
  auto update = next.getDefiningOp<AddPtrOp>();
  if (!update || update.getPtr() != carried ||
      dependsOnAny(update.getOffset(), loop.getRegionIterArgs()))
    return false;

  llvm::SmallPtrSet<Value, 32> visited;
  SmallVector<Value> worklist{carried};
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *owner = use.getOwner();
      if (auto load = dyn_cast<LoadOp>(owner)) {
        if (use.get() != load.getPtr())
          return false;
        continue;
      }
      if (auto addPtr = dyn_cast<AddPtrOp>(owner)) {
        if (use.get() != addPtr.getPtr() ||
            dependsOnAny(addPtr.getOffset(), loop.getRegionIterArgs()))
          return false;
        llvm::append_range(worklist, addPtr->getResults());
        continue;
      }
      StringRef name = owner->getName().getStringRef();
      if (name == "tt.splat" || name == "tt.broadcast" ||
          name == "tt.expand_dims" || name == "tt.reshape" ||
          name == "tt.trans") {
        llvm::append_range(worklist, owner->getResults());
        continue;
      }
      if (owner == yield.getOperation() && use.getOperandNumber() == index &&
          value == next)
        continue;
      return false;
    }
  }
  return true;
}

bool hasOnlyAffineReadOnlyPointerDependencies(Value pointer,
                                              scf::ForOp loop) {
  for (auto [index, carried] : llvm::enumerate(loop.getRegionIterArgs()))
    if (valueDependsOn(pointer, carried) &&
        !isAffineReadOnlyPointerInduction(loop, index))
      return false;
  return true;
}

// Prove the coarser causal boundary needed when the source recurrence is not
// one exact add.  The route may expose read/address work from later iterations
// before earlier state updates, but it never changes the order of those state
// updates.  Consequently this proof deliberately does not identify max/argmax,
// dot, or any workload-specific combiner: it proves only that every moved
// read is non-volatile, independent of data-carrying state (or depends only on
// a proved affine read-only pointer induction), and cannot cross a write or
// unknown effect.
LoopDependenceCertificate certifyOrderPreservingReadExposure(
    scf::ForOp loop, SmallVectorImpl<Operation *> &loadSlice,
    SmallVectorImpl<Operation *> &computeSlice,
    SmallVectorImpl<Operation *> &stateSlice,
    bool requireVectorizableTensorLoad) {
  loadSlice.clear();
  computeSlice.clear();
  stateSlice.clear();
  auto trip = exactStaticTripCount(loop);
  if (trip && *trip < 2)
    return {false, "", "iteration domain has fewer than two iterations"};
  auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
  if (!yield || yield.getNumOperands() != loop.getInitArgs().size())
    return {false, "", "loop-carried result arity is not closed"};

  SmallVector<LoadOp> loads;
  bool hasAffinePointerInduction = false;
  bool rejected = false;
  std::string reason;
  loop.walk([&](Operation *operation) {
    if (rejected || operation == loop.getOperation())
      return;
    if (isa<scf::ForOp>(operation)) {
      rejected = true;
      reason = "nested scf.for requires a separately composed certificate";
      return;
    }
    if (auto load = dyn_cast<LoadOp>(operation)) {
      if (load->getBlock() != loop.getBody()) {
        rejected = true;
        reason = "nested-region load cannot be phase-exposed";
        return;
      }
      bool pointerDependsOnCarried =
          dependsOnAny(load.getPtr(), loop.getRegionIterArgs());
      if (load.getIsVolatile() ||
          (pointerDependsOnCarried &&
           !hasOnlyAffineReadOnlyPointerDependencies(load.getPtr(), loop)) ||
          (load.getMask() &&
           dependsOnAny(load.getMask(), loop.getRegionIterArgs())) ||
          (load.getOther() &&
           dependsOnAny(load.getOther(), loop.getRegionIterArgs()))) {
        rejected = true;
        reason = "read address, mask, or fill depends on loop-carried state";
        return;
      }
      hasAffinePointerInduction |= pointerDependsOnCarried;
      loads.push_back(load);
      return;
    }
    if (isa<StoreOp, AtomicRMWOp, AtomicCASOp>(operation)) {
      rejected = true;
      reason = "iteration contains a write or atomic effect";
      return;
    }
    if (!operation->hasTrait<OpTrait::IsTerminator>() &&
        !isMemoryEffectFree(operation)) {
      rejected = true;
      reason = "iteration contains an unknown non-read effect";
    }
  });
  if (rejected)
    return {false, "", reason};
  if (loads.empty())
    return {false, "", "iteration has no read service to expose"};

  bool hasVectorizableTensorLoad = false;
  for (LoadOp load : loads) {
    if (!isa<RankedTensorType>(load.getPtr().getType()))
      continue;
    if (!load.getBoundaryCheck().empty() || load.getPadding())
      continue;
    hasVectorizableTensorLoad = true;
  }
  if (requireVectorizableTensorLoad && !hasVectorizableTensorLoad)
    return {false, "", "iteration has no join-compatible tensor load"};

  llvm::SmallPtrSet<Operation *, 32> loadClosure;
  SmallVector<Operation *> worklist;
  for (LoadOp load : loads) {
    loadClosure.insert(load.getOperation());
    worklist.push_back(load.getOperation());
  }
  while (!worklist.empty()) {
    Operation *operation = worklist.pop_back_val();
    for (Value operand : operation->getOperands()) {
      Operation *definition = operand.getDefiningOp();
      if (!definition || definition->getBlock() != loop.getBody() ||
          loadClosure.contains(definition))
        continue;
      if (!isMemoryEffectFree(definition))
        return {false, "", "read dependency has an observable effect"};
      loadClosure.insert(definition);
      worklist.push_back(definition);
    }
  }

  auto operationDependsOnCarried = [&](Operation *root) {
    bool depends = llvm::any_of(root->getOperands(), [&](Value operand) {
      return dependsOnAny(operand, loop.getRegionIterArgs());
    });
    root->walk([&](Operation *nested) {
      if (nested != root)
        depends |= llvm::any_of(nested->getOperands(), [&](Value operand) {
          return dependsOnAny(operand, loop.getRegionIterArgs());
        });
    });
    return depends;
  };
  for (Operation &operation : loop.getBody()->without_terminator()) {
    if (loadClosure.contains(&operation))
      loadSlice.push_back(&operation);
    else if (operationDependsOnCarried(&operation))
      stateSlice.push_back(&operation);
    else
      computeSlice.push_back(&operation);
  }
  if (loadSlice.empty() || (computeSlice.empty() && stateSlice.empty()))
    return {false, "", "iteration does not expose a nontrivial read/work split"};

  return {
      true,
      hasAffinePointerInduction
          ? (requireVectorizableTensorLoad
                 ? "existing_affine_pointer_read_load_vectorization_v1"
                 : "existing_affine_pointer_read_exposure_v1")
          : (requireVectorizableTensorLoad
                 ? "existing_order_preserving_load_vectorization_v1"
                 : "existing_order_preserving_read_exposure_v1"),
      ""};
}
