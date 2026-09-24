// General program-effect closure and address partition certificate.
// scf.for exposes recursive memory effects, so no prefix-pattern exception
// is needed to inspect its leaf operations.
bool isProspectivelyClosedPureCall(CallOp call) {
  auto callee = dyn_cast_or_null<FuncOp>(call.resolveCallable());
  if (!callee || callee.getBody().empty())
    return false;
  bool pure = true;
  callee.walk([&](Operation *operation) {
    if (!pure || operation == callee.getOperation() ||
        operation->hasTrait<OpTrait::IsTerminator>())
      return;
    if (isa<CallOp, LoadOp, StoreOp, AtomicRMWOp, AtomicCASOp>(operation) ||
        !isMemoryEffectFree(operation))
      pure = false;
  });
  return pure;
}

LoopDependenceCertificate certifyBridgeProgramIndependence(
    FuncOp entry, GetProgramIdOp pid) {
  SmallVector<LoadOp> loads;
  SmallVector<StoreOp> stores;
  std::string reason;
  bool rejectedEffect = false;
  entry.walk([&](Operation *op) {
    if (rejectedEffect)
      return;
    if (isa<AtomicRMWOp, AtomicCASOp>(op)) {
      rejectedEffect = true;
      reason = "atomic memory operation";
      return;
    }
    if (auto load = dyn_cast<LoadOp>(op)) {
      if (load.getIsVolatile()) {
        rejectedEffect = true;
        reason = "volatile load";
      }
      loads.push_back(load);
      return;
    }
    if (auto store = dyn_cast<StoreOp>(op)) {
      stores.push_back(store);
      return;
    }
    // RecursiveMemoryEffects marks a structural container whose complete
    // memory behavior is carried by the operations in its regions.  walk()
    // continues into those regions after this callback, so accepting the
    // container does not accept any leaf effect: volatile loads, stores,
    // atomics, calls and unknown leaves are still certified below.
    if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>() ||
        isa<FuncOp, ReturnOp, scf::IfOp, scf::YieldOp, ReduceOp,
            ReduceReturnOp>(op))
      return;
    if (auto call = dyn_cast<CallOp>(op)) {
      if (!isProspectivelyClosedPureCall(call)) {
        rejectedEffect = true;
        reason = "program call is not prospectively pure";
      }
      return;
    }
    if (!isMemoryEffectFree(op)) {
      rejectedEffect = true;
      reason = (Twine("unclosed program effect: ") +
                op->getName().getStringRef()).str();
    }
  });
  if (rejectedEffect)
    return {false, "", reason};
  llvm::SmallPtrSet<Value, 8> loadRoots;
  llvm::SmallPtrSet<Value, 8> storeRoots;
  llvm::DenseMap<Value, AffinePidFootprint> storeRootFootprints;
  llvm::SmallPtrSet<Value, 8> symbolicStoreRoots;
  for (LoadOp load : loads) {
    auto root = uniquePointerRoot(load.getPtr());
    if (!root)
      return {false, "", "load pointer does not have one auditable base"};
    loadRoots.insert(*root);
  }
  unsigned storeOrdinal = 0;
  for (StoreOp store : stores) {
    auto root = uniquePointerRoot(store.getPtr());
    if (!root)
      return {false, "", "store pointer does not have one auditable base"};
    storeRoots.insert(*root);
    auto footprint = pointerFootprint(store.getPtr(), *root, pid.getResult());
    bool numericallyDisjoint =
        footprint && provesDisjointPrograms(*footprint);
    if (!numericallyDisjoint &&
        !provesSymbolicPidPartition(store, *root, pid.getResult()) &&
        !provesMixedRadixPidPartition(store, *root, pid.getResult())) {
      std::string detail;
      llvm::raw_string_ostream stream(detail);
      stream << "store " << storeOrdinal
             << (footprint ? " affine interval overlaps adjacent program"
                           : " affine pid footprint is unavailable");
      if (footprint)
        stream << " (stride=" << footprint->pidStride
               << ", local_min=" << footprint->localMin
               << ", local_max=" << footprint->localMax << ")";
      stream.flush();
      return {false, "", detail};
    }
    if (footprint) {
      if (symbolicStoreRoots.contains(*root))
        return {false, "", "same output base mixes symbolic and affine stores"};
      auto [position, inserted] = storeRootFootprints.try_emplace(
          *root, *footprint);
      if (!inserted) {
        AffinePidFootprint &combined = position->second;
        if (combined.pidStride != footprint->pidStride)
          return {false, "", "same output base has incompatible pid strides"};
        combined.localMin = std::min(
            combined.localMin, footprint->localMin);
        combined.localMax = std::max(
            combined.localMax, footprint->localMax);
        if (!provesDisjointPrograms(combined) &&
            !provesBoundedStoreUnion(stores, *root, pid))
          return {false, "", "same output base has overlapping cross-program stores"};
      }
    } else {
      if (storeRootFootprints.contains(*root) ||
          !symbolicStoreRoots.insert(*root).second)
        return {false, "", "same output base has uncomposed symbolic stores"};
    }
    ++storeOrdinal;
  }
  for (Value root : storeRoots)
    if (loadRoots.contains(root))
      return {false, "", "read/write base alias is not disproven"};
  return {true, "bridge_pid_partitioned_disjoint_v1", ""};
}

struct DirectPidPartition {
  arith::DivSIOp quotient;
  arith::RemSIOp remainder;
  Value extent;
};

std::optional<DirectPidPartition>
findDirectPidPartition(FuncOp entry, GetProgramIdOp pid) {
  SmallVector<arith::DivSIOp> quotients;
  SmallVector<arith::RemSIOp> remainders;
  entry.walk([&](arith::DivSIOp op) {
    if (op.getLhs() == pid.getResult())
      quotients.push_back(op);
  });
  entry.walk([&](arith::RemSIOp op) {
    if (op.getLhs() == pid.getResult())
      remainders.push_back(op);
  });
  if (quotients.size() != 1 || remainders.size() != 1 ||
      quotients.front().getRhs() != remainders.front().getRhs())
    return std::nullopt;
  return DirectPidPartition{quotients.front(), remainders.front(),
                            quotients.front().getRhs()};
}
