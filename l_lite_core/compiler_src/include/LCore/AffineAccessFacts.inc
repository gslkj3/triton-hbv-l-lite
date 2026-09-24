// Direct affine geometry for an independently reorderable iteration.  This
// is deliberately expressed relative to the loop-carried induction, not an
// operator, axis name, tensor shape, or observed performance.  A stride of
// one and a large stride have different transaction/coalescing service even
// when trip count and byte count are identical, so the distinction belongs in
// the strong state consumed by any quantitative phase authority.
std::optional<int64_t> independentIterationAddressStride(
    scf::ForOp loop, bool stores) {
  int64_t maximum = 0;
  bool found = false;
  auto inspect = [&](Value pointer) {
    auto root = uniquePointerRoot(pointer);
    if (!root)
      return;
    for (Value carried : loop.getRegionIterArgs()) {
      auto footprint = pointerFootprint(pointer, *root, carried);
      if (!footprint || footprint->pidStride == 0)
        continue;
      __int128 magnitude = footprint->pidStride;
      if (magnitude < 0)
        magnitude = -magnitude;
      if (magnitude > std::numeric_limits<int64_t>::max())
        continue;
      maximum = std::max(maximum, static_cast<int64_t>(magnitude));
      found = true;
    }
  };
  loop.walk([&](Operation *operation) {
    if (stores) {
      if (auto store = dyn_cast<StoreOp>(operation))
        inspect(store.getPtr());
    } else if (auto load = dyn_cast<LoadOp>(operation)) {
      inspect(load.getPtr());
    }
  });
  return found ? std::optional<int64_t>(maximum) : std::nullopt;
}

int64_t independentInnerReductionWidth(scf::ForOp loop) {
  int64_t maximum = 0;
  loop.walk([&](ReduceOp reduce) {
    for (Value source : reduce.getSrcs())
      maximum = std::max(maximum, staticElementCount(source.getType()));
  });
  return maximum;
}

// Complete per-access affine geometry for an independent reduction.  This is
// structural TTIR evidence: no target rate, kernel identity, observed timing,
// or profitable configuration enters the record.  Unknown roots or nonlinear
// address expressions remain explicit through `complete=false` rather than
// being collapsed into a maximum stride.
struct IndependentAffineAccessFact {
  bool write = false;
  int64_t rootArgument = -1;
  int64_t elementBytes = 0;
  int64_t laneCount = 0;
  int64_t iterationStride = 0;
  std::array<int64_t, 3> programAxisStrides{0, 0, 0};
  std::array<bool, 3> programAxisDependencies{false, false, false};
  std::array<bool, 3> programAxisAffineComplete{true, true, true};
  int64_t localMinimum = 0;
  int64_t localMaximum = 0;
  bool masked = false;
  bool complete = false;
};

SmallVector<IndependentAffineAccessFact>
independentAffineAccessFacts(scf::ForOp loop) {
  SmallVector<IndependentAffineAccessFact> result;
  auto function = loop->getParentOfType<FuncOp>();
  std::array<SmallVector<Value>, 3> pids;
  if (function)
    function.walk([&](GetProgramIdOp pid) {
      int axis = pid.getAxisAsInt();
      if (axis >= 0 && axis < 3)
        pids[axis].push_back(pid.getResult());
    });

  auto inspect = [&](Value pointer, Value payload, bool write, bool masked) {
    IndependentAffineAccessFact fact;
    fact.write = write;
    fact.masked = masked;
    fact.laneCount = staticElementCount(payload.getType());
    Type element = getElementTypeOrSelf(payload.getType());
    if (fact.laneCount < 1 || !element.isIntOrFloat()) {
      result.push_back(fact);
      return;
    }
    fact.elementBytes = (element.getIntOrFloatBitWidth() + 7) / 8;
    auto root = uniquePointerRoot(pointer);
    if (!root) {
      result.push_back(fact);
      return;
    }
    if (auto argument = dyn_cast<BlockArgument>(*root))
      fact.rootArgument = argument.getArgNumber();
    if (fact.rootArgument < 0) {
      result.push_back(fact);
      return;
    }

    bool iterationFound = false;
    for (Value carried : loop.getRegionIterArgs()) {
      auto footprint = pointerFootprint(pointer, *root, carried);
      if (!footprint || footprint->pidStride == 0)
        continue;
      if (iterationFound) {
        result.push_back(fact);
        return;
      }
      iterationFound = true;
      fact.iterationStride = footprint->pidStride;
      fact.localMinimum = footprint->localMin;
      fact.localMaximum = footprint->localMax;
    }
    if (!iterationFound) {
      result.push_back(fact);
      return;
    }

    for (int axis = 0; axis < 3; ++axis) {
      bool axisFound = false;
      int64_t stride = 0;
      for (Value pid : pids[axis]) {
        bool depends = valueDependsOn(pointer, pid);
        fact.programAxisDependencies[axis] |= depends;
        auto footprint = pointerFootprint(pointer, *root, pid);
        if (!footprint) {
          if (depends)
            fact.programAxisAffineComplete[axis] = false;
          continue;
        }
        if (footprint->pidStride == 0)
          continue;
        if (axisFound && stride != footprint->pidStride) {
          fact.programAxisAffineComplete[axis] = false;
          continue;
        }
        axisFound = true;
        stride = footprint->pidStride;
        fact.localMinimum = std::min(
            fact.localMinimum, footprint->localMin);
        fact.localMaximum = std::max(
            fact.localMaximum, footprint->localMax);
      }
      fact.programAxisStrides[axis] = stride;
    }
    fact.complete = llvm::all_of(
        fact.programAxisAffineComplete, [](bool value) { return value; });
    result.push_back(fact);
  };
  loop.walk([&](Operation *operation) {
    if (auto load = dyn_cast<LoadOp>(operation))
      inspect(load.getPtr(), load.getResult(), false,
              static_cast<bool>(load.getMask()));
    else if (auto store = dyn_cast<StoreOp>(operation))
      inspect(store.getPtr(), store.getValue(), true,
              static_cast<bool>(store.getMask()));
  });
  return result;
}

// Observation-only projection of packed lanes. Keep this separate from
// integerFootprint: expanding observation coverage must not change legality.
std::optional<AffinePidFootprint> packedIntegerObservation(
    Value value, Value pid, SmallVector<unsigned> selections, unsigned &budget) {
  if (!budget) return std::nullopt;
  --budget;
  if (selections.empty()) {
    llvm::SmallPtrSet<Value, 32> active;
    if (auto known = integerFootprint(value, pid, active)) return known;
  }
  Operation *op = value.getDefiningOp();
  if (!op) return std::nullopt;
  if (auto split = dyn_cast<SplitOp>(op)) {
    selections.push_back(cast<OpResult>(value).getResultNumber());
    return packedIntegerObservation(split.getSrc(), pid, selections, budget);
  }
  if (auto join = dyn_cast<JoinOp>(op)) {
    if (selections.empty()) return std::nullopt;
    unsigned selected = selections.pop_back_val();
    if (selected > 1) return std::nullopt;
    return packedIntegerObservation(op->getOperand(selected), pid, selections, budget);
  }
  // Splat constants have the same value in every selected component.
  if (auto constant = splatInteger(value))
    return AffinePidFootprint{0, *constant, *constant};
  StringRef name = op->getName().getStringRef();
  if (name == "tt.splat")
    return packedIntegerObservation(op->getOperand(0), pid, {}, budget);
  if (name == "arith.extsi")
    return packedIntegerObservation(op->getOperand(0), pid, selections, budget);
  if (selections.empty() && (name == "tt.broadcast" || name == "tt.expand_dims" ||
                            name == "tt.reshape" || name == "tt.trans"))
    return packedIntegerObservation(op->getOperand(0), pid, {}, budget);
  if ((name == "arith.addi" || name == "arith.subi") && op->getNumOperands() == 2) {
    auto lhs = packedIntegerObservation(op->getOperand(0), pid, selections, budget);
    auto rhs = packedIntegerObservation(op->getOperand(1), pid, selections, budget);
    if (!lhs || !rhs) return std::nullopt;
    return combineFootprints(*lhs, *rhs, name == "arith.subi");
  }
  if (name == "arith.muli" && op->getNumOperands() == 2) {
    for (unsigned i = 0; i < 2; ++i)
      if (auto scale = splatInteger(op->getOperand(i))) {
        auto input = packedIntegerObservation(op->getOperand(1-i), pid, selections, budget);
        return input ? scaleFootprint(*input, *scale) : std::nullopt;
      }
  }
  return std::nullopt;
}

std::optional<AffinePidFootprint> packedPointerObservation(
    Value pointer, Value root, Value pid, unsigned &budget) {
  if (!budget) return std::nullopt;
  --budget;
  if (pointer == root) return AffinePidFootprint{};
  Operation *op = pointer.getDefiningOp();
  if (!op) return std::nullopt;
  StringRef name = op->getName().getStringRef();
  if (name == "tt.splat" || name == "tt.broadcast" || name == "tt.expand_dims" ||
      name == "tt.reshape" || name == "tt.trans")
    return packedPointerObservation(op->getOperand(0), root, pid, budget);
  if (auto add = dyn_cast<AddPtrOp>(op)) {
    auto base = packedPointerObservation(add.getPtr(), root, pid, budget);
    auto offset = packedIntegerObservation(add.getOffset(), pid, {}, budget);
    if (base && offset) return combineFootprints(*base, *offset, false);
  }
  return std::nullopt;
}

// Complete per-access geometry relative to launch program IDs.  Unlike the
// independent-reduction facts above, this describes the source program before
// Bridge grouping and therefore has no synthetic iteration coordinate.  It is
// an upper-footprint proof input, not a cache-residency or profitability claim.
SmallVector<IndependentAffineAccessFact>
programAffineAccessFacts(FuncOp function) {
  SmallVector<IndependentAffineAccessFact> result;
  std::array<SmallVector<Value>, 3> pids;
  function.walk([&](GetProgramIdOp pid) {
    int axis = pid.getAxisAsInt();
    if (axis >= 0 && axis < 3)
      pids[axis].push_back(pid.getResult());
  });
  auto inspect = [&](Value pointer, Value payload, bool write, bool masked) {
    IndependentAffineAccessFact fact;
    fact.write = write;
    fact.masked = masked;
    fact.iterationStride = 1;
    fact.laneCount = staticElementCount(payload.getType());
    Type element = getElementTypeOrSelf(payload.getType());
    if (fact.laneCount < 1 || !element.isIntOrFloat()) {
      result.push_back(fact);
      return;
    }
    fact.elementBytes = (element.getIntOrFloatBitWidth() + 7) / 8;
    auto root = uniquePointerRoot(pointer);
    if (!root) {
      result.push_back(fact);
      return;
    }
    if (auto argument = dyn_cast<BlockArgument>(*root))
      fact.rootArgument = argument.getArgNumber();
    if (fact.rootArgument < 0) {
      result.push_back(fact);
      return;
    }
    bool sawFootprint = false;
    for (int axis = 0; axis < 3; ++axis) {
      bool axisFound = false;
      int64_t stride = 0;
      for (Value pid : pids[axis]) {
        bool depends = valueDependsOn(pointer, pid);
        fact.programAxisDependencies[axis] |= depends;
        auto footprint = pointerFootprint(pointer, *root, pid);
        if (!footprint) {
          unsigned observationBudget = 512;
          footprint = packedPointerObservation(pointer, *root, pid, observationBudget);
        }
        if (!footprint) {
          if (depends)
            fact.programAxisAffineComplete[axis] = false;
          continue;
        }
        sawFootprint = true;
        fact.localMinimum =
            std::min(fact.localMinimum, footprint->localMin);
        fact.localMaximum =
            std::max(fact.localMaximum, footprint->localMax);
        if (footprint->pidStride == 0)
          continue;
        if (axisFound && stride != footprint->pidStride) {
          fact.programAxisAffineComplete[axis] = false;
          continue;
        }
        axisFound = true;
        stride = footprint->pidStride;
      }
      fact.programAxisStrides[axis] = stride;
    }
    fact.complete = sawFootprint && llvm::all_of(
        fact.programAxisAffineComplete, [](bool value) { return value; });
    result.push_back(fact);
  };
  function.walk([&](Operation *operation) {
    if (auto load = dyn_cast<LoadOp>(operation))
      inspect(load.getPtr(), load.getResult(), false,
              static_cast<bool>(load.getMask()));
    else if (auto store = dyn_cast<StoreOp>(operation))
      inspect(store.getPtr(), store.getValue(), true,
              static_cast<bool>(store.getMask()));
  });
  return result;
}
