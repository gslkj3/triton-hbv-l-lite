// A launch axis may enter an address through a quotient/remainder partition
// rather than one affine stride.  This normalized footprint preserves the
// source-level mixed-radix mechanism without naming a row/column layout or a
// kernel.  All coefficients are element offsets and must be compiler-known at
// the planning cut; otherwise `complete` remains false.
struct ProgramMixedRadixAccessFact {
  bool write = false;
  int64_t rootArgument = -1;
  int64_t elementBytes = 0;
  int64_t laneCount = 0;
  int64_t programAxis = -1;
  int64_t divisor = 0;
  int64_t quotientStride = 0;
  int64_t remainderStride = 0;
  int64_t localMinimum = 0;
  int64_t localMaximum = 0;
  bool masked = false;
  bool complete = false;
};

struct MixedRadixFootprint {
  int64_t quotientStride = 0;
  int64_t remainderStride = 0;
  int64_t localMinimum = 0;
  int64_t localMaximum = 0;
};

std::optional<MixedRadixFootprint> combineMixedRadixFootprints(
    const MixedRadixFootprint &lhs, const MixedRadixFootprint &rhs,
    bool subtract) {
  auto quotient = checkedI64(
      static_cast<__int128>(lhs.quotientStride) +
      (subtract ? -static_cast<__int128>(rhs.quotientStride)
                : static_cast<__int128>(rhs.quotientStride)));
  auto remainder = checkedI64(
      static_cast<__int128>(lhs.remainderStride) +
      (subtract ? -static_cast<__int128>(rhs.remainderStride)
                : static_cast<__int128>(rhs.remainderStride)));
  auto minimum = checkedI64(
      static_cast<__int128>(lhs.localMinimum) +
      (subtract ? -static_cast<__int128>(rhs.localMaximum)
                : static_cast<__int128>(rhs.localMinimum)));
  auto maximum = checkedI64(
      static_cast<__int128>(lhs.localMaximum) +
      (subtract ? -static_cast<__int128>(rhs.localMinimum)
                : static_cast<__int128>(rhs.localMaximum)));
  if (!quotient || !remainder || !minimum || !maximum)
    return std::nullopt;
  return MixedRadixFootprint{*quotient, *remainder, *minimum, *maximum};
}

std::optional<MixedRadixFootprint>
scaleMixedRadixFootprint(const MixedRadixFootprint &input, int64_t scale) {
  auto quotient = checkedI64(
      static_cast<__int128>(input.quotientStride) * scale);
  auto remainder = checkedI64(
      static_cast<__int128>(input.remainderStride) * scale);
  auto first = checkedI64(
      static_cast<__int128>(input.localMinimum) * scale);
  auto second = checkedI64(
      static_cast<__int128>(input.localMaximum) * scale);
  if (!quotient || !remainder || !first || !second)
    return std::nullopt;
  return MixedRadixFootprint{*quotient, *remainder,
                             std::min(*first, *second),
                             std::max(*first, *second)};
}

void collectMixedRadixDivisors(Value value, Value pid,
                               std::set<int64_t> &divisors,
                               llvm::SmallPtrSetImpl<Value> &visited) {
  if (!visited.insert(value).second)
    return;
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return;
  StringRef name = definition->getName().getStringRef();
  if ((name == "arith.divsi" || name == "arith.divui" ||
       name == "arith.remsi" || name == "arith.remui") &&
      definition->getNumOperands() == 2 &&
      stripShapeOnlyIntegerCasts(definition->getOperand(0)) == pid) {
    if (auto divisor = splatInteger(definition->getOperand(1));
        divisor && *divisor >= 1)
      divisors.insert(*divisor);
  }
  for (Value operand : definition->getOperands())
    collectMixedRadixDivisors(operand, pid, divisors, visited);
}

std::optional<MixedRadixFootprint> integerMixedRadixFootprint(
    Value value, Value pid, int64_t divisor,
    llvm::SmallPtrSetImpl<Value> &active) {
  value = stripShapeOnlyIntegerCasts(value);
  if (value == pid)
    return MixedRadixFootprint{divisor, divisor == 1 ? 0 : 1, 0, 0};
  if (auto constant = splatInteger(value))
    return MixedRadixFootprint{0, 0, *constant, *constant};
  if (auto range = value.getDefiningOp<MakeRangeOp>())
    return MixedRadixFootprint{0, 0, range.getStartAttr().getInt(),
                               range.getEndAttr().getInt() - 1};
  // Independence from this program axis is not a zero local extent. Recurse
  // through supported arithmetic even for axis-independent values; unknown
  // scalar translations and lane-varying values remain incomplete rather than
  // silently acquiring a fabricated absolute offset or span.
  if (!active.insert(value).second)
    return std::nullopt;
  auto erase = llvm::make_scope_exit([&] { active.erase(value); });
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return std::nullopt;
  StringRef name = definition->getName().getStringRef();
  if ((name == "arith.divsi" || name == "arith.divui" ||
       name == "arith.remsi" || name == "arith.remui") &&
      definition->getNumOperands() == 2 &&
      stripShapeOnlyIntegerCasts(definition->getOperand(0)) == pid) {
    auto candidate = splatInteger(definition->getOperand(1));
    if (!candidate || *candidate != divisor)
      return std::nullopt;
    if (name == "arith.divsi" || name == "arith.divui")
      return MixedRadixFootprint{1, 0, 0, 0};
    if (divisor == 1)
      return MixedRadixFootprint{};
    return MixedRadixFootprint{0, 1, 0, 0};
  }
  if ((name == "arith.addi" || name == "arith.subi") &&
      definition->getNumOperands() == 2) {
    auto lhs = integerMixedRadixFootprint(
        definition->getOperand(0), pid, divisor, active);
    auto rhs = integerMixedRadixFootprint(
        definition->getOperand(1), pid, divisor, active);
    if (!lhs || !rhs)
      return std::nullopt;
    return combineMixedRadixFootprints(
        *lhs, *rhs, name == "arith.subi");
  }
  if (name == "arith.muli" && definition->getNumOperands() == 2) {
    if (auto rhs = splatInteger(definition->getOperand(1))) {
      auto lhs = integerMixedRadixFootprint(
          definition->getOperand(0), pid, divisor, active);
      return lhs ? scaleMixedRadixFootprint(*lhs, *rhs) : std::nullopt;
    }
    if (auto lhs = splatInteger(definition->getOperand(0))) {
      auto rhs = integerMixedRadixFootprint(
          definition->getOperand(1), pid, divisor, active);
      return rhs ? scaleMixedRadixFootprint(*rhs, *lhs) : std::nullopt;
    }
  }
  return std::nullopt;
}

std::optional<MixedRadixFootprint> pointerMixedRadixFootprint(
    Value pointer, Value root, Value pid, int64_t divisor,
    llvm::SmallPtrSetImpl<Value> &active) {
  if (pointer == root)
    return MixedRadixFootprint{};
  if (!active.insert(pointer).second)
    return std::nullopt;
  auto erase = llvm::make_scope_exit([&] { active.erase(pointer); });
  Operation *definition = pointer.getDefiningOp();
  if (!definition)
    return std::nullopt;
  StringRef name = definition->getName().getStringRef();
  if (name == "tt.splat" || name == "tt.broadcast" ||
      name == "tt.expand_dims" || name == "tt.reshape" ||
      name == "tt.trans") {
    if (definition->getNumOperands() != 1)
      return std::nullopt;
    return pointerMixedRadixFootprint(
        definition->getOperand(0), root, pid, divisor, active);
  }
  auto addPtr = dyn_cast<AddPtrOp>(definition);
  if (!addPtr)
    return std::nullopt;
  auto base = pointerMixedRadixFootprint(
      addPtr.getPtr(), root, pid, divisor, active);
  llvm::SmallPtrSet<Value, 32> integerActive;
  auto offset = integerMixedRadixFootprint(
      addPtr.getOffset(), pid, divisor, integerActive);
  if (!base || !offset)
    return std::nullopt;
  return combineMixedRadixFootprints(*base, *offset, false);
}

SmallVector<ProgramMixedRadixAccessFact>
programMixedRadixAccessFacts(FuncOp function) {
  SmallVector<ProgramMixedRadixAccessFact> result;
  SmallVector<GetProgramIdOp> pids;
  function.walk([&](GetProgramIdOp pid) { pids.push_back(pid); });
  auto inspect = [&](Value pointer, Value payload, bool write, bool masked) {
    for (GetProgramIdOp pidOp : pids) {
      Value pid = pidOp.getResult();
      if (!valueDependsOn(pointer, pid))
        continue;
      std::set<int64_t> divisors;
      llvm::SmallPtrSet<Value, 32> visited;
      collectMixedRadixDivisors(pointer, pid, divisors, visited);
      if (divisors.empty())
        continue;
      ProgramMixedRadixAccessFact fact;
      fact.write = write;
      fact.masked = masked;
      fact.programAxis = pidOp.getAxisAsInt();
      fact.laneCount = staticElementCount(payload.getType());
      Type element = getElementTypeOrSelf(payload.getType());
      if (fact.laneCount > 0 && element.isIntOrFloat())
        fact.elementBytes = (element.getIntOrFloatBitWidth() + 7) / 8;
      auto root = uniquePointerRoot(pointer);
      if (root)
        if (auto argument = dyn_cast<BlockArgument>(*root))
          fact.rootArgument = argument.getArgNumber();
      if (divisors.size() == 1 && root && fact.rootArgument >= 0 &&
          fact.elementBytes > 0 && fact.laneCount > 0) {
        fact.divisor = *divisors.begin();
        llvm::SmallPtrSet<Value, 32> active;
        if (auto footprint = pointerMixedRadixFootprint(
                pointer, *root, pid, fact.divisor, active)) {
          fact.quotientStride = footprint->quotientStride;
          fact.remainderStride = footprint->remainderStride;
          fact.localMinimum = footprint->localMinimum;
          fact.localMaximum = footprint->localMaximum;
          fact.complete = true;
        }
      }
      result.push_back(fact);
    }
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
