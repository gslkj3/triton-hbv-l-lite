// Address-footprint algebra and bounded, no-wrap store-union proof.
// A footprint is a relative-coordinate abstraction, not an absolute address
// observation. Unconstrained translations must not justify finite-domain
// independence; boundedPointerDoesNotWrap closes that additional obligation.
std::optional<AffinePidFootprint>
integerFootprint(Value value, Value pid, llvm::SmallPtrSetImpl<Value> &active) {
  if (value == pid)
    return AffinePidFootprint{1, 0, 0};
  if (auto constant = splatInteger(value))
    return AffinePidFootprint{0, *constant, *constant};
  if (!active.insert(value).second)
    return std::nullopt;
  auto erase = llvm::make_scope_exit([&] { active.erase(value); });

  // A local loop coordinate is not a zero-width translation.  Preserve its
  // exact closed interval when all three controls are static and the positive
  // SCF step defines at least one executed iteration.  Runtime controls remain
  // unavailable until a separate predecision bound proof exists.
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (auto loop = dyn_cast_or_null<scf::ForOp>(
            argument.getOwner()->getParentOp())) {
      if (argument.getArgNumber() == 0) {
        auto lower = splatInteger(loop.getLowerBound());
        auto upper = splatInteger(loop.getUpperBound());
        auto step = splatInteger(loop.getStep());
        if (!lower || !upper || !step || *step <= 0 || *lower >= *upper)
          return std::nullopt;
        __int128 distance = static_cast<__int128>(*upper) - *lower - 1;
        auto last = checkedI64(static_cast<__int128>(*lower) +
                               (distance / *step) * *step);
        if (!last)
          return std::nullopt;
        return AffinePidFootprint{0, *lower, *last};
      }
      unsigned iterIndex = argument.getArgNumber() - 1;
      Operation *terminator = loop.getBody()->getTerminator();
      if (iterIndex < loop.getInitArgs().size() &&
          iterIndex < terminator->getNumOperands() &&
          terminator->getOperand(iterIndex) == argument)
        return integerFootprint(loop.getInitArgs()[iterIndex], pid, active);
      return std::nullopt;
    }
  }

  if (auto range = value.getDefiningOp<MakeRangeOp>()) {
    int64_t start = range.getStartAttr().getInt();
    int64_t end = range.getEndAttr().getInt();
    if (end <= start)
      return std::nullopt;
    return AffinePidFootprint{0, start, end - 1};
  }
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return isa<ShapedType>(value.getType())
               ? std::nullopt
               : std::optional<AffinePidFootprint>(
                     AffinePidFootprint{0, 0, 0});
  if (auto otherPid = dyn_cast<GetProgramIdOp>(definition)) {
    if (otherPid.getResult() == pid)
      return AffinePidFootprint{1, 0, 0};
    // This proof is conditional on one grouped target axis.  Every other
    // scalar program ID is fixed while those virtual target-axis programs are
    // traversed, so it is a common translation rather than a local interval.
    // This alone is not a joint multi-axis proof. Multi-axis construction must
    // additionally prove that different coordinate tuples cannot collide.
    return AffinePidFootprint{0, 0, 0};
  }

  // An unchanged loop result has exactly its preheader footprint.  A result
  // with any recurrence is deliberately unavailable: inferring even a simple
  // update here would silently turn this proof into a recurrence solver.
  if (auto loop = dyn_cast<scf::ForOp>(definition)) {
    auto result = dyn_cast<OpResult>(value);
    if (!result || result.getResultNumber() >= loop.getInitArgs().size())
      return std::nullopt;
    unsigned index = result.getResultNumber();
    Operation *terminator = loop.getBody()->getTerminator();
    BlockArgument carried = loop.getRegionIterArgs()[index];
    if (index >= terminator->getNumOperands() ||
        terminator->getOperand(index) != carried)
      return std::nullopt;
    return integerFootprint(loop.getInitArgs()[index], pid, active);
  }
  StringRef name = definition->getName().getStringRef();
  if (name == "tt.splat" || name == "tt.broadcast" ||
      name == "tt.expand_dims" || name == "tt.reshape" ||
      name == "tt.trans" || name == "arith.extsi" ||
      name == "arith.extui") {
    if (definition->getNumOperands() != 1)
      return std::nullopt;
    return integerFootprint(definition->getOperand(0), pid, active);
  }
  if ((name == "arith.addi" || name == "arith.subi") &&
      definition->getNumOperands() == 2) {
    auto lhs = integerFootprint(definition->getOperand(0), pid, active);
    auto rhs = integerFootprint(definition->getOperand(1), pid, active);
    if (!lhs || !rhs)
      return std::nullopt;
    return combineFootprints(*lhs, *rhs, name == "arith.subi");
  }
  if (name == "arith.muli" && definition->getNumOperands() == 2) {
    if (auto rhs = splatInteger(definition->getOperand(1))) {
      auto lhs = integerFootprint(definition->getOperand(0), pid, active);
      return lhs ? scaleFootprint(*lhs, *rhs) : std::nullopt;
    }
    if (auto lhs = splatInteger(definition->getOperand(0))) {
      auto rhs = integerFootprint(definition->getOperand(1), pid, active);
      return rhs ? scaleFootprint(*rhs, *lhs) : std::nullopt;
    }
  }
  // Bridge groups only one launch axis.  A remaining scalar expression that
  // is independent of that axis is a common translation across the grouped
  // virtual programs and cancels from pairwise disjointness.  Ranked local
  // values are not zeroed: they may contain an unmodelled lane interval.
  if (!valueDependsOn(value, pid) && !isa<ShapedType>(value.getType()))
    return AffinePidFootprint{0, 0, 0};
  // Truncation, division, remainder, select, and nonlinear products can fold
  // different physical programs onto one address and are deliberately rejected.
  return std::nullopt;
}

std::optional<AffinePidFootprint>
pointerFootprint(Value pointer, Value root, Value pid,
                 llvm::SmallPtrSetImpl<Value> &active) {
  if (pointer == root)
    return AffinePidFootprint{};
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
    return pointerFootprint(definition->getOperand(0), root, pid, active);
  }
  auto addPtr = dyn_cast<AddPtrOp>(definition);
  if (!addPtr)
    return std::nullopt;
  auto base = pointerFootprint(addPtr.getPtr(), root, pid, active);
  llvm::SmallPtrSet<Value, 32> integerActive;
  auto offset = integerFootprint(addPtr.getOffset(), pid, integerActive);
  if (!base || !offset)
    return std::nullopt;
  return combineFootprints(*base, *offset, false);
}

std::optional<AffinePidFootprint> pointerFootprint(Value pointer, Value root,
                                                   Value pid) {
  llvm::SmallPtrSet<Value, 32> active;
  return pointerFootprint(pointer, root, pid, active);
}

bool provesDisjointPrograms(const AffinePidFootprint &footprint) {
  __int128 span = static_cast<__int128>(footprint.localMax) -
                  static_cast<__int128>(footprint.localMin);
  __int128 stride = footprint.pidStride;
  if (stride < 0)
    stride = -stride;
  return stride > span;
}

// The finite-domain extension below must not inherit the older footprint
// abstraction's unconstrained scalar translations or mathematical-integer
// arithmetic. Check every offset intermediate at its actual signed bit width.
bool boundedOffsetDoesNotWrap(Value value, Value pid, int64_t extent,
                              unsigned &budget) {
  if (!budget--)
    return false;
  auto type = dyn_cast<IntegerType>(getElementTypeOrSelf(value.getType()));
  if (!type || type.getWidth() > 64)
    return false;
  Operation *op = value.getDefiningOp();
  bool leaf = value == pid || isa_and_nonnull<arith::ConstantOp, MakeRangeOp>(op);
  if (!op && splatInteger(value))
    leaf = true;
  if (!leaf) {
    if (!op)
      return false;
    StringRef name = op->getName().getStringRef();
    if (name != "tt.splat" && name != "tt.broadcast" &&
        name != "tt.expand_dims" && name != "tt.reshape" &&
        name != "tt.trans" && name != "arith.extsi" &&
        name != "arith.addi" && name != "arith.subi" && name != "arith.muli")
      return false;
    for (Value input : op->getOperands())
      if (!boundedOffsetDoesNotWrap(input, pid, extent, budget))
        return false;
  }
  llvm::SmallPtrSet<Value, 32> active;
  auto f = integerFootprint(value, pid, active);
  if (!f)
    return false;
  __int128 delta = static_cast<__int128>(f->pidStride) * (extent - 1);
  __int128 low = f->localMin + std::min(__int128(0), delta);
  __int128 high = f->localMax + std::max(__int128(0), delta);
  __int128 limit = __int128(1) << (type.getWidth() - 1);
  return low >= -limit && high < limit;
}

bool boundedPointerDoesNotWrap(Value pointer, Value root, Value pid,
                               int64_t extent, unsigned &budget) {
  if (!budget--)
    return false;
  if (pointer == root)
    return true;
  Operation *op = pointer.getDefiningOp();
  if (!op)
    return false;
  if (auto add = dyn_cast<AddPtrOp>(op))
    return boundedPointerDoesNotWrap(add.getPtr(), root, pid, extent, budget) &&
           boundedOffsetDoesNotWrap(add.getOffset(), pid, extent, budget);
  StringRef name = op->getName().getStringRef();
  return (name == "tt.splat" || name == "tt.broadcast" ||
          name == "tt.expand_dims" || name == "tt.reshape" || name == "tt.trans") &&
         op->getNumOperands() == 1 &&
         boundedPointerDoesNotWrap(op->getOperand(0), root, pid, extent, budget);
}

bool provesBoundedStoreUnion(ArrayRef<StoreOp> stores, Value root,
                             GetProgramIdOp pid) {
  auto module = pid->getParentOfType<ModuleOp>();
  auto object = module->getAttrOfType<DictionaryAttr>(kBridgeRuntimeScalarsAttr);
  if (!object || pid.getAxisAsInt() != 0)
    return false;
  auto grid = object.getAs<DenseIntElementsAttr>("grid");
  auto schema = object.getAs<StringAttr>("schema");
  if (!schema || schema.getValue() != "triton.loop-bridge.runtime-scalars.v2" ||
      !grid || grid.getType().getRank() != 1 ||
      grid.getType().getElementTypeBitWidth() != 32 ||
      grid.getNumElements() < 1 || grid.getNumElements() > 3)
    return false;
  int64_t extent = (*grid.getValues<APInt>().begin()).getSExtValue();
  auto bound = pid->getAttrOfType<IntegerAttr>(kBridgeAxisExtentAttr);
  if (extent < 1 || extent > INT32_MAX ||
      !bound || bound.getInt() != extent)
    return false;
  // Other varying launch axes require their own joint-domain proof.
  for (auto [axis, value] : llvm::enumerate(grid.getValues<APInt>()))
    if (axis > 0 && value.getSExtValue() != 1)
      return false;
  SmallVector<AffinePidFootprint> parts;
  for (StoreOp store : stores) {
    auto base = uniquePointerRoot(store.getPtr());
    if (!base || *base != root)
      continue;
    unsigned budget = 512;
    if (!boundedPointerDoesNotWrap(store.getPtr(), root, pid.getResult(),
                                   extent, budget))
      return false;
    auto f = pointerFootprint(store.getPtr(), root, pid.getResult());
    if (!f || !provesDisjointPrograms(*f))
      return false;
    parts.push_back(*f);
  }
  if (parts.empty())
    return false;
  __int128 stride = parts.front().pidStride;
  if (stride < 0)
    stride = -stride;
  if (!stride)
    return false;
  auto floorDiv = [](__int128 n, __int128 d) {
    __int128 q = n / d;
    return n % d < 0 ? q - 1 : q;
  };
  for (const auto &a : parts)
    for (const auto &b : parts) {
      if (a.pidStride != b.pidStride)
        return false;
      __int128 low = -floorDiv(-(static_cast<__int128>(b.localMin) - a.localMax), stride);
      __int128 high = floorDiv(static_cast<__int128>(b.localMax) - a.localMin, stride);
      low = std::max(low, -static_cast<__int128>(extent - 1));
      high = std::min(high, static_cast<__int128>(extent - 1));
      if (low <= high && (low < 0 || high > 0))
        return false;
    }
  return true;
}
