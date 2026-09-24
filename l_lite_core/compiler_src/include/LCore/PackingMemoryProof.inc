// Delaying corresponding memory operations to their final member is not
// justified by SSA dominance alone. Preserve intervening observable effects,
// including synchronization and possibly aliasing accesses. This necessary
// movement proof is separate from each adapter's group-internal legality.
template <typename MemoryOp>
bool hasNoInterveningMemoryEffects(ArrayRef<MemoryOp> operations) {
  if (operations.empty())
    return false;
  llvm::SmallPtrSet<Operation *, 16> members;
  for (MemoryOp operation : operations)
    members.insert(operation.getOperation());
  MemoryOp firstOp = operations.front(), lastOp = operations.back();
  Operation *last = lastOp.getOperation();
  for (Operation *operation = firstOp.getOperation();
       operation != last; operation = operation->getNextNode()) {
    if (!operation || operation->getBlock() != last->getBlock())
      return false;
    if (!members.contains(operation) && !isMemoryEffectFree(operation)) {
      // Ordinary reads commute with other ordinary reads, even when their
      // addresses alias. SSA dominance is checked by the grouping caller.
      // This exception never applies to stores, volatile accesses or barriers.
      if (isa<LoadOp>(firstOp.getOperation())) {
        auto interveningLoad = dyn_cast<LoadOp>(operation);
        if (interveningLoad && !interveningLoad.getIsVolatile())
          continue;
      }
      return false;
    }
  }
  return true;
}

// A closed interval, with every integer intermediate checked at its real bit
// width. Unknown scalar values are NOT silently treated as zero translations.
// Only an identical scalar pointer SSA value can serve as a common translation
// for the pointer proof below.
std::optional<AffinePidFootprint> closedStoreOffset(Value value,
                                                  unsigned &budget,
                                                  SmallVector<unsigned> selections = {}) {
  if (!budget)
    return std::nullopt;
  --budget;
  auto type = dyn_cast<IntegerType>(getElementTypeOrSelf(value.getType()));
  if (!type || type.getWidth() > 64)
    return std::nullopt;
  if (auto split = value.getDefiningOp<SplitOp>()) {
    selections.push_back(cast<OpResult>(value).getResultNumber());
    return closedStoreOffset(split.getSrc(), budget, selections);
  }
  if (auto join = value.getDefiningOp<JoinOp>()) {
    if (selections.empty()) return std::nullopt;
    unsigned selected = selections.pop_back_val();
    return closedStoreOffset(join->getOperand(selected), budget, selections);
  }
  std::optional<AffinePidFootprint> result;
  if (auto constant = splatInteger(value))
    result = AffinePidFootprint{0, *constant, *constant};
  else if (auto range = value.getDefiningOp<MakeRangeOp>()) {
    if (!selections.empty()) return std::nullopt;
    if (range.getStart() >= range.getEnd())
      return std::nullopt;
    result = AffinePidFootprint{0, range.getStart(), range.getEnd() - 1};
  } else if (Operation *op = value.getDefiningOp()) {
    StringRef name = op->getName().getStringRef();
    if ((name == "tt.splat" || name == "tt.broadcast" ||
         name == "tt.expand_dims" || name == "tt.reshape" ||
         name == "tt.trans" || name == "arith.extsi") &&
        op->getNumOperands() == 1)
      result = (name == "tt.splat" || selections.empty())
                   ? closedStoreOffset(op->getOperand(0), budget)
                   : name == "arith.extsi"
                         ? closedStoreOffset(op->getOperand(0), budget, selections)
                         : std::nullopt;
    else if ((name == "arith.addi" || name == "arith.subi") &&
             op->getNumOperands() == 2) {
      auto left = closedStoreOffset(op->getOperand(0), budget, selections);
      auto right = closedStoreOffset(op->getOperand(1), budget, selections);
      if (left && right)
        result = combineFootprints(*left, *right, name == "arith.subi");
    } else if (name == "arith.muli" && op->getNumOperands() == 2) {
      for (unsigned i = 0; i < 2 && !result; ++i)
        if (auto scale = splatInteger(op->getOperand(i)))
          if (auto input = closedStoreOffset(op->getOperand(1-i), budget, selections))
            result = scaleFootprint(*input, *scale);
    }
  }
  __int128 limit = static_cast<__int128>(1) << (type.getWidth() - 1);
  if (!result || result->localMin < -limit || result->localMax >= limit)
    return std::nullopt;
  return result;
}

std::optional<AffinePidFootprint> closedStorePointerOffset(
    Value pointer, Value commonBase, unsigned &budget,
    SmallVector<unsigned> selections = {}) {
  if (!budget)
    return std::nullopt;
  --budget;
  if (pointer == commonBase)
    return AffinePidFootprint{0, 0, 0};
  Operation *op = pointer.getDefiningOp();
  if (!op)
    return std::nullopt;
  if (auto split = dyn_cast<SplitOp>(op)) {
    selections.push_back(cast<OpResult>(pointer).getResultNumber());
    return closedStorePointerOffset(split.getSrc(), commonBase, budget, selections);
  }
  if (auto join = dyn_cast<JoinOp>(op)) {
    if (selections.empty()) return std::nullopt;
    unsigned selected = selections.pop_back_val();
    return closedStorePointerOffset(join->getOperand(selected), commonBase, budget, selections);
  }
  if (auto add = dyn_cast<AddPtrOp>(op)) {
    auto base = closedStorePointerOffset(add.getPtr(), commonBase, budget, selections);
    auto offset = closedStoreOffset(add.getOffset(), budget, selections);
    if (base && offset)
      return combineFootprints(*base, *offset, false);
    return std::nullopt;
  }
  StringRef name = op->getName().getStringRef();
  if ((name == "tt.splat" || name == "tt.broadcast" ||
       name == "tt.expand_dims" || name == "tt.reshape" || name == "tt.trans") &&
      op->getNumOperands() == 1)
    return (name == "tt.splat" || selections.empty())
               ? closedStorePointerOffset(op->getOperand(0), commonBase, budget)
               : std::nullopt;
  return std::nullopt;
}

bool provesPackedStoresDisjoint(ArrayRef<StoreOp> stores) {
  if (stores.empty())
    return false;
  // Try scalar pointer ancestors, nearest first. A shared tensor pointer is
  // insufficient: its unknown lanes could overlap after translation.
  StoreOp first = stores.front();
  auto pointerType =
      dyn_cast<PointerType>(getElementTypeOrSelf(first.getPtr().getType()));
  if (!pointerType || !isa<IntegerType, FloatType>(pointerType.getPointeeType()))
    return false;
  uint64_t elementBytes =
      (pointerType.getPointeeType().getIntOrFloatBitWidth() + 7) / 8;
  Value base = first.getPtr();
  for (unsigned depth = 0; base && depth < 128; ++depth) {
    if (isa<PointerType>(base.getType())) {
      SmallVector<AffinePidFootprint> spans;
      bool known = true;
      for (StoreOp store : stores) {
        unsigned budget = 512;
        auto span = closedStorePointerOffset(store.getPtr(), base, budget);
        if (!span) { known = false; break; }
        spans.push_back(*span);
      }
      if (known) {
        bool disjoint = true;
        for (size_t i = 0; i < spans.size(); ++i)
          for (size_t j = i + 1; j < spans.size(); ++j) {
            disjoint &= spans[i].localMax < spans[j].localMin ||
                        spans[j].localMax < spans[i].localMin;
            __int128 spread = static_cast<__int128>(
                std::max(spans[i].localMax, spans[j].localMax)) -
                std::min(spans[i].localMin, spans[j].localMin);
            disjoint &= spread * elementBytes < (static_cast<__int128>(1) << 64);
          }
        if (disjoint)
          return true;
      }
    }
    Operation *op = base.getDefiningOp();
    if (!op) break;
    if (auto add = dyn_cast<AddPtrOp>(op)) base = add.getPtr();
    else if (auto split = dyn_cast<SplitOp>(op)) base = split.getSrc();
    else if (auto join = dyn_cast<JoinOp>(op)) base = join->getOperand(0);
    else if (op->getNumOperands() == 1 &&
             (isa<SplatOp, BroadcastOp, ExpandDimsOp, ReshapeOp, TransOp>(op)))
      base = op->getOperand(0);
    else break;
  }
  return false;
}
