// Symbolic and quotient/remainder partition rules; no operator identity.
// Relative partition proofs are not absolute address or timing observations.
// Prove the common symbolic partition
//   root + pid * extent + local, 0 <= local < extent.
// This is stronger and more general than replacing a runtime extent with a
// benchmark constant.  The store mask itself supplies the upper bound; if the
// extent is non-positive no lane stores and disjointness holds vacuously.
bool provesSymbolicPidPartition(StoreOp store, Value root, Value pid) {
  auto localAdd = store.getPtr().getDefiningOp<AddPtrOp>();
  if (!localAdd)
    return false;
  Value local = localAdd.getOffset();
  llvm::SmallPtrSet<Value, 32> localActive;
  auto localFootprint = integerFootprint(local, pid, localActive);
  if (!localFootprint || localFootprint->pidStride != 0 ||
      localFootprint->localMin < 0)
    return false;
  Value scalarBase = stripShapeOnlyIntegerCasts(localAdd.getPtr());
  auto pidAdd = scalarBase.getDefiningOp<AddPtrOp>();
  if (!pidAdd)
    return false;
  auto baseRoot = uniquePointerRoot(pidAdd.getPtr());
  if (!baseRoot || *baseRoot != root)
    return false;
  Value extent;
  if (!matchesPidTimesExtent(pidAdd.getOffset(), pid, extent))
    return false;
  return maskProvesLocalBelowExtent(store.getMask(), local, extent);
}

struct MixedRadixRangeTerm {
  Value basis;
  int64_t coefficient = 0;
  int64_t lower = 0;
  int64_t upper = 0;
};

struct MixedRadixExpression {
  int64_t quotientCoefficient = 0;
  int64_t remainderCoefficient = 0;
  SmallVector<MixedRadixRangeTerm> ranges;
};

struct MixedRadixContext {
  Value pid;
  Value quotient;
  Value remainder;
  int64_t partitionExtent = 0;
  int64_t launchBound = 0;
};

bool valueContainsMakeRange(Value value,
                            llvm::SmallPtrSetImpl<Value> &visited) {
  if (!visited.insert(value).second)
    return false;
  if (value.getDefiningOp<MakeRangeOp>())
    return true;
  Operation *definition = value.getDefiningOp();
  return definition && llvm::any_of(
      definition->getOperands(), [&](Value operand) {
        return valueContainsMakeRange(operand, visited);
      });
}

bool valueContainsMakeRange(Value value) {
  llvm::SmallPtrSet<Value, 32> visited;
  return valueContainsMakeRange(value, visited);
}

bool addMixedRadixTerm(MixedRadixExpression &expression,
                       MixedRadixRangeTerm term) {
  for (MixedRadixRangeTerm &existing : expression.ranges) {
    if (existing.basis != term.basis)
      continue;
    auto coefficient = checkedI64(
        static_cast<__int128>(existing.coefficient) + term.coefficient);
    if (!coefficient || existing.lower != term.lower ||
        existing.upper != term.upper)
      return false;
    existing.coefficient = *coefficient;
    return true;
  }
  expression.ranges.push_back(term);
  return true;
}

std::optional<MixedRadixExpression>
combineMixedRadixExpressions(const MixedRadixExpression &lhs,
                             const MixedRadixExpression &rhs,
                             bool subtract) {
  auto quotient = checkedI64(
      static_cast<__int128>(lhs.quotientCoefficient) +
      (subtract ? -static_cast<__int128>(rhs.quotientCoefficient)
                : static_cast<__int128>(rhs.quotientCoefficient)));
  auto remainder = checkedI64(
      static_cast<__int128>(lhs.remainderCoefficient) +
      (subtract ? -static_cast<__int128>(rhs.remainderCoefficient)
                : static_cast<__int128>(rhs.remainderCoefficient)));
  if (!quotient || !remainder)
    return std::nullopt;
  MixedRadixExpression result{*quotient, *remainder, lhs.ranges};
  for (MixedRadixRangeTerm term : rhs.ranges) {
    if (subtract) {
      auto coefficient = checkedI64(-static_cast<__int128>(term.coefficient));
      if (!coefficient)
        return std::nullopt;
      term.coefficient = *coefficient;
    }
    if (!addMixedRadixTerm(result, term))
      return std::nullopt;
  }
  return result;
}

std::optional<MixedRadixExpression>
scaleMixedRadixExpression(const MixedRadixExpression &input, int64_t scale) {
  auto quotient = checkedI64(
      static_cast<__int128>(input.quotientCoefficient) * scale);
  auto remainder = checkedI64(
      static_cast<__int128>(input.remainderCoefficient) * scale);
  if (!quotient || !remainder)
    return std::nullopt;
  MixedRadixExpression result{*quotient, *remainder, {}};
  for (MixedRadixRangeTerm term : input.ranges) {
    auto coefficient = checkedI64(
        static_cast<__int128>(term.coefficient) * scale);
    if (!coefficient)
      return std::nullopt;
    term.coefficient = *coefficient;
    result.ranges.push_back(term);
  }
  return result;
}

std::optional<MixedRadixExpression>
mixedRadixIntegerExpression(Value value, const MixedRadixContext &context,
                            llvm::SmallPtrSetImpl<Value> &active) {
  if (value == context.quotient)
    return MixedRadixExpression{1, 0, {}};
  if (value == context.remainder)
    return MixedRadixExpression{0, 1, {}};
  if (value == context.pid)
    return MixedRadixExpression{
        context.partitionExtent, 1, {}};
  if (auto range = value.getDefiningOp<MakeRangeOp>())
    return MixedRadixExpression{
        0, 0,
        {{value, 1, range.getStartAttr().getInt(),
          range.getEndAttr().getInt() - 1}}};
  if (splatInteger(value))
    return MixedRadixExpression{};
  if (!active.insert(value).second)
    return std::nullopt;
  auto erase = llvm::make_scope_exit([&] { active.erase(value); });
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return std::nullopt;
  StringRef name = definition->getName().getStringRef();
  if (name == "tt.splat" || name == "tt.broadcast" ||
      name == "tt.expand_dims" || name == "tt.reshape" ||
      name == "tt.trans" || name == "arith.extsi" ||
      name == "arith.extui") {
    if (definition->getNumOperands() != 1)
      return std::nullopt;
    return mixedRadixIntegerExpression(
        definition->getOperand(0), context, active);
  }
  if ((name == "arith.addi" || name == "arith.subi") &&
      definition->getNumOperands() == 2) {
    auto lhs = mixedRadixIntegerExpression(
        definition->getOperand(0), context, active);
    auto rhs = mixedRadixIntegerExpression(
        definition->getOperand(1), context, active);
    if (!lhs || !rhs)
      return std::nullopt;
    return combineMixedRadixExpressions(
        *lhs, *rhs, name == "arith.subi");
  }
  if (name == "arith.muli" && definition->getNumOperands() == 2) {
    if (auto rhs = splatInteger(definition->getOperand(1))) {
      auto lhs = mixedRadixIntegerExpression(
          definition->getOperand(0), context, active);
      return lhs ? scaleMixedRadixExpression(*lhs, *rhs) : std::nullopt;
    }
    if (auto lhs = splatInteger(definition->getOperand(0))) {
      auto rhs = mixedRadixIntegerExpression(
          definition->getOperand(1), context, active);
      return rhs ? scaleMixedRadixExpression(*rhs, *lhs) : std::nullopt;
    }
  }
  if (!valueDependsOn(value, context.pid) && !valueContainsMakeRange(value))
    return MixedRadixExpression{};
  return std::nullopt;
}

std::optional<MixedRadixExpression>
mixedRadixPointerExpression(Value pointer, Value root,
                            const MixedRadixContext &context,
                            llvm::SmallPtrSetImpl<Value> &active) {
  if (pointer == root)
    return MixedRadixExpression{};
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
    return mixedRadixPointerExpression(
        definition->getOperand(0), root, context, active);
  }
  auto addPtr = dyn_cast<AddPtrOp>(definition);
  if (!addPtr)
    return std::nullopt;
  auto base = mixedRadixPointerExpression(
      addPtr.getPtr(), root, context, active);
  llvm::SmallPtrSet<Value, 32> integerActive;
  auto offset = mixedRadixIntegerExpression(
      addPtr.getOffset(), context, integerActive);
  if (!base || !offset)
    return std::nullopt;
  return combineMixedRadixExpressions(*base, *offset, false);
}

void collectMaskedRangeUpperBounds(Value mask,
                                   llvm::DenseMap<Value, int64_t> &bounds) {
  if (!mask)
    return;
  mask = stripShapeOnlyIntegerCasts(mask);
  Operation *definition = mask.getDefiningOp();
  if (!definition)
    return;
  if (auto conjunction = dyn_cast<arith::AndIOp>(definition)) {
    collectMaskedRangeUpperBounds(conjunction.getLhs(), bounds);
    collectMaskedRangeUpperBounds(conjunction.getRhs(), bounds);
    return;
  }
  auto comparison = dyn_cast<arith::CmpIOp>(definition);
  if (!comparison ||
      (comparison.getPredicate() != arith::CmpIPredicate::slt &&
       comparison.getPredicate() != arith::CmpIPredicate::ult))
    return;
  Value local = stripShapeOnlyIntegerCasts(comparison.getLhs());
  auto range = local.getDefiningOp<MakeRangeOp>();
  auto upper = splatInteger(comparison.getRhs());
  if (!range || !upper || *upper <= range.getStartAttr().getInt())
    return;
  int64_t clipped = std::min<int64_t>(
      range.getEndAttr().getInt() - 1, *upper - 1);
  auto position = bounds.find(local);
  if (position == bounds.end())
    bounds[local] = clipped;
  else
    position->second = std::min(position->second, clipped);
}

std::optional<int64_t>
mixedRadixResidualSpan(const MixedRadixExpression &expression,
                       std::optional<unsigned> excludedRange) {
  __int128 span = 0;
  for (unsigned index = 0; index < expression.ranges.size(); ++index) {
    if (excludedRange && *excludedRange == index)
      continue;
    const MixedRadixRangeTerm &term = expression.ranges[index];
    __int128 coefficient = term.coefficient;
    if (coefficient < 0)
      coefficient = -coefficient;
    span += coefficient * (term.upper - term.lower);
  }
  return checkedI64(span);
}

bool mixedRadixSeparated(const MixedRadixExpression &expression,
                         const MixedRadixContext &context,
                         std::optional<unsigned> innerRange) {
  int64_t secondaryCount = context.partitionExtent;
  int64_t secondaryCoefficient = expression.remainderCoefficient;
  if (innerRange) {
    const MixedRadixRangeTerm &term = expression.ranges[*innerRange];
    int64_t innerCount = term.upper - term.lower + 1;
    if (innerCount < 2 || term.coefficient == 0)
      return false;
    if (context.partitionExtent > 1) {
      auto recombined = checkedI64(
          static_cast<__int128>(term.coefficient) * innerCount);
      if (!recombined || expression.remainderCoefficient != *recombined)
        return false;
    }
    auto count = checkedI64(
        static_cast<__int128>(secondaryCount) * innerCount);
    if (!count)
      return false;
    secondaryCount = *count;
    secondaryCoefficient = term.coefficient;
  }
  auto spanValue = mixedRadixResidualSpan(expression, innerRange);
  if (!spanValue)
    return false;
  __int128 span = *spanValue;
  __int128 quotientStep = expression.quotientCoefficient;
  __int128 secondaryStep = secondaryCoefficient;
  if (quotientStep < 0)
    quotientStep = -quotientStep;
  if (secondaryStep < 0)
    secondaryStep = -secondaryStep;
  int64_t quotientCount =
      (context.launchBound + context.partitionExtent - 1) /
      context.partitionExtent;
  if (secondaryCount == 1)
    return quotientStep > span;
  if (quotientCount == 1)
    return secondaryStep > span;
  if (static_cast<__int128>(expression.quotientCoefficient) *
          secondaryCoefficient <=
      0)
    return false;
  bool quotientMajor =
      secondaryStep > span &&
      quotientStep >
          static_cast<__int128>(secondaryCount - 1) * secondaryStep + span;
  bool secondaryMajor =
      quotientStep > span &&
      secondaryStep >
          static_cast<__int128>(quotientCount - 1) * quotientStep + span;
  return quotientMajor || secondaryMajor;
}

// Prove a bounded mixed-radix store partition from
//   q = pid / E, r = pid % E,
// and an address linear in q/r plus static tensor ranges.  One tensor range
// may extend r into a nested radix (for example vectorized groups).  Store
// masks may only tighten a recognized make_range upper bound.  The accepted
// q-major/r-major inequalities are sufficient and identity-free; unknown,
// non-positive, overflowing, or overlapping forms remain typed unsupported.
bool provesMixedRadixPidPartition(StoreOp store, Value root, Value pid) {
  FuncOp entry = store->getParentOfType<FuncOp>();
  if (!entry)
    return false;
  SmallVector<arith::DivSIOp> quotients;
  SmallVector<arith::RemSIOp> remainders;
  entry.walk([&](arith::DivSIOp op) {
    if (op.getLhs() == pid)
      quotients.push_back(op);
  });
  entry.walk([&](arith::RemSIOp op) {
    if (op.getLhs() == pid)
      remainders.push_back(op);
  });
  if (quotients.size() != 1 || remainders.size() != 1 ||
      quotients.front().getRhs() != remainders.front().getRhs())
    return false;
  auto extent = splatInteger(quotients.front().getRhs());
  auto launch = pid.getDefiningOp<GetProgramIdOp>()->getAttrOfType<IntegerAttr>(
      kBridgeAxisExtentAttr);
  if (!extent || *extent < 1 || !launch || launch.getInt() < 1)
    return false;
  MixedRadixContext context{
      pid, quotients.front().getResult(), remainders.front().getResult(),
      *extent, launch.getInt()};
  llvm::SmallPtrSet<Value, 32> active;
  auto expression = mixedRadixPointerExpression(
      store.getPtr(), root, context, active);
  if (!expression)
    return false;
  llvm::DenseMap<Value, int64_t> maskedBounds;
  collectMaskedRangeUpperBounds(store.getMask(), maskedBounds);
  for (MixedRadixRangeTerm &term : expression->ranges)
    if (auto position = maskedBounds.find(term.basis);
        position != maskedBounds.end())
      term.upper = std::min(term.upper, position->second);
  if (mixedRadixSeparated(*expression, context, std::nullopt))
    return true;
  for (unsigned index = 0; index < expression->ranges.size(); ++index)
    if (mixedRadixSeparated(*expression, context, index))
      return true;
  return false;
}
