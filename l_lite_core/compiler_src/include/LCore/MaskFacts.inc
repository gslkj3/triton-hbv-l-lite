// Mask dataflow observations. Necessary-condition upper bounds are not exact
// activity counts, nor measurements of backend lane execution.
Value stripShapeOnlyIntegerCasts(Value value) {
  while (Operation *definition = value.getDefiningOp()) {
    StringRef name = definition->getName().getStringRef();
    if (name != "tt.splat" && name != "tt.broadcast" &&
        name != "tt.expand_dims" && name != "tt.reshape" &&
        name != "tt.trans" && name != "arith.extsi" &&
        name != "arith.extui")
      break;
    if (definition->getNumOperands() != 1)
      break;
    value = definition->getOperand(0);
  }
  return value;
}

bool matchesPidTimesExtent(Value value, Value pid, Value &extent) {
  value = stripShapeOnlyIntegerCasts(value);
  auto multiply = value.getDefiningOp<arith::MulIOp>();
  if (!multiply)
    return false;
  Value lhs = stripShapeOnlyIntegerCasts(multiply.getLhs());
  Value rhs = stripShapeOnlyIntegerCasts(multiply.getRhs());
  if (lhs == pid && !valueDependsOn(rhs, pid)) {
    extent = rhs;
    return true;
  }
  if (rhs == pid && !valueDependsOn(lhs, pid)) {
    extent = lhs;
    return true;
  }
  return false;
}

bool maskProvesLocalBelowExtent(Value mask, Value local, Value extent) {
  if (!mask)
    return false;
  Operation *definition = mask.getDefiningOp();
  if (!definition)
    return false;
  if (auto conjunction = dyn_cast<arith::AndIOp>(definition))
    return maskProvesLocalBelowExtent(
               conjunction.getLhs(), local, extent) ||
           maskProvesLocalBelowExtent(
               conjunction.getRhs(), local, extent);
  auto comparison = dyn_cast<arith::CmpIOp>(definition);
  if (!comparison ||
      (comparison.getPredicate() != arith::CmpIPredicate::slt &&
       comparison.getPredicate() != arith::CmpIPredicate::ult))
    return false;
  return stripShapeOnlyIntegerCasts(comparison.getLhs()) ==
             stripShapeOnlyIntegerCasts(local) &&
         stripShapeOnlyIntegerCasts(comparison.getRhs()) ==
             stripShapeOnlyIntegerCasts(extent);
}

int64_t staticElementCount(Type type);

struct RuntimeActiveExtentFact {
  int64_t argumentIndex = -1;
  int64_t containerWidth = 0;
};

// A runtime scalar can control one logical tensor axis even when the complete
// memory operation is a higher-rank broadcast.  This fact is deliberately
// separate from RuntimeActiveExtentFact: the latter owns exact whole-operation
// work accounting, while this one records only the compiler-visible mask axis
// and its replication.  Conflating the two would undercount a 2-D mask as one
// active row.
struct RuntimeMaskScalarFact {
  bool write = false;
  int64_t argumentIndex = -1;
  int64_t containerWidth = 0;
  int64_t maskElementCount = 0;
  int64_t replicationFactor = 0;
  int64_t boundValue = 0;
  int64_t divisibility = 1;
  bool boundValueKnown = false;
  bool complete = false;
};

// A necessary mask condition bounds activity; it does not prove that all
// remaining lanes execute. Keep this observation separate from complete masks.
std::optional<int64_t> maskLaunchScalar(Value value, unsigned &budget) {
  if (!budget) return std::nullopt;
  --budget;
  Operation *op = value.getDefiningOp();
  if (op) {
    StringRef name = op->getName().getStringRef();
    if (name == "tt.splat" || name == "tt.broadcast" ||
        name == "tt.expand_dims" || name == "tt.reshape" || name == "tt.trans" ||
        name == "arith.extsi" || name == "arith.extui") {
      auto observed = maskLaunchScalar(op->getOperand(0), budget);
      if (name == "arith.extui" && observed && *observed < 0)
        return std::nullopt;
      return observed;
    }
    if (isa<GetProgramIdOp>(op)) {
      auto extent = op->getAttrOfType<IntegerAttr>(kBridgeAxisExtentAttr);
      return extent && extent.getInt() == 1 ? std::optional<int64_t>(0)
                                           : std::nullopt;
    }
    if (name == "arith.muli") {
      auto lhs = maskLaunchScalar(op->getOperand(0), budget);
      auto rhs = maskLaunchScalar(op->getOperand(1), budget);
      // Both operands must be observable: 0 times unknown/poison is not proof.
      if (lhs && rhs && (*lhs == 0 || *rhs == 0)) return 0;
    }
  }
  return launchIntegerObservation(value, budget);
}

std::optional<int64_t> maskZeroBasedRangeWidth(Value value, unsigned &budget) {
  if (!budget) return std::nullopt;
  --budget;
  Operation *op = value.getDefiningOp();
  if (!op) return std::nullopt;
  if (auto range = dyn_cast<MakeRangeOp>(op)) {
    int64_t end = range.getEndAttr().getInt();
    return range.getStartAttr().getInt() == 0 && end > 0
               ? std::optional<int64_t>(end) : std::nullopt;
  }
  StringRef name = op->getName().getStringRef();
  if (name == "tt.splat" || name == "tt.broadcast" ||
      name == "tt.expand_dims" || name == "tt.reshape" || name == "tt.trans" ||
      name == "arith.extsi" || name == "arith.extui")
    return maskZeroBasedRangeWidth(op->getOperand(0), budget);
  if (auto add = dyn_cast<arith::AddIOp>(op)) {
    for (unsigned side = 0; side < 2; ++side) {
      auto scalar = maskLaunchScalar(add->getOperand(side), budget);
      if (scalar && *scalar == 0)
        if (auto width = maskZeroBasedRangeWidth(add->getOperand(1-side), budget))
          return width;
    }
  }
  return std::nullopt;
}

std::optional<int64_t> necessaryMaskActiveUpper(Value value, int64_t elements,
                                               unsigned &budget) {
  if (!budget || elements < 1) return std::nullopt;
  --budget;
  Operation *op = value.getDefiningOp();
  if (!op) return std::nullopt;
  StringRef name = op->getName().getStringRef();
  if (name == "tt.splat" || name == "tt.broadcast" ||
      name == "tt.expand_dims" || name == "tt.reshape" || name == "tt.trans")
    return necessaryMaskActiveUpper(op->getOperand(0), elements, budget);
  if (auto both = dyn_cast<arith::AndIOp>(op)) {
    auto a = necessaryMaskActiveUpper(both.getLhs(), elements, budget);
    auto b = necessaryMaskActiveUpper(both.getRhs(), elements, budget);
    if (a && b) return std::min(*a, *b);
    return a ? a : b;
  }
  auto cmp = dyn_cast<arith::CmpIOp>(op);
  if (!cmp || (cmp.getPredicate() != arith::CmpIPredicate::slt &&
               cmp.getPredicate() != arith::CmpIPredicate::ult))
    return std::nullopt;
  auto width = maskZeroBasedRangeWidth(cmp.getLhs(), budget);
  auto extent = maskLaunchScalar(cmp.getRhs(), budget);
  if (!width || !extent || *extent < 0 || elements % *width)
    return std::nullopt;
  return std::min(*extent, *width) * (elements / *width);
}

std::optional<RuntimeMaskScalarFact>
runtimeMaskScalarFromMask(Value mask, FuncOp entry, bool write) {
  if (!mask)
    return std::nullopt;
  int64_t maskElements = staticElementCount(mask.getType());
  Value normalizedMask = stripShapeOnlyIntegerCasts(mask);
  auto comparison = normalizedMask.getDefiningOp<arith::CmpIOp>();
  if (!comparison ||
      (comparison.getPredicate() != arith::CmpIPredicate::slt &&
       comparison.getPredicate() != arith::CmpIPredicate::ult))
    return std::nullopt;
  Value local = stripShapeOnlyIntegerCasts(comparison.getLhs());
  Value extent = stripShapeOnlyIntegerCasts(comparison.getRhs());
  auto range = local.getDefiningOp<MakeRangeOp>();
  auto argument = dyn_cast<BlockArgument>(extent);
  if (!range || !argument || argument.getOwner() != &entry.getBody().front() ||
      range.getStartAttr().getInt() != 0 ||
      range.getEndAttr().getInt() < 1)
    return std::nullopt;
  int64_t width = staticElementCount(local.getType());
  if (width != range.getEndAttr().getInt() || maskElements < width ||
      maskElements % width != 0)
    return std::nullopt;
  RuntimeMaskScalarFact fact;
  fact.write = write;
  fact.argumentIndex = argument.getArgNumber();
  fact.containerWidth = width;
  fact.maskElementCount = maskElements;
  fact.replicationFactor = maskElements / width;
  if (auto bound = entry.getArgAttrOfType<IntegerAttr>(
          argument.getArgNumber(), kBridgeBoundScalarAttr)) {
    fact.boundValue = bound.getInt();
    fact.boundValueKnown = true;
  }
  if (auto divisibility = entry.getArgAttrOfType<IntegerAttr>(
          argument.getArgNumber(), "tt.divisibility"))
    fact.divisibility = divisibility.getInt();
  fact.complete = fact.argumentIndex >= 0 && fact.containerWidth > 0 &&
                  fact.replicationFactor > 0 && fact.divisibility > 0 &&
                  (!fact.boundValueKnown ||
                   (fact.boundValue > 0 &&
                    fact.boundValue <= fact.containerWidth));
  return fact;
}

// Recognize a prospective active-lane fact directly from TTIR dataflow:
//   make_range(0, W) < splat(entry_argument).
// The runtime value is intentionally not inspected here.  The fact only
// records which launch argument supplies the extent and the statically proven
// container bound.  This is independent of operator, shape, group-size value,
// benchmark identity and profitability.
std::optional<RuntimeActiveExtentFact>
runtimeActiveExtentFromMask(Value mask, FuncOp entry) {
  if (!mask)
    return std::nullopt;
  Operation *definition = mask.getDefiningOp();
  if (!definition)
    return std::nullopt;
  if (auto conjunction = dyn_cast<arith::AndIOp>(definition)) {
    if (auto lhs = runtimeActiveExtentFromMask(conjunction.getLhs(), entry))
      return lhs;
    return runtimeActiveExtentFromMask(conjunction.getRhs(), entry);
  }
  auto comparison = dyn_cast<arith::CmpIOp>(definition);
  if (!comparison ||
      (comparison.getPredicate() != arith::CmpIPredicate::slt &&
       comparison.getPredicate() != arith::CmpIPredicate::ult))
    return std::nullopt;
  Value local = stripShapeOnlyIntegerCasts(comparison.getLhs());
  Value extent = stripShapeOnlyIntegerCasts(comparison.getRhs());
  auto range = local.getDefiningOp<MakeRangeOp>();
  auto argument = dyn_cast<BlockArgument>(extent);
  if (!range || !argument || argument.getOwner() != &entry.getBody().front() ||
      range.getStartAttr().getInt() != 0 ||
      range.getEndAttr().getInt() < 1)
    return std::nullopt;
  int64_t width = staticElementCount(local.getType());
  if (width != range.getEndAttr().getInt())
    return std::nullopt;
  return RuntimeActiveExtentFact{
      static_cast<int64_t>(argument.getArgNumber()), width};
}
