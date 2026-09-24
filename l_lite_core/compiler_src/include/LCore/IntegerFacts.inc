// Integer observations used by address and control-flow proofs.
// Runtime bindings are observations, not static unroll authorization.
std::optional<int64_t> splatIntegerImpl(Value value, unsigned &budget) {
  if (!budget)
    return std::nullopt;
  --budget;
  Type elementType = getElementTypeOrSelf(value.getType());
  auto integerType = dyn_cast<IntegerType>(elementType);
  unsigned width = isa<IndexType>(elementType) ? 64 :
                   integerType ? integerType.getWidth() : 0;
  if (!width || width > 64)
    return std::nullopt;
  auto representable = [&](const __int128 result) -> std::optional<int64_t> {
    auto integer = checkedI64(result);
    if (!integer || !llvm::isIntN(width, *integer))
      return std::nullopt;
    return integer;
  };
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (auto function = dyn_cast_or_null<FuncOp>(
            argument.getOwner()->getParentOp()))
      if (auto bound = function.getArgAttrOfType<IntegerAttr>(
              argument.getArgNumber(), kBridgeBoundScalarAttr))
        return representable(bound.getInt());
  }
  APInt scalar;
  if (matchPattern(value, m_ConstantInt(&scalar)))
    return scalar.getSExtValue();
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  auto dense = constant ? dyn_cast<DenseIntElementsAttr>(constant.getValue())
                        : DenseIntElementsAttr();
  if (dense && dense.isSplat())
    return dense.getSplatValue<APInt>().getSExtValue();
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return std::nullopt;
  StringRef name = definition->getName().getStringRef();
  if (definition->getNumOperands() == 1 &&
      (name == "tt.splat" || name == "tt.broadcast" ||
      name == "tt.expand_dims" || name == "tt.reshape" ||
      name == "tt.trans" || name == "arith.extsi"))
    return splatIntegerImpl(definition->getOperand(0), budget);
  if (name == "arith.extui" && definition->getNumOperands() == 1) {
    Value input = definition->getOperand(0);
    auto inputType = dyn_cast<IntegerType>(getElementTypeOrSelf(input.getType()));
    auto observed = splatIntegerImpl(input, budget);
    if (!inputType || !observed || inputType.getWidth() >= width)
      return std::nullopt;
    return representable(APInt(inputType.getWidth(), *observed).getZExtValue());
  }
  if (definition->getNumOperands() == 2) {
    if (auto flags = definition->getAttrOfType<arith::IntegerOverflowFlagsAttr>(
            "overflowFlags"))
      if (flags.getValue() != arith::IntegerOverflowFlags::none)
        return std::nullopt;
    auto lhs = splatIntegerImpl(definition->getOperand(0), budget);
    auto rhs = splatIntegerImpl(definition->getOperand(1), budget);
    if (!lhs || !rhs)
      return std::nullopt;
    if (name == "arith.addi")
      return representable(static_cast<__int128>(*lhs) + *rhs);
    if (name == "arith.subi")
      return representable(static_cast<__int128>(*lhs) - *rhs);
    if (name == "arith.muli")
      return representable(static_cast<__int128>(*lhs) * *rhs);
    if (name == "arith.divui" && (*lhs < 0 || *rhs < 0))
      return std::nullopt;
    if ((name == "arith.divsi" || name == "arith.divui") && *rhs != 0)
      return representable(static_cast<__int128>(*lhs) / *rhs);
    if ((name == "arith.ceildivsi" || name == "arith.ceildivui") &&
        *lhs >= 0 && *rhs > 0)
      return representable(
          (static_cast<__int128>(*lhs) + *rhs - 1) / *rhs);
  }
  return std::nullopt;
}

std::optional<int64_t> splatInteger(Value value) {
  unsigned budget = 256;
  return splatIntegerImpl(value, budget);
}


// Observation only: do not reuse this launch binding as static unroll proof.
// Reject overflowing intermediates rather than approximate fixed-width wrap
// or poison semantics. A finite visit budget bounds arbitrary SSA expression cost.
std::optional<int64_t> launchIntegerObservation(Value value, unsigned &budget) {
  if (!budget)
    return std::nullopt;
  --budget;
  auto type = dyn_cast<IntegerType>(value.getType());
  if (!type || type.getWidth() > 64)
    return std::nullopt;
  auto representable = [&](const __int128 result) -> std::optional<int64_t> {
    auto integer = checkedI64(result);
    if (!integer || !llvm::isIntN(type.getWidth(), *integer))
      return std::nullopt;
    return integer;
  };
  APInt constant;
  if (matchPattern(value, m_ConstantInt(&constant)))
    return constant.getSExtValue();
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    auto function = dyn_cast_or_null<FuncOp>(argument.getOwner()->getParentOp());
    if (!function || argument.getOwner() != &function.getBody().front())
      return std::nullopt;
    auto binding = function.getArgAttrOfType<IntegerAttr>(
        argument.getArgNumber(), kBridgeBoundScalarAttr);
    return binding ? representable(binding.getInt()) : std::nullopt;
  }
  Operation *op = value.getDefiningOp();
  if (!op || op->getNumOperands() != 2)
    return std::nullopt;
  StringRef name = op->getName().getStringRef();
  if (name != "arith.addi" && name != "arith.subi" &&
      name != "arith.muli" && name != "arith.divsi" &&
      name != "arith.divui" && name != "arith.remsi")
    return std::nullopt;
  // Signed representability alone does not prove unsigned no-wrap flags.
  // Keep flagged expressions unknown until those contracts are evaluated.
  if (auto flags = op->getAttrOfType<arith::IntegerOverflowFlagsAttr>(
          "overflowFlags"))
    if (flags.getValue() != arith::IntegerOverflowFlags::none)
      return std::nullopt;
  auto lhs = launchIntegerObservation(op->getOperand(0), budget);
  auto rhs = launchIntegerObservation(op->getOperand(1), budget);
  if (!lhs || !rhs)
    return std::nullopt;
  if (name == "arith.addi")
    return representable(static_cast<__int128>(*lhs) + *rhs);
  if (name == "arith.subi")
    return representable(static_cast<__int128>(*lhs) - *rhs);
  if (name == "arith.muli")
    return representable(static_cast<__int128>(*lhs) * *rhs);
  if (*rhs == 0)
    return std::nullopt;
  // For unsigned division, the signed observation representation is exact
  // only on the nonnegative common range. Do not reinterpret negative values
  // as large unsigned bounds or silently approximate them.
  if (name == "arith.divui" && (*lhs < 0 || *rhs < 0))
    return std::nullopt;
  if (name == "arith.remsi") {
    // Conservatively reject the signed minimum / -1 overflow case as well.
    if (!representable(static_cast<__int128>(*lhs) / *rhs))
      return std::nullopt;
    return representable(static_cast<__int128>(*lhs) % *rhs);
  }
  return representable(static_cast<__int128>(*lhs) / *rhs);
}
