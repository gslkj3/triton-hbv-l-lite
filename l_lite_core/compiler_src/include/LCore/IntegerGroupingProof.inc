// Structural signatures for exact integer grouping. These are shared by
// admission and postcondition checks, not workload-specific patterns.
bool isRankOneIntegerVector(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.getRank() == 1 && !type.isDynamicDim(0) &&
         type.getDimSize(0) > 0 && isa<IntegerType>(type.getElementType());
}

bool isLogicalIntegerGroupingReduction(ReduceOp reduce,
                                       int64_t groupingFactor) {
  if (reduce.getSrcs().size() != 1 || reduce.getResults().size() != 1 ||
      reduce.getAxis() != 0)
    return false;
  auto sourceType = dyn_cast<RankedTensorType>(
      reduce.getSrcs().front().getType());
  auto resultType = dyn_cast<RankedTensorType>(
      reduce.getResult().front().getType());
  if (!sourceType || !resultType || sourceType.getRank() != 2 ||
      resultType.getRank() != 1 || sourceType.isDynamicDim(0) ||
      sourceType.isDynamicDim(1) || resultType.isDynamicDim(0) ||
      sourceType.getDimSize(0) != groupingFactor ||
      sourceType.getDimSize(1) != resultType.getDimSize(0) ||
      sourceType.getElementType() != resultType.getElementType() ||
      !isa<IntegerType>(sourceType.getElementType()))
    return false;
  Block &body = reduce.getCombineOp().front();
  if (body.getNumArguments() != 2 ||
      std::distance(body.begin(), body.end()) != 2)
    return false;
  auto add = dyn_cast<arith::AddIOp>(body.front());
  auto yield = dyn_cast<ReduceReturnOp>(body.back());
  return add && yield && yield.getNumOperands() == 1 &&
         yield.getOperand(0) == add.getResult() &&
         ((add.getLhs() == body.getArgument(0) &&
           add.getRhs() == body.getArgument(1)) ||
          (add.getRhs() == body.getArgument(0) &&
           add.getLhs() == body.getArgument(1)));
}
