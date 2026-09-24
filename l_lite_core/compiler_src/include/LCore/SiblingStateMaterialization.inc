// Shared exact sibling-state packing; no plan parser or route selection.
Value joinStateAxisTensor(OpBuilder &builder, Location loc,
                          ArrayRef<Value> leaves) {
  assert(!leaves.empty() && llvm::isPowerOf2_64(leaves.size()));
  if (leaves.size() == 1)
    return leaves.front();
  size_t middle = leaves.size() / 2;
  Value lhs = joinStateAxisTensor(builder, loc, leaves.take_front(middle));
  Value rhs = joinStateAxisTensor(builder, loc, leaves.drop_front(middle));
  return JoinOp::create(builder, loc, lhs, rhs);
}

void splitStateAxisTensor(OpBuilder &builder, Location loc, Value value,
                          int64_t leafCount,
                          SmallVectorImpl<Value> &leaves) {
  assert(leafCount >= 1 && llvm::isPowerOf2_64(leafCount));
  if (leafCount == 1) {
    leaves.push_back(value);
    return;
  }
  auto halves = SplitOp::create(builder, loc, value);
  splitStateAxisTensor(builder, loc, halves.getOutLHS(), leafCount / 2,
                       leaves);
  splitStateAxisTensor(builder, loc, halves.getOutRHS(), leafCount / 2,
                       leaves);
}

Type packedStateAxisType(Type type, int64_t paddedCardinality) {
  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    SmallVector<int64_t> shape(tensor.getShape());
    shape.push_back(paddedCardinality);
    return RankedTensorType::get(
        shape, tensor.getElementType(), tensor.getEncoding());
  }
  if (isa<FloatType>(type))
    return RankedTensorType::get({paddedCardinality}, type);
  return Type();
}

class StateAxisBundlePacker {
public:
  StateAxisBundlePacker(OpBuilder &builder, Location location,
                        int64_t paddedCardinality)
      : builder(builder), loc(location), padded(paddedCardinality) {}

  FailureOr<Value> pack(ArrayRef<Value> values) {
    for (auto &item : memo)
      if (item.first == values)
        return item.second;
    if (values.empty())
      return failure();
    Type resultType = packedStateAxisType(values.front().getType(), padded);
    if (!resultType || llvm::any_of(values, [&](Value value) {
          return value.getType() != values.front().getType();
        }))
      return failure();
    FailureOr<Value> result = allSameStateAxisValues(values)
                                  ? broadcast(values.front(), resultType)
                                  : transformOrBoundary(values, resultType);
    if (succeeded(result))
      memo.push_back({SmallVector<Value>(values), *result});
    return result;
  }

private:
  FailureOr<Value> broadcast(Value value, Type resultType) {
    auto packedType = cast<RankedTensorType>(resultType);
    if (isa<FloatType>(value.getType()))
      return SplatOp::create(builder, loc, packedType, value).getResult();
    auto source = dyn_cast<RankedTensorType>(value.getType());
    if (!source || source.getRank() != 1)
      return failure();
    Value expanded = ExpandDimsOp::create(builder, loc, value, 1);
    return BroadcastOp::create(builder, loc, packedType, expanded).getResult();
  }

  FailureOr<Value> zeroLike(Type type) {
    Type elementType = isa<RankedTensorType>(type)
                           ? cast<RankedTensorType>(type).getElementType()
                           : type;
    auto floatType = dyn_cast<FloatType>(elementType);
    // Boundary padding is part of the floating-point sibling-state contract.
    // Scalar integer/index operands can occur in a recursive operation slice,
    // but inventing a zero of that type would silently change the state-axis
    // semantics.  Return a typed failure instead of asserting in MLIR's cast.
    if (!floatType)
      return failure();
    Value zero = arith::ConstantOp::create(
        builder, loc, FloatAttr::get(floatType, 0.0));
    if (auto tensor = dyn_cast<RankedTensorType>(type))
      return SplatOp::create(builder, loc, tensor, zero).getResult();
    return zero;
  }

  FailureOr<Value> boundary(ArrayRef<Value> values, Type resultType) {
    SmallVector<Value> leaves;
    auto originalTensor = dyn_cast<RankedTensorType>(values.front().getType());
    for (Value value : values) {
      if (originalTensor)
        leaves.push_back(value);
      else {
        auto singleton = RankedTensorType::get({1}, value.getType());
        leaves.push_back(SplatOp::create(builder, loc, singleton, value));
      }
    }
    while (leaves.size() < static_cast<size_t>(padded)) {
      FailureOr<Value> zero = zeroLike(values.front().getType());
      if (failed(zero))
        return failure();
      if (originalTensor)
        leaves.push_back(*zero);
      else {
        auto singleton = RankedTensorType::get({1}, zero->getType());
        leaves.push_back(SplatOp::create(builder, loc, singleton, *zero));
      }
    }
    Value joined = joinStateAxisTensor(builder, loc, leaves);
    return ReshapeOp::create(
        builder, loc, cast<RankedTensorType>(resultType).getShape(), joined,
        /*allowReorder=*/false).getResult();
  }

  FailureOr<Value> transformOrBoundary(ArrayRef<Value> values,
                                       Type resultType) {
    Operation *prototype = nullptr;
    if (!stateAxisOperationsEquivalent(values, prototype))
      return boundary(values, resultType);
    SmallVector<Value> operands;
    for (unsigned operand = 0; operand < prototype->getNumOperands(); ++operand) {
      SmallVector<Value> laneOperands;
      for (Value value : values)
        laneOperands.push_back(value.getDefiningOp()->getOperand(operand));
      auto packedOperand = pack(laneOperands);
      if (failed(packedOperand))
        return failure();
      operands.push_back(*packedOperand);
    }
    if (auto reduce = dyn_cast<ReduceOp>(prototype)) {
      auto packedReduce = ReduceOp::create(
          builder, loc, ValueRange{operands.front()}, reduce.getAxis());
      IRMapping mapping;
      reduce.getCombineOp().cloneInto(
          &packedReduce.getCombineOp(), mapping);
      packedReduce->setAttr(
          kStateAxisArtifactAttr,
          builder.getStringAttr("packed_node"));
      packedReduce->setAttr(kStateAxisPackableNodeAttr,
                            builder.getUnitAttr());
      return packedReduce.getResult().front();
    }
    if (isa<SplatOp>(prototype)) {
      auto resultTensor = cast<RankedTensorType>(resultType);
      Value expanded = ExpandDimsOp::create(
          builder, loc, operands.front(), 0);
      Value broadcast = BroadcastOp::create(
          builder, loc, resultTensor, expanded).getResult();
      broadcast.getDefiningOp()->setAttr(
          kStateAxisArtifactAttr, builder.getStringAttr("packed_node"));
      broadcast.getDefiningOp()->setAttr(kStateAxisPackableNodeAttr,
                                         builder.getUnitAttr());
      return broadcast;
    }
    OperationState state(loc, prototype->getName());
    state.addOperands(operands);
    state.addTypes(resultType);
    state.addAttributes(prototype->getAttrs());
    Operation *created = builder.create(state);
    created->setAttr(kStateAxisArtifactAttr,
                     builder.getStringAttr("packed_node"));
    created->setAttr(kStateAxisPackableNodeAttr, builder.getUnitAttr());
    return created->getResult(0);
  }

  OpBuilder &builder;
  Location loc;
  int64_t padded;
  SmallVector<std::pair<SmallVector<Value>, Value>> memo;
};

FailureOr<Value> cloneStateAxisCrossGraph(
    OpBuilder &builder, Value value, ArrayRef<Value> oldLeaves,
    ArrayRef<Value> newLeaves, llvm::DenseMap<Value, Value> &memo) {
  for (auto [oldValue, newValue] : llvm::zip(oldLeaves, newLeaves))
    if (value == oldValue)
      return newValue;
  if (auto found = memo.find(value); found != memo.end())
    return found->second;
  bool depends = llvm::any_of(oldLeaves, [&](Value leaf) {
    return valueDependsOn(value, leaf);
  });
  if (!depends)
    return value;
  Operation *operation = value.getDefiningOp();
  if (!operation || operation->getNumResults() != 1 ||
      operation->getNumRegions() != 0 ||
      !isStateAxisElementwiseCapability(operation))
    return failure();
  IRMapping mapping;
  for (Value operand : operation->getOperands()) {
    auto replacement = cloneStateAxisCrossGraph(
        builder, operand, oldLeaves, newLeaves, memo);
    if (failed(replacement))
      return failure();
    mapping.map(operand, *replacement);
  }
  Operation *cloned = builder.clone(*operation, mapping);
  // The subject marker belongs to the original Provider anchor.  Copying it
  // into the cloned cross-state graph would counterfeit a second admitted
  // subject and break one-to-one Plan/artifact correspondence.
  cloned->removeAttr(kSubjectAttr);
  memo[value] = cloned->getResult(0);
  return cloned->getResult(0);
}

LogicalResult materializeStateAxisNormalization(
    StateAxisNormalizationRegion subject) {
  Location loc = subject.finalLaneOperations.front()->getLoc();
  OpBuilder builder(subject.finalLaneOperations.front());
  // State-axis packing is a tensor-domain materializer.  A scalar carried
  // value may still be present in the same kernel (for example an i32 prefix
  // accumulator), but it is not a valid sibling lane.  The previous direct
  // cast turned this ordinary structural mismatch into an LLVM assertion and
  // aborted the whole TritonGPU pipeline.  Keep the rule generic: reject the
  // subject with a typed pass failure and let the caller retain Original.
  auto inputType = dyn_cast<RankedTensorType>(subject.laneValues[0].getType());
  if (!inputType || inputType.getRank() != 1 ||
      !isa<FloatType>(inputType.getElementType()))
    return failure();
  int64_t padded = subject.paddedStateCardinality;
  auto packedType = dyn_cast<RankedTensorType>(
      packedStateAxisType(inputType, padded));
  if (!packedType)
    return failure();
  StateAxisBundlePacker packer(builder, loc, padded);
  auto packedLaneValues = packer.pack(subject.laneValues);
  if (failed(packedLaneValues) ||
      (*packedLaneValues).getType() != packedType)
    return failure();
  (*packedLaneValues).getDefiningOp()->setAttr(
      kStateAxisArtifactAttr, builder.getStringAttr("packed_root"));
  SmallVector<int64_t> nestedShape{subject.stateWidth};
  for (int64_t count = padded; count > 1; count /= 2)
    nestedShape.push_back(2);
  Value nestedRows = ReshapeOp::create(
      builder, loc, nestedShape, *packedLaneValues,
      /*allowReorder=*/false);
  SmallVector<Value> unpackedLanes;
  splitStateAxisTensor(builder, loc, nestedRows, padded, unpackedLanes);
  llvm::DenseMap<Value, Value> crossMemo;
  auto candidateDenominator = cloneStateAxisCrossGraph(
      builder, subject.columnDenominator, subject.laneValues,
      ArrayRef<Value>(unpackedLanes).take_front(subject.stateCardinality),
      crossMemo);
  if (failed(candidateDenominator))
    return failure();
  Value expandedColumns = ExpandDimsOp::create(
      builder, loc, *candidateDenominator, 1);
  Value broadcastColumns = BroadcastOp::create(
      builder, loc, packedType, expandedColumns);
  Operation *prototype = subject.finalLaneOperations.front();
  OperationState finalState(loc, prototype->getName());
  finalState.addOperands(subject.laneOperandIndex == 0
                             ? ValueRange{*packedLaneValues, broadcastColumns}
                             : ValueRange{broadcastColumns, *packedLaneValues});
  finalState.addTypes(packedType);
  finalState.addAttributes(prototype->getAttrs());
  Operation *packedFinal = builder.create(finalState);
  packedFinal->setAttr(
      kStateAxisArtifactAttr,
      builder.getStringAttr("cross_state_consumer"));
  Value nestedOutputs = ReshapeOp::create(
      builder, loc, nestedShape, packedFinal->getResult(0),
      /*allowReorder=*/false);
  SmallVector<Value> outputs;
  splitStateAxisTensor(builder, loc, nestedOutputs, padded, outputs);
  for (unsigned index = 0; index < subject.stateCardinality; ++index)
    subject.outputs[index].replaceAllUsesWith(outputs[index]);
  for (Operation *operation : subject.finalLaneOperations)
    operation->erase();
  return success();
}
