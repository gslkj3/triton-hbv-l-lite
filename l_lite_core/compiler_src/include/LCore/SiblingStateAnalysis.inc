// Isomorphic sibling-state analysis. No workload or fixed-cardinality dispatch.
bool isStateAxisElementwiseCapability(Operation *operation) {
  if (!operation || operation->getNumResults() != 1 ||
      operation->getNumOperands() == 0 || operation->getNumRegions() != 0 ||
      !operation->hasTrait<OpTrait::Elementwise>() ||
      !operation->hasTrait<OpTrait::SameOperandsAndResultType>() ||
      !isMemoryEffectFree(operation) || !isSpeculatable(operation))
    return false;
  // Appending a sibling axis preserves an elementwise operation. This packer
  // supports floating scalar/rank-one values and zero padding; conversions,
  // mixed types and operations with regions need their own transport proof.
  Type type = operation->getResult(0).getType();
  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    if (tensor.getRank() != 1)
      return false;
    type = tensor.getElementType();
  }
  return isa<FloatType>(type);
}

std::optional<StringRef> stateAxisReductionCombiner(ReduceOp reduce) {
  if (reduce.getSrcs().size() != 1 || reduce.getResults().size() != 1 ||
      !llvm::hasSingleElement(reduce.getCombineOp()))
    return std::nullopt;
  Block &block = reduce.getCombineOp().front();
  if (block.getNumArguments() != 2 ||
      std::distance(block.begin(), block.end()) != 2)
    return std::nullopt;
  Operation &combine = block.front();
  auto result = dyn_cast<ReduceReturnOp>(block.back());
  if (!result || combine.getNumResults() != 1 ||
      result.getNumOperands() != 1 ||
      result.getOperand(0) != combine.getResult(0) ||
      !isStateAxisElementwiseCapability(&combine))
    return std::nullopt;
  return combine.getName().getStringRef();
}

bool allSameStateAxisValues(ArrayRef<Value> values) {
  return !values.empty() &&
         llvm::all_of(values.drop_front(),
                      [&](Value value) { return value == values.front(); });
}

bool stateAxisOperationsEquivalent(ArrayRef<Value> values,
                                   Operation *&prototype) {
  if (values.empty() || allSameStateAxisValues(values))
    return false;
  prototype = values.front().getDefiningOp();
  if (!prototype || prototype->getNumResults() != 1 ||
      prototype->getResult(0) != values.front())
    return false;
  for (Value value : values.drop_front()) {
    Operation *operation = value.getDefiningOp();
    if (!operation || operation->getName() != prototype->getName() ||
        operation->getNumOperands() != prototype->getNumOperands() ||
        operation->getNumResults() != 1 ||
        operation->getResult(0) != value ||
        operation->getAttrDictionary() != prototype->getAttrDictionary())
      return false;
  }
  if (auto reduce = dyn_cast<ReduceOp>(prototype)) {
    auto combiner = stateAxisReductionCombiner(reduce);
    return combiner && llvm::all_of(values.drop_front(), [&](Value value) {
      auto other = cast<ReduceOp>(value.getDefiningOp());
      return other.getAxis() == reduce.getAxis() &&
             stateAxisReductionCombiner(other) == combiner &&
             OperationEquivalence::isRegionEquivalentTo(
                 &reduce.getCombineOp(), &other.getCombineOp(),
                 OperationEquivalence::Flags::IgnoreLocations);
    });
  }
  if (isa<SplatOp>(prototype))
    return true;
  return isStateAxisElementwiseCapability(prototype);
}

bool stateAxisCrossGraphClosed(Value value, ArrayRef<Value> laneValues,
                               Block *block,
                               std::set<std::string> &capabilities,
                               llvm::SmallPtrSetImpl<Value> &active) {
  if (llvm::is_contained(laneValues, value) ||
      llvm::none_of(laneValues, [&](Value lane) {
        return valueDependsOn(value, lane);
      }))
    return true;
  if (!active.insert(value).second)
    return false;
  auto erase = llvm::make_scope_exit([&] { active.erase(value); });
  Operation *operation = value.getDefiningOp();
  if (!operation || operation->getBlock() != block ||
      operation->getNumResults() != 1 || operation->getResult(0) != value ||
      operation->getNumRegions() != 0 ||
      !isStateAxisElementwiseCapability(operation))
    return false;
  capabilities.insert("cross_graph:" +
                      operation->getName().getStringRef().str());
  return llvm::all_of(operation->getOperands(), [&](Value operand) {
    return stateAxisCrossGraphClosed(operand, laneValues, block,
                                     capabilities, active);
  });
}

std::string describeStateAxisBundleImpl(
    ArrayRef<Value> values, std::set<std::string> &capabilities,
    int64_t &packableNodeCount,
    SmallVectorImpl<SmallVector<Value>> &visitedBundles) {
  if (values.empty())
    return "empty";
  if (allSameStateAxisValues(values)) {
    capabilities.insert("uniform_broadcast");
    return "uniform";
  }
  Operation *prototype = nullptr;
  if (!stateAxisOperationsEquivalent(values, prototype)) {
    capabilities.insert("boundary_pack");
    return "boundary";
  }
  for (auto [index, bundle] : llvm::enumerate(visitedBundles))
    if (ArrayRef<Value>(bundle) == values)
      return "ref" + std::to_string(index);
  visitedBundles.push_back(SmallVector<Value>(values));
  ++packableNodeCount;
  std::string kind = prototype->getName().getStringRef().str();
  if (auto reduce = dyn_cast<ReduceOp>(prototype))
    kind += "[axis=" + std::to_string(reduce.getAxis()) +
            ",combiner=" + stateAxisReductionCombiner(reduce)->str() + "]";
  capabilities.insert(kind);
  std::string signature = kind + "(";
  for (unsigned operand = 0; operand < prototype->getNumOperands(); ++operand) {
    if (operand)
      signature += ",";
    SmallVector<Value> operands;
    for (Value value : values)
      operands.push_back(value.getDefiningOp()->getOperand(operand));
    signature += describeStateAxisBundleImpl(
        operands, capabilities, packableNodeCount, visitedBundles);
  }
  signature += ")";
  return signature;
}

std::string describeStateAxisBundle(
    ArrayRef<Value> values, std::set<std::string> &capabilities,
    int64_t &packableNodeCount) {
  SmallVector<SmallVector<Value>> visitedBundles;
  return describeStateAxisBundleImpl(
      values, capabilities, packableNodeCount, visitedBundles);
}

std::optional<StateAxisNormalizationRegion>
matchStateAxisNormalization(Value denominator) {
  auto tensorType = dyn_cast<RankedTensorType>(denominator.getType());
  if (!tensorType || tensorType.getRank() != 1 ||
      !isa<FloatType>(tensorType.getElementType()))
    return std::nullopt;

  // The shared cross-sibling value is discovered from its isomorphic lane
  // consumers, not from a named normalization sequence.  Any registered
  // binary operation may consume it; the other operand supplies each sibling
  // lane.  The exact upstream cross-lane graph is proved and later cloned
  // without reassociation.
  SmallVector<Operation *> users(denominator.getUsers());
  for (Operation *prototype : users) {
    if (!isStateAxisElementwiseCapability(prototype) ||
        prototype->getNumOperands() != 2 ||
        prototype->getNumResults() != 1)
      continue;
    int64_t sharedOperand = prototype->getOperand(0) == denominator ? 0
                            : prototype->getOperand(1) == denominator ? 1
                                                                      : -1;
    if (sharedOperand < 0)
      continue;
    int64_t laneOperand = 1 - sharedOperand;
    Value firstLane = prototype->getOperand(laneOperand);
    if (firstLane.getType() != tensorType)
      continue;
    SmallVector<Operation *> laneOperations;
    SmallVector<Value> laneValues;
    for (Operation *candidate : users) {
      if (candidate->getName() != prototype->getName() ||
          candidate->getAttrDictionary() != prototype->getAttrDictionary() ||
          candidate->getBlock() != prototype->getBlock() ||
          candidate->getNumOperands() != 2 ||
          candidate->getNumResults() != 1 ||
          candidate->getOperand(sharedOperand) != denominator ||
          candidate->getOperand(laneOperand).getType() != tensorType)
        continue;
      laneOperations.push_back(candidate);
    }
    llvm::sort(laneOperations, [](Operation *lhs, Operation *rhs) {
      return lhs->isBeforeInBlock(rhs);
    });
    for (Operation *operation : laneOperations)
      laneValues.push_back(operation->getOperand(laneOperand));
    if (laneValues.size() < 2 ||
        llvm::SmallPtrSet<Value, 8>(laneValues.begin(), laneValues.end()).size()
            != laneValues.size() ||
        llvm::any_of(laneValues, [&](Value lane) {
          return !valueDependsOn(denominator, lane);
        }))
      continue;

    StateAxisNormalizationRegion result;
    result.laneValues = laneValues;
    result.columnDenominator = denominator;
    result.finalLaneOperations = laneOperations;
    result.laneOperandIndex = laneOperand;
    for (Operation *operation : laneOperations)
      result.outputs.push_back(operation->getResult(0));
    result.stateCardinality = laneValues.size();
    result.paddedStateCardinality = llvm::PowerOf2Ceil(
        static_cast<uint64_t>(result.stateCardinality));
    result.stateWidth = tensorType.getShape()[0];
    auto elementType = cast<FloatType>(tensorType.getElementType());
    result.elementBytes = (elementType.getWidth() + 7) / 8;
    // Provider admission is structural.  Acquisition/code-size policy may
    // later reject a large bundle, but a local literal capacity must not turn
    // one training shape into the semantic definition of this mechanism.
    if (result.paddedStateCardinality <= 0 || result.stateWidth <= 0)
      continue;
    llvm::SmallPtrSet<Value, 32> active;
    if (!stateAxisCrossGraphClosed(
            denominator, result.laneValues, prototype->getBlock(),
            result.operationCapabilities, active))
      continue;
    result.operationCapabilities.insert(
        "cross_state_consumer:" +
        prototype->getName().getStringRef().str());
    result.graphSignature = describeStateAxisBundle(
        result.laneValues, result.operationCapabilities,
        result.packableNodeCount);
    result.graphSignature =
        "shared_cross_graph{" + result.graphSignature + "}->" +
        prototype->getName().getStringRef().str() +
        "[lane_operand=" + std::to_string(laneOperand) + "]";
    if (result.packableNodeCount < 1)
      continue;
    return result;
  }
  return std::nullopt;
}

SmallVector<StateAxisNormalizationRegion, 2>
collectStateAxisNormalizations(FuncOp entry) {
  SmallVector<StateAxisNormalizationRegion, 2> result;
  llvm::SmallPtrSet<Value, 8> seenDenominators;
  entry.walk([&](Operation *operation) {
    if (operation->getNumResults() != 1)
      return;
    Value denominator = operation->getResult(0);
    if (!seenDenominators.insert(denominator).second)
      return;
    if (auto region = matchStateAxisNormalization(denominator))
      result.push_back(*region);
  });
  return result;
}
