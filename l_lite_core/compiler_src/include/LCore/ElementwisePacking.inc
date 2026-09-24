// Pack corresponding copies of any regionless elementwise tensor operation
// whose exact higher-rank form is admitted by MLIR's Elementwise trait.  This
// is an operation capability rule: it names neither a workload nor a favored
// opcode category.  Memory service and reductions retain their separate exact
// adapters because their packed semantics are not obtained by merely adding a
// leading tensor dimension.
int64_t materializeExactElementwiseOperationGroups(
    FuncOp func, StringRef groupAttribute, int64_t factor,
    std::optional<StringRef> subject = std::nullopt,
    ArrayRef<StringRef> auxiliaryLineage = {}) {
  if (factor < 2 || !llvm::isPowerOf2_64(factor))
    return 0;
  using GroupKey = std::pair<Block *, std::string>;
  std::map<GroupKey, SmallVector<Operation *>> groups;
  SmallVector<GroupKey> order;
  func.walk([&](Operation *operation) {
    auto owner = operation->getAttrOfType<StringAttr>(kRoleSubjectAttr);
    if (subject && (!owner || owner.getValue() != *subject))
      return;
    auto group = operation->getAttrOfType<StringAttr>(groupAttribute);
    auto partition = operation->getAttrOfType<StringAttr>(
        kUnrollPartitionLineageAttr);
    if (!group || (partition && partition.getValue() == "tail"))
      return;
    GroupKey key{operation->getBlock(), group.getValue().str()};
    auto [position, inserted] = groups.try_emplace(key);
    if (inserted)
      order.push_back(key);
    position->second.push_back(operation);
  });

  auto strippedAttributes = [&](Operation *operation) {
    NamedAttrList attributes(operation->getAttrs());
    attributes.erase(groupAttribute);
    for (StringRef attribute : auxiliaryLineage)
      attributes.erase(attribute);
    attributes.erase(kRoleAttr);
    attributes.erase(kRoleSubjectAttr);
    attributes.erase(kRoleIndexAttr);
    attributes.erase(kUnrollPartitionLineageAttr);
    return attributes.getDictionary(operation->getContext());
  };
  auto joinValues = [](OpBuilder &builder, Location loc,
                       ArrayRef<Value> values) -> FailureOr<Value> {
    if (values.empty() || !llvm::isPowerOf2_64(values.size()))
      return failure();
    SmallVector<Value> current(values.begin(), values.end());
    while (current.size() > 1) {
      SmallVector<Value> next;
      for (size_t index = 0; index < current.size(); index += 2)
        next.push_back(JoinOp::create(
            builder, loc, current[index], current[index + 1]));
      current = std::move(next);
    }
    return current.front();
  };
  auto splitValue = [](OpBuilder &builder, Location loc, Value input,
                       int64_t count) {
    SmallVector<Value> current{input};
    for (int64_t width = 1; width < count; width *= 2) {
      SmallVector<Value> next;
      for (Value value : current) {
        auto split = SplitOp::create(builder, loc, value);
        next.push_back(split.getOutLHS());
        next.push_back(split.getOutRHS());
      }
      current = std::move(next);
    }
    return current;
  };

  int64_t packedCount = 0;
  for (const GroupKey &key : order) {
    SmallVector<Operation *> operations = groups[key];
    if (operations.size() != static_cast<size_t>(factor))
      continue;
    llvm::sort(operations, [](Operation *left, Operation *right) {
      return left->isBeforeInBlock(right);
    });
    Operation *prototype = operations.front();
    if (!isGenericExactPackableOperation(prototype))
      continue;
    DictionaryAttr attributes = strippedAttributes(prototype);
    bool compatible = llvm::all_of(
        llvm::drop_begin(operations), [&](Operation *operation) {
          return operation->getName() == prototype->getName() &&
                 operation->getNumOperands() == prototype->getNumOperands() &&
                 operation->getNumResults() == prototype->getNumResults() &&
                 isGenericExactPackableOperation(operation) &&
                 strippedAttributes(operation) == attributes;
        });
    if (!compatible)
      continue;
    llvm::SmallPtrSet<Operation *, 16> groupSet(
        operations.begin(), operations.end());
    if (prototype->getNumOperands() == 0)
      continue;
    SmallVector<Type> resultElementTypes;
    for (unsigned resultIndex = 0;
         resultIndex < prototype->getNumResults(); ++resultIndex) {
      auto resultType = dyn_cast<RankedTensorType>(
          prototype->getResult(resultIndex).getType());
      if (!resultType || llvm::any_of(
              llvm::drop_begin(operations), [&](Operation *operation) {
                return operation->getResult(resultIndex).getType() !=
                       resultType;
              })) {
        compatible = false;
        break;
      }
      resultElementTypes.push_back(resultType.getElementType());
    }
    if (!compatible)
      continue;
    SmallVector<SmallVector<Value>> operandLanes(
        prototype->getNumOperands());
    for (unsigned operandIndex = 0;
         operandIndex < prototype->getNumOperands(); ++operandIndex) {
      Type expected = prototype->getOperand(operandIndex).getType();
      if (!isa<RankedTensorType>(expected)) {
        compatible = false;
        break;
      }
      for (Operation *operation : operations) {
        Value operand = operation->getOperand(operandIndex);
        Operation *definition = operand.getDefiningOp();
        if (operand.getType() != expected ||
            (definition && groupSet.contains(definition))) {
          compatible = false;
          break;
        }
        operandLanes[operandIndex].push_back(operand);
      }
      if (!compatible)
        break;
    }
    if (!compatible)
      continue;

    // The packed operation and its split leaves are inserted after the last
    // source-lane operation.  That point must dominate every original use.
    // A loop-carried chain can force lane 0's consumer to precede lane 1's
    // corresponding producer even when the two producers share a source
    // identity.  Packing that group would create use-before-definition IR.
    // This is an exact SSA boundary, not an operation-category restriction;
    // leave the group scalar and continue to other registered capabilities.
    Operation *insertionAnchor = operations.back();
    bool replacementDominatesAllUses = llvm::all_of(
        operations, [&](Operation *operation) {
          return llvm::all_of(operation->getResults(), [&](Value result) {
            return llvm::all_of(result.getUses(), [&](OpOperand &use) {
              Operation *topLevelUser =
                  key.first->findAncestorOpInBlock(*use.getOwner());
              return topLevelUser &&
                     insertionAnchor->isBeforeInBlock(topLevelUser);
            });
          });
        });
    if (!replacementDominatesAllUses)
      continue;

    OpBuilder builder(operations.back());
    builder.setInsertionPointAfter(operations.back());
    SmallVector<Value> packedOperands;
    for (ArrayRef<Value> lanes : operandLanes) {
      FailureOr<Value> packed = joinValues(
          builder, prototype->getLoc(), lanes);
      if (failed(packed)) {
        compatible = false;
        break;
      }
      packedOperands.push_back(*packed);
    }
    if (!compatible)
      continue;
    auto packedShapeType = dyn_cast<RankedTensorType>(
        packedOperands.front().getType());
    if (!packedShapeType)
      continue;
    SmallVector<Type> packedResultTypes;
    for (Type elementType : resultElementTypes)
      packedResultTypes.push_back(RankedTensorType::get(
          packedShapeType.getShape(), elementType,
          packedShapeType.getEncoding()));
    // Clone the registered operation so dialect properties that are not
    // represented in the generic attribute dictionary remain exact.  Only
    // operands, result tensor shape, and transient HBV lineage change.
    Operation *packed = prototype->clone();
    packed->setOperands(packedOperands);
    for (auto [result, type] :
         llvm::zip(packed->getResults(), packedResultTypes))
      result.setType(type);
    packed->removeAttr(groupAttribute);
    for (StringRef attribute : auxiliaryLineage)
      packed->removeAttr(attribute);
    packed->removeAttr(kRoleAttr);
    packed->removeAttr(kRoleSubjectAttr);
    packed->removeAttr(kRoleIndexAttr);
    packed->removeAttr(kUnrollPartitionLineageAttr);
    // Logical packing adds one or more tensor dimensions.  Axis-analysis
    // hints copied from the scalar-lane prototype describe the old rank and
    // are therefore not facts about this result.  Keeping them can give
    // AxisInfo unequal dimension-vector lengths and, more importantly, would
    // assert a layout property that this exact transformation did not prove.
    // Drop only these derived hints and let the downstream analysis infer a
    // conservative, shape-consistent state for the packed operation.
    packed->removeAttr("tt.contiguity");
    packed->removeAttr("tt.divisibility");
    packed->removeAttr("tt.constancy");
    builder.insert(packed);
    for (unsigned resultIndex = 0; resultIndex < packed->getNumResults();
         ++resultIndex) {
      SmallVector<Value> pieces = splitValue(
          builder, prototype->getLoc(), packed->getResult(resultIndex), factor);
      for (auto [operation, piece] : llvm::zip(operations, pieces))
        operation->getResult(resultIndex).replaceAllUsesWith(piece);
    }
    for (Operation *operation : llvm::reverse(operations))
      operation->erase();
    ++packedCount;
  }
  if (packedCount) {
    int64_t prior = 0;
    if (auto count = func->getAttrOfType<IntegerAttr>(
            kVectorizedOperationGroupCountAttr))
      prior = count.getInt();
    func->setAttr(
        kVectorizedOperationGroupCountAttr,
        IntegerAttr::get(IntegerType::get(func.getContext(), 64),
                         prior + packedCount));
  }
  return packedCount;
}
