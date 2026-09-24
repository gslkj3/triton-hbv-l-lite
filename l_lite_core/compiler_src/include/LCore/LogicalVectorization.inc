// Exact corresponding-operation vectorization for an existing loop.  The
// operation-neutral scheduler first exposes source-operation groups.  A
// regionless elementwise tensor group may then use its generic packed form;
// loads retain a dedicated exact adapter because memory semantics are not an
// instance of the Elementwise trait.  No opcode category is required.
LogicalResult materializeExactOperationVectorization(
    FuncOp func, std::optional<StringRef> subject, int64_t unrollFactor,
    std::string *failureReason = nullptr) {
  auto reject = [&](StringRef reason) -> LogicalResult {
    if (failureReason)
      *failureReason = reason.str();
    return failure();
  };
  if (unrollFactor < 2 || !llvm::isPowerOf2_64(unrollFactor))
    return reject("existing_loop_route_factor_is_not_a_power_of_two_at_least_two");
  if (failed(materializeOperationNeutralPhase(
          func, subject, /*clearLineage=*/false)))
    return reject("existing_loop_operation_neutral_phase_schedule_failed");

  int64_t groupedElementwise = materializeExactElementwiseOperationGroups(
      func, kOperationGroupRefAttr, unrollFactor, subject);

  llvm::DenseMap<Block *, std::map<std::string, SmallVector<LoadOp>>>
      groupsByBlock;
  SmallVector<Block *> blocks;
  bool malformed = false;
  func.walk([&](Operation *op) {
    auto role = op->getAttrOfType<StringAttr>(kRoleAttr);
    if (!role)
      return;
    auto roleSubject = op->getAttrOfType<StringAttr>(kRoleSubjectAttr);
    if (subject && (!roleSubject || roleSubject.getValue() != *subject))
      return;
    auto load = dyn_cast<LoadOp>(op);
    if (!load)
      return;
    auto group = op->getAttrOfType<StringAttr>(kOperationGroupRefAttr);
    auto partition = op->getAttrOfType<StringAttr>(
        kUnrollPartitionLineageAttr);
    if (partition && partition.getValue() == "tail")
      return;
    if (!group) {
      malformed = true;
      return;
    }
    Block *block = op->getBlock();
    if (!groupsByBlock.contains(block))
      blocks.push_back(block);
    groupsByBlock[block][group.getValue().str()].push_back(load);
  });
  if (malformed)
    return reject("existing_loop_load_clone_has_no_source_operation_group");

  auto joinValues = [](OpBuilder &builder, Location loc,
                       ArrayRef<Value> inputs) -> FailureOr<Value> {
    if (inputs.empty() || !llvm::isPowerOf2_64(inputs.size()))
      return failure();
    SmallVector<Value> current(inputs.begin(), inputs.end());
    while (current.size() > 1) {
      SmallVector<Value> next;
      for (size_t index = 0; index < current.size(); index += 2)
        next.push_back(
            JoinOp::create(builder, loc, current[index], current[index + 1]));
      current = std::move(next);
    }
    return current.front();
  };
  auto splitValue = [](OpBuilder &builder, Location loc, Value input,
                       int64_t count) -> SmallVector<Value> {
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

  using StoreGroupKey = std::pair<Block *, std::string>;
  std::map<StoreGroupKey, SmallVector<StoreOp>> storeGroups;
  SmallVector<StoreGroupKey> storeOrder;
  func.walk([&](StoreOp store) {
    auto roleSubject = store->getAttrOfType<StringAttr>(kRoleSubjectAttr);
    if (subject && (!roleSubject || roleSubject.getValue() != *subject))
      return;
    auto group = store->getAttrOfType<StringAttr>(kOperationGroupRefAttr);
    auto partition = store->getAttrOfType<StringAttr>(
        kUnrollPartitionLineageAttr);
    if (!group || (partition && partition.getValue() == "tail"))
      return;
    StoreGroupKey key{store->getBlock(), group.getValue().str()};
    auto [position, inserted] = storeGroups.try_emplace(key);
    if (inserted)
      storeOrder.push_back(key);
    position->second.push_back(store);
  });

  int64_t groupedLoadCount = 0;
  for (Block *block : blocks) {
    auto &groups = groupsByBlock[block];
    for (auto &entry : groups) {
      SmallVector<LoadOp> &loads = entry.second;
      if (loads.size() != static_cast<size_t>(unrollFactor))
        return reject("existing_loop_load_group_cardinality_differs_from_route_factor");
      llvm::sort(loads, [](LoadOp left, LoadOp right) {
        return left->isBeforeInBlock(right);
      });
      LoadOp first = loads.front();
      // Scalar reads stay phase-exposed but cannot form a tensor lane.  At
      // least one tensor group must materialize for route success below.
      if (!isa<RankedTensorType>(first.getPtr().getType()))
        continue;
      bool masked = static_cast<bool>(first.getMask());
      bool hasOther = static_cast<bool>(first.getOther());
      if (!first.getBoundaryCheck().empty() || first.getPadding() ||
          llvm::any_of(loads, [&](LoadOp load) {
            return load.getPtr().getType() != first.getPtr().getType() ||
                   load.getResult().getType() != first.getResult().getType() ||
                   static_cast<bool>(load.getMask()) != masked ||
                   static_cast<bool>(load.getOther()) != hasOther ||
                   (masked && load.getMask().getType() !=
                                  first.getMask().getType()) ||
                   (hasOther && load.getOther().getType() !=
                                    first.getOther().getType()) ||
                   !load.getBoundaryCheck().empty() || load.getPadding() ||
                   load.getCache() != first.getCache() ||
                   load.getEvict() != first.getEvict() ||
                   load.getIsVolatile() != first.getIsVolatile();
          }))
        return reject("existing_loop_load_group_memory_semantics_are_not_exactly_packable");

      // The joined load is inserted after the last source-lane load.  It can
      // therefore replace a source result only when that insertion point
      // dominates every use of every lane result.  Dependent gather chains
      // commonly consume an earlier load to construct a later load's pointer;
      // grouping such a producer would otherwise create use-before-definition
      // IR.  This is a structural SSA precondition, independent of kernel or
      // operator identity.  Keep such producer groups scalar and allow later,
      // independent leaf-load groups to vectorize.
      Operation *insertionAnchor = loads.back();
      bool replacementDominatesAllUses = llvm::all_of(loads, [&](LoadOp load) {
        return llvm::all_of(load.getResult().getUses(), [&](OpOperand &use) {
          Operation *topLevelUser =
              block->findAncestorOpInBlock(*use.getOwner());
          return topLevelUser &&
                 insertionAnchor->isBeforeInBlock(topLevelUser);
        });
      });
      if (!replacementDominatesAllUses)
        continue;

      if (first.getIsVolatile() ||
          !hasNoInterveningMemoryEffects<LoadOp>(loads))
        continue;

      OpBuilder builder(loads.back());
      builder.setInsertionPointAfter(loads.back());
      SmallVector<Value> pointers, masks, others;
      for (LoadOp load : loads) {
        pointers.push_back(load.getPtr());
        if (masked)
          masks.push_back(load.getMask());
        if (hasOther)
          others.push_back(load.getOther());
      }
      FailureOr<Value> pointer = joinValues(
          builder, first.getLoc(), pointers);
      if (failed(pointer))
        return reject("existing_loop_load_pointer_group_cannot_be_joined_exactly");
      Value mask;
      if (masked) {
        FailureOr<Value> joined = joinValues(
            builder, first.getLoc(), masks);
        if (failed(joined))
          return reject("existing_loop_load_mask_group_cannot_be_joined_exactly");
        mask = *joined;
      }
      Value other;
      if (hasOther) {
        FailureOr<Value> joined = joinValues(
            builder, first.getLoc(), others);
        if (failed(joined))
          return reject("existing_loop_load_other_group_cannot_be_joined_exactly");
        other = *joined;
      }
      auto grouped = LoadOp::create(
          builder, first.getLoc(), *pointer, mask, other,
          first.getCache(), first.getEvict(), first.getIsVolatile());
      SmallVector<Value> pieces = splitValue(
          builder, first.getLoc(), grouped.getResult(), unrollFactor);
      for (auto [load, piece] : llvm::zip(loads, pieces))
        load.getResult().replaceAllUsesWith(piece);
      for (LoadOp load : llvm::reverse(loads))
        load.erase();
      ++groupedLoadCount;
    }
  }

  int64_t groupedStoreCount = 0;
  for (const StoreGroupKey &key : storeOrder) {
    SmallVector<StoreOp> stores = storeGroups[key];
    if (stores.size() != static_cast<size_t>(unrollFactor))
      continue;
    llvm::sort(stores, [](StoreOp left, StoreOp right) {
      return left->isBeforeInBlock(right);
    });
    StoreOp first = stores.front();
    if (!isExactPackableMemoryOperation(first))
      continue;
    bool masked = static_cast<bool>(first.getMask());
    if (llvm::any_of(stores, [&](StoreOp store) {
          return !isExactPackableMemoryOperation(store) ||
                 store.getPtr().getType() != first.getPtr().getType() ||
                 store.getValue().getType() != first.getValue().getType() ||
                 static_cast<bool>(store.getMask()) != masked ||
                 (masked && store.getMask().getType() !=
                                first.getMask().getType()) ||
                 store.getCache() != first.getCache() ||
                 store.getEvict() != first.getEvict();
        }))
      return reject("existing_loop_store_group_memory_semantics_are_not_exactly_packable");

    if (!hasNoInterveningMemoryEffects<StoreOp>(stores) ||
        !provesPackedStoresDisjoint(stores))
      continue;

    OpBuilder builder(stores.back());
    builder.setInsertionPointAfter(stores.back());
    SmallVector<Value> pointers, values, masks;
    for (StoreOp store : stores) {
      pointers.push_back(store.getPtr());
      values.push_back(store.getValue());
      if (masked)
        masks.push_back(store.getMask());
    }
    FailureOr<Value> pointer = joinValues(
        builder, first.getLoc(), pointers);
    FailureOr<Value> value = joinValues(
        builder, first.getLoc(), values);
    if (failed(pointer) || failed(value))
      return reject("existing_loop_store_pointer_or_value_group_cannot_be_joined_exactly");
    if (masked) {
      FailureOr<Value> mask = joinValues(
          builder, first.getLoc(), masks);
      if (failed(mask))
        return reject("existing_loop_store_mask_group_cannot_be_joined_exactly");
      StoreOp::create(builder, first.getLoc(), *pointer, *value, *mask,
                      first.getCache(), first.getEvict());
    } else {
      StoreOp::create(builder, first.getLoc(), *pointer, *value,
                      first.getCache(), first.getEvict());
    }
    for (StoreOp store : llvm::reverse(stores))
      store.erase();
    ++groupedStoreCount;
  }
  if (groupedLoadCount == 0 && groupedElementwise == 0 &&
      groupedStoreCount == 0)
    return reject("existing_loop_subject_has_no_exactly_packable_operation_group");

  SmallVector<Operation *> lineage;
  func.walk([&](Operation *op) {
    auto role = op->getAttrOfType<StringAttr>(kRoleAttr);
    if (!role)
      return;
    auto roleSubject = op->getAttrOfType<StringAttr>(kRoleSubjectAttr);
    if (!subject || (roleSubject && roleSubject.getValue() == *subject))
      lineage.push_back(op);
  });
  func.walk([&](Operation *operation) {
    auto roleSubject = operation->getAttrOfType<StringAttr>(
        kRoleSubjectAttr);
    if (subject && (!roleSubject || roleSubject.getValue() != *subject))
      return;
    // Existing-loop discovery attaches role_subject to every source-body
    // operation so operation-neutral phase scheduling has complete ownership.
    // Exact logical packing consumes that ownership even when a particular
    // address, predicate or shape operation has no load/compute/reduce role.
    // Clear all transient lineage for the successfully materialized subject;
    // limiting cleanup to the role-bearing subset leaves otherwise valid
    // kernels indistinguishable from a partially applied transformation.
    operation->removeAttr(kRoleAttr);
    operation->removeAttr(kRoleSubjectAttr);
    operation->removeAttr(kRoleIndexAttr);
    operation->removeAttr(kOperationGroupRefAttr);
    operation->removeAttr(kUnrollPartitionLineageAttr);
  });
  clearRoles(lineage);
  int64_t prior = 0;
  if (auto count =
          func->getAttrOfType<IntegerAttr>(kVectorizedLoadGroupCountAttr))
    prior = count.getInt();
  func->setAttr(kVectorizedLoadGroupCountAttr,
                IntegerAttr::get(IntegerType::get(func.getContext(), 64),
                                 prior + groupedLoadCount));
  if (groupedStoreCount) {
    int64_t operationPrior = 0;
    if (auto count = func->getAttrOfType<IntegerAttr>(
            kVectorizedOperationGroupCountAttr))
      operationPrior = count.getInt();
    func->setAttr(
        kVectorizedOperationGroupCountAttr,
        IntegerAttr::get(IntegerType::get(func.getContext(), 64),
                         operationPrior + groupedStoreCount));
  }
  return success();
}

LogicalResult materializeLogical(FuncOp func,
                                 std::optional<StringRef> subject = std::nullopt,
                                 int64_t unrollFactor = 4,
                                 Block *onlyBlock = nullptr,
                                 std::string *failureReason = nullptr) {
  SmallVector<Operation *> loads, computes, reductions;
  if (!collectRoles(func, loads, computes, reductions, subject))
    return failure();
  auto retainBlock = [onlyBlock](SmallVectorImpl<Operation *> &operations) {
    if (onlyBlock)
      llvm::erase_if(operations, [onlyBlock](Operation *operation) {
        return operation->getBlock() != onlyBlock;
      });
  };
  retainBlock(loads);
  retainBlock(computes);
  retainBlock(reductions);

  bool narrowIntegerReduction =
      computes.size() == static_cast<size_t>(unrollFactor) &&
      reductions.size() == static_cast<size_t>(unrollFactor) &&
      llvm::all_of(computes, [](Operation *op) {
        return op->getNumResults() == 1 &&
               isRankOneIntegerVector(op->getResult(0));
      });
  if (!onlyBlock && !narrowIntegerReduction)
    return materializeExactOperationVectorization(
        func, subject, unrollFactor, failureReason);

  // Match the native unroller's dynamic main/remainder shape.  Logical lane
  // fusion applies to the factor-wide main block; the scalar tail retains the
  // original exact recurrence and is intentionally not vectorized.
  if (!onlyBlock) {
    SmallVector<Block *> roleBlocks;
    auto rememberBlock = [&](Operation *operation) {
      if (!llvm::is_contained(roleBlocks, operation->getBlock()))
        roleBlocks.push_back(operation->getBlock());
    };
    llvm::for_each(loads, rememberBlock);
    llvm::for_each(computes, rememberBlock);
    llvm::for_each(reductions, rememberBlock);
    if (roleBlocks.size() > 1) {
      for (Block *block : roleBlocks) {
        unsigned blockComputes = llvm::count_if(
            computes, [block](Operation *op) { return op->getBlock() == block; });
        unsigned blockReductions = llvm::count_if(
            reductions,
            [block](Operation *op) { return op->getBlock() == block; });
        if (blockComputes == 1 && blockReductions == 1) {
          SmallVector<Operation *> scalarTail;
          llvm::copy_if(loads, std::back_inserter(scalarTail),
                        [block](Operation *op) {
                          return op->getBlock() == block;
                        });
          llvm::copy_if(computes, std::back_inserter(scalarTail),
                        [block](Operation *op) {
                          return op->getBlock() == block;
                        });
          llvm::copy_if(reductions, std::back_inserter(scalarTail),
                        [block](Operation *op) {
                          return op->getBlock() == block;
                        });
          clearRoles(scalarTail);
          continue;
        }
        if (failed(materializeLogical(
                func, subject, unrollFactor, block, failureReason)))
          return failure();
      }
      return success();
    }
  }

  auto vectorType = computes.empty()
                        ? RankedTensorType()
                        : dyn_cast<RankedTensorType>(
                              computes.front()->getResult(0).getType());
  if (loads.empty() || computes.size() != static_cast<size_t>(unrollFactor) ||
      reductions.size() != static_cast<size_t>(unrollFactor) ||
      unrollFactor < 2 || !llvm::isPowerOf2_64(unrollFactor) ||
      !vectorType || !isRankOneIntegerVector(computes.front()->getResult(0)) ||
      llvm::any_of(computes, [&](Operation *op) {
        return op->getNumResults() != 1 ||
               op->getResult(0).getType() != vectorType;
      }))
    return failure();

  // The narrow integer certificate describes a loop-carried additive
  // recurrence, not merely `factor` independent values.  Native unrolling
  // exposes the exact chain
  //
  //   next_0 = incoming + value_0
  //   next_i = next_(i-1) + value_i
  //
  // A factor-local vector reduction may replace the value part, but it must
  // preserve `incoming`.  Omitting it is observationally harmless only when
  // the focal loop is fully eliminated and the initial state happens to be
  // zero; for every retained factor-group loop it silently resets the
  // recurrence at each group.  Prove the complete chain from SSA before
  // materializing the associative integer rewrite.
  Value incomingAccumulator;
  for (size_t index = 0; index < reductions.size(); ++index) {
    auto add = dyn_cast<arith::AddIOp>(reductions[index]);
    if (!add)
      return failure();
    Value value = computes[index]->getResult(0);
    Value expectedCarry = index == 0
                              ? Value()
                              : reductions[index - 1]->getResult(0);
    Value lhs = add.getLhs();
    Value rhs = add.getRhs();
    if (index == 0) {
      if (lhs == value)
        incomingAccumulator = rhs;
      else if (rhs == value)
        incomingAccumulator = lhs;
      else
        return failure();
    } else if (!((lhs == value && rhs == expectedCarry) ||
                 (rhs == value && lhs == expectedCarry))) {
      return failure();
    }
  }
  if (!incomingAccumulator ||
      incomingAccumulator.getType() != vectorType)
    return failure();
  OpBuilder builder(computes.back());
  builder.setInsertionPointAfter(computes.back());
  Location loc = reductions.front()->getLoc();
  SmallVector<Value> joinedValues;
  llvm::transform(computes, std::back_inserter(joinedValues),
                  [](Operation *operation) {
                    return operation->getResult(0);
                  });
  while (joinedValues.size() > 1) {
    SmallVector<Value> next;
    for (size_t i = 0; i < joinedValues.size(); i += 2)
      next.push_back(builder
                         .create<JoinOp>(loc, joinedValues[i],
                                         joinedValues[i + 1])
                         .getResult());
    joinedValues = std::move(next);
  }
  auto reshaped = builder.create<ReshapeOp>(
      loc, ArrayRef<int64_t>{vectorType.getDimSize(0), unrollFactor},
      joinedValues.front(), false);
  auto grouped = builder.create<TransOp>(loc, reshaped.getResult(),
                                         ArrayRef<int32_t>{1, 0});
  auto reduced = builder.create<ReduceOp>(loc, ValueRange{grouped.getResult()}, 0u);
  Block &body = reduced.getCombineOp().emplaceBlock();
  Type elementType = vectorType.getElementType();
  body.addArguments({elementType, elementType}, {loc, loc});
  OpBuilder bodyBuilder(&body, body.end());
  auto sum = bodyBuilder.create<arith::AddIOp>(loc, body.getArgument(0),
                                               body.getArgument(1));
  bodyBuilder.create<ReduceReturnOp>(loc, sum.getResult());
  // This is the same rule-derived logical grouping proof consumed by the
  // mandatory postcondition below.  Preserve it as a generic realized-effect
  // counter so downstream attestation does not mistake the reduction branch
  // for a failed vector-packing route merely because it does not use the
  // load/elementwise adapter counters.
  int64_t priorLogicalReductions = 0;
  if (auto count =
          func->getAttrOfType<IntegerAttr>(kLogicalReductionGroupCountAttr))
    priorLogicalReductions = count.getInt();
  func->setAttr(
      kLogicalReductionGroupCountAttr,
      IntegerAttr::get(IntegerType::get(func.getContext(), 64),
                       priorLogicalReductions + 1));
  auto nextAccumulator = builder.create<arith::AddIOp>(
      loc, incomingAccumulator, reduced.getResult().front());
  reductions.back()->getResult(0).replaceAllUsesWith(
      nextAccumulator.getResult());
  for (Operation *op : llvm::reverse(reductions))
    op->erase();
  clearRoles(loads);
  clearRoles(computes);
  return success();
}
