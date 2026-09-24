// Rule-based early-return normalization for ordinary Bridge construction.
// Requires native IR utilities, splatInteger, and the CFG evidence attribute.
struct BridgeCFGNormalizationResult {
  bool matched = false;
  bool normalized = false;
  std::string reason;
};

bool isVoidReturnOnlyBlock(Block *block) {
  if (!block || block->getNumArguments() != 0 ||
      !block->without_terminator().empty())
    return false;
  auto returnOp = dyn_cast<ReturnOp>(block->getTerminator());
  return returnOp && returnOp.getNumOperands() == 0;
}

bool isIntegerDivisionOrRemainder(Operation *operation) {
  StringRef name = operation->getName().getStringRef();
  return name == "arith.divsi" || name == "arith.divui" ||
         name == "arith.ceildivsi" || name == "arith.ceildivui" ||
         name == "arith.floordivsi" || name == "arith.remsi" ||
         name == "arith.remui";
}

bool hasCertifiedSafeIntegerDivision(Operation *operation) {
  if (!isIntegerDivisionOrRemainder(operation))
    return false;
  if (operation->getNumOperands() != 2)
    return false;
  auto divisor = splatInteger(operation->getOperand(1));
  if (!divisor || *divisor == 0)
    return false;
  StringRef name = operation->getName().getStringRef();
  if (*divisor != -1 || (name != "arith.divsi" &&
                        name != "arith.ceildivsi" &&
                        name != "arith.floordivsi"))
    return true;
  // MLIR defines signed minimum / -1 as undefined for these three operations.
  // A nonzero divisor alone cannot authorize moving them out of a branch.
  auto numerator = splatInteger(operation->getOperand(0));
  Type element = getElementTypeOrSelf(operation->getOperand(0).getType());
  auto integer = dyn_cast<IntegerType>(element);
  if (!numerator || !integer || integer.getWidth() > 64)
    return false;
  return *numerator != APInt::getSignedMinValue(integer.getWidth()).getSExtValue();
}

bool continuationOperationIsPredicatable(Operation *operation);

bool reductionCombinerIsPredicatable(ReduceOp reduce) {
  Region &combiner = reduce.getCombineOp();
  if (!llvm::hasSingleElement(combiner))
    return false;
  for (Operation &operation : combiner.front()) {
    if (isa<ReduceReturnOp>(operation))
      continue;
    // A reduction combiner is certified as a closed pure computation.  It
    // may not acquire memory service merely because its enclosing ReduceOp is
    // Pure, and every potentially undefined integer divisor retains the same
    // nonzero and signed-division overflow proof required in the continuation.
    if (isa<LoadOp, StoreOp>(operation) ||
        !continuationOperationIsPredicatable(&operation))
      return false;
  }
  return true;
}

bool continuationOperationIsPredicatable(Operation *operation) {
  if (auto load = dyn_cast<LoadOp>(operation))
    return load.getBoundaryCheck().empty() && !load.getPadding();
  if (auto store = dyn_cast<StoreOp>(operation))
    return store.getBoundaryCheck().empty();
  // Triton reductions are single-block Pure regions, but do not implement
  // MLIR's generic speculatability interface.  Treat them as predicatable
  // only after recursively proving the combiner contains no memory service,
  // nested control, non-speculatable operation or unbounded integer divide.
  // This is an operation-structural rule, independent of kernel/operator
  // identity and tensor shape.
  if (auto reduce = dyn_cast<ReduceOp>(operation))
    return isMemoryEffectFree(operation) &&
           reductionCombinerIsPredicatable(reduce);
  if (operation->getNumRegions() != 0 || !isMemoryEffectFree(operation))
    return false;
  if (isIntegerDivisionOrRemainder(operation))
    return hasCertifiedSafeIntegerDivision(operation);
  return isSpeculatable(operation);
}

Block *argumentFreeForwardTarget(Block *block) {
  if (!block || block->getNumArguments() != 0 ||
      !block->without_terminator().empty())
    return nullptr;
  auto branch = dyn_cast<cf::BranchOp>(block->getTerminator());
  if (!branch || !branch.getDestOperands().empty() ||
      branch.getDest() == block || branch.getDest()->getNumArguments() != 0)
    return nullptr;
  return branch.getDest();
}

// Convert one narrowly certified early-return CFG into a straight-line,
// predicated program body.  This is the earliest representation at which all
// three downstream routes can consume the same Bridge subject: native
// pipelining sees the loads directly in the constructed scf.for, while both
// full-unroll routes see the same load/compute/store lineage.
//
// The accepted shape is intentionally small and proof-carrying:
//   entry --cond--> void return
//         `-------> one unique continuation --void return
// The continuation has no block arguments or nested control.  Its only
// effects are Triton loads/stores, which receive the continuation predicate;
// every other operation is memory-effect-free and speculatable.  Integer
// division/remainder additionally requires an exact nonzero denominator from
// a constant or the existing Bridge bound-scalar provider fact. Signed division
// by -1 additionally excludes the signed minimum numerator. A masked
// load may produce an unspecified inactive value, but every observable memory
// effect derived from it is guarded by the same predicate.
BridgeCFGNormalizationResult
normalizeSingleEarlyVoidReturnCFG(FuncOp entry, bool materialize = true) {
  BridgeCFGNormalizationResult result;
  Region &body = entry.getBody();

  // Triton's frontend can encode the same source-level early return with an
  // argument-free forwarding block and an unreachable forwarding predecessor
  // of the continuation.  Canonicalize only that empty CFG scaffolding before
  // matching the proof shape.  No operation carrying a value or effect is
  // erased or moved by this step, so the accepted causal object remains the
  // same conditional return plus one continuation.
  llvm::SmallPtrSet<Block *, 8> reachable;
  SmallVector<Block *> worklist{&body.front()};
  while (!worklist.empty()) {
    Block *block = worklist.pop_back_val();
    if (!reachable.insert(block).second)
      continue;
    for (Block *successor : block->getSuccessors())
      worklist.push_back(successor);
  }
  SmallVector<Block *> unreachableForwarders;
  for (Block &block : llvm::drop_begin(body.getBlocks()))
    if (!reachable.contains(&block) && argumentFreeForwardTarget(&block))
      unreachableForwarders.push_back(&block);
  if (materialize)
    for (Block *block : unreachableForwarders)
      block->erase();

  Block *entryBlock = &body.front();
  SmallVector<Block *> foldedForwarders;
  llvm::DenseMap<Block *, Block *> forwardedTargets;
  if (auto branch = dyn_cast<cf::CondBranchOp>(entryBlock->getTerminator())) {
    for (unsigned successorIndex = 0; successorIndex < 2; ++successorIndex) {
      Block *successor = branch->getSuccessor(successorIndex);
      Block *target = argumentFreeForwardTarget(successor);
      if (!target || successor->getSinglePredecessor() != entryBlock)
        continue;
      if (materialize)
        branch->setSuccessor(target, successorIndex);
      forwardedTargets[successor] = target;
      foldedForwarders.push_back(successor);
    }
    if (materialize)
      for (Block *block : foldedForwarders)
        block->erase();
  }

  size_t effectiveBlockCount = body.getBlocks().size();
  if (!materialize)
    effectiveBlockCount -= unreachableForwarders.size() + foldedForwarders.size();
  if (effectiveBlockCount != 3) {
    result.reason = "cfg_not_three_blocks";
    return result;
  }
  auto branch = dyn_cast<cf::CondBranchOp>(entryBlock->getTerminator());
  if (!branch) {
    result.reason = "entry_not_single_conditional_branch";
    return result;
  }
  result.matched = true;
  if (!branch.getTrueDestOperands().empty() ||
      !branch.getFalseDestOperands().empty() ||
      branch.getTrueDest() == branch.getFalseDest()) {
    result.reason = "conditional_branch_has_phi_operands_or_shared_target";
    return result;
  }

  Block *trueBlock = branch.getTrueDest();
  Block *falseBlock = branch.getFalseDest();
  if (!materialize) {
    if (forwardedTargets.count(trueBlock))
      trueBlock = forwardedTargets.lookup(trueBlock);
    if (forwardedTargets.count(falseBlock))
      falseBlock = forwardedTargets.lookup(falseBlock);
    if (trueBlock == falseBlock) {
      result.reason = "conditional_branch_has_phi_operands_or_shared_target";
      return result;
    }
  }
  bool trueReturns = isVoidReturnOnlyBlock(trueBlock);
  bool falseReturns = isVoidReturnOnlyBlock(falseBlock);
  if (trueReturns == falseReturns) {
    result.reason = "cfg_requires_exactly_one_immediate_void_return";
    return result;
  }
  Block *returnBlock = trueReturns ? trueBlock : falseBlock;
  Block *continuation = trueReturns ? falseBlock : trueBlock;
  auto hasUniqueEntryPredecessor = [&](Block *block) {
    if (materialize)
      return block->getSinglePredecessor() == entryBlock;
    unsigned count = 0;
    for (Block *predecessor : block->getPredecessors()) {
      if (llvm::is_contained(unreachableForwarders, predecessor))
        continue;
      if (llvm::is_contained(foldedForwarders, predecessor))
        predecessor = entryBlock;
      if (predecessor != entryBlock)
        return false;
      ++count;
    }
    return count == 1;
  };
  if (continuation->getNumArguments() != 0 ||
      !hasUniqueEntryPredecessor(continuation) ||
      !hasUniqueEntryPredecessor(returnBlock)) {
    result.reason = "cfg_successors_are_not_unique_argument_free_blocks";
    return result;
  }
  auto continuationReturn = dyn_cast<ReturnOp>(continuation->getTerminator());
  if (!continuationReturn || continuationReturn.getNumOperands() != 0) {
    result.reason = "continuation_has_no_void_return";
    return result;
  }

  bool hasMemoryEffect = false;
  for (Operation &operation : continuation->without_terminator()) {
    if (!continuationOperationIsPredicatable(&operation)) {
      result.reason =
          (Twine("continuation_contains_nonpredicatable_or_unbounded_") +
           operation.getName().getStringRef())
              .str();
      return result;
    }
    hasMemoryEffect |= isa<LoadOp, StoreOp>(operation);
    for (Value value : operation.getResults())
      for (OpOperand &use : value.getUses())
        if (use.getOwner()->getBlock() != continuation) {
          result.reason = "continuation_result_escapes_cfg_region";
          return result;
        }
  }
  if (!hasMemoryEffect) {
    result.reason = "continuation_has_no_predicatable_memory_service";
    return result;
  }

  // Read-only source-state analysis shares the exact operation/SSA proof with
  // materialization. Forwarding scaffolding above is simulated, never edited.
  if (!materialize) {
    result.normalized = true; // Proven normalizable; no IR mutation performed.
    result.reason = "single_early_void_return_predicated_v1";
    return result;
  }

  OpBuilder entryBuilder(branch);
  Value active = branch.getCondition();
  if (trueReturns) {
    Value one = arith::ConstantOp::create(
        entryBuilder, branch.getLoc(), entryBuilder.getBoolAttr(true));
    active = arith::XOrIOp::create(entryBuilder, branch.getLoc(), active, one);
  }
  for (Operation &operation : continuation->without_terminator()) {
    IRRewriter rewriter(&operation);
    if (auto load = dyn_cast<LoadOp>(operation)) {
      Value mask = getPredMask(rewriter, load.getPtr().getType(),
                               load.getMask(), active);
      load.getMaskMutable().assign(mask);
    } else if (auto store = dyn_cast<StoreOp>(operation)) {
      Value mask = getPredMask(rewriter, store.getPtr().getType(),
                               store.getMask(), active);
      store.getMaskMutable().assign(mask);
    }
  }

  // Move both the original continuation and the mask expressions inserted
  // above as one ordered list, preserving SSA dominance.
  entryBlock->getOperations().splice(
      Block::iterator(branch), continuation->getOperations(),
      continuation->begin(), Block::iterator(continuationReturn));
  ReturnOp::create(entryBuilder, branch.getLoc());
  branch.erase();
  continuationReturn.erase();
  returnBlock->erase();
  continuation->erase();
  entry->setAttr(
      kBridgeCFGPredicationAttr,
      StringAttr::get(entry.getContext(),
                      "single_early_void_return_predicated_v1"));
  result.normalized = true;
  result.reason = "single_early_void_return_predicated_v1";
  return result;
}
