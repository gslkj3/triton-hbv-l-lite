// Operation semantics, not workload names or loop provenance, define packing.
bool isGenericExactPackableOperation(Operation *operation) {
  return operation->getNumRegions() == 0 &&
         operation->hasTrait<OpTrait::Elementwise>() &&
         isMemoryEffectFree(operation) && isSpeculatable(operation) &&
         operation->getNumOperands() > 0 && operation->getNumResults() > 0 &&
         llvm::all_of(operation->getOperandTypes(),
                      [](Type type) { return isa<RankedTensorType>(type); }) &&
         llvm::all_of(operation->getResultTypes(),
                      [](Type type) { return isa<RankedTensorType>(type); });
}

bool isExactPackableMemoryOperation(Operation *operation) {
  if (auto load = dyn_cast<LoadOp>(operation))
    return isa<RankedTensorType>(load.getPtr().getType()) &&
           !load.getIsVolatile() && load.getBoundaryCheck().empty() &&
           !load.getPadding();
  if (auto store = dyn_cast<StoreOp>(operation))
    return isa<RankedTensorType>(store.getPtr().getType()) &&
           isa<RankedTensorType>(store.getValue().getType()) &&
           store.getBoundaryCheck().empty();
  return false;
}
