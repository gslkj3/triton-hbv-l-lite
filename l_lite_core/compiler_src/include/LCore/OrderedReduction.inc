LoopDependenceCertificate certifyExistingUnrollReorder(
    scf::ForOp loop, LoadOp load, Operation *compute, Operation *reduce) {
  if (load.getIsVolatile())
    return {false, "", "volatile load cannot be exposed across iterations"};
  if (!compute || !reduce || loop.getRegionIterArgs().size() != 1)
    return {false, "", "missing unique loop-carried reduction"};
  Value accumulator = loop.getRegionIterArgs().front();
  if (valueDependsOn(compute->getResult(0), accumulator))
    return {false, "", "compute depends on prior iteration accumulator"};
  if (valueDependsOn(load.getPtr(), accumulator) ||
      (load.getMask() && valueDependsOn(load.getMask(), accumulator)) ||
      (load.getOther() && valueDependsOn(load.getOther(), accumulator)))
    return {false, "", "load address/mask/value is loop-carried"};
  // Phase-major reordering moves only the load/compute slices.  The cloned
  // combiner operations retain their original sequential order, so floating
  // addition does not require reassociation or fast-math authority here.
  // Logical grouping remains separately restricted because that route does
  // change the reduction tree.
  auto integerAdd = dyn_cast<arith::AddIOp>(reduce);
  auto floatingAdd = dyn_cast<arith::AddFOp>(reduce);
  if (!integerAdd && !floatingAdd)
    return {false, "", "combiner_is_not_order_preserved_addition"};
  Value lhs = integerAdd ? integerAdd.getLhs() : floatingAdd.getLhs();
  Value rhs = integerAdd ? integerAdd.getRhs() : floatingAdd.getRhs();
  bool exactOperands =
      (lhs == accumulator && rhs == compute->getResult(0)) ||
      (rhs == accumulator && lhs == compute->getResult(0));
  if (!exactOperands)
    return {false, "", "reduction does not combine accumulator and current compute"};
  for (OpOperand &use : accumulator.getUses())
    if (use.getOwner() != reduce)
      return {false, "", "loop accumulator has another cross-iteration consumer"};
  for (OpOperand &use : compute->getResult(0).getUses())
    if (use.getOwner() != reduce)
      return {false, "", "current compute has a non-reduction consumer"};
  return {true,
          integerAdd ? "existing_exact_integer_addition_v2"
                     : "existing_order_preserved_floating_addition_v2",
          ""};
}
