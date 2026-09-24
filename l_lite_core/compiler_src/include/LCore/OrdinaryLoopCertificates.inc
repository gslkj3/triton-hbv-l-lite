// Ordinary loop opportunity proofs. Materialization still checks dependence
// and actual groups; no workload identity or Bridge-origin dispatch.
std::optional<int64_t> exactStaticTripCount(scf::ForOp loop) {
  APInt lower, upper, step;
  if (!matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) ||
      !matchPattern(loop.getUpperBound(), m_ConstantInt(&upper)) ||
      !matchPattern(loop.getStep(), m_ConstantInt(&step)))
    return std::nullopt;
  if (lower.getBitWidth() > 64 || upper.getBitWidth() > 64 || step.getBitWidth() > 64)
    return std::nullopt;
  int64_t lb = lower.getSExtValue();
  int64_t ub = upper.getSExtValue();
  int64_t stride = step.getSExtValue();
  if (stride <= 0)
    return std::nullopt;
  if (ub <= lb)
    return 0;
  // Positive-step SCF loops execute ceil((upper-lower)/step) times.
  // The difference can exceed int64 even when the final count fits.
  __int128 distance = static_cast<__int128>(ub) - lb;
  return lcore::checkedI64((distance + stride - 1) / stride);
}

// Full unrolling itself preserves the source loop's dynamic order.  The
// operation-neutral phase scheduler may subsequently choose another
// topological order, but only from actual SSA dependencies and an exact
// observable-effect chain.  Consequently the Provider need not demand a
// load, store, reduction, combiner, dtype or workload shape.  It only needs a
// nontrivial loop body whose carried-result boundary is structurally closed;
// nested loops remain separately composed subjects.
LoopDependenceCertificate
certifyOperationNeutralFullUnrollReorder(scf::ForOp loop) {
  auto trip = exactStaticTripCount(loop);
  if (trip && *trip < 2)
    return {false, "", "iteration domain has fewer than two iterations"};
  auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
  if (!yield || yield.getNumOperands() != loop.getInitArgs().size())
    return {false, "", "loop-carried result arity is not closed"};
  bool nonempty = false;
  for (Operation &operation : loop.getBody()->without_terminator()) {
    nonempty = true;
    if (isa<scf::ForOp>(operation))
      return {false, "", "nested scf.for requires a separately composed certificate"};
  }
  if (!nonempty)
    return {false, "", "iteration body is empty"};
  return {true, "existing_operation_neutral_topological_reorder_v1", ""};
}

#include "LCore/OperationCapabilities.inc"

// Certify that full unrolling exposes at least one corresponding operation
// group with an exact generic packed form.  This is deliberately an operation
// capability predicate, not an opcode allowlist: any regionless, pure,
// speculatable Elementwise operation over ranked tensors qualifies.  A
// candidate that feeds the loop yield is excluded because its copies may form
// a cross-iteration recurrence rather than independent lanes.
LoopDependenceCertificate
certifyOperationNeutralExactPacking(scf::ForOp loop) {
  LoopDependenceCertificate reorder =
      certifyOperationNeutralFullUnrollReorder(loop);
  if (!reorder.safe)
    return reorder;
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  for (Operation &operation : loop.getBody()->without_terminator()) {
    if (!isGenericExactPackableOperation(&operation) &&
        !isExactPackableMemoryOperation(&operation))
      continue;
    bool feedsCarriedState = llvm::any_of(
        operation.getResults(), [&](Value result) {
          return llvm::any_of(yield.getOperands(), [&](Value yielded) {
            return valueDependsOn(yielded, result);
          });
        });
    if (!feedsCarriedState)
      return {true, "existing_operation_neutral_exact_packing_v2", ""};
  }
  return {false, "",
          "iteration has no recurrence-independent exact-packable operation"};
}
