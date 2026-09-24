LoopDependenceCertificate
certifyNestedInnerDimension(scf::ForOp loop) {
  auto parent = loop->getParentOfType<scf::ForOp>();
  if (!parent || loop->getBlock() != parent.getBody())
    return {false, "", "eligible inner loop has no direct scf.for parent"};
  // Contextual substitution, not outer-iteration independence: the selected
  // leaf must separately satisfy its route's dependence/effect certificate.
  // Only that leaf is marked/unrolled. Parent bounds, iter_args, yield and
  // surrounding effects keep their order; no outer flattening is authorized.
  bool hasNestedLoop = false;
  loop.walk([&](scf::ForOp inner) {
    hasNestedLoop |= inner != loop;
  });
  if (hasNestedLoop)
    return {false, "", "local nested substitution requires a leaf loop"};
  return {true, "nested_inner_local_equivalence_v2", ""};
}
