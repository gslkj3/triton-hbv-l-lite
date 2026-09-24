#pragma once
#include <string>
#include <string_view>

namespace mlir::triton::lcore {
// A named structural proof, never a probability or a performance prediction.
// Rejection keeps its reason explicit; unknown facts are not successful proof.
struct LoopDependenceCertificate {
  bool safe = false;
  std::string kind;
  std::string reason;
};

// Wire identities emitted by the current ordinary-loop proof producers.
// This is not an operator/factor/model support list. It prevents historical
// composition certificates from authorizing a different materialization path.
inline bool isOrdinaryUnrollCertificate(std::string_view kind) {
  constexpr std::string_view current[] = {
      "existing_exact_integer_addition_v2",
      "existing_order_preserved_floating_addition_v2",
      "existing_order_preserving_read_exposure_v1",
      "existing_order_preserving_load_vectorization_v1",
      "existing_affine_pointer_read_exposure_v1",
      "existing_affine_pointer_read_load_vectorization_v1",
      "existing_operation_neutral_topological_reorder_v1",
      "existing_operation_neutral_exact_packing_v2",
      "nested_inner_local_equivalence_v2",
      "per_loop_exact_operation_vectorization_v3",
      "per_loop_operation_neutral_topological_reorder_v1"};
  for (auto value : current)
    if (value == kind)
      return true;
  return false;
}
} // namespace mlir::triton::lcore
