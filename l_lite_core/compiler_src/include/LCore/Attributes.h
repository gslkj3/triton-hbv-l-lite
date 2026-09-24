#pragma once
#include "llvm/ADT/StringRef.h"

namespace mlir::triton::lcore {
inline constexpr llvm::StringLiteral kCoreFactsAttr = "tt.hbv.l.static_facts";
inline constexpr llvm::StringLiteral kNativeDefaultStagesAttr =
    "tt.hbv.l.native_default_num_stages";
inline constexpr llvm::StringLiteral kStateAxisArtifactAttr =
    "tt.hbv.l.state_axis_artifact";
inline constexpr llvm::StringLiteral kStateAxisPackableNodeAttr =
    "tt.hbv.l.state_axis_packable_node";
inline constexpr llvm::StringLiteral kBridgeRuntimeBindingAppliedAttr =
    "tt.loop_bridge.runtime_binding_applied";
inline constexpr llvm::StringLiteral kBridgeDiscoveryAttr =
    "tt.loop_bridge.discovery";
inline constexpr llvm::StringLiteral kBridgeRuntimeScalarsAttr =
    "tt.loop_bridge.runtime_scalars";
inline constexpr llvm::StringLiteral kBridgeBoundScalarAttr =
    "tt.loop_bridge.bound_scalar";
inline constexpr llvm::StringLiteral kBridgeCFGPredicationAttr =
    "tt.loop_bridge.cfg_predication";
// Current L transport vocabulary. Historical combination protocols are not
// part of this header. Attributes describe evidence or transformation requests,
// never a second profitability decision.
inline constexpr llvm::StringLiteral kBundleAttr = "tt.hbv.plan_bundle";
inline constexpr llvm::StringLiteral kRouteAttr = "tt.hbv.l.route";
inline constexpr llvm::StringLiteral kMechanismRouteAttr = "tt.hbv.l.mechanism_route";
inline constexpr llvm::StringLiteral kRouteSubtypeAttr = "tt.hbv.l.route_subtype";
inline constexpr llvm::StringLiteral kArtifactRouteAttr = "tt.hbv.l.artifact_route";
inline constexpr llvm::StringLiteral kSubjectRefAttr = "tt.hbv.l.subject_ref";
inline constexpr llvm::StringLiteral kPipelineRoute = "l.nvidia.software_pipeline.v1";
inline constexpr llvm::StringLiteral kSourceExactTripCountAttr =
    "tt.hbv.l.source_exact_trip_count";
inline constexpr llvm::StringLiteral kSubjectAttr = "tt.hbv.l.subject";
inline constexpr llvm::StringLiteral kRoleAttr = "tt.hbv.l.role";
inline constexpr llvm::StringLiteral kRoleSubjectAttr = "tt.hbv.l.role_subject";
inline constexpr llvm::StringLiteral kRoleIndexAttr = "tt.hbv.l.role_index";
inline constexpr llvm::StringLiteral kRealizedAttr = "tt.hbv.l.realized";
inline constexpr llvm::StringLiteral kPostconditionAttr = "tt.hbv.l.postcondition";
inline constexpr llvm::StringLiteral kVectorizedLoadGroupCountAttr =
    "tt.hbv.l.vectorized_load_group_count";
inline constexpr llvm::StringLiteral kMainTailAttr = "tt.hbv.l.main_tail";
inline constexpr llvm::StringLiteral kUnrollPartitionLineageAttr =
    "tt.hbv.l.unroll_partition_lineage";
inline constexpr llvm::StringLiteral kProviderBoundMembersAttr =
    "tt.hbv.l.provider_bound_members";
inline constexpr llvm::StringLiteral kPipelineMemberRefAttr =
    "tt.hbv.l.pipeline_member_ref";
inline constexpr llvm::StringLiteral kBridgeFactorAttr = "tt.loop_bridge.factor";
inline constexpr llvm::StringLiteral kBridgeCardinalityAttr =
    "tt.loop_bridge.grouped_program_count";
inline constexpr llvm::StringLiteral kBridgeRequestedDivisorsAttr =
    "tt.loop_bridge.requested_divisors";
inline constexpr llvm::StringLiteral kBridgeOriginAttr = "tt.loop_bridge.origin";
inline constexpr llvm::StringLiteral kBridgeSubjectAttr = "tt.loop_bridge.subject";
inline constexpr llvm::StringLiteral kBridgeSourceLoopAttr = "tt.loop_bridge.source_loop";
inline constexpr llvm::StringLiteral kBridgeGridDivisorAttr = "tt.loop_bridge.grid_divisor_x";
inline constexpr llvm::StringLiteral kBridgeGridDivisorYAttr = "tt.loop_bridge.grid_divisor_y";
inline constexpr llvm::StringLiteral kBridgeGridDivisorZAttr = "tt.loop_bridge.grid_divisor_z";
inline constexpr llvm::StringLiteral kBridgeAxisExtentAttr =
    "tt.loop_bridge.axis_extent";
inline constexpr llvm::StringLiteral kBridgeRoleAttr = "tt.loop_bridge.role";
inline constexpr llvm::StringLiteral kOperationGroupRefAttr =
    "tt.hbv.l.operation_group_ref";
inline constexpr llvm::StringLiteral kPhaseOperationGroupCountAttr =
    "tt.hbv.l.phase_operation_group_count";
inline constexpr llvm::StringLiteral kPhaseSourceOperationGroupCountAttr =
    "tt.hbv.l.phase_source_operation_group_count";
inline constexpr llvm::StringLiteral kPhaseOperationReorderedCountAttr =
    "tt.hbv.l.phase_operation_reordered_count";
inline constexpr llvm::StringLiteral kPhaseOperationGroupReorderedCountAttr =
    "tt.hbv.l.phase_operation_group_reordered_count";
inline constexpr llvm::StringLiteral kVectorizedOperationGroupCountAttr =
    "tt.hbv.l.vectorized_operation_group_count";
inline constexpr llvm::StringLiteral kLogicalReductionGroupCountAttr =
    "tt.hbv.l.logical_reduction_group_count";
inline constexpr llvm::StringLiteral kBridgePartitionRecurrenceAttr =
    "tt.loop_bridge.partition_recurrence";
inline constexpr llvm::StringLiteral kBridgeCFGPredicationRejectionAttr =
    "tt.loop_bridge.cfg_predication_rejection";
inline constexpr llvm::StringLiteral kDependenceAttr = "tt.hbv.l.dependence_certificate";
inline constexpr llvm::StringLiteral kPhaseRoute = "l.ttir.full_unroll_phase_major.v1";
inline constexpr llvm::StringLiteral kLogicalRoute = "l.ttir.full_unroll_logical_group.v1";
} // namespace mlir::triton::lcore
