#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MathExtras.h"
#include <array>
#include <optional>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <tuple>

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONHBVLOOPDECISION
#define GEN_PASS_DEF_TRITONHBVLOOPFACTS
#define GEN_PASS_DEF_TRITONLOOPBRIDGEDISCOVER
#define GEN_PASS_DEF_TRITONLOOPBRIDGEPROGRAMCOARSENING
#define GEN_PASS_DEF_TRITONHBVLOOPMATERIALIZE
#define GEN_PASS_DEF_TRITONHBVVALIDATELOOPPLAN
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

constexpr StringLiteral kBundleAttr = "tt.hbv.plan_bundle";
constexpr StringLiteral kSubjectAttr = "tt.hbv.l.subject";
constexpr StringLiteral kRoleAttr = "tt.hbv.l.role";
constexpr StringLiteral kRoleSubjectAttr = "tt.hbv.l.role_subject";
constexpr StringLiteral kRoleIndexAttr = "tt.hbv.l.role_index";
constexpr StringLiteral kRouteAttr = "tt.hbv.l.route";
constexpr StringLiteral kMechanismRouteAttr = "tt.hbv.l.mechanism_route";
constexpr StringLiteral kRouteSubtypeAttr = "tt.hbv.l.route_subtype";
constexpr StringLiteral kArtifactRouteAttr = "tt.hbv.l.artifact_route";
constexpr StringLiteral kSubjectRefAttr = "tt.hbv.l.subject_ref";
constexpr StringLiteral kRealizedAttr = "tt.hbv.l.realized";
constexpr StringLiteral kPostconditionAttr = "tt.hbv.l.postcondition";
constexpr StringLiteral kCompositionSchemaAttr =
    "tt.hbv.l.composition_schema";
constexpr StringLiteral kCompositionBridgeFactorAttr =
    "tt.hbv.l.composition_bridge_factor";
constexpr StringLiteral kCompositionRouteFactorAttr =
    "tt.hbv.l.composition_route_factor";
constexpr StringLiteral kCompositionBridgeAxisDivisorsAttr =
    "tt.hbv.l.composition_bridge_axis_divisors";
constexpr StringLiteral kCompositionPostconditionAttr =
    "tt.hbv.l.composition_postcondition";
constexpr StringLiteral kRuntimeGuardAttr = "tt.hbv.l.runtime_guard";
constexpr StringLiteral kSplitElisionAttr = "tt.hbv.l.split_elision";
constexpr StringLiteral kVectorizedLoadGroupCountAttr =
    "tt.hbv.l.vectorized_load_group_count";
constexpr StringLiteral kMainTailAttr = "tt.hbv.l.main_tail";
constexpr StringLiteral kUnrollPartitionLineageAttr =
    "tt.hbv.l.unroll_partition_lineage";
constexpr StringLiteral kSourceExactTripCountAttr =
    "tt.hbv.l.source_exact_trip_count";
constexpr StringLiteral kProviderBoundMembersAttr =
    "tt.hbv.l.provider_bound_members";
constexpr StringLiteral kBridgeFactorAttr = "tt.loop_bridge.factor";
constexpr StringLiteral kBridgeCardinalityAttr =
    "tt.loop_bridge.grouped_program_count";
constexpr StringLiteral kBridgeRequestedDivisorsAttr =
    "tt.loop_bridge.requested_divisors";
constexpr StringLiteral kBridgeOriginAttr = "tt.loop_bridge.origin";
constexpr StringLiteral kBridgeSubjectAttr = "tt.loop_bridge.subject";
constexpr StringLiteral kBridgeGridDivisorAttr = "tt.loop_bridge.grid_divisor_x";
constexpr StringLiteral kBridgeGridDivisorYAttr = "tt.loop_bridge.grid_divisor_y";
constexpr StringLiteral kBridgeGridDivisorZAttr = "tt.loop_bridge.grid_divisor_z";
constexpr StringLiteral kBridgeRuntimeScalarsAttr =
    "tt.loop_bridge.runtime_scalars";
constexpr StringLiteral kBridgeBoundScalarAttr =
    "tt.loop_bridge.bound_scalar";
constexpr StringLiteral kBridgeAxisExtentAttr =
    "tt.loop_bridge.axis_extent";
constexpr StringLiteral kBridgeDiscoveryAttr = "tt.loop_bridge.discovery";
constexpr StringLiteral kBridgeRoleAttr = "tt.loop_bridge.role";
constexpr StringLiteral kBridgeOrdinalAttr = "tt.loop_bridge.ordinal";
constexpr StringLiteral kBridgePartitionRecurrenceAttr =
    "tt.loop_bridge.partition_recurrence";
constexpr StringLiteral kExactPrefixVectorElementBudgetAttr =
    "tt.hbv.l.exact_prefix_vector_element_budget";
constexpr StringLiteral kBridgeCFGPredicationAttr =
    "tt.loop_bridge.cfg_predication";
constexpr StringLiteral kBridgeCFGPredicationRejectionAttr =
    "tt.loop_bridge.cfg_predication_rejection";
constexpr StringLiteral kPrefixReductionAttr =
    "tt.hbv.l.prefix_reduction";
constexpr StringLiteral kStateAxisGroupCountAttr =
    "tt.hbv.l.state_axis_sibling_group_count";
constexpr StringLiteral kStateAxisArtifactAttr =
    "tt.hbv.l.state_axis_artifact";
constexpr StringLiteral kStateAxisPackableNodeAttr =
    "tt.hbv.l.state_axis_packable_node";
constexpr StringLiteral kDependenceAttr = "tt.hbv.l.dependence_certificate";
constexpr StringLiteral kStaticFactsAttr = "tt.hbv.l.static_facts";
constexpr StringLiteral kNativeDefaultStagesAttr =
    "tt.hbv.l.native_default_num_stages";
constexpr StringLiteral kFailureAttr = "tt.hbv.materialization_failure";
constexpr StringLiteral kPipelineRoute = "l.nvidia.software_pipeline.v1";
constexpr StringLiteral kPhaseRoute = "l.ttir.full_unroll_phase_major.v1";
constexpr StringLiteral kLogicalRoute = "l.ttir.full_unroll_logical_group.v1";
constexpr StringLiteral kExactPrefixRoute =
    "l.ttir.predicated_exact_prefix_reduction.v1";
constexpr StringLiteral kOriginalRoute = "l.original.default";
constexpr StringLiteral kPinnedCompilerCommit =
    "7c56a5e40f7fd928dfd5c72902d5def0097db73a";

struct ParsedLoopPlan {
  bool controlled = false;
  bool multiSubject = false;
  bool nestedSubject = false;
  bool runtimeGuardedLogical = false;
  bool bridgeConstructed = false;
  bool bridgeStaticPartition = false;
  bool composedIntervention = false;
  bool compositionV2 = false;
  bool compositionV3 = false;
  bool bridgeAxisVector = false;
  bool providerClosedStatic = false;
  bool exactPrefixReduction = false;
  bool stateAxisLogical = false;
  bool affineRuntimePartial = false;
  bool bridgeInvariantHoisting = false;
  bool bridgeTensorLaneFusion = false;
  bool bridgeExactSplitElision = true;
  bool providerBoundSubjectSet = false;
  int64_t adapterVersion = 1;
  int64_t bridgeFactor = 1;
  int64_t routeFactor = 1;
  int64_t unrollFactor = 1;
  int64_t stageCount = 0;
  int64_t exactPrefixActiveExtent = 0;
  int64_t exactPrefixContainerWidth = 0;
  int64_t exactPrefixElementBytes = 0;
  int64_t stateAxisGroupCount = 0;
  SmallVector<int64_t> unrollFactors;
  SmallVector<int64_t, 3> bridgeAxisDivisors;
  struct StateAxisGroup {
    int64_t ordinal = -1;
    int64_t stateCardinality = 0;
    int64_t paddedStateCardinality = 0;
    int64_t stateWidth = 0;
    int64_t elementBytes = 0;
    int64_t laneOperandIndex = -1;
    int64_t packableNodeCount = 0;
    std::string graphSignature;
    SmallVector<std::string> operationCapabilities;
  };
  struct ProviderBoundMember {
    std::string providerLoopLocator;
    std::string memberRef;
    std::string routeCapabilityCertificateRef;
    std::string runtimeMainTailCertificateRef;
    std::string parentLoopLocator;
    std::string nestedContextCertificateRef;
    int64_t nestingDepth = 0;
    int64_t exactStaticTripCount = 0;
    int64_t routeFactor = 1;
  };
  SmallVector<StateAxisGroup> stateAxisGroups;
  SmallVector<ProviderBoundMember> providerBoundMembers;
  std::string route;
  std::string mechanismRoute;
  std::string routeSubtype;
  std::string artifactRoute;
  std::string decisionRef;
  std::string subjectRef;
  std::string compositionSchema;
  std::string affineRuntimeMaterializationPolicy = "provider_selected";
  std::string providerBoundMemberSignature;
};

struct StateAxisNormalizationRegion {
  SmallVector<Value> laneValues;
  SmallVector<Value> outputs;
  SmallVector<Operation *> finalLaneOperations;
  Value columnDenominator;
  int64_t laneOperandIndex = -1;
  std::string graphSignature;
  std::set<std::string> operationCapabilities;
  int64_t packableNodeCount = 0;
  int64_t stateCardinality = 0;
  int64_t paddedStateCardinality = 0;
  int64_t stateWidth = 0;
  int64_t elementBytes = 0;
};

struct LoopDependenceCertificate {
  bool safe = false;
  std::string kind;
  std::string reason;
};

// Defined with the existing-loop materialization predicates below.  The
// passive Provider uses the same predicate so a nested subject cannot be
// admitted from a weaker census than the decision pass later rederives.
LoopDependenceCertificate certifyNestedInnerDimension(scf::ForOp loop);

std::optional<StringRef> structuralArgumentName(BlockArgument argument) {
  Location location = argument.getLoc();
  while (true) {
    if (auto name = dyn_cast<NameLoc>(location))
      return name.getName().getValue();
    if (auto callsite = dyn_cast<CallSiteLoc>(location)) {
      location = callsite.getCallee();
      continue;
    }
    break;
  }
  return std::nullopt;
}

bool valueDependsOn(Value value, Value target,
                    llvm::SmallPtrSetImpl<Value> &visited) {
  if (value == target)
    return true;
  if (!visited.insert(value).second)
    return false;
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return false;
  return llvm::any_of(definition->getOperands(), [&](Value operand) {
    return valueDependsOn(operand, target, visited);
  });
}

bool valueDependsOn(Value value, Value target) {
  llvm::SmallPtrSet<Value, 32> visited;
  return valueDependsOn(value, target, visited);
}

bool isPointerLike(Type type) {
  if (isa<PointerType>(type))
    return true;
  auto shaped = dyn_cast<ShapedType>(type);
  return shaped && isa<PointerType>(shaped.getElementType());
}

void collectPointerRoots(Value value, llvm::SmallPtrSetImpl<Value> &roots,
                         llvm::SmallPtrSetImpl<Value> &visited) {
  if (!visited.insert(value).second)
    return;
  if (isa<BlockArgument>(value)) {
    if (isPointerLike(value.getType()))
      roots.insert(value);
    return;
  }
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return;
  for (Value operand : definition->getOperands())
    if (isPointerLike(operand.getType()))
      collectPointerRoots(operand, roots, visited);
}

std::optional<Value> uniquePointerRoot(Value pointer) {
  llvm::SmallPtrSet<Value, 4> roots;
  llvm::SmallPtrSet<Value, 32> visited;
  collectPointerRoots(pointer, roots, visited);
  if (roots.size() != 1)
    return std::nullopt;
  return *roots.begin();
}

struct AffinePidFootprint {
  int64_t pidStride = 0;
  int64_t localMin = 0;
  int64_t localMax = 0;
};

std::optional<int64_t> checkedI64(__int128 value);

std::optional<int64_t> splatInteger(Value value) {
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (auto function = dyn_cast_or_null<FuncOp>(
            argument.getOwner()->getParentOp()))
      if (auto bound = function.getArgAttrOfType<IntegerAttr>(
              argument.getArgNumber(), kBridgeBoundScalarAttr))
        return bound.getInt();
  }
  APInt scalar;
  if (matchPattern(value, m_ConstantInt(&scalar)))
    return scalar.getSExtValue();
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  auto dense = constant ? dyn_cast<DenseIntElementsAttr>(constant.getValue())
                        : DenseIntElementsAttr();
  if (dense && dense.isSplat())
    return dense.getSplatValue<APInt>().getSExtValue();
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return std::nullopt;
  StringRef name = definition->getName().getStringRef();
  if (definition->getNumOperands() == 1 &&
      (name == "tt.splat" || name == "tt.broadcast" ||
      name == "tt.expand_dims" || name == "tt.reshape" ||
      name == "tt.trans" || name == "arith.extsi" ||
      name == "arith.extui"))
    return splatInteger(definition->getOperand(0));
  if (definition->getNumOperands() == 2) {
    auto lhs = splatInteger(definition->getOperand(0));
    auto rhs = splatInteger(definition->getOperand(1));
    if (!lhs || !rhs)
      return std::nullopt;
    if (name == "arith.addi")
      return checkedI64(static_cast<__int128>(*lhs) + *rhs);
    if (name == "arith.subi")
      return checkedI64(static_cast<__int128>(*lhs) - *rhs);
    if (name == "arith.muli")
      return checkedI64(static_cast<__int128>(*lhs) * *rhs);
    if ((name == "arith.divsi" || name == "arith.divui") && *rhs != 0)
      return *lhs / *rhs;
    if ((name == "arith.ceildivsi" || name == "arith.ceildivui") &&
        *lhs >= 0 && *rhs > 0)
      return checkedI64(
          (static_cast<__int128>(*lhs) + *rhs - 1) / *rhs);
  }
  return std::nullopt;
}

std::optional<int64_t> checkedI64(__int128 value) {
  if (value < std::numeric_limits<int64_t>::min() ||
      value > std::numeric_limits<int64_t>::max())
    return std::nullopt;
  return static_cast<int64_t>(value);
}

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

bool hasCertifiedNonzeroIntegerDivisor(Operation *operation) {
  if (!isIntegerDivisionOrRemainder(operation))
    return false;
  if (operation->getNumOperands() != 2)
    return false;
  auto divisor = splatInteger(operation->getOperand(1));
  return divisor && *divisor != 0;
}

bool continuationOperationIsPredicatable(Operation *operation) {
  if (auto load = dyn_cast<LoadOp>(operation))
    return load.getBoundaryCheck().empty() && !load.getPadding();
  if (auto store = dyn_cast<StoreOp>(operation))
    return store.getBoundaryCheck().empty();
  if (operation->getNumRegions() != 0 || !isMemoryEffectFree(operation))
    return false;
  if (isIntegerDivisionOrRemainder(operation))
    return hasCertifiedNonzeroIntegerDivisor(operation);
  return isSpeculatable(operation);
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
// a constant or the existing Bridge bound-scalar provider fact.  A masked
// load may produce an unspecified inactive value, but every observable memory
// effect derived from it is guarded by the same predicate.
BridgeCFGNormalizationResult
normalizeSingleEarlyVoidReturnCFG(FuncOp entry) {
  BridgeCFGNormalizationResult result;
  Region &body = entry.getBody();
  if (body.getBlocks().size() != 3) {
    result.reason = "cfg_not_three_blocks";
    return result;
  }
  Block *entryBlock = &body.front();
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
  bool trueReturns = isVoidReturnOnlyBlock(trueBlock);
  bool falseReturns = isVoidReturnOnlyBlock(falseBlock);
  if (trueReturns == falseReturns) {
    result.reason = "cfg_requires_exactly_one_immediate_void_return";
    return result;
  }
  Block *returnBlock = trueReturns ? trueBlock : falseBlock;
  Block *continuation = trueReturns ? falseBlock : trueBlock;
  if (continuation->getNumArguments() != 0 ||
      continuation->getSinglePredecessor() != entryBlock ||
      returnBlock->getSinglePredecessor() != entryBlock) {
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

std::optional<AffinePidFootprint>
combineFootprints(const AffinePidFootprint &lhs,
                  const AffinePidFootprint &rhs, bool subtract) {
  auto stride = checkedI64(static_cast<__int128>(lhs.pidStride) +
                           (subtract ? -static_cast<__int128>(rhs.pidStride)
                                     : static_cast<__int128>(rhs.pidStride)));
  auto minimum = checkedI64(static_cast<__int128>(lhs.localMin) +
                            (subtract ? -static_cast<__int128>(rhs.localMax)
                                      : static_cast<__int128>(rhs.localMin)));
  auto maximum = checkedI64(static_cast<__int128>(lhs.localMax) +
                            (subtract ? -static_cast<__int128>(rhs.localMin)
                                      : static_cast<__int128>(rhs.localMax)));
  if (!stride || !minimum || !maximum)
    return std::nullopt;
  return AffinePidFootprint{*stride, *minimum, *maximum};
}

std::optional<AffinePidFootprint>
scaleFootprint(const AffinePidFootprint &input, int64_t scale) {
  auto stride = checkedI64(static_cast<__int128>(input.pidStride) * scale);
  auto first = checkedI64(static_cast<__int128>(input.localMin) * scale);
  auto second = checkedI64(static_cast<__int128>(input.localMax) * scale);
  if (!stride || !first || !second)
    return std::nullopt;
  return AffinePidFootprint{*stride, std::min(*first, *second),
                            std::max(*first, *second)};
}

std::optional<AffinePidFootprint>
integerFootprint(Value value, Value pid, llvm::SmallPtrSetImpl<Value> &active) {
  if (value == pid)
    return AffinePidFootprint{1, 0, 0};
  if (auto constant = splatInteger(value))
    return AffinePidFootprint{0, *constant, *constant};
  // Follow an scf.for region iteration argument to its preheader value when
  // resolving another affine basis such as a program ID.  The induction
  // argument itself is independent of that basis; loop-carried values retain
  // the affine program-axis relation established before the loop.
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (auto loop = dyn_cast_or_null<scf::ForOp>(
            argument.getOwner()->getParentOp())) {
      if (argument.getArgNumber() == 0)
        return AffinePidFootprint{0, 0, 0};
      unsigned iterIndex = argument.getArgNumber() - 1;
      if (iterIndex < loop.getInitArgs().size())
        return integerFootprint(loop.getInitArgs()[iterIndex], pid, active);
    }
  }
  // Bridge groups only one launch axis.  Any expression independent of that
  // axis (including another program ID and its quotient/remainder tree) is
  // fixed across the virtual iterations of one physical program and therefore
  // cancels from the pairwise disjointness proof.
  if (!valueDependsOn(value, pid))
    return AffinePidFootprint{0, 0, 0};
  if (!active.insert(value).second)
    return std::nullopt;
  auto erase = llvm::make_scope_exit([&] { active.erase(value); });
  if (auto range = value.getDefiningOp<MakeRangeOp>())
    return AffinePidFootprint{0, range.getStartAttr().getInt(),
                              range.getEndAttr().getInt() - 1};
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return std::nullopt;
  if (auto otherPid = dyn_cast<GetProgramIdOp>(definition)) {
    if (otherPid.getResult() == pid)
      return AffinePidFootprint{1, 0, 0};
    if (auto extent = otherPid->getAttrOfType<IntegerAttr>(
            kBridgeAxisExtentAttr))
      return AffinePidFootprint{0, 0, extent.getInt() - 1};
    return std::nullopt;
  }
  StringRef name = definition->getName().getStringRef();
  if (name == "tt.splat" || name == "tt.broadcast" ||
      name == "tt.expand_dims" || name == "tt.reshape" ||
      name == "tt.trans" || name == "arith.extsi" ||
      name == "arith.extui") {
    if (definition->getNumOperands() != 1)
      return std::nullopt;
    return integerFootprint(definition->getOperand(0), pid, active);
  }
  if ((name == "arith.addi" || name == "arith.subi") &&
      definition->getNumOperands() == 2) {
    auto lhs = integerFootprint(definition->getOperand(0), pid, active);
    auto rhs = integerFootprint(definition->getOperand(1), pid, active);
    if (!lhs || !rhs)
      return std::nullopt;
    return combineFootprints(*lhs, *rhs, name == "arith.subi");
  }
  if (name == "arith.muli" && definition->getNumOperands() == 2) {
    if (auto rhs = splatInteger(definition->getOperand(1))) {
      auto lhs = integerFootprint(definition->getOperand(0), pid, active);
      return lhs ? scaleFootprint(*lhs, *rhs) : std::nullopt;
    }
    if (auto lhs = splatInteger(definition->getOperand(0))) {
      auto rhs = integerFootprint(definition->getOperand(1), pid, active);
      return rhs ? scaleFootprint(*rhs, *lhs) : std::nullopt;
    }
  }
  // Truncation, division, remainder, select, and nonlinear products can fold
  // different physical programs onto one address and are deliberately rejected.
  return std::nullopt;
}

std::optional<AffinePidFootprint>
pointerFootprint(Value pointer, Value root, Value pid,
                 llvm::SmallPtrSetImpl<Value> &active) {
  if (pointer == root)
    return AffinePidFootprint{};
  if (!active.insert(pointer).second)
    return std::nullopt;
  auto erase = llvm::make_scope_exit([&] { active.erase(pointer); });
  Operation *definition = pointer.getDefiningOp();
  if (!definition)
    return std::nullopt;
  StringRef name = definition->getName().getStringRef();
  if (name == "tt.splat" || name == "tt.broadcast" ||
      name == "tt.expand_dims" || name == "tt.reshape" ||
      name == "tt.trans") {
    if (definition->getNumOperands() != 1)
      return std::nullopt;
    return pointerFootprint(definition->getOperand(0), root, pid, active);
  }
  auto addPtr = dyn_cast<AddPtrOp>(definition);
  if (!addPtr)
    return std::nullopt;
  auto base = pointerFootprint(addPtr.getPtr(), root, pid, active);
  llvm::SmallPtrSet<Value, 32> integerActive;
  auto offset = integerFootprint(addPtr.getOffset(), pid, integerActive);
  if (!base || !offset)
    return std::nullopt;
  return combineFootprints(*base, *offset, false);
}

std::optional<AffinePidFootprint> pointerFootprint(Value pointer, Value root,
                                                   Value pid) {
  llvm::SmallPtrSet<Value, 32> active;
  return pointerFootprint(pointer, root, pid, active);
}

bool provesDisjointPrograms(const AffinePidFootprint &footprint) {
  __int128 span = static_cast<__int128>(footprint.localMax) -
                  static_cast<__int128>(footprint.localMin);
  __int128 stride = footprint.pidStride;
  if (stride < 0)
    stride = -stride;
  return stride > span;
}

Value stripShapeOnlyIntegerCasts(Value value) {
  while (Operation *definition = value.getDefiningOp()) {
    StringRef name = definition->getName().getStringRef();
    if (name != "tt.splat" && name != "tt.broadcast" &&
        name != "tt.expand_dims" && name != "tt.reshape" &&
        name != "tt.trans" && name != "arith.extsi" &&
        name != "arith.extui")
      break;
    if (definition->getNumOperands() != 1)
      break;
    value = definition->getOperand(0);
  }
  return value;
}

bool matchesPidTimesExtent(Value value, Value pid, Value &extent) {
  value = stripShapeOnlyIntegerCasts(value);
  auto multiply = value.getDefiningOp<arith::MulIOp>();
  if (!multiply)
    return false;
  Value lhs = stripShapeOnlyIntegerCasts(multiply.getLhs());
  Value rhs = stripShapeOnlyIntegerCasts(multiply.getRhs());
  if (lhs == pid && !valueDependsOn(rhs, pid)) {
    extent = rhs;
    return true;
  }
  if (rhs == pid && !valueDependsOn(lhs, pid)) {
    extent = lhs;
    return true;
  }
  return false;
}

bool maskProvesLocalBelowExtent(Value mask, Value local, Value extent) {
  if (!mask)
    return false;
  Operation *definition = mask.getDefiningOp();
  if (!definition)
    return false;
  if (auto conjunction = dyn_cast<arith::AndIOp>(definition))
    return maskProvesLocalBelowExtent(
               conjunction.getLhs(), local, extent) ||
           maskProvesLocalBelowExtent(
               conjunction.getRhs(), local, extent);
  auto comparison = dyn_cast<arith::CmpIOp>(definition);
  if (!comparison ||
      (comparison.getPredicate() != arith::CmpIPredicate::slt &&
       comparison.getPredicate() != arith::CmpIPredicate::ult))
    return false;
  return stripShapeOnlyIntegerCasts(comparison.getLhs()) ==
             stripShapeOnlyIntegerCasts(local) &&
         stripShapeOnlyIntegerCasts(comparison.getRhs()) ==
             stripShapeOnlyIntegerCasts(extent);
}

int64_t staticElementCount(Type type);

struct RuntimeActiveExtentFact {
  int64_t argumentIndex = -1;
  int64_t containerWidth = 0;
};

// A runtime scalar can control one logical tensor axis even when the complete
// memory operation is a higher-rank broadcast.  This fact is deliberately
// separate from RuntimeActiveExtentFact: the latter owns exact whole-operation
// work accounting, while this one records only the compiler-visible mask axis
// and its replication.  Conflating the two would undercount a 2-D mask as one
// active row.
struct RuntimeMaskScalarFact {
  bool write = false;
  int64_t argumentIndex = -1;
  int64_t containerWidth = 0;
  int64_t maskElementCount = 0;
  int64_t replicationFactor = 0;
  int64_t boundValue = 0;
  int64_t divisibility = 1;
  bool boundValueKnown = false;
  bool complete = false;
};

std::optional<RuntimeMaskScalarFact>
runtimeMaskScalarFromMask(Value mask, FuncOp entry, bool write) {
  if (!mask)
    return std::nullopt;
  int64_t maskElements = staticElementCount(mask.getType());
  Value normalizedMask = stripShapeOnlyIntegerCasts(mask);
  auto comparison = normalizedMask.getDefiningOp<arith::CmpIOp>();
  if (!comparison ||
      (comparison.getPredicate() != arith::CmpIPredicate::slt &&
       comparison.getPredicate() != arith::CmpIPredicate::ult))
    return std::nullopt;
  Value local = stripShapeOnlyIntegerCasts(comparison.getLhs());
  Value extent = stripShapeOnlyIntegerCasts(comparison.getRhs());
  auto range = local.getDefiningOp<MakeRangeOp>();
  auto argument = dyn_cast<BlockArgument>(extent);
  if (!range || !argument || argument.getOwner() != &entry.getBody().front() ||
      range.getStartAttr().getInt() != 0 ||
      range.getEndAttr().getInt() < 1)
    return std::nullopt;
  int64_t width = staticElementCount(local.getType());
  if (width != range.getEndAttr().getInt() || maskElements < width ||
      maskElements % width != 0)
    return std::nullopt;
  RuntimeMaskScalarFact fact;
  fact.write = write;
  fact.argumentIndex = argument.getArgNumber();
  fact.containerWidth = width;
  fact.maskElementCount = maskElements;
  fact.replicationFactor = maskElements / width;
  if (auto bound = entry.getArgAttrOfType<IntegerAttr>(
          argument.getArgNumber(), kBridgeBoundScalarAttr)) {
    fact.boundValue = bound.getInt();
    fact.boundValueKnown = true;
  }
  if (auto divisibility = entry.getArgAttrOfType<IntegerAttr>(
          argument.getArgNumber(), "tt.divisibility"))
    fact.divisibility = divisibility.getInt();
  fact.complete = fact.argumentIndex >= 0 && fact.containerWidth > 0 &&
                  fact.replicationFactor > 0 && fact.divisibility > 0 &&
                  (!fact.boundValueKnown ||
                   (fact.boundValue > 0 &&
                    fact.boundValue <= fact.containerWidth));
  return fact;
}

// Recognize a prospective active-lane fact directly from TTIR dataflow:
//   make_range(0, W) < splat(entry_argument).
// The runtime value is intentionally not inspected here.  The fact only
// records which launch argument supplies the extent and the statically proven
// container bound.  This is independent of operator, shape, group-size value,
// benchmark identity and profitability.
std::optional<RuntimeActiveExtentFact>
runtimeActiveExtentFromMask(Value mask, FuncOp entry) {
  if (!mask)
    return std::nullopt;
  Operation *definition = mask.getDefiningOp();
  if (!definition)
    return std::nullopt;
  if (auto conjunction = dyn_cast<arith::AndIOp>(definition)) {
    if (auto lhs = runtimeActiveExtentFromMask(conjunction.getLhs(), entry))
      return lhs;
    return runtimeActiveExtentFromMask(conjunction.getRhs(), entry);
  }
  auto comparison = dyn_cast<arith::CmpIOp>(definition);
  if (!comparison ||
      (comparison.getPredicate() != arith::CmpIPredicate::slt &&
       comparison.getPredicate() != arith::CmpIPredicate::ult))
    return std::nullopt;
  Value local = stripShapeOnlyIntegerCasts(comparison.getLhs());
  Value extent = stripShapeOnlyIntegerCasts(comparison.getRhs());
  auto range = local.getDefiningOp<MakeRangeOp>();
  auto argument = dyn_cast<BlockArgument>(extent);
  if (!range || !argument || argument.getOwner() != &entry.getBody().front() ||
      range.getStartAttr().getInt() != 0 ||
      range.getEndAttr().getInt() < 1)
    return std::nullopt;
  int64_t width = staticElementCount(local.getType());
  if (width != range.getEndAttr().getInt())
    return std::nullopt;
  return RuntimeActiveExtentFact{
      static_cast<int64_t>(argument.getArgNumber()), width};
}

// Prove the common symbolic partition
//   root + pid * extent + local, 0 <= local < extent.
// This is stronger and more general than replacing a runtime extent with a
// benchmark constant.  The store mask itself supplies the upper bound; if the
// extent is non-positive no lane stores and disjointness holds vacuously.
bool provesSymbolicPidPartition(StoreOp store, Value root, Value pid) {
  auto localAdd = store.getPtr().getDefiningOp<AddPtrOp>();
  if (!localAdd)
    return false;
  Value local = localAdd.getOffset();
  llvm::SmallPtrSet<Value, 32> localActive;
  auto localFootprint = integerFootprint(local, pid, localActive);
  if (!localFootprint || localFootprint->pidStride != 0 ||
      localFootprint->localMin < 0)
    return false;
  Value scalarBase = stripShapeOnlyIntegerCasts(localAdd.getPtr());
  auto pidAdd = scalarBase.getDefiningOp<AddPtrOp>();
  if (!pidAdd)
    return false;
  auto baseRoot = uniquePointerRoot(pidAdd.getPtr());
  if (!baseRoot || *baseRoot != root)
    return false;
  Value extent;
  if (!matchesPidTimesExtent(pidAdd.getOffset(), pid, extent))
    return false;
  return maskProvesLocalBelowExtent(store.getMask(), local, extent);
}

struct MixedRadixRangeTerm {
  Value basis;
  int64_t coefficient = 0;
  int64_t lower = 0;
  int64_t upper = 0;
};

struct MixedRadixExpression {
  int64_t quotientCoefficient = 0;
  int64_t remainderCoefficient = 0;
  SmallVector<MixedRadixRangeTerm> ranges;
};

struct MixedRadixContext {
  Value pid;
  Value quotient;
  Value remainder;
  int64_t partitionExtent = 0;
  int64_t launchBound = 0;
};

bool valueContainsMakeRange(Value value,
                            llvm::SmallPtrSetImpl<Value> &visited) {
  if (!visited.insert(value).second)
    return false;
  if (value.getDefiningOp<MakeRangeOp>())
    return true;
  Operation *definition = value.getDefiningOp();
  return definition && llvm::any_of(
      definition->getOperands(), [&](Value operand) {
        return valueContainsMakeRange(operand, visited);
      });
}

bool valueContainsMakeRange(Value value) {
  llvm::SmallPtrSet<Value, 32> visited;
  return valueContainsMakeRange(value, visited);
}

bool addMixedRadixTerm(MixedRadixExpression &expression,
                       MixedRadixRangeTerm term) {
  for (MixedRadixRangeTerm &existing : expression.ranges) {
    if (existing.basis != term.basis)
      continue;
    auto coefficient = checkedI64(
        static_cast<__int128>(existing.coefficient) + term.coefficient);
    if (!coefficient || existing.lower != term.lower ||
        existing.upper != term.upper)
      return false;
    existing.coefficient = *coefficient;
    return true;
  }
  expression.ranges.push_back(term);
  return true;
}

std::optional<MixedRadixExpression>
combineMixedRadixExpressions(const MixedRadixExpression &lhs,
                             const MixedRadixExpression &rhs,
                             bool subtract) {
  auto quotient = checkedI64(
      static_cast<__int128>(lhs.quotientCoefficient) +
      (subtract ? -static_cast<__int128>(rhs.quotientCoefficient)
                : static_cast<__int128>(rhs.quotientCoefficient)));
  auto remainder = checkedI64(
      static_cast<__int128>(lhs.remainderCoefficient) +
      (subtract ? -static_cast<__int128>(rhs.remainderCoefficient)
                : static_cast<__int128>(rhs.remainderCoefficient)));
  if (!quotient || !remainder)
    return std::nullopt;
  MixedRadixExpression result{*quotient, *remainder, lhs.ranges};
  for (MixedRadixRangeTerm term : rhs.ranges) {
    if (subtract) {
      auto coefficient = checkedI64(-static_cast<__int128>(term.coefficient));
      if (!coefficient)
        return std::nullopt;
      term.coefficient = *coefficient;
    }
    if (!addMixedRadixTerm(result, term))
      return std::nullopt;
  }
  return result;
}

std::optional<MixedRadixExpression>
scaleMixedRadixExpression(const MixedRadixExpression &input, int64_t scale) {
  auto quotient = checkedI64(
      static_cast<__int128>(input.quotientCoefficient) * scale);
  auto remainder = checkedI64(
      static_cast<__int128>(input.remainderCoefficient) * scale);
  if (!quotient || !remainder)
    return std::nullopt;
  MixedRadixExpression result{*quotient, *remainder, {}};
  for (MixedRadixRangeTerm term : input.ranges) {
    auto coefficient = checkedI64(
        static_cast<__int128>(term.coefficient) * scale);
    if (!coefficient)
      return std::nullopt;
    term.coefficient = *coefficient;
    result.ranges.push_back(term);
  }
  return result;
}

std::optional<MixedRadixExpression>
mixedRadixIntegerExpression(Value value, const MixedRadixContext &context,
                            llvm::SmallPtrSetImpl<Value> &active) {
  if (value == context.quotient)
    return MixedRadixExpression{1, 0, {}};
  if (value == context.remainder)
    return MixedRadixExpression{0, 1, {}};
  if (value == context.pid)
    return MixedRadixExpression{
        context.partitionExtent, 1, {}};
  if (auto range = value.getDefiningOp<MakeRangeOp>())
    return MixedRadixExpression{
        0, 0,
        {{value, 1, range.getStartAttr().getInt(),
          range.getEndAttr().getInt() - 1}}};
  if (splatInteger(value))
    return MixedRadixExpression{};
  if (!active.insert(value).second)
    return std::nullopt;
  auto erase = llvm::make_scope_exit([&] { active.erase(value); });
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return std::nullopt;
  StringRef name = definition->getName().getStringRef();
  if (name == "tt.splat" || name == "tt.broadcast" ||
      name == "tt.expand_dims" || name == "tt.reshape" ||
      name == "tt.trans" || name == "arith.extsi" ||
      name == "arith.extui") {
    if (definition->getNumOperands() != 1)
      return std::nullopt;
    return mixedRadixIntegerExpression(
        definition->getOperand(0), context, active);
  }
  if ((name == "arith.addi" || name == "arith.subi") &&
      definition->getNumOperands() == 2) {
    auto lhs = mixedRadixIntegerExpression(
        definition->getOperand(0), context, active);
    auto rhs = mixedRadixIntegerExpression(
        definition->getOperand(1), context, active);
    if (!lhs || !rhs)
      return std::nullopt;
    return combineMixedRadixExpressions(
        *lhs, *rhs, name == "arith.subi");
  }
  if (name == "arith.muli" && definition->getNumOperands() == 2) {
    if (auto rhs = splatInteger(definition->getOperand(1))) {
      auto lhs = mixedRadixIntegerExpression(
          definition->getOperand(0), context, active);
      return lhs ? scaleMixedRadixExpression(*lhs, *rhs) : std::nullopt;
    }
    if (auto lhs = splatInteger(definition->getOperand(0))) {
      auto rhs = mixedRadixIntegerExpression(
          definition->getOperand(1), context, active);
      return rhs ? scaleMixedRadixExpression(*rhs, *lhs) : std::nullopt;
    }
  }
  if (!valueDependsOn(value, context.pid) && !valueContainsMakeRange(value))
    return MixedRadixExpression{};
  return std::nullopt;
}

std::optional<MixedRadixExpression>
mixedRadixPointerExpression(Value pointer, Value root,
                            const MixedRadixContext &context,
                            llvm::SmallPtrSetImpl<Value> &active) {
  if (pointer == root)
    return MixedRadixExpression{};
  if (!active.insert(pointer).second)
    return std::nullopt;
  auto erase = llvm::make_scope_exit([&] { active.erase(pointer); });
  Operation *definition = pointer.getDefiningOp();
  if (!definition)
    return std::nullopt;
  StringRef name = definition->getName().getStringRef();
  if (name == "tt.splat" || name == "tt.broadcast" ||
      name == "tt.expand_dims" || name == "tt.reshape" ||
      name == "tt.trans") {
    if (definition->getNumOperands() != 1)
      return std::nullopt;
    return mixedRadixPointerExpression(
        definition->getOperand(0), root, context, active);
  }
  auto addPtr = dyn_cast<AddPtrOp>(definition);
  if (!addPtr)
    return std::nullopt;
  auto base = mixedRadixPointerExpression(
      addPtr.getPtr(), root, context, active);
  llvm::SmallPtrSet<Value, 32> integerActive;
  auto offset = mixedRadixIntegerExpression(
      addPtr.getOffset(), context, integerActive);
  if (!base || !offset)
    return std::nullopt;
  return combineMixedRadixExpressions(*base, *offset, false);
}

void collectMaskedRangeUpperBounds(Value mask,
                                   llvm::DenseMap<Value, int64_t> &bounds) {
  if (!mask)
    return;
  mask = stripShapeOnlyIntegerCasts(mask);
  Operation *definition = mask.getDefiningOp();
  if (!definition)
    return;
  if (auto conjunction = dyn_cast<arith::AndIOp>(definition)) {
    collectMaskedRangeUpperBounds(conjunction.getLhs(), bounds);
    collectMaskedRangeUpperBounds(conjunction.getRhs(), bounds);
    return;
  }
  auto comparison = dyn_cast<arith::CmpIOp>(definition);
  if (!comparison ||
      (comparison.getPredicate() != arith::CmpIPredicate::slt &&
       comparison.getPredicate() != arith::CmpIPredicate::ult))
    return;
  Value local = stripShapeOnlyIntegerCasts(comparison.getLhs());
  auto range = local.getDefiningOp<MakeRangeOp>();
  auto upper = splatInteger(comparison.getRhs());
  if (!range || !upper || *upper <= range.getStartAttr().getInt())
    return;
  int64_t clipped = std::min<int64_t>(
      range.getEndAttr().getInt() - 1, *upper - 1);
  auto position = bounds.find(local);
  if (position == bounds.end())
    bounds[local] = clipped;
  else
    position->second = std::min(position->second, clipped);
}

std::optional<int64_t>
mixedRadixResidualSpan(const MixedRadixExpression &expression,
                       std::optional<unsigned> excludedRange) {
  __int128 span = 0;
  for (unsigned index = 0; index < expression.ranges.size(); ++index) {
    if (excludedRange && *excludedRange == index)
      continue;
    const MixedRadixRangeTerm &term = expression.ranges[index];
    __int128 coefficient = term.coefficient;
    if (coefficient < 0)
      coefficient = -coefficient;
    span += coefficient * (term.upper - term.lower);
  }
  return checkedI64(span);
}

bool mixedRadixSeparated(const MixedRadixExpression &expression,
                         const MixedRadixContext &context,
                         std::optional<unsigned> innerRange) {
  int64_t secondaryCount = context.partitionExtent;
  int64_t secondaryCoefficient = expression.remainderCoefficient;
  if (innerRange) {
    const MixedRadixRangeTerm &term = expression.ranges[*innerRange];
    int64_t innerCount = term.upper - term.lower + 1;
    if (innerCount < 2 || term.coefficient == 0)
      return false;
    if (context.partitionExtent > 1) {
      auto recombined = checkedI64(
          static_cast<__int128>(term.coefficient) * innerCount);
      if (!recombined || expression.remainderCoefficient != *recombined)
        return false;
    }
    auto count = checkedI64(
        static_cast<__int128>(secondaryCount) * innerCount);
    if (!count)
      return false;
    secondaryCount = *count;
    secondaryCoefficient = term.coefficient;
  }
  auto spanValue = mixedRadixResidualSpan(expression, innerRange);
  if (!spanValue)
    return false;
  __int128 span = *spanValue;
  __int128 quotientStep = expression.quotientCoefficient;
  __int128 secondaryStep = secondaryCoefficient;
  if (quotientStep < 0)
    quotientStep = -quotientStep;
  if (secondaryStep < 0)
    secondaryStep = -secondaryStep;
  int64_t quotientCount =
      (context.launchBound + context.partitionExtent - 1) /
      context.partitionExtent;
  if (secondaryCount == 1)
    return quotientStep > span;
  if (quotientCount == 1)
    return secondaryStep > span;
  if (static_cast<__int128>(expression.quotientCoefficient) *
          secondaryCoefficient <=
      0)
    return false;
  bool quotientMajor =
      secondaryStep > span &&
      quotientStep >
          static_cast<__int128>(secondaryCount - 1) * secondaryStep + span;
  bool secondaryMajor =
      quotientStep > span &&
      secondaryStep >
          static_cast<__int128>(quotientCount - 1) * quotientStep + span;
  return quotientMajor || secondaryMajor;
}

// Prove a bounded mixed-radix store partition from
//   q = pid / E, r = pid % E,
// and an address linear in q/r plus static tensor ranges.  One tensor range
// may extend r into a nested radix (for example vectorized groups).  Store
// masks may only tighten a recognized make_range upper bound.  The accepted
// q-major/r-major inequalities are sufficient and identity-free; unknown,
// non-positive, overflowing, or overlapping forms remain typed unsupported.
bool provesMixedRadixPidPartition(StoreOp store, Value root, Value pid) {
  FuncOp entry = store->getParentOfType<FuncOp>();
  if (!entry)
    return false;
  SmallVector<arith::DivSIOp> quotients;
  SmallVector<arith::RemSIOp> remainders;
  entry.walk([&](arith::DivSIOp op) {
    if (op.getLhs() == pid)
      quotients.push_back(op);
  });
  entry.walk([&](arith::RemSIOp op) {
    if (op.getLhs() == pid)
      remainders.push_back(op);
  });
  if (quotients.size() != 1 || remainders.size() != 1 ||
      quotients.front().getRhs() != remainders.front().getRhs())
    return false;
  auto extent = splatInteger(quotients.front().getRhs());
  auto launch = pid.getDefiningOp<GetProgramIdOp>()->getAttrOfType<IntegerAttr>(
      kBridgeAxisExtentAttr);
  if (!extent || *extent < 1 || !launch || launch.getInt() < 1)
    return false;
  MixedRadixContext context{
      pid, quotients.front().getResult(), remainders.front().getResult(),
      *extent, launch.getInt()};
  llvm::SmallPtrSet<Value, 32> active;
  auto expression = mixedRadixPointerExpression(
      store.getPtr(), root, context, active);
  if (!expression)
    return false;
  llvm::DenseMap<Value, int64_t> maskedBounds;
  collectMaskedRangeUpperBounds(store.getMask(), maskedBounds);
  for (MixedRadixRangeTerm &term : expression->ranges)
    if (auto position = maskedBounds.find(term.basis);
        position != maskedBounds.end())
      term.upper = std::min(term.upper, position->second);
  if (mixedRadixSeparated(*expression, context, std::nullopt))
    return true;
  for (unsigned index = 0; index < expression->ranges.size(); ++index)
    if (mixedRadixSeparated(*expression, context, index))
      return true;
  return false;
}

// A dynamic prefix recurrence is an algebraic subject, not an operator
// template.  This recognizer accepts exactly
//
//   acc = 0
//   for i in [0, pid_x): acc = acc + load(root + i)
//
// over an integer type.  The launch-bound axis extent is a predecision fact
// installed by LoopBridgeDiscoverPass.  It supplies the smallest power-of-two
// container for an equivalent predicated vector reduction.
struct ExactPrefixReduction {
  scf::ForOp loop;
  GetProgramIdOp upperPid;
  AddPtrOp pointer;
  LoadOp load;
  arith::AddIOp combine;
  int64_t activeExtent = 0;
  int64_t vectorWidth = 0;
};

std::optional<ExactPrefixReduction>
matchExactPrefixReduction(scf::ForOp loop) {
  if (loop.getNumRegionIterArgs() != 1 || loop.getInitArgs().size() != 1 ||
      loop.getNumResults() != 1 || !isa<IntegerType>(loop.getResult(0).getType()))
    return std::nullopt;
  APInt lower, step, init;
  if (!matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) ||
      !matchPattern(loop.getStep(), m_ConstantInt(&step)) ||
      !matchPattern(loop.getInitArgs().front(), m_ConstantInt(&init)) ||
      !lower.isZero() || !init.isZero() || step.getSExtValue() != 1)
    return std::nullopt;
  auto upperPid = loop.getUpperBound().getDefiningOp<GetProgramIdOp>();
  auto extent = upperPid
                    ? upperPid->getAttrOfType<IntegerAttr>(
                          kBridgeAxisExtentAttr)
                    : IntegerAttr();
  if (!upperPid || upperPid.getAxisAsInt() != 0 || !extent ||
      extent.getInt() < 1)
    return std::nullopt;
  uint64_t width = llvm::PowerOf2Ceil(
      static_cast<uint64_t>(extent.getInt()));
  if (width == 0 ||
      width > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return std::nullopt;

  SmallVector<AddPtrOp> pointers;
  SmallVector<LoadOp> loads;
  bool unsupported = false;
  for (Operation &operation : loop.getBody()->without_terminator()) {
    if (auto pointer = dyn_cast<AddPtrOp>(operation))
      pointers.push_back(pointer);
    else if (auto load = dyn_cast<LoadOp>(operation))
      loads.push_back(load);
    else if (isa<StoreOp, AtomicRMWOp, AtomicCASOp, scf::ForOp>(operation) ||
             !isMemoryEffectFree(&operation))
      unsupported = true;
  }
  auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
  if (unsupported || pointers.size() != 1 || loads.size() != 1 ||
      !yield || yield.getNumOperands() != 1 ||
      loads.front().getIsVolatile() || loads.front().getMask() ||
      loads.front().getOther() || pointers.front().getOffset() !=
                                      loop.getInductionVar() ||
      loads.front().getPtr() != pointers.front().getResult())
    return std::nullopt;
  Value carried = loop.getRegionIterArgs().front();
  Value loaded = loads.front().getResult();
  arith::AddIOp add = yield.getOperand(0).getDefiningOp<arith::AddIOp>();
  if (!add)
    return std::nullopt;
  bool exact = (add.getLhs() == carried && add.getRhs() == loaded) ||
               (add.getRhs() == carried && add.getLhs() == loaded);
  if (!exact || yield.getOperand(0) != add.getResult())
    return std::nullopt;
  return ExactPrefixReduction{loop, upperPid, pointers.front(), loads.front(),
                              add, extent.getInt(),
                              static_cast<int64_t>(width)};
}

SmallVector<ExactPrefixReduction> collectExactPrefixReductions(FuncOp entry) {
  SmallVector<ExactPrefixReduction> result;
  entry.walk([&](scf::ForOp loop) {
    if (auto reduction = matchExactPrefixReduction(loop))
      result.push_back(*reduction);
  });
  return result;
}

LogicalResult materializeExactPrefixReduction(
    ExactPrefixReduction subject, int64_t elementBudget) {
  if (elementBudget < 1 || subject.vectorWidth > elementBudget)
    return failure();
  scf::ForOp loop = subject.loop;
  Location loc = loop.getLoc();
  OpBuilder builder(loop);
  Type elementType = loop.getResult(0).getType();
  auto valueType = RankedTensorType::get(
      {subject.vectorWidth}, elementType);
  auto pointerType = RankedTensorType::get(
      {subject.vectorWidth}, subject.pointer.getPtr().getType());
  Value lanes = MakeRangeOp::create(
      builder, loc, valueType, 0, subject.vectorWidth);
  Value upper = SplatOp::create(
      builder, loc, valueType, loop.getUpperBound());
  Value mask = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::slt, lanes, upper);
  Value base = SplatOp::create(
      builder, loc, pointerType, subject.pointer.getPtr());
  Value pointers = AddPtrOp::create(
      builder, loc, pointerType, base, lanes);
  Value identity = SplatOp::create(
      builder, loc, valueType, loop.getInitArgs().front());
  auto loaded = LoadOp::create(
      builder, loc, pointers, mask, identity, subject.load.getCache(),
      subject.load.getEvict(), subject.load.getIsVolatile());
  auto reduced = ReduceOp::create(
      builder, loc, ValueRange{loaded.getResult()}, 0u);
  Block &body = reduced.getCombineOp().emplaceBlock();
  body.addArguments({elementType, elementType}, {loc, loc});
  OpBuilder combineBuilder(&body, body.end());
  Value sum = arith::AddIOp::create(
      combineBuilder, loc, body.getArgument(0), body.getArgument(1));
  ReduceReturnOp::create(combineBuilder, loc, sum);
  loop.getResult(0).replaceAllUsesWith(reduced.getResult().front());
  loop.erase();
  return success();
}

std::optional<APFloat> splatFloatConstant(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return std::nullopt;
  if (auto scalar = dyn_cast<FloatAttr>(constant.getValue()))
    return scalar.getValue();
  if (auto dense = dyn_cast<DenseFPElementsAttr>(constant.getValue());
      dense && dense.isSplat())
    return dense.getSplatValue<APFloat>();
  return std::nullopt;
}

bool isAddFReduction(ReduceOp reduce) {
  if (reduce.getAxis() != 0 || reduce.getSrcs().size() != 1 ||
      reduce.getResults().size() != 1 ||
      !llvm::hasSingleElement(reduce.getCombineOp()))
    return false;
  Block &block = reduce.getCombineOp().front();
  if (block.getNumArguments() != 2 ||
      std::distance(block.begin(), block.end()) != 2)
    return false;
  auto add = dyn_cast<arith::AddFOp>(block.front());
  auto result = dyn_cast<ReduceReturnOp>(block.back());
  return add && result && result.getNumOperands() == 1 &&
         result.getOperand(0) == add.getResult() &&
         ((add.getLhs() == block.getArgument(0) &&
           add.getRhs() == block.getArgument(1)) ||
          (add.getLhs() == block.getArgument(1) &&
           add.getRhs() == block.getArgument(0)));
}

bool isStateAxisElementwiseCapability(Operation *operation) {
  if (!operation || operation->getNumResults() != 1 ||
      !isMemoryEffectFree(operation))
    return false;
  StringRef name = operation->getName().getStringRef();
  return name == "arith.addf" || name == "arith.subf" ||
         name == "arith.mulf" || name == "arith.divf" ||
         name == "arith.maxnumf" || name == "arith.minnumf" ||
         name == "math.exp";
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
             stateAxisReductionCombiner(other) == combiner;
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

bool stateAxisSubjectMatchesPlan(
    const StateAxisNormalizationRegion &subject,
    const ParsedLoopPlan::StateAxisGroup &group) {
  if (subject.stateCardinality != group.stateCardinality ||
      subject.paddedStateCardinality != group.paddedStateCardinality ||
      subject.stateWidth != group.stateWidth ||
      subject.elementBytes != group.elementBytes ||
      subject.laneOperandIndex != group.laneOperandIndex ||
      subject.packableNodeCount != group.packableNodeCount ||
      subject.graphSignature != group.graphSignature ||
      subject.operationCapabilities.size() !=
          group.operationCapabilities.size())
    return false;
  return std::equal(subject.operationCapabilities.begin(),
                    subject.operationCapabilities.end(),
                    group.operationCapabilities.begin());
}

Value joinStateAxisTensor(OpBuilder &builder, Location loc,
                          ArrayRef<Value> leaves) {
  assert(!leaves.empty() && llvm::isPowerOf2_64(leaves.size()));
  if (leaves.size() == 1)
    return leaves.front();
  size_t middle = leaves.size() / 2;
  Value lhs = joinStateAxisTensor(builder, loc, leaves.take_front(middle));
  Value rhs = joinStateAxisTensor(builder, loc, leaves.drop_front(middle));
  return JoinOp::create(builder, loc, lhs, rhs);
}

void splitStateAxisTensor(OpBuilder &builder, Location loc, Value value,
                          int64_t leafCount,
                          SmallVectorImpl<Value> &leaves) {
  assert(leafCount >= 1 && llvm::isPowerOf2_64(leafCount));
  if (leafCount == 1) {
    leaves.push_back(value);
    return;
  }
  auto halves = SplitOp::create(builder, loc, value);
  splitStateAxisTensor(builder, loc, halves.getOutLHS(), leafCount / 2,
                       leaves);
  splitStateAxisTensor(builder, loc, halves.getOutRHS(), leafCount / 2,
                       leaves);
}

Type packedStateAxisType(Type type, int64_t paddedCardinality) {
  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    SmallVector<int64_t> shape(tensor.getShape());
    shape.push_back(paddedCardinality);
    return RankedTensorType::get(
        shape, tensor.getElementType(), tensor.getEncoding());
  }
  if (isa<FloatType>(type))
    return RankedTensorType::get({paddedCardinality}, type);
  return Type();
}

class StateAxisBundlePacker {
public:
  StateAxisBundlePacker(OpBuilder &builder, Location location,
                        int64_t paddedCardinality)
      : builder(builder), loc(location), padded(paddedCardinality) {}

  FailureOr<Value> pack(ArrayRef<Value> values) {
    for (auto &item : memo)
      if (item.first == values)
        return item.second;
    if (values.empty())
      return failure();
    Type resultType = packedStateAxisType(values.front().getType(), padded);
    if (!resultType || llvm::any_of(values, [&](Value value) {
          return value.getType() != values.front().getType();
        }))
      return failure();
    FailureOr<Value> result = allSameStateAxisValues(values)
                                  ? broadcast(values.front(), resultType)
                                  : transformOrBoundary(values, resultType);
    if (succeeded(result))
      memo.push_back({SmallVector<Value>(values), *result});
    return result;
  }

private:
  FailureOr<Value> broadcast(Value value, Type resultType) {
    auto packedType = cast<RankedTensorType>(resultType);
    if (isa<FloatType>(value.getType()))
      return SplatOp::create(builder, loc, packedType, value).getResult();
    auto source = dyn_cast<RankedTensorType>(value.getType());
    if (!source || source.getRank() != 1)
      return failure();
    Value expanded = ExpandDimsOp::create(builder, loc, value, 1);
    return BroadcastOp::create(builder, loc, packedType, expanded).getResult();
  }

  Value zeroLike(Type type) {
    Type elementType = isa<RankedTensorType>(type)
                           ? cast<RankedTensorType>(type).getElementType()
                           : type;
    auto floatType = cast<FloatType>(elementType);
    Value zero = arith::ConstantOp::create(
        builder, loc, FloatAttr::get(floatType, 0.0));
    if (auto tensor = dyn_cast<RankedTensorType>(type))
      return SplatOp::create(builder, loc, tensor, zero);
    return zero;
  }

  FailureOr<Value> boundary(ArrayRef<Value> values, Type resultType) {
    SmallVector<Value> leaves;
    auto originalTensor = dyn_cast<RankedTensorType>(values.front().getType());
    for (Value value : values) {
      if (originalTensor)
        leaves.push_back(value);
      else {
        auto singleton = RankedTensorType::get({1}, value.getType());
        leaves.push_back(SplatOp::create(builder, loc, singleton, value));
      }
    }
    while (leaves.size() < static_cast<size_t>(padded)) {
      Value zero = zeroLike(values.front().getType());
      if (originalTensor)
        leaves.push_back(zero);
      else {
        auto singleton = RankedTensorType::get({1}, zero.getType());
        leaves.push_back(SplatOp::create(builder, loc, singleton, zero));
      }
    }
    Value joined = joinStateAxisTensor(builder, loc, leaves);
    return ReshapeOp::create(
        builder, loc, cast<RankedTensorType>(resultType).getShape(), joined,
        /*allowReorder=*/false).getResult();
  }

  FailureOr<Value> transformOrBoundary(ArrayRef<Value> values,
                                       Type resultType) {
    Operation *prototype = nullptr;
    if (!stateAxisOperationsEquivalent(values, prototype))
      return boundary(values, resultType);
    SmallVector<Value> operands;
    for (unsigned operand = 0; operand < prototype->getNumOperands(); ++operand) {
      SmallVector<Value> laneOperands;
      for (Value value : values)
        laneOperands.push_back(value.getDefiningOp()->getOperand(operand));
      auto packedOperand = pack(laneOperands);
      if (failed(packedOperand))
        return failure();
      operands.push_back(*packedOperand);
    }
    if (auto reduce = dyn_cast<ReduceOp>(prototype)) {
      auto packedReduce = ReduceOp::create(
          builder, loc, ValueRange{operands.front()}, reduce.getAxis());
      IRMapping mapping;
      reduce.getCombineOp().cloneInto(
          &packedReduce.getCombineOp(), mapping);
      packedReduce->setAttr(
          kStateAxisArtifactAttr,
          builder.getStringAttr("packed_node"));
      packedReduce->setAttr(kStateAxisPackableNodeAttr,
                            builder.getUnitAttr());
      return packedReduce.getResult().front();
    }
    if (isa<SplatOp>(prototype)) {
      auto resultTensor = cast<RankedTensorType>(resultType);
      Value expanded = ExpandDimsOp::create(
          builder, loc, operands.front(), 0);
      Value broadcast = BroadcastOp::create(
          builder, loc, resultTensor, expanded).getResult();
      broadcast.getDefiningOp()->setAttr(
          kStateAxisArtifactAttr, builder.getStringAttr("packed_node"));
      broadcast.getDefiningOp()->setAttr(kStateAxisPackableNodeAttr,
                                         builder.getUnitAttr());
      return broadcast;
    }
    OperationState state(loc, prototype->getName());
    state.addOperands(operands);
    state.addTypes(resultType);
    state.addAttributes(prototype->getAttrs());
    Operation *created = builder.create(state);
    created->setAttr(kStateAxisArtifactAttr,
                     builder.getStringAttr("packed_node"));
    created->setAttr(kStateAxisPackableNodeAttr, builder.getUnitAttr());
    return created->getResult(0);
  }

  OpBuilder &builder;
  Location loc;
  int64_t padded;
  SmallVector<std::pair<SmallVector<Value>, Value>> memo;
};

FailureOr<Value> cloneStateAxisCrossGraph(
    OpBuilder &builder, Value value, ArrayRef<Value> oldLeaves,
    ArrayRef<Value> newLeaves, llvm::DenseMap<Value, Value> &memo) {
  for (auto [oldValue, newValue] : llvm::zip(oldLeaves, newLeaves))
    if (value == oldValue)
      return newValue;
  if (auto found = memo.find(value); found != memo.end())
    return found->second;
  bool depends = llvm::any_of(oldLeaves, [&](Value leaf) {
    return valueDependsOn(value, leaf);
  });
  if (!depends)
    return value;
  Operation *operation = value.getDefiningOp();
  if (!operation || operation->getNumResults() != 1 ||
      operation->getNumRegions() != 0 ||
      !isStateAxisElementwiseCapability(operation))
    return failure();
  IRMapping mapping;
  for (Value operand : operation->getOperands()) {
    auto replacement = cloneStateAxisCrossGraph(
        builder, operand, oldLeaves, newLeaves, memo);
    if (failed(replacement))
      return failure();
    mapping.map(operand, *replacement);
  }
  Operation *cloned = builder.clone(*operation, mapping);
  // The subject marker belongs to the original Provider anchor.  Copying it
  // into the cloned cross-state graph would counterfeit a second admitted
  // subject and break one-to-one Plan/artifact correspondence.
  cloned->removeAttr(kSubjectAttr);
  memo[value] = cloned->getResult(0);
  return cloned->getResult(0);
}

LogicalResult materializeStateAxisNormalization(
    StateAxisNormalizationRegion subject) {
  Location loc = subject.finalLaneOperations.front()->getLoc();
  OpBuilder builder(subject.finalLaneOperations.front());
  auto inputType = cast<RankedTensorType>(subject.laneValues[0].getType());
  int64_t padded = subject.paddedStateCardinality;
  auto packedType = cast<RankedTensorType>(
      packedStateAxisType(inputType, padded));
  StateAxisBundlePacker packer(builder, loc, padded);
  auto packedLaneValues = packer.pack(subject.laneValues);
  if (failed(packedLaneValues) ||
      (*packedLaneValues).getType() != packedType)
    return failure();
  (*packedLaneValues).getDefiningOp()->setAttr(
      kStateAxisArtifactAttr, builder.getStringAttr("packed_root"));
  SmallVector<int64_t> nestedShape{subject.stateWidth};
  for (int64_t count = padded; count > 1; count /= 2)
    nestedShape.push_back(2);
  Value nestedRows = ReshapeOp::create(
      builder, loc, nestedShape, *packedLaneValues,
      /*allowReorder=*/false);
  SmallVector<Value> unpackedLanes;
  splitStateAxisTensor(builder, loc, nestedRows, padded, unpackedLanes);
  llvm::DenseMap<Value, Value> crossMemo;
  auto candidateDenominator = cloneStateAxisCrossGraph(
      builder, subject.columnDenominator, subject.laneValues,
      ArrayRef<Value>(unpackedLanes).take_front(subject.stateCardinality),
      crossMemo);
  if (failed(candidateDenominator))
    return failure();
  Value expandedColumns = ExpandDimsOp::create(
      builder, loc, *candidateDenominator, 1);
  Value broadcastColumns = BroadcastOp::create(
      builder, loc, packedType, expandedColumns);
  Operation *prototype = subject.finalLaneOperations.front();
  OperationState finalState(loc, prototype->getName());
  finalState.addOperands(subject.laneOperandIndex == 0
                             ? ValueRange{*packedLaneValues, broadcastColumns}
                             : ValueRange{broadcastColumns, *packedLaneValues});
  finalState.addTypes(packedType);
  finalState.addAttributes(prototype->getAttrs());
  Operation *packedFinal = builder.create(finalState);
  packedFinal->setAttr(
      kStateAxisArtifactAttr,
      builder.getStringAttr("cross_state_consumer"));
  Value nestedOutputs = ReshapeOp::create(
      builder, loc, nestedShape, packedFinal->getResult(0),
      /*allowReorder=*/false);
  SmallVector<Value> outputs;
  splitStateAxisTensor(builder, loc, nestedOutputs, padded, outputs);
  for (unsigned index = 0; index < subject.stateCardinality; ++index)
    subject.outputs[index].replaceAllUsesWith(outputs[index]);
  for (Operation *operation : subject.finalLaneOperations)
    operation->erase();
  return success();
}

LoopDependenceCertificate certifyBridgeProgramIndependence(
    FuncOp entry, GetProgramIdOp pid) {
  SmallVector<LoadOp> loads;
  SmallVector<StoreOp> stores;
  llvm::SmallPtrSet<Operation *, 4> exactPrefixLoops;
  for (ExactPrefixReduction subject :
       collectExactPrefixReductions(entry))
    exactPrefixLoops.insert(subject.loop.getOperation());
  std::string reason;
  bool rejectedEffect = false;
  entry.walk([&](Operation *op) {
    if (isa<AtomicRMWOp, AtomicCASOp>(op)) {
      rejectedEffect = true;
      reason = "atomic memory operation";
      return;
    }
    if (auto load = dyn_cast<LoadOp>(op)) {
      if (load.getIsVolatile()) {
        rejectedEffect = true;
        reason = "volatile load";
      }
      loads.push_back(load);
      return;
    }
    if (auto store = dyn_cast<StoreOp>(op)) {
      stores.push_back(store);
      return;
    }
    if (exactPrefixLoops.contains(op) ||
        isa<FuncOp, ReturnOp, scf::IfOp, scf::YieldOp, ReduceOp,
            ReduceReturnOp>(op))
      return;
    if (!isMemoryEffectFree(op)) {
      rejectedEffect = true;
      reason = "unknown or unsupported side effect";
    }
  });
  if (rejectedEffect)
    return {false, "", reason};
  if (stores.empty())
    return {false, "", "program has no pid-partitioned output store"};

  llvm::SmallPtrSet<Value, 8> loadRoots;
  llvm::SmallPtrSet<Value, 8> storeRoots;
  llvm::DenseMap<Value, AffinePidFootprint> storeRootFootprints;
  llvm::SmallPtrSet<Value, 8> symbolicStoreRoots;
  for (LoadOp load : loads) {
    auto root = uniquePointerRoot(load.getPtr());
    if (!root)
      return {false, "", "load pointer does not have one auditable base"};
    loadRoots.insert(*root);
  }
  unsigned storeOrdinal = 0;
  for (StoreOp store : stores) {
    auto root = uniquePointerRoot(store.getPtr());
    if (!root)
      return {false, "", "store pointer does not have one auditable base"};
    storeRoots.insert(*root);
    auto footprint = pointerFootprint(store.getPtr(), *root, pid.getResult());
    bool numericallyDisjoint =
        footprint && provesDisjointPrograms(*footprint);
    if (!numericallyDisjoint &&
        !provesSymbolicPidPartition(store, *root, pid.getResult()) &&
        !provesMixedRadixPidPartition(store, *root, pid.getResult())) {
      std::string detail;
      llvm::raw_string_ostream stream(detail);
      stream << "store " << storeOrdinal
             << (footprint ? " affine interval overlaps adjacent program"
                           : " affine pid footprint is unavailable");
      if (footprint)
        stream << " (stride=" << footprint->pidStride
               << ", local_min=" << footprint->localMin
               << ", local_max=" << footprint->localMax << ")";
      stream.flush();
      return {false, "", detail};
    }
    if (footprint) {
      if (symbolicStoreRoots.contains(*root))
        return {false, "", "same output base mixes symbolic and affine stores"};
      auto [position, inserted] = storeRootFootprints.try_emplace(
          *root, *footprint);
      if (!inserted) {
        AffinePidFootprint &combined = position->second;
        if (combined.pidStride != footprint->pidStride)
          return {false, "", "same output base has incompatible pid strides"};
        combined.localMin = std::min(
            combined.localMin, footprint->localMin);
        combined.localMax = std::max(
            combined.localMax, footprint->localMax);
        if (!provesDisjointPrograms(combined))
          return {false, "", "same output base has overlapping cross-program stores"};
      }
    } else {
      if (storeRootFootprints.contains(*root) ||
          !symbolicStoreRoots.insert(*root).second)
        return {false, "", "same output base has uncomposed symbolic stores"};
    }
    ++storeOrdinal;
  }
  for (Value root : storeRoots)
    if (loadRoots.contains(root))
      return {false, "", "read/write base alias is not disproven"};
  return {true, "bridge_pid_partitioned_disjoint_v1", ""};
}

struct DirectPidPartition {
  arith::DivSIOp quotient;
  arith::RemSIOp remainder;
  Value extent;
};

std::optional<DirectPidPartition>
findDirectPidPartition(FuncOp entry, GetProgramIdOp pid) {
  SmallVector<arith::DivSIOp> quotients;
  SmallVector<arith::RemSIOp> remainders;
  entry.walk([&](arith::DivSIOp op) {
    if (op.getLhs() == pid.getResult())
      quotients.push_back(op);
  });
  entry.walk([&](arith::RemSIOp op) {
    if (op.getLhs() == pid.getResult())
      remainders.push_back(op);
  });
  if (quotients.size() != 1 || remainders.size() != 1 ||
      quotients.front().getRhs() != remainders.front().getRhs())
    return std::nullopt;
  return DirectPidPartition{quotients.front(), remainders.front(),
                            quotients.front().getRhs()};
}

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

std::optional<int64_t> exactStaticTripCount(scf::ForOp loop) {
  APInt lower, upper, step;
  if (!matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) ||
      !matchPattern(loop.getUpperBound(), m_ConstantInt(&upper)) ||
      !matchPattern(loop.getStep(), m_ConstantInt(&step)))
    return std::nullopt;
  int64_t lb = lower.getSExtValue();
  int64_t ub = upper.getSExtValue();
  int64_t stride = step.getSExtValue();
  if (stride <= 0 || ub <= lb || (ub - lb) % stride != 0)
    return std::nullopt;
  return (ub - lb) / stride;
}

bool hasExactIntegerReductionCombiner(ReduceOp reduce) {
  if (reduce.getCombineOp().empty() ||
      reduce.getCombineOp().getBlocks().size() != 1)
    return false;
  Block &block = reduce.getCombineOp().front();
  auto terminator = dyn_cast<ReduceReturnOp>(block.getTerminator());
  if (!terminator || terminator.getNumOperands() != 1 ||
      block.getNumArguments() != 2)
    return false;
  Operation *combiner = terminator.getOperand(0).getDefiningOp();
  if (!combiner || combiner->getNumOperands() != 2 ||
      combiner->getNumResults() != 1 ||
      !isa<IntegerType>(combiner->getResult(0).getType()))
    return false;
  bool arguments =
      (combiner->getOperand(0) == block.getArgument(0) &&
       combiner->getOperand(1) == block.getArgument(1)) ||
      (combiner->getOperand(1) == block.getArgument(0) &&
       combiner->getOperand(0) == block.getArgument(1));
  if (!arguments)
    return false;
  if (isa<arith::OrIOp, arith::AddIOp>(combiner))
    return true;
  auto call = dyn_cast<CallOp>(combiner);
  auto function = call
                      ? dyn_cast_or_null<FuncOp>(call.resolveCallable())
                      : FuncOp();
  if (!function || function.getBody().empty())
    return false;
  SmallVector<Operation *> body;
  for (Operation &op : function.getBody().front())
    if (!isa<ReturnOp>(op))
      body.push_back(&op);
  if (body.size() != 1 || !isa<arith::OrIOp, arith::AddIOp>(body.front()))
    return false;
  auto returned = dyn_cast<ReturnOp>(function.getBody().front().getTerminator());
  return returned && returned.getNumOperands() == 1 &&
         returned.getOperand(0) == body.front()->getResult(0);
}

LoopDependenceCertificate
certifyIndependentIterationExactInnerReduction(scf::ForOp loop) {
  auto trip = exactStaticTripCount(loop);
  if (!trip || *trip < 2)
    return {false, "", "iteration domain is not nontrivial and exact static"};
  auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
  if (!yield || yield.getNumOperands() != loop.getRegionIterArgs().size() ||
      yield.getNumOperands() == 0)
    return {false, "", "iteration induction map is not closed"};

  SmallVector<LoadOp> loads;
  SmallVector<StoreOp> stores;
  SmallVector<ReduceOp> reductions;
  bool rejected = false;
  std::string reason;
  loop.walk([&](Operation *op) {
    if (op == loop.getOperation())
      return;
    if (op != loop.getOperation() && isa<scf::ForOp>(op)) {
      rejected = true;
      reason = "nested scf.for requires a separately composed certificate";
      return;
    }
    if (isa<AtomicRMWOp, AtomicCASOp>(op)) {
      rejected = true;
      reason = "atomic effect is not iteration independent";
      return;
    }
    if (auto load = dyn_cast<LoadOp>(op)) {
      if (load.getIsVolatile()) {
        rejected = true;
        reason = "volatile load is not iteration reorderable";
      }
      loads.push_back(load);
      return;
    }
    if (auto store = dyn_cast<StoreOp>(op)) {
      stores.push_back(store);
      return;
    }
    if (auto reduce = dyn_cast<ReduceOp>(op)) {
      if (!hasExactIntegerReductionCombiner(reduce)) {
        rejected = true;
        reason = "inner reduction lacks an exact integer combiner";
      }
      reductions.push_back(reduce);
      return;
    }
    // The operations inside a reduction's combine region are part of the
    // already-checked exact combiner.  In particular, Triton commonly lowers
    // a named combine_fn to a tt.call; CallOp does not advertise generic
    // memory effects, so reclassifying it below would incorrectly reject the
    // closed OR/add reduction that has just been proved above.
    if (auto parent = op->getParentOfType<ReduceOp>();
        parent && hasExactIntegerReductionCombiner(parent))
      return;
    if (isa<ReduceReturnOp, scf::YieldOp>(op))
      return;
    if (!isMemoryEffectFree(op)) {
      rejected = true;
      reason = ("iteration body contains an unclosed side effect: " +
                op->getName().getStringRef()).str();
    }
  });
  if (rejected || loads.empty() || stores.empty() || reductions.empty())
    return {false, "", reason.empty()
                              ? "iteration lacks load, store, or exact reduction"
                              : reason};

  // Every carried value must be a pure affine induction recurrence.  This is
  // more general than naming row/column variables: the increment and all
  // pointer strides are ordinary predecision SSA facts.
  for (auto [carried, next] :
       llvm::zip(loop.getRegionIterArgs(), yield.getOperands())) {
    auto add = next.getDefiningOp<arith::AddIOp>();
    if (!add)
      return {false, "", "carried state is not an affine additive induction"};
    Value increment;
    if (add.getLhs() == carried)
      increment = add.getRhs();
    else if (add.getRhs() == carried)
      increment = add.getLhs();
    else
      return {false, "", "carried state does not advance from itself"};
    auto constant = splatInteger(increment);
    if (!constant || *constant == 0)
      return {false, "", "carried-state step is not a nonzero exact integer"};
    for (OpOperand &use : carried.getUses()) {
      Operation *owner = use.getOwner();
      if (owner == add || isMemoryEffectFree(owner))
        continue;
      return {false, "", "carried state has a non-affine iteration consumer"};
    }
  }

  llvm::SmallPtrSet<Value, 8> loadRoots;
  llvm::SmallPtrSet<Value, 8> storeRoots;
  for (LoadOp load : loads) {
    auto root = uniquePointerRoot(load.getPtr());
    if (!root)
      return {false, "", "iteration load has no unique base"};
    loadRoots.insert(*root);
  }
  for (StoreOp store : stores) {
    auto root = uniquePointerRoot(store.getPtr());
    if (!root)
      return {false, "", "iteration store has no unique base"};
    bool disjoint = false;
    for (Value carried : loop.getRegionIterArgs()) {
      auto footprint = pointerFootprint(store.getPtr(), *root, carried);
      if (footprint && provesDisjointPrograms(*footprint)) {
        disjoint = true;
        break;
      }
    }
    if (!disjoint)
      return {false, "", "store footprint is not disjoint across iterations"};
    storeRoots.insert(*root);
  }
  for (Value root : storeRoots)
    if (loadRoots.contains(root))
      return {false, "", "iteration input/output alias is not disproven"};
  return {true,
          "independent_iteration_exact_inner_reduction_v1", ""};
}

bool dependsOnAny(Value value, ValueRange targets) {
  return llvm::any_of(targets,
                      [&](Value target) { return valueDependsOn(value, target); });
}

// A pointer induction is semantic state, but not data-dependent state.  Moving
// the cloned address/read slice after unrolling is legal when the next pointer
// is exactly ``addptr(carried, invariant_step)`` and the pointer lineage ends
// only at reads or that matching yield.  This covers the ordinary Triton
// matmul idiom without naming dot, matmul, a benchmark, or a backend schedule.
bool isAffineReadOnlyPointerInduction(scf::ForOp loop, unsigned index) {
  if (index >= loop.getRegionIterArgs().size())
    return false;
  auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
  if (!yield || yield.getNumOperands() != loop.getRegionIterArgs().size())
    return false;
  Value carried = loop.getRegionIterArg(index);
  Value next = yield.getOperand(index);
  auto update = next.getDefiningOp<AddPtrOp>();
  if (!update || update.getPtr() != carried ||
      dependsOnAny(update.getOffset(), loop.getRegionIterArgs()))
    return false;

  llvm::SmallPtrSet<Value, 32> visited;
  SmallVector<Value> worklist{carried};
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *owner = use.getOwner();
      if (auto load = dyn_cast<LoadOp>(owner)) {
        if (use.get() != load.getPtr())
          return false;
        continue;
      }
      if (auto addPtr = dyn_cast<AddPtrOp>(owner)) {
        if (use.get() != addPtr.getPtr() ||
            dependsOnAny(addPtr.getOffset(), loop.getRegionIterArgs()))
          return false;
        llvm::append_range(worklist, addPtr->getResults());
        continue;
      }
      StringRef name = owner->getName().getStringRef();
      if (name == "tt.splat" || name == "tt.broadcast" ||
          name == "tt.expand_dims" || name == "tt.reshape" ||
          name == "tt.trans") {
        llvm::append_range(worklist, owner->getResults());
        continue;
      }
      if (owner == yield.getOperation() && use.getOperandNumber() == index &&
          value == next)
        continue;
      return false;
    }
  }
  return true;
}

bool hasOnlyAffineReadOnlyPointerDependencies(Value pointer,
                                              scf::ForOp loop) {
  for (auto [index, carried] : llvm::enumerate(loop.getRegionIterArgs()))
    if (valueDependsOn(pointer, carried) &&
        !isAffineReadOnlyPointerInduction(loop, index))
      return false;
  return true;
}

// Prove the coarser causal boundary needed when the source recurrence is not
// one exact add.  The route may expose read/address work from later iterations
// before earlier state updates, but it never changes the order of those state
// updates.  Consequently this proof deliberately does not identify max/argmax,
// dot, or any workload-specific combiner: it proves only that every moved
// read is non-volatile, independent of data-carrying state (or depends only on
// a proved affine read-only pointer induction), and cannot cross a write or
// unknown effect.
LoopDependenceCertificate certifyOrderPreservingReadExposure(
    scf::ForOp loop, SmallVectorImpl<Operation *> &loadSlice,
    SmallVectorImpl<Operation *> &computeSlice,
    SmallVectorImpl<Operation *> &stateSlice,
    bool requireVectorizableTensorLoad) {
  loadSlice.clear();
  computeSlice.clear();
  stateSlice.clear();
  auto trip = exactStaticTripCount(loop);
  if (trip && *trip < 2)
    return {false, "", "iteration domain has fewer than two iterations"};
  auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
  if (!yield || yield.getNumOperands() != loop.getInitArgs().size())
    return {false, "", "loop-carried result arity is not closed"};

  SmallVector<LoadOp> loads;
  bool hasAffinePointerInduction = false;
  bool rejected = false;
  std::string reason;
  loop.walk([&](Operation *operation) {
    if (rejected || operation == loop.getOperation())
      return;
    if (isa<scf::ForOp>(operation)) {
      rejected = true;
      reason = "nested scf.for requires a separately composed certificate";
      return;
    }
    if (auto load = dyn_cast<LoadOp>(operation)) {
      if (load->getBlock() != loop.getBody()) {
        rejected = true;
        reason = "nested-region load cannot be phase-exposed";
        return;
      }
      bool pointerDependsOnCarried =
          dependsOnAny(load.getPtr(), loop.getRegionIterArgs());
      if (load.getIsVolatile() ||
          (pointerDependsOnCarried &&
           !hasOnlyAffineReadOnlyPointerDependencies(load.getPtr(), loop)) ||
          (load.getMask() &&
           dependsOnAny(load.getMask(), loop.getRegionIterArgs())) ||
          (load.getOther() &&
           dependsOnAny(load.getOther(), loop.getRegionIterArgs()))) {
        rejected = true;
        reason = "read address, mask, or fill depends on loop-carried state";
        return;
      }
      hasAffinePointerInduction |= pointerDependsOnCarried;
      loads.push_back(load);
      return;
    }
    if (isa<StoreOp, AtomicRMWOp, AtomicCASOp>(operation)) {
      rejected = true;
      reason = "iteration contains a write or atomic effect";
      return;
    }
    if (!operation->hasTrait<OpTrait::IsTerminator>() &&
        !isMemoryEffectFree(operation)) {
      rejected = true;
      reason = "iteration contains an unknown non-read effect";
    }
  });
  if (rejected)
    return {false, "", reason};
  if (loads.empty())
    return {false, "", "iteration has no read service to expose"};

  bool hasVectorizableTensorLoad = false;
  for (LoadOp load : loads) {
    if (!isa<RankedTensorType>(load.getPtr().getType()))
      continue;
    if (!load.getBoundaryCheck().empty() || load.getPadding())
      continue;
    hasVectorizableTensorLoad = true;
  }
  if (requireVectorizableTensorLoad && !hasVectorizableTensorLoad)
    return {false, "", "iteration has no join-compatible tensor load"};

  llvm::SmallPtrSet<Operation *, 32> loadClosure;
  SmallVector<Operation *> worklist;
  for (LoadOp load : loads) {
    loadClosure.insert(load.getOperation());
    worklist.push_back(load.getOperation());
  }
  while (!worklist.empty()) {
    Operation *operation = worklist.pop_back_val();
    for (Value operand : operation->getOperands()) {
      Operation *definition = operand.getDefiningOp();
      if (!definition || definition->getBlock() != loop.getBody() ||
          loadClosure.contains(definition))
        continue;
      if (!isMemoryEffectFree(definition))
        return {false, "", "read dependency has an observable effect"};
      loadClosure.insert(definition);
      worklist.push_back(definition);
    }
  }

  auto operationDependsOnCarried = [&](Operation *root) {
    bool depends = llvm::any_of(root->getOperands(), [&](Value operand) {
      return dependsOnAny(operand, loop.getRegionIterArgs());
    });
    root->walk([&](Operation *nested) {
      if (nested != root)
        depends |= llvm::any_of(nested->getOperands(), [&](Value operand) {
          return dependsOnAny(operand, loop.getRegionIterArgs());
        });
    });
    return depends;
  };
  for (Operation &operation : loop.getBody()->without_terminator()) {
    if (loadClosure.contains(&operation))
      loadSlice.push_back(&operation);
    else if (operationDependsOnCarried(&operation))
      stateSlice.push_back(&operation);
    else
      computeSlice.push_back(&operation);
  }
  if (loadSlice.empty() || (computeSlice.empty() && stateSlice.empty()))
    return {false, "", "iteration does not expose a nontrivial read/work split"};

  return {
      true,
      hasAffinePointerInduction
          ? (requireVectorizableTensorLoad
                 ? "existing_affine_pointer_read_load_vectorization_v1"
                 : "existing_affine_pointer_read_exposure_v1")
          : (requireVectorizableTensorLoad
                 ? "existing_order_preserving_load_vectorization_v1"
                 : "existing_order_preserving_read_exposure_v1"),
      ""};
}

LoopDependenceCertificate certifyProviderClosedStaticNest(
    scf::ForOp root, SmallVectorImpl<scf::ForOp> &loops,
    SmallVectorImpl<scf::ForOp> &leaves) {
  auto independent =
      certifyIndependentIterationExactInnerReduction(root);
  if (independent.safe) {
    loops.push_back(root);
    leaves.push_back(root);
    return independent;
  }
  bool containsInnerReduction = false;
  root.walk([&](ReduceOp) { containsInnerReduction = true; });
  // A reduction-bearing iteration is governed by the stronger certificate
  // above.  Falling through to the legacy pure-static-nest recognizer both
  // loses the precise rejection cause and can never make its stores legal.
  if (containsInnerReduction)
    return independent;
  unsigned operationCount = 0;
  bool rejected = false;
  std::string reason;
  root.walk<WalkOrder::PreOrder>([&](Operation *op) {
    ++operationCount;
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      loops.push_back(loop);
      auto trip = exactStaticTripCount(loop);
      if (!trip || *trip < 2) {
        rejected = true;
        reason = "a nest dimension has no nontrivial exact static trip count";
        return;
      }
      auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
      if (!yield || yield.getNumOperands() != loop.getRegionIterArgs().size() ||
          yield.getNumOperands() == 0) {
        rejected = true;
        reason = "a nest dimension has no closed loop-carried sink map";
        return;
      }
      for (auto [result, carried] :
           llvm::zip(yield.getOperands(), loop.getRegionIterArgs())) {
        if (!valueDependsOn(result, carried)) {
          rejected = true;
          reason = "a yielded value is not derived from its loop-carried state";
          return;
        }
      }
      bool nested = false;
      for (Operation &bodyOp : loop.getBody()->without_terminator())
        nested |= isa<scf::ForOp>(bodyOp);
      if (!nested)
        leaves.push_back(loop);
      return;
    }
    if (auto load = dyn_cast<LoadOp>(op)) {
      if (load.getIsVolatile()) {
        rejected = true;
        reason = "volatile load cannot enter a phase-major static nest";
      }
      return;
    }
    if (isa<StoreOp>(op) || op->getName().getStringRef().starts_with("tt.atomic")) {
      rejected = true;
      reason = "static nest contains a write or atomic effect";
      return;
    }
    if (!isa<scf::YieldOp>(op) && !isMemoryEffectFree(op)) {
      rejected = true;
      reason = "static nest contains an unclosed side effect";
    }
  });
  // Semantic closure is independent of acquisition/code-growth policy.  The
  // predecision controller owns that policy and emits no candidate plan when
  // acquisition is deferred.  Here we only reject arithmetic overflow; a
  // fixed operation count must not redefine a legal mechanism domain.
  uint64_t expansion = operationCount;
  for (scf::ForOp loop : rejected ? ArrayRef<scf::ForOp>()
                                  : ArrayRef<scf::ForOp>(loops)) {
    auto trip = exactStaticTripCount(loop);
    if (!trip || expansion >
                     std::numeric_limits<uint64_t>::max() /
                         static_cast<uint64_t>(*trip)) {
      rejected = true;
      reason = "provider-closed nest expansion overflows its structural count";
      break;
    }
    expansion *= static_cast<uint64_t>(*trip);
  }
  if (rejected || loops.empty() || leaves.empty())
    return {false, "", reason.empty() ? "static nest is empty" : reason};

  for (scf::ForOp leaf : leaves) {
    SmallVector<Value> carried;
    for (scf::ForOp loop = leaf; loop; loop = loop->getParentOfType<scf::ForOp>())
      llvm::append_range(carried, loop.getRegionIterArgs());
    unsigned loads = 0;
    leaf.walk([&](LoadOp load) {
      ++loads;
      if (dependsOnAny(load.getPtr(), carried) ||
          (load.getMask() && dependsOnAny(load.getMask(), carried)) ||
          (load.getOther() && dependsOnAny(load.getOther(), carried))) {
        rejected = true;
        reason = "load address, mask, or fill depends on loop-carried state";
      }
    });
    if (loads == 0) {
      rejected = true;
      reason = "leaf dimension has no load service to expose";
    }
  }
  return rejected
             ? LoopDependenceCertificate{false, "", reason}
             : LoopDependenceCertificate{
                   true, "provider_closed_complete_static_nest_v1", ""};
}

LogicalResult tagProviderClosedLeaf(scf::ForOp leaf, StringRef roleSubject,
                                    Builder &builder) {
  llvm::SmallPtrSet<Operation *, 32> loadPhase;
  SmallVector<Operation *> worklist;
  for (Operation &op : leaf.getBody()->without_terminator())
    if (isa<LoadOp>(op))
      worklist.push_back(&op);
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!loadPhase.insert(op).second)
      continue;
    for (Value operand : op->getOperands()) {
      Operation *definition = operand.getDefiningOp();
      if (definition && definition->getBlock() == op->getBlock()) {
        if (!isMemoryEffectFree(definition) && !isa<LoadOp>(definition))
          return failure();
        worklist.push_back(definition);
      }
    }
  }
  bool sawReduce = false;
  auto roleSubjectAttr = builder.getStringAttr(roleSubject);
  for (Operation &op : leaf.getBody()->without_terminator()) {
    StringRef role;
    if (loadPhase.contains(&op)) {
      role = "load";
    } else if (isa<StoreOp>(op)) {
      role = "reduce";
      sawReduce = true;
    } else {
      if (!isMemoryEffectFree(&op))
        return failure();
      bool carried = llvm::any_of(op.getResults(), [&](Value result) {
        return dependsOnAny(result, leaf.getRegionIterArgs());
      });
      role = carried ? "reduce" : "compute";
      sawReduce |= carried;
    }
    op.setAttr(kRoleAttr, builder.getStringAttr(role));
    op.setAttr(kRoleSubjectAttr, roleSubjectAttr);
  }
  // A load can feed the loop-carried reduction directly.  That is a closed
  // two-phase load/reduce DAG, not a missing compute template.
  return success(!loadPhase.empty() && sawReduce);
}

bool exactKeys(const llvm::json::Object &object, ArrayRef<StringRef> keys) {
  if (object.size() != keys.size())
    return false;
  return llvm::all_of(keys,
                      [&](StringRef key) { return object.get(key) != nullptr; });
}

bool validateCommonContract(const llvm::json::Object &contract,
                            StringRef route, StringRef subjectRef,
                            int64_t adapterVersion,
                            StringRef subjectGuard,
                            StringRef lineageGuard = "l.unroll.lineage") {
  const auto *locator = contract.getObject("subject_locator");
  const auto *fallback = contract.getObject("fallback_binding");
  const auto *provenance = contract.getObject("minimal_provenance");
  const auto *guards = contract.getArray("dynamic_guard_assumptions");
  const auto *feedback = contract.getArray("requested_feedback_fields");
  if (!locator ||
      !exactKeys(*locator, {"anchor_or_marker_ref", "kind", "subject_ref"}) ||
      locator->getString("kind") != "hbv_typed_marker" ||
      locator->getString("anchor_or_marker_ref") != kSubjectAttr ||
      locator->getString("subject_ref") != subjectRef || !fallback ||
      !exactKeys(*fallback, {"max_original_route_retries",
                             "original_route_ref", "retry_loop_guard"}) ||
      fallback->getString("original_route_ref") != kOriginalRoute ||
      fallback->getInteger("max_original_route_retries") != 1 ||
      fallback->getString("retry_loop_guard") != "hbv_disable_decision" ||
      !provenance ||
      !exactKeys(*provenance, {"adapter_version", "compiler_commit",
                               "producer_schema", "source_ref"}) ||
      provenance->getInteger("adapter_version") != adapterVersion ||
      provenance->getString("compiler_commit") != kPinnedCompilerCommit ||
      provenance->getString("producer_schema") != "hbv.plan_contract.v1" ||
      !provenance->getString("source_ref") || !guards || guards->size() != 11 ||
      !feedback || feedback->size() != 4)
    return false;

  SmallVector<StringRef> expectedGuards = {
      "l.bundle.schema",          "l.target.binding",
      subjectGuard,
      "l.subject.structure",
      "l.effects_dependencies",  "l.parameters.closed",
      "l.route.mutual_exclusion", lineageGuard,
      "l.route.postcondition",    "l.ir.verify",
      "l.observation.correspondence"};
  for (auto [index, value] : llvm::enumerate(*guards)) {
    const auto *guard = value.getAsObject();
    if (!guard ||
        !exactKeys(*guard, {"guard_id", "required", "responsible_stage",
                            "verifier_or_legality_binding"}) ||
        guard->getString("guard_id") != expectedGuards[index] ||
        guard->getBoolean("required") != true ||
        !guard->getString("responsible_stage") ||
        !guard->getString("verifier_or_legality_binding"))
      return false;
  }
  SmallVector<StringRef> expectedFeedback = {
      "loop.route.realized", "loop.route.parameters",
      "loop.route.postcondition", "codegen.instruction_family_counts"};
  for (auto [index, value] : llvm::enumerate(*feedback)) {
    const auto *field = value.getAsObject();
    if (!field ||
        !exactKeys(*field, {"availability_stage", "evidence_sink", "field_id",
                            "required", "source_kind"}) ||
        field->getString("field_id") != expectedFeedback[index] ||
        field->getBoolean("required") != true ||
        !field->getString("availability_stage") ||
        !field->getString("evidence_sink") || !field->getString("source_kind"))
      return false;
  }
  return route == kPipelineRoute || route == kPhaseRoute ||
         route == kLogicalRoute || route == kExactPrefixRoute;
}

FailureOr<ParsedLoopPlan> parseLoopPlan(StringRef payload,
                                       std::string &reason) {
  auto parsed = llvm::json::parse(payload);
  if (!parsed) {
    reason = "Loop PlanBundle is not valid JSON";
    llvm::consumeError(parsed.takeError());
    return failure();
  }
  const auto *bundle = parsed->getAsObject();
  if (!bundle || !exactKeys(*bundle, {"bundle_id", "contract", "schema_version"}) ||
      bundle->getInteger("schema_version") != 1 || !bundle->getString("bundle_id")) {
    reason = "Loop PlanBundle envelope is malformed or unknown";
    return failure();
  }
  const auto *contract = bundle->getObject("contract");
  if (!contract || contract->getInteger("schema_version") != 1 ||
      contract->getString("project_kind") != "loop") {
    reason = "Loop PlanContract envelope is malformed or cross-project";
    return failure();
  }
  auto route = contract->getString("route_ref");
  auto decision = contract->getString("decision_ref");
  if (!route || !decision) {
    reason = "Loop PlanContract route or decision is missing";
    return failure();
  }
  ParsedLoopPlan result;
  result.route = route->str();
  result.mechanismRoute = route->str();
  result.routeSubtype = "base";
  result.artifactRoute = route->str();
  result.decisionRef = decision->str();
  if (*route == kOriginalRoute) {
    if (!exactKeys(*contract, {"candidate_parameters", "decision_ref",
                               "dynamic_guard_assumptions",
                               "dynamic_materialization_bindings",
                               "fallback_binding", "minimal_provenance",
                               "project_kind", "requested_feedback_fields",
                               "route_ref", "schema_version"})) {
      reason = "selected-default Loop contract has unknown semantics";
      return failure();
    }
    const auto *parameters = contract->getObject("candidate_parameters");
    const auto *bindings = contract->getArray("dynamic_materialization_bindings");
    const auto *guards = contract->getArray("dynamic_guard_assumptions");
    const auto *feedback = contract->getArray("requested_feedback_fields");
    if (!parameters || !exactKeys(*parameters, {"kind"}) ||
        parameters->getString("kind") != "default" || !bindings ||
        !bindings->empty() || !guards || !guards->empty() || !feedback ||
        !feedback->empty()) {
      reason = "selected-default Loop contract is not empty/default";
      return failure();
    }
    return result;
  }
  if (!exactKeys(*contract, {"candidate_parameters", "decision_ref",
                             "dynamic_guard_assumptions",
                             "dynamic_materialization_bindings",
                             "fallback_binding", "minimal_provenance",
                             "project_kind", "requested_feedback_fields",
                             "route_ref", "schema_version", "subject_locator"})) {
    reason = "selected Loop contract has unknown semantics";
    return failure();
  }
  const auto *parameters = contract->getObject("candidate_parameters");
  const auto *bindings = contract->getArray("dynamic_materialization_bindings");
  if (!parameters || parameters->getString("kind") != *route) {
    reason = "Loop route and parameter kind are not closed";
    return failure();
  }
  auto subject = parameters->getString("subject_ref");
  auto adapterVersion = parameters->getInteger("adapter_version");
  auto bridgeFactor = parameters->getInteger("bridge_factor");
  auto routeFactor = parameters->getInteger("route_factor");
  auto compositionSchema = parameters->getString("composition_schema");
  bool compositionV2 =
      (*route == kPipelineRoute && adapterVersion == 7) ||
      (*route == kPhaseRoute && adapterVersion == 9) ||
      (*route == kLogicalRoute && adapterVersion == 7);
  bool compositionV3 =
      *route == kPhaseRoute && adapterVersion == 10 &&
      compositionSchema &&
      *compositionSchema == "hbv.loop.bridge-route-composition.v3";
  bool bridgeAxisVector =
      compositionV3 && parameters->getString("subject_policy") ==
                           "provider_closed_bridge_axis_vector_program_region";
  bool composedIntervention =
      (*route == kPipelineRoute && adapterVersion == 6) ||
      (*route == kPhaseRoute && adapterVersion == 8) ||
      (*route == kLogicalRoute && adapterVersion == 6) || compositionV2 ||
      compositionV3;
  bool runtimeGuardedLogical =
      *route == kLogicalRoute && adapterVersion == 2;
  bool bridgeStaticPartition =
      *route == kPhaseRoute &&
      (adapterVersion == 3 || adapterVersion == 8 || adapterVersion == 9) &&
      parameters->getString("subject_policy") ==
          "static_consecutive_virtual_program_partition_recurrence";
  bool legacyBridgeConstructed =
      ((*route == kPipelineRoute || *route == kPhaseRoute) &&
       adapterVersion == 2) ||
      (*route == kLogicalRoute &&
       (adapterVersion == 3 || adapterVersion == 4 || adapterVersion == 5)) ||
      bridgeStaticPartition;
  bool bridgeConstructed =
      legacyBridgeConstructed ||
      (composedIntervention && bridgeFactor && *bridgeFactor > 1);
  bool multiSubject =
      (*route == kPhaseRoute || *route == kLogicalRoute) &&
      parameters->getString("subject_policy") ==
          "all_independent_eligible_existing_scf_for";
  bool nestedSubject =
      (*route == kPhaseRoute || *route == kLogicalRoute) &&
      parameters->getString("subject_policy") ==
          "provider_closed_nested_inner_dimension_scf_for";
  bool providerBoundSubjectSet =
      (*route == kPhaseRoute || *route == kLogicalRoute) &&
      *adapterVersion == 11 &&
      (parameters->getString("subject_policy") ==
           "provider_bound_independent_existing_scf_for_set" ||
       parameters->getString("subject_policy") ==
           "provider_bound_nested_inner_dimension_scf_for");
  bool providerClosedStatic =
      *route == kPhaseRoute && *adapterVersion == 4 &&
      parameters->getString("subject_policy") ==
          "provider_closed_complete_static_scf_for_nest";
  bool exactPrefixReduction =
      *route == kExactPrefixRoute &&
      (*adapterVersion == 5 || *adapterVersion == 6) &&
      parameters->getString("subject_policy") ==
          "provider_closed_dynamic_exact_integer_prefix";
  bool stateAxisLogical =
      *route == kLogicalRoute && *adapterVersion == 9 &&
      parameters->getString("subject_policy") ==
          "provider_closed_sibling_state_operation_graph";
  bool affineRuntimePartial =
      *route == kPhaseRoute &&
      (*adapterVersion == 6 || *adapterVersion == 7) &&
      parameters->getString("subject_policy") ==
          "provider_closed_affine_runtime_grid_stride_order_preserving";
  bool historicalOnlyAdapter =
      (*route == kPipelineRoute &&
       (*adapterVersion == 2 || *adapterVersion == 6)) ||
      (*route == kPhaseRoute &&
       (*adapterVersion == 1 || *adapterVersion == 2 ||
        *adapterVersion == 3 || *adapterVersion == 8)) ||
      (*route == kLogicalRoute &&
       (*adapterVersion == 1 || *adapterVersion == 2 ||
        *adapterVersion == 3 || *adapterVersion == 4 ||
        *adapterVersion == 5 || *adapterVersion == 6));
  if (historicalOnlyAdapter) {
    reason =
        "historical Loop adapter is evidence-readable but not production-executable";
    return failure();
  }
  StringRef expectedSubjectPolicy =
      bridgeStaticPartition
          ? "static_consecutive_virtual_program_partition_recurrence"
      : bridgeAxisVector
          ? "provider_closed_bridge_axis_vector_program_region"
      : bridgeConstructed ? "unique_bridge_constructed_program_loop"
      : runtimeGuardedLogical ? "unique_eligible_runtime_topk_scf_for"
      : providerClosedStatic ? "provider_closed_complete_static_scf_for_nest"
      : exactPrefixReduction
          ? "provider_closed_dynamic_exact_integer_prefix"
      : stateAxisLogical
          ? "provider_closed_sibling_state_operation_graph"
      : affineRuntimePartial
          ? "provider_closed_affine_runtime_grid_stride_order_preserving"
      : providerBoundSubjectSet
          ? parameters->getString("subject_policy").value_or("")
      : multiSubject ? "all_independent_eligible_existing_scf_for"
      : nestedSubject ? "provider_closed_nested_inner_dimension_scf_for"
                              : "unique_eligible_existing_scf_for";
  if (!subject || !adapterVersion) {
    reason = "selected Loop subject or adapter version is absent";
    return failure();
  }
  if (parameters->getString("subject_policy") != expectedSubjectPolicy) {
    reason = "selected Loop subject policy does not match its route origin";
    return failure();
  }
  if (!runtimeGuardedLogical && !bridgeConstructed &&
      !providerClosedStatic && !exactPrefixReduction &&
      !stateAxisLogical && !affineRuntimePartial && !composedIntervention &&
      !providerBoundSubjectSet &&
      *adapterVersion != 1) {
    reason = "selected legacy Loop adapter version is unsupported";
    return failure();
  }
  if (composedIntervention &&
      (!bridgeFactor || *bridgeFactor < 1 ||
      !llvm::isPowerOf2_64(*bridgeFactor) ||
       !routeFactor || *routeFactor < 1 ||
       !compositionSchema ||
       *compositionSchema !=
           (compositionV3 ? "hbv.loop.bridge-route-composition.v3"
            : compositionV2 ? "hbv.loop.bridge-route-composition.v2"
                          : "hbv.loop.bridge-route-composition.v1"))) {
    reason = "selected Loop composition factors or schema are invalid";
    return failure();
  }
  if (compositionV2 || compositionV3) {
    auto routeSubjectRef = parameters->getString("route_subject_ref");
    auto exactTrip = parameters->getInteger(
        "route_subject_exact_trip_count");
    auto runtimeCertificate = parameters->getString(
        "runtime_main_tail_certificate_ref");
    auto admissionRef = parameters->getString("factor_admission_ref");
    bool runtimeSubject = runtimeCertificate && !runtimeCertificate->empty();
    bool exactSubject = exactTrip && *exactTrip > 0;
    bool fullRoute = *route == kPhaseRoute || *route == kLogicalRoute;
    if (!routeSubjectRef || routeSubjectRef->empty() || !admissionRef ||
        admissionRef->empty() || exactSubject == runtimeSubject ||
        (bridgeFactor && *bridgeFactor > 1 &&
         (!exactSubject || *exactTrip != *bridgeFactor)) ||
        (fullRoute && (*routeFactor < 2 ||
                       (exactSubject && *routeFactor > *exactTrip)))) {
      reason = "selected Loop V2 subject/factor admission is invalid";
      return failure();
    }
  }
  SmallVector<int64_t, 3> bridgeAxisDivisors;
  if (compositionV3) {
    const auto *divisors = parameters->getArray("bridge_axis_divisors");
    int64_t product = 1;
    unsigned groupedAxes = 0;
    if (!divisors || divisors->size() != 3) {
      reason = "selected Loop axis-vector divisors are absent or malformed";
      return failure();
    }
    for (const llvm::json::Value &value : *divisors) {
      auto divisor = value.getAsInteger();
      if (!divisor || *divisor < 1 || !llvm::isPowerOf2_64(*divisor) ||
          product > std::numeric_limits<int64_t>::max() / *divisor) {
        reason = "selected Loop axis-vector divisor is invalid";
        return failure();
      }
      groupedAxes += *divisor > 1;
      product *= *divisor;
      bridgeAxisDivisors.push_back(*divisor);
    }
    if (groupedAxes < 2 || !bridgeFactor || product != *bridgeFactor ||
        !routeFactor || *routeFactor != product) {
      reason =
          "selected Loop axis-vector cardinality or phase factor is not closed";
      return failure();
    }
  }
  SmallVector<ParsedLoopPlan::ProviderBoundMember> providerBoundMembers;
  std::string providerBoundMemberSignature;
  if (providerBoundSubjectSet) {
    const auto *members = parameters->getArray("members");
    auto providerRef = parameters->getString("provider_ref");
    auto providerSchema = parameters->getString("provider_schema");
    auto policy = parameters->getString("subject_policy");
    bool nested = policy ==
        "provider_bound_nested_inner_dimension_scf_for";
    if (!members || members->empty() || !providerRef ||
        providerRef->empty() ||
        providerSchema != "hbv.loop-provider.bound-subject-set.v1" ||
        (nested && members->size() != 1) ||
        (!nested && members->size() < 2)) {
      reason = "provider-bound Loop subject set envelope is malformed";
      return failure();
    }
    int64_t previousOrdinal = -1;
    std::set<std::string> locators;
    std::set<std::string> memberRefs;
    for (const llvm::json::Value &value : *members) {
      const auto *member = value.getAsObject();
      if (!member ||
          !exactKeys(
              *member,
              {"exact_static_trip_count", "factor_admission_ref",
               "member_ref", "nested_context_certificate_ref",
               "nesting_depth", "parent_loop_locator",
               "provider_loop_locator",
               "route_capability_certificate_ref", "route_factor",
               "route_factor_kind", "runtime_main_tail_certificate_ref",
               "schema"})) {
        reason = "provider-bound Loop member schema is malformed";
        return failure();
      }
      auto locator = member->getString("provider_loop_locator");
      auto memberRef = member->getString("member_ref");
      auto capability =
          member->getString("route_capability_certificate_ref");
      auto runtime =
          member->getString("runtime_main_tail_certificate_ref");
      auto parent = member->getString("parent_loop_locator");
      auto nestedCertificate =
          member->getString("nested_context_certificate_ref");
      auto factorAdmission = member->getString("factor_admission_ref");
      auto factorKind = member->getString("route_factor_kind");
      auto nestingDepth = member->getInteger("nesting_depth");
      auto exactTrip = member->getInteger("exact_static_trip_count");
      auto factor = member->getInteger("route_factor");
      StringRef locatorSuffix = locator.value_or("");
      int64_t locatorOrdinal = -1;
      bool locatorValid =
          locatorSuffix.consume_front("planning-cut.loop.") &&
          !locatorSuffix.empty() &&
          !locatorSuffix.getAsInteger(10, locatorOrdinal) &&
          locatorOrdinal > previousOrdinal;
      bool exact = exactTrip && *exactTrip > 0;
      bool runtimeSubject = runtime && !runtime->empty();
      bool memberNested = parent && !parent->empty() &&
                          nestedCertificate &&
                          !nestedCertificate->empty();
      if (!locatorValid || !memberRef || memberRef->empty() ||
          !capability || capability->empty() || !factorAdmission ||
          factorAdmission->empty() || !runtime || !parent ||
          !nestedCertificate || !factor || *factor < 2 ||
          !llvm::isPowerOf2_64(*factor) || !nestingDepth ||
          *nestingDepth < 0 || exact == runtimeSubject ||
          (exact && *factor > *exactTrip) ||
          factorKind != (*route == kPhaseRoute
                              ? "phase_reorder_grouping_width"
                              : "logical_vector_grouping_width") ||
          member->getString("schema") !=
              "hbv.loop.provider-bound-route-member.v1" ||
          !locators.insert(locator->str()).second ||
          !memberRefs.insert(memberRef->str()).second ||
          (nested &&
           (!memberNested || *nestingDepth < 1 ||
            nestedCertificate !=
                "nested_inner_dimension_independent_sink_v1")) ||
          (!nested &&
           (*nestingDepth != 0 || memberNested ||
            (parent && !parent->empty()) ||
            (nestedCertificate && !nestedCertificate->empty())))) {
        reason = "provider-bound Loop member semantics are not closed";
        return failure();
      }
      ParsedLoopPlan::ProviderBoundMember parsedMember;
      parsedMember.providerLoopLocator = locator->str();
      parsedMember.memberRef = memberRef->str();
      parsedMember.routeCapabilityCertificateRef = capability->str();
      parsedMember.runtimeMainTailCertificateRef = runtime->str();
      parsedMember.parentLoopLocator = parent->str();
      parsedMember.nestedContextCertificateRef = nestedCertificate->str();
      parsedMember.nestingDepth = *nestingDepth;
      parsedMember.exactStaticTripCount = exact ? *exactTrip : 0;
      parsedMember.routeFactor = *factor;
      if (!providerBoundMemberSignature.empty())
        providerBoundMemberSignature += ";";
      providerBoundMemberSignature += parsedMember.providerLoopLocator +
                                      "=" +
                                      std::to_string(*factor);
      providerBoundMembers.push_back(std::move(parsedMember));
      previousOrdinal = locatorOrdinal;
    }
  }
  if (bridgeConstructed &&
      (!bridgeFactor || *bridgeFactor < 2 ||
       !llvm::isPowerOf2_64(*bridgeFactor))) {
    reason = "selected Loop Bridge factor is invalid";
    return failure();
  }
  if (!validateCommonContract(
          *contract, *route, *subject, *adapterVersion,
          providerClosedStatic
              ? "l.subject.provider_closed_static_nest"
          : exactPrefixReduction
              ? "l.subject.exact_prefix_reduction"
          : stateAxisLogical
              ? "l.subject.state_axis_sibling_group"
          : affineRuntimePartial
              ? "l.subject.affine_runtime_grid_stride"
          : providerBoundSubjectSet
              ? (parameters->getString("subject_policy") ==
                         "provider_bound_nested_inner_dimension_scf_for"
                     ? "l.subject.provider_bound_nested_inner"
                     : "l.subject.provider_bound_independent_set")
          : nestedSubject
              ? "l.subject.provider_closed_nested_dimension"
          : multiSubject ? "l.subject.independent_fresh_set"
                         : "l.subject.unique_fresh",
          stateAxisLogical ? "l.state_axis.lineage"
                           : "l.unroll.lineage")) {
    reason = "selected Loop guards, feedback, fallback, or provenance are invalid";
    return failure();
  }
  bool routeClosed = false;
  if (*route == kPipelineRoute) {
    if (composedIntervention) {
      bool parameterKeysClosed = compositionV2
          ? exactKeys(*parameters,
                      {"adapter_version", "bridge_factor",
                       "composition_schema", "factor_admission_ref", "kind",
                       "route_factor", "route_factor_kind",
                       "route_subject_exact_trip_count", "route_subject_ref",
                       "runtime_main_tail_certificate_ref", "subject_policy",
                       "subject_ref", "target_arch"})
          : exactKeys(*parameters,
                      {"adapter_version", "bridge_factor",
                       "composition_schema", "kind", "route_factor",
                       "route_factor_kind", "subject_policy", "subject_ref",
                       "target_arch"});
      routeClosed =
          parameterKeysClosed &&
          parameters->getString("route_factor_kind") ==
              "pipeline_stage_count" &&
          routeFactor && *routeFactor > 0 &&
          parameters->getInteger("target_arch") > 0 && bindings &&
          bindings->size() == 2;
    } else {
      routeClosed = exactKeys(
                      *parameters,
                      bridgeConstructed
                          ? ArrayRef<StringRef>({"adapter_version", "bridge_factor", "kind",
                                                "stage_count", "subject_policy", "subject_ref",
                                                "target_arch"})
                          : ArrayRef<StringRef>({"adapter_version", "kind", "stage_count",
                                                "subject_policy", "subject_ref", "target_arch"})) &&
                  parameters->getInteger("stage_count") > 0 &&
                  parameters->getInteger("target_arch") > 0 && bindings &&
                  bindings->size() == 1;
    }
  } else if (*route == kPhaseRoute) {
    if (providerBoundSubjectSet) {
      routeClosed =
          exactKeys(*parameters,
                    {"adapter_version", "kind", "members", "provider_ref",
                     "provider_schema", "subject_policy", "subject_ref"}) &&
          bindings &&
          bindings->size() == providerBoundMembers.size() * 2;
    } else if (composedIntervention) {
      bool staticComposition = bridgeStaticPartition;
      bool parameterKeysClosed = compositionV3
          ? exactKeys(
                *parameters,
                {"adapter_version", "bridge_axis_divisors",
                 "bridge_factor", "composition_schema",
                 "factor_admission_ref", "kind", "phase_order",
                 "reduction_order", "route_factor", "route_factor_kind",
                 "route_subject_exact_trip_count", "route_subject_ref",
                 "runtime_main_tail_certificate_ref", "subject_policy",
                 "subject_ref", "tail_policy"})
          : compositionV2
          ? exactKeys(
                *parameters,
                staticComposition
                    ? ArrayRef<StringRef>({
                          "adapter_version", "bridge_factor",
                          "composition_schema", "factor_admission_ref", "kind",
                          "partition_policy", "phase_order", "reduction_order",
                          "route_factor", "route_factor_kind",
                          "route_subject_exact_trip_count", "route_subject_ref",
                          "runtime_main_tail_certificate_ref", "subject_policy",
                          "subject_ref", "tail_policy"})
                    : ArrayRef<StringRef>({
                          "adapter_version", "bridge_factor",
                          "composition_schema", "factor_admission_ref", "kind",
                          "phase_order", "reduction_order", "route_factor",
                          "route_factor_kind", "route_subject_exact_trip_count",
                          "route_subject_ref", "runtime_main_tail_certificate_ref",
                          "subject_policy", "subject_ref", "tail_policy"}))
          : exactKeys(
                *parameters,
                staticComposition
                    ? ArrayRef<StringRef>({
                          "adapter_version", "bridge_factor",
                          "composition_schema", "kind", "partition_policy",
                          "phase_order", "reduction_order", "route_factor",
                          "route_factor_kind", "subject_policy", "subject_ref",
                          "tail_policy"})
                    : ArrayRef<StringRef>({
                          "adapter_version", "bridge_factor",
                          "composition_schema", "kind", "phase_order",
                          "reduction_order", "route_factor",
                          "route_factor_kind", "subject_policy", "subject_ref",
                          "tail_policy"}));
      routeClosed =
          parameterKeysClosed &&
          parameters->getString("route_factor_kind") ==
              ((compositionV2 || compositionV3)
                   ? "phase_reorder_grouping_width"
                             : "loop_unroll_factor") &&
          routeFactor &&
          ((compositionV2 || compositionV3) ? *routeFactor >= 2
                         : ((bridgeConstructed &&
                             *routeFactor == *bridgeFactor) ||
                            (!bridgeConstructed && *routeFactor >= 2))) &&
          parameters->getString("phase_order") == "load_compute_store" &&
          parameters->getString("reduction_order") ==
              "operator_preserving" &&
          parameters->getString("tail_policy") ==
              (bridgeConstructed ? "exact_grid_divisible"
                                 : "native_unroll_main_ordered_remainder") &&
          (!staticComposition ||
           parameters->getString("partition_policy") ==
               "consecutive_pid_signed_quotient_remainder_recurrence_v1") &&
          bindings && bindings->size() == 3;
    } else if (affineRuntimePartial) {
      auto factor = parameters->getInteger("unroll_factor");
      bool explicitPolicy = *adapterVersion == 7;
      auto policy = parameters->getString("materialization_policy");
      routeClosed = exactKeys(
                        *parameters,
                        explicitPolicy
                            ? ArrayRef<StringRef>({
                                  "adapter_version", "kind", "phase_order",
                                  "provider_schema", "reduction_order",
                                  "subject_policy", "subject_ref",
                                  "tail_policy", "unroll_factor",
                                  "materialization_policy"})
                            : ArrayRef<StringRef>({
                                  "adapter_version", "kind", "phase_order",
                                  "provider_schema", "reduction_order",
                                  "subject_policy", "subject_ref",
                                  "tail_policy", "unroll_factor"})) &&
                    factor && *factor >= 2 &&
                    parameters->getString("provider_schema") ==
                        "hbv.loop-provider.ordered-owned-iteration.v3" &&
                    parameters->getString("phase_order") ==
                        (explicitPolicy
                             ? "explicit_main_tail_or_guarded_lanes"
                             : "provider_selected_main_tail_or_guarded_lanes") &&
                    parameters->getString("reduction_order") ==
                        "operator_preserving" &&
                    parameters->getString("tail_policy") ==
                        "quotient_remainder_main_tail_or_guarded_lanes" &&
                    (!explicitPolicy ||
                     (policy && (*policy == "provider_selected" ||
                                 *policy == "guarded_lanes" ||
                                 *policy == "main_tail"))) &&
                    bindings && bindings->size() == 1;
    } else if (providerClosedStatic) {
      const auto *unrollFactors = parameters->getArray("unroll_factors");
      routeClosed = exactKeys(*parameters, {"adapter_version", "kind",
                                            "phase_order", "provider_schema",
                                            "reduction_order", "subject_policy",
                                            "subject_ref", "tail_policy",
                                            "unroll_factors"}) &&
                    unrollFactors && !unrollFactors->empty() &&
                    llvm::all_of(*unrollFactors, [](const llvm::json::Value &value) {
                      auto factor = value.getAsInteger();
                      return factor && *factor >= 2;
                    }) &&
                    parameters->getString("provider_schema") ==
                        "hbv.loop-provider.static-scf.v2" &&
                    parameters->getString("phase_order") ==
                        "load_compute_reduce" &&
                    parameters->getString("reduction_order") ==
                        "dependency_preserving" &&
                    parameters->getString("tail_policy") ==
                        "exact_static_domain" &&
                    bindings && bindings->size() == 2;
    } else if (bridgeStaticPartition) {
      routeClosed =
          exactKeys(*parameters,
                    {"adapter_version", "bridge_factor", "kind",
                     "partition_policy", "phase_order", "reduction_order",
                     "subject_policy", "subject_ref", "tail_policy",
                     "unroll_factor"}) &&
          parameters->getInteger("unroll_factor") == bridgeFactor &&
          parameters->getString("partition_policy") ==
              "consecutive_pid_signed_quotient_remainder_recurrence_v1" &&
          parameters->getString("phase_order") == "load_compute_store" &&
          parameters->getString("reduction_order") ==
              "operator_preserving" &&
          parameters->getString("tail_policy") == "exact_grid_divisible" &&
          bindings && bindings->size() == 2;
    } else if (bridgeConstructed) {
      routeClosed = exactKeys(*parameters, {"adapter_version", "bridge_factor", "kind",
                                            "phase_order", "reduction_order", "subject_policy",
                                            "subject_ref", "tail_policy", "unroll_factor"}) &&
                    parameters->getInteger("unroll_factor") == bridgeFactor &&
                    parameters->getString("phase_order") == "load_compute_store" &&
                    parameters->getString("reduction_order") == "operator_preserving" &&
                    parameters->getString("tail_policy") == "exact_grid_divisible" &&
                    bindings && bindings->size() == 2;
    } else {
      routeClosed = exactKeys(*parameters, {"adapter_version", "kind", "phase_order",
                                            "reduction_order", "subject_policy",
                                            "subject_ref", "tail_policy",
                                            "unroll_factor"}) &&
                    parameters->getInteger("unroll_factor") == 4 &&
                    parameters->getString("phase_order") == "load_compute_reduce" &&
                    parameters->getString("reduction_order") == "dependency_preserving" &&
                    parameters->getString("tail_policy") == "exact_static_trip" &&
                    bindings && bindings->size() == 2;
    }
  } else if (*route == kLogicalRoute) {
    const auto *shape = parameters->getArray("group_shape");
    if (stateAxisLogical) {
      const auto *groups = parameters->getArray("sibling_groups");
      const auto *binding =
          bindings && bindings->size() == 1
              ? (*bindings)[0].getAsObject()
              : nullptr;
      routeClosed =
          exactKeys(*parameters,
                    {"adapter_version", "backend_adapter_ref",
                     "equivalence_policy", "graph_schema", "kind",
                     "materialization_policy", "provider_certificate",
                     "provider_ref", "provider_schema", "route_subtype",
                     "sibling_groups", "subject_policy", "subject_ref"}) &&
          groups && !groups->empty() &&
          parameters->getString("provider_schema") ==
              "hbv.loop-provider.state-axis-logical.v3" &&
          parameters->getString("route_subtype") ==
              "sibling_state_axis_slp_vectorization" &&
          parameters->getString("graph_schema") ==
              "hbv.loop.state-axis-sibling-graph.v1" &&
          parameters->getString("backend_adapter_ref") ==
              "hbv.loop.state-axis.backend.ttir.v1" &&
          parameters->getString("equivalence_policy") ==
              "provider_exact_graph_order_types_attributes_and_effects" &&
          parameters->getString("materialization_policy") ==
              "state_axis_registered_graph_pack_execute_unpack_v2" &&
          parameters->getString("provider_certificate") ==
              "provider_closed_sibling_state_operation_graph_v1" &&
          parameters->getString("provider_ref") &&
          !parameters->getString("provider_ref")->empty() && binding &&
          exactKeys(*binding,
                    {"binding_schema_version", "native_key_or_adapter_id",
                     "native_owner_or_binding_kind", "required",
                     "semantic_role", "typed_value_or_typed_reference"}) &&
          binding->getString("semantic_role") ==
              "loop_state_axis_logical_vectorization" &&
          binding->getString("native_owner_or_binding_kind") ==
              "triton_hbv_loop_adapter" &&
          binding->getString("native_key_or_adapter_id") ==
              "hbv.loop.state_axis_logical.v2" &&
          binding->getString("typed_value_or_typed_reference") == subject &&
          binding->getBoolean("required") == true &&
          binding->getInteger("binding_schema_version") == 1;
      if (routeClosed) {
        for (auto [ordinal, value] : llvm::enumerate(*groups)) {
          const auto *group = value.getAsObject();
          if (!group ||
              !exactKeys(*group,
                         {"element_bytes", "graph_signature",
                          "lane_operand_index", "operation_capabilities",
                          "ordinal", "packable_node_count",
                          "padded_state_cardinality", "state_cardinality",
                          "state_width"})) {
            routeClosed = false;
            break;
          }
          auto parsedOrdinal = group->getInteger("ordinal");
          auto cardinality = group->getInteger("state_cardinality");
          auto padded = group->getInteger("padded_state_cardinality");
          auto width = group->getInteger("state_width");
          auto elementBytes = group->getInteger("element_bytes");
          auto laneOperand = group->getInteger("lane_operand_index");
          auto packableNodes = group->getInteger("packable_node_count");
          auto signature = group->getString("graph_signature");
          const auto *capabilities =
              group->getArray("operation_capabilities");
          ParsedLoopPlan::StateAxisGroup parsedGroup;
          bool valid =
              parsedOrdinal && *parsedOrdinal ==
                                   static_cast<int64_t>(ordinal) &&
              cardinality && *cardinality >= 2 && padded &&
              *padded >= *cardinality && llvm::isPowerOf2_64(*padded) &&
              width && *width >= 1 && elementBytes && *elementBytes >= 1 &&
              laneOperand && (*laneOperand == 0 || *laneOperand == 1) &&
              packableNodes && *packableNodes >= 1 && signature &&
              !signature->empty() && capabilities && !capabilities->empty();
          std::string previous;
          if (valid)
            for (const llvm::json::Value &capability : *capabilities) {
              auto name = capability.getAsString();
              if (!name || name->empty() ||
                  (!previous.empty() &&
                   StringRef(previous).compare(*name) >= 0)) {
                valid = false;
                break;
              }
              previous = name->str();
              parsedGroup.operationCapabilities.push_back(previous);
            }
          if (!valid) {
            routeClosed = false;
            break;
          }
          parsedGroup.ordinal = *parsedOrdinal;
          parsedGroup.stateCardinality = *cardinality;
          parsedGroup.paddedStateCardinality = *padded;
          parsedGroup.stateWidth = *width;
          parsedGroup.elementBytes = *elementBytes;
          parsedGroup.laneOperandIndex = *laneOperand;
          parsedGroup.packableNodeCount = *packableNodes;
          parsedGroup.graphSignature = signature->str();
          result.stateAxisGroups.push_back(std::move(parsedGroup));
        }
        result.stateAxisGroupCount = result.stateAxisGroups.size();
      }
    } else if (providerBoundSubjectSet) {
      routeClosed =
          exactKeys(*parameters,
                    {"adapter_version", "kind", "members", "provider_ref",
                     "provider_schema", "subject_policy", "subject_ref"}) &&
          bindings &&
          bindings->size() == providerBoundMembers.size() * 2;
    } else if (composedIntervention) {
      bool parameterKeysClosed = compositionV2
          ? exactKeys(*parameters,
                      {"adapter_version", "bridge_factor", "combiner",
                       "composition_schema", "exact_split_elision",
                       "factor_admission_ref", "invariant_hoisting",
                       "iteration_axis", "kind", "mask_tail_policy",
                       "route_factor", "route_factor_kind",
                       "route_subject_exact_trip_count", "route_subject_ref",
                       "runtime_main_tail_certificate_ref", "subject_policy",
                       "subject_ref", "tensor_lane_fusion"})
          : exactKeys(*parameters,
                      {"adapter_version", "bridge_factor", "combiner",
                       "composition_schema", "exact_split_elision",
                       "invariant_hoisting", "iteration_axis", "kind",
                       "mask_tail_policy", "route_factor",
                       "route_factor_kind", "subject_policy", "subject_ref",
                       "tensor_lane_fusion"});
      routeClosed =
          parameterKeysClosed &&
          parameters->getString("route_factor_kind") ==
              (compositionV2 ? "logical_vector_grouping_width"
                             : "logical_group_unroll_factor") &&
          routeFactor &&
          (compositionV2 ? *routeFactor >= 2
                         : ((bridgeConstructed &&
                             *routeFactor == *bridgeFactor) ||
                            (!bridgeConstructed && *routeFactor >= 2))) &&
          parameters->getInteger("iteration_axis") == 0 &&
          parameters->getString("combiner") == "operator_preserving" &&
          parameters->getString("mask_tail_policy") ==
              (bridgeConstructed ? "exact_grid_divisible"
                                 : "native_unroll_main_ordered_remainder") &&
          parameters->getBoolean("invariant_hoisting").has_value() &&
          parameters->getBoolean("tensor_lane_fusion").has_value() &&
          parameters->getBoolean("exact_split_elision").has_value() &&
          (!*parameters->getBoolean("exact_split_elision") ||
           *parameters->getBoolean("tensor_lane_fusion")) &&
          bindings && bindings->size() == 3;
    } else if (runtimeGuardedLogical) {
      routeClosed = exactKeys(*parameters, {"adapter_version", "combiner",
                                            "group_shape", "iteration_axis", "kind",
                                            "mask_tail_policy", "runtime_guard_value",
                                            "subject_policy", "subject_ref",
                                            "unroll_factor"}) &&
                    parameters->getInteger("unroll_factor") == 4 && shape &&
                    shape->size() == 2 && (*shape)[0].getAsInteger() == 4 &&
                    (*shape)[1].getAsInteger() == 128 &&
                    parameters->getInteger("iteration_axis") == 0 &&
                    parameters->getString("combiner") == "addf" &&
                    parameters->getString("mask_tail_policy") ==
                        "runtime_eq_guard_original_else" &&
                    parameters->getInteger("runtime_guard_value") == 4 &&
                    bindings && bindings->size() == 2;
    } else if (bridgeConstructed) {
      bool factorized = adapterVersion == 4 || adapterVersion == 5;
      bool explicitSplitElision = adapterVersion == 5;
      routeClosed = exactKeys(
                        *parameters,
                        explicitSplitElision
                            ? ArrayRef<StringRef>({"adapter_version", "bridge_factor", "combiner",
                                                   "invariant_hoisting", "iteration_axis", "kind",
                                                   "mask_tail_policy", "subject_policy", "subject_ref",
                                                   "tensor_lane_fusion", "exact_split_elision",
                                                   "unroll_factor"})
                        : factorized
                            ? ArrayRef<StringRef>({"adapter_version", "bridge_factor", "combiner",
                                                   "invariant_hoisting", "iteration_axis", "kind",
                                                   "mask_tail_policy", "subject_policy", "subject_ref",
                                                   "tensor_lane_fusion", "unroll_factor"})
                            : ArrayRef<StringRef>({"adapter_version", "bridge_factor", "combiner",
                                                   "iteration_axis", "kind", "mask_tail_policy",
                                                   "subject_policy", "subject_ref", "unroll_factor"})) &&
                    parameters->getInteger("unroll_factor") == bridgeFactor &&
                    parameters->getInteger("iteration_axis") == 0 &&
                    parameters->getString("combiner") == "operator_preserving" &&
                    parameters->getString("mask_tail_policy") == "exact_grid_divisible" &&
                    (!factorized ||
                     (parameters->getBoolean("invariant_hoisting").has_value() &&
                      parameters->getBoolean("tensor_lane_fusion").has_value() &&
                      (*parameters->getBoolean("invariant_hoisting") ||
                       *parameters->getBoolean("tensor_lane_fusion")) &&
                      (!explicitSplitElision ||
                       parameters->getBoolean("exact_split_elision").has_value() &&
                       (!*parameters->getBoolean("exact_split_elision") ||
                        *parameters->getBoolean("tensor_lane_fusion"))))) &&
                    bindings && bindings->size() == 2;
    } else {
      routeClosed = exactKeys(*parameters, {"adapter_version", "combiner",
                                            "group_shape", "iteration_axis", "kind",
                                            "mask_tail_policy", "subject_policy",
                                            "subject_ref", "unroll_factor"}) &&
                    parameters->getInteger("unroll_factor") == 4 && shape &&
                    shape->size() == 2 && (*shape)[0].getAsInteger() == 4 &&
                    (*shape)[1].getAsInteger() == 32 &&
                    parameters->getInteger("iteration_axis") == 0 &&
                    parameters->getString("combiner") == "addi" &&
                    parameters->getString("mask_tail_policy") == "exact_static_trip" &&
                    bindings && bindings->size() == 2;
    }
  } else if (*route == kExactPrefixRoute) {
    bool logicalSubtype = *adapterVersion == 6;
    routeClosed =
        exactKeys(
            *parameters,
            logicalSubtype
                ? ArrayRef<StringRef>({
                      "active_extent", "adapter_version",
                      "artifact_adapter_ref", "combiner", "element_bytes",
                      "exact_reduction_certificate", "factor_admission_ref",
                      "kind", "mechanism_route_ref", "provider_schema",
                      "reduction_container_width", "route_factor",
                      "route_factor_kind", "route_subject_ref",
                      "route_subtype", "subject_policy", "subject_ref",
                      "tail_policy"})
                : ArrayRef<StringRef>({
                      "adapter_version", "combiner", "kind",
                      "provider_schema", "subject_policy", "subject_ref",
                      "tail_policy"})) &&
        parameters->getString("provider_schema") ==
            "hbv.loop-provider.exact-prefix.v1" &&
        parameters->getString("combiner") == "addi" &&
        parameters->getString("tail_policy") ==
            "predicated_identity_zero" &&
        (!logicalSubtype ||
         (parameters->getString("mechanism_route_ref") == kLogicalRoute &&
          parameters->getString("route_subtype") ==
              "predicated_exact_prefix_vectorization" &&
          parameters->getInteger("route_factor") == 1 &&
          parameters->getString("route_factor_kind") ==
              "exact_prefix_vectorization_adapter" &&
          parameters->getString("artifact_adapter_ref") ==
              "hbv.loop.predicated_exact_prefix_reduction.v1" &&
          parameters->getInteger("active_extent") > 0 &&
          parameters->getInteger("reduction_container_width") >=
              parameters->getInteger("active_extent") &&
          parameters->getInteger("element_bytes") > 0 &&
          parameters->getString("exact_reduction_certificate") ==
              "predicated_exact_prefix_reduction_v1" &&
          parameters->getString("route_subject_ref") &&
          !parameters->getString("route_subject_ref")->empty() &&
          parameters->getString("factor_admission_ref") &&
          !parameters->getString("factor_admission_ref")->empty())) &&
        bindings && bindings->size() == 1;
  }
  if (!routeClosed) {
    reason = "selected Loop route parameters or binding cardinality are not closed";
    return failure();
  }
  result.controlled = true;
  result.multiSubject =
      multiSubject ||
      (providerBoundSubjectSet &&
       parameters->getString("subject_policy") ==
           "provider_bound_independent_existing_scf_for_set");
  result.nestedSubject =
      nestedSubject ||
      (providerBoundSubjectSet &&
       parameters->getString("subject_policy") ==
           "provider_bound_nested_inner_dimension_scf_for");
  result.runtimeGuardedLogical = runtimeGuardedLogical;
  result.bridgeConstructed = bridgeConstructed;
  result.bridgeStaticPartition = bridgeStaticPartition;
  result.composedIntervention = composedIntervention;
  result.compositionV2 = compositionV2;
  result.compositionV3 = compositionV3;
  result.bridgeAxisVector = bridgeAxisVector;
  result.bridgeAxisDivisors = bridgeAxisDivisors;
  result.compositionSchema =
      composedIntervention ? compositionSchema->str() : "";
  result.providerClosedStatic = providerClosedStatic;
  result.exactPrefixReduction = exactPrefixReduction;
  result.stateAxisLogical = stateAxisLogical;
  if (stateAxisLogical) {
    result.mechanismRoute = kLogicalRoute.str();
    result.routeSubtype = parameters->getString("route_subtype")->str();
  }
  if (exactPrefixReduction && *adapterVersion == 6) {
    result.mechanismRoute =
        parameters->getString("mechanism_route_ref")->str();
    result.routeSubtype = parameters->getString("route_subtype")->str();
    result.exactPrefixActiveExtent =
        *parameters->getInteger("active_extent");
    result.exactPrefixContainerWidth =
        *parameters->getInteger("reduction_container_width");
    result.exactPrefixElementBytes =
        *parameters->getInteger("element_bytes");
  }
  result.affineRuntimePartial = affineRuntimePartial;
  result.bridgeInvariantHoisting =
      (adapterVersion == 4 ||
       (*route == kLogicalRoute && composedIntervention)) &&
      parameters->getBoolean("invariant_hoisting").value_or(false);
  result.bridgeTensorLaneFusion =
      adapterVersion == 3 ||
      ((adapterVersion == 4 || adapterVersion == 5) &&
       parameters->getBoolean("tensor_lane_fusion").value_or(false)) ||
      (*route == kLogicalRoute && composedIntervention &&
       parameters->getBoolean("tensor_lane_fusion").value_or(false));
  result.bridgeExactSplitElision =
      (*route == kLogicalRoute && composedIntervention)
          ? parameters->getBoolean("exact_split_elision").value_or(false)
          : adapterVersion != 5 ||
                parameters->getBoolean("exact_split_elision").value_or(false);
  result.providerBoundSubjectSet = providerBoundSubjectSet;
  result.providerBoundMembers = std::move(providerBoundMembers);
  result.providerBoundMemberSignature = providerBoundMemberSignature;
  if (affineRuntimePartial && *adapterVersion == 7)
    result.affineRuntimeMaterializationPolicy =
        parameters->getString("materialization_policy")->str();
  result.adapterVersion = *adapterVersion;
  result.bridgeFactor =
      composedIntervention ? *bridgeFactor
                           : bridgeConstructed ? *bridgeFactor : 1;
  result.routeFactor =
      stateAxisLogical ? 1
      : composedIntervention ? *routeFactor
                           : bridgeConstructed ? *bridgeFactor : 1;
  result.unrollFactor = affineRuntimePartial
                            ? *parameters->getInteger("unroll_factor")
                        : stateAxisLogical ? 1
                        : composedIntervention ? result.routeFactor
                        : bridgeConstructed    ? *bridgeFactor
                                               : 1;
  if (*route == kPipelineRoute)
    result.stageCount = composedIntervention
                            ? result.routeFactor
                            : *parameters->getInteger("stage_count");
  if (providerClosedStatic)
    for (const llvm::json::Value &value :
         *parameters->getArray("unroll_factors"))
      result.unrollFactors.push_back(*value.getAsInteger());
  result.subjectRef = subject->str();
  return result;
}

void reportFailure(ModuleOp module, const Twine &reason) {
  module->setAttr(kFailureAttr,
                  StringAttr::get(module.getContext(), reason.str()));
  module.emitError() << "[HBV_MATERIALIZATION] " << reason;
}

FailureOr<FuncOp> findEntryPoint(ModuleOp module) {
  SmallVector<FuncOp> entries;
  for (FuncOp func : module.getOps<FuncOp>())
    if (func.isPublic())
      entries.push_back(func);
  if (entries.size() != 1)
    return failure();
  return entries.front();
}

bool isRankOneIntegerVector(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.getRank() == 1 && !type.isDynamicDim(0) &&
         type.getDimSize(0) > 0 && isa<IntegerType>(type.getElementType());
}

bool isLogicalIntegerGroupingReduction(ReduceOp reduce,
                                       int64_t groupingFactor) {
  if (reduce.getSrcs().size() != 1 || reduce.getResults().size() != 1 ||
      reduce.getAxis() != 0)
    return false;
  auto sourceType = dyn_cast<RankedTensorType>(
      reduce.getSrcs().front().getType());
  auto resultType = dyn_cast<RankedTensorType>(
      reduce.getResult().front().getType());
  if (!sourceType || !resultType || sourceType.getRank() != 2 ||
      resultType.getRank() != 1 || sourceType.isDynamicDim(0) ||
      sourceType.isDynamicDim(1) || resultType.isDynamicDim(0) ||
      sourceType.getDimSize(0) != groupingFactor ||
      sourceType.getDimSize(1) != resultType.getDimSize(0) ||
      sourceType.getElementType() != resultType.getElementType() ||
      !isa<IntegerType>(sourceType.getElementType()))
    return false;
  Block &body = reduce.getCombineOp().front();
  if (body.getNumArguments() != 2 ||
      std::distance(body.begin(), body.end()) != 2)
    return false;
  auto add = dyn_cast<arith::AddIOp>(body.front());
  auto yield = dyn_cast<ReduceReturnOp>(body.back());
  return add && yield && yield.getNumOperands() == 1 &&
         yield.getOperand(0) == add.getResult() &&
         ((add.getLhs() == body.getArgument(0) &&
           add.getRhs() == body.getArgument(1)) ||
          (add.getRhs() == body.getArgument(0) &&
           add.getLhs() == body.getArgument(1)));
}

bool isVectorF32(Value value, int64_t width) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.getRank() == 1 && type.getDimSize(0) == width &&
         type.getElementType().isF32();
}

bool matchI32ConstantThroughIdentityBitcast(Value value, int64_t expected) {
  APInt constant;
  if (matchPattern(value, m_ConstantInt(&constant)))
    return constant.getSExtValue() == expected;
  auto bitcast = value.getDefiningOp<arith::BitcastOp>();
  return bitcast && bitcast.getIn().getType() == bitcast.getOut().getType() &&
         matchPattern(bitcast.getIn(), m_ConstantInt(&constant)) &&
         constant.getSExtValue() == expected;
}

bool isZeroTensorConstant(Value value) {
  if (matchPattern(value, m_Zero()))
    return true;
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  auto dense = constant ? dyn_cast<DenseElementsAttr>(constant.getValue())
                        : DenseElementsAttr();
  return dense && dense.isSplat() && dense.getElementType().isF32() &&
         dense.getSplatValue<APFloat>().isZero();
}

bool isStructurallyZeroTensor(Value value) {
  if (isZeroTensorConstant(value))
    return true;
  auto call = value.getDefiningOp<CallOp>();
  if (!call || call.getNumOperands() != 0 || call.getNumResults() != 1)
    return false;
  auto callable = dyn_cast_or_null<FuncOp>(call.resolveCallable());
  if (!callable || callable.getBody().empty())
    return false;
  auto returnOp =
      dyn_cast<ReturnOp>(callable.getBody().front().getTerminator());
  return returnOp && returnOp.getNumOperands() == 1 &&
         returnOp.getOperand(0).getType() == value.getType() &&
         isZeroTensorConstant(returnOp.getOperand(0));
}

int64_t staticElementCount(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped)
    return 1;
  if (!shaped.hasStaticShape())
    return 0;
  return shaped.getNumElements();
}

int64_t staticValueBytes(Value value) {
  Type type = value.getType();
  int64_t elements = staticElementCount(type);
  Type element = getElementTypeOrSelf(type);
  if (elements < 1 || !element.isIntOrFloat())
    return 0;
  unsigned bits = element.getIntOrFloatBitWidth();
  return elements * ((bits + 7) / 8);
}

// Direct affine geometry for an independently reorderable iteration.  This
// is deliberately expressed relative to the loop-carried induction, not an
// operator, axis name, tensor shape, or observed performance.  A stride of
// one and a large stride have different transaction/coalescing service even
// when trip count and byte count are identical, so the distinction belongs in
// the strong state consumed by any quantitative phase authority.
std::optional<int64_t> independentIterationAddressStride(
    scf::ForOp loop, bool stores) {
  int64_t maximum = 0;
  bool found = false;
  auto inspect = [&](Value pointer) {
    auto root = uniquePointerRoot(pointer);
    if (!root)
      return;
    for (Value carried : loop.getRegionIterArgs()) {
      auto footprint = pointerFootprint(pointer, *root, carried);
      if (!footprint || footprint->pidStride == 0)
        continue;
      __int128 magnitude = footprint->pidStride;
      if (magnitude < 0)
        magnitude = -magnitude;
      if (magnitude > std::numeric_limits<int64_t>::max())
        continue;
      maximum = std::max(maximum, static_cast<int64_t>(magnitude));
      found = true;
    }
  };
  loop.walk([&](Operation *operation) {
    if (stores) {
      if (auto store = dyn_cast<StoreOp>(operation))
        inspect(store.getPtr());
    } else if (auto load = dyn_cast<LoadOp>(operation)) {
      inspect(load.getPtr());
    }
  });
  return found ? std::optional<int64_t>(maximum) : std::nullopt;
}

int64_t independentInnerReductionWidth(scf::ForOp loop) {
  int64_t maximum = 0;
  loop.walk([&](ReduceOp reduce) {
    for (Value source : reduce.getSrcs())
      maximum = std::max(maximum, staticElementCount(source.getType()));
  });
  return maximum;
}

// Complete per-access affine geometry for an independent reduction.  This is
// structural TTIR evidence: no target rate, kernel identity, observed timing,
// or profitable configuration enters the record.  Unknown roots or nonlinear
// address expressions remain explicit through `complete=false` rather than
// being collapsed into a maximum stride.
struct IndependentAffineAccessFact {
  bool write = false;
  int64_t rootArgument = -1;
  int64_t elementBytes = 0;
  int64_t laneCount = 0;
  int64_t iterationStride = 0;
  std::array<int64_t, 3> programAxisStrides{0, 0, 0};
  std::array<bool, 3> programAxisDependencies{false, false, false};
  std::array<bool, 3> programAxisAffineComplete{true, true, true};
  int64_t localMinimum = 0;
  int64_t localMaximum = 0;
  bool masked = false;
  bool complete = false;
};

// A launch axis may enter an address through a quotient/remainder partition
// rather than one affine stride.  This normalized footprint preserves the
// source-level mixed-radix mechanism without naming a row/column layout or a
// kernel.  All coefficients are element offsets and must be compiler-known at
// the planning cut; otherwise `complete` remains false.
struct ProgramMixedRadixAccessFact {
  bool write = false;
  int64_t rootArgument = -1;
  int64_t elementBytes = 0;
  int64_t laneCount = 0;
  int64_t programAxis = -1;
  int64_t divisor = 0;
  int64_t quotientStride = 0;
  int64_t remainderStride = 0;
  int64_t localMinimum = 0;
  int64_t localMaximum = 0;
  bool masked = false;
  bool complete = false;
};

struct MixedRadixFootprint {
  int64_t quotientStride = 0;
  int64_t remainderStride = 0;
  int64_t localMinimum = 0;
  int64_t localMaximum = 0;
};

std::optional<MixedRadixFootprint> combineMixedRadixFootprints(
    const MixedRadixFootprint &lhs, const MixedRadixFootprint &rhs,
    bool subtract) {
  auto quotient = checkedI64(
      static_cast<__int128>(lhs.quotientStride) +
      (subtract ? -static_cast<__int128>(rhs.quotientStride)
                : static_cast<__int128>(rhs.quotientStride)));
  auto remainder = checkedI64(
      static_cast<__int128>(lhs.remainderStride) +
      (subtract ? -static_cast<__int128>(rhs.remainderStride)
                : static_cast<__int128>(rhs.remainderStride)));
  auto minimum = checkedI64(
      static_cast<__int128>(lhs.localMinimum) +
      (subtract ? -static_cast<__int128>(rhs.localMaximum)
                : static_cast<__int128>(rhs.localMinimum)));
  auto maximum = checkedI64(
      static_cast<__int128>(lhs.localMaximum) +
      (subtract ? -static_cast<__int128>(rhs.localMinimum)
                : static_cast<__int128>(rhs.localMaximum)));
  if (!quotient || !remainder || !minimum || !maximum)
    return std::nullopt;
  return MixedRadixFootprint{*quotient, *remainder, *minimum, *maximum};
}

std::optional<MixedRadixFootprint>
scaleMixedRadixFootprint(const MixedRadixFootprint &input, int64_t scale) {
  auto quotient = checkedI64(
      static_cast<__int128>(input.quotientStride) * scale);
  auto remainder = checkedI64(
      static_cast<__int128>(input.remainderStride) * scale);
  auto first = checkedI64(
      static_cast<__int128>(input.localMinimum) * scale);
  auto second = checkedI64(
      static_cast<__int128>(input.localMaximum) * scale);
  if (!quotient || !remainder || !first || !second)
    return std::nullopt;
  return MixedRadixFootprint{*quotient, *remainder,
                             std::min(*first, *second),
                             std::max(*first, *second)};
}

void collectMixedRadixDivisors(Value value, Value pid,
                               std::set<int64_t> &divisors,
                               llvm::SmallPtrSetImpl<Value> &visited) {
  if (!visited.insert(value).second)
    return;
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return;
  StringRef name = definition->getName().getStringRef();
  if ((name == "arith.divsi" || name == "arith.divui" ||
       name == "arith.remsi" || name == "arith.remui") &&
      definition->getNumOperands() == 2 &&
      stripShapeOnlyIntegerCasts(definition->getOperand(0)) == pid) {
    if (auto divisor = splatInteger(definition->getOperand(1));
        divisor && *divisor >= 1)
      divisors.insert(*divisor);
  }
  for (Value operand : definition->getOperands())
    collectMixedRadixDivisors(operand, pid, divisors, visited);
}

std::optional<MixedRadixFootprint> integerMixedRadixFootprint(
    Value value, Value pid, int64_t divisor,
    llvm::SmallPtrSetImpl<Value> &active) {
  value = stripShapeOnlyIntegerCasts(value);
  if (value == pid)
    return MixedRadixFootprint{divisor, divisor == 1 ? 0 : 1, 0, 0};
  if (auto constant = splatInteger(value))
    return MixedRadixFootprint{0, 0, *constant, *constant};
  if (auto range = value.getDefiningOp<MakeRangeOp>())
    return MixedRadixFootprint{0, 0, range.getStartAttr().getInt(),
                               range.getEndAttr().getInt() - 1};
  if (!valueDependsOn(value, pid))
    return MixedRadixFootprint{};
  if (!active.insert(value).second)
    return std::nullopt;
  auto erase = llvm::make_scope_exit([&] { active.erase(value); });
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return std::nullopt;
  StringRef name = definition->getName().getStringRef();
  if ((name == "arith.divsi" || name == "arith.divui" ||
       name == "arith.remsi" || name == "arith.remui") &&
      definition->getNumOperands() == 2 &&
      stripShapeOnlyIntegerCasts(definition->getOperand(0)) == pid) {
    auto candidate = splatInteger(definition->getOperand(1));
    if (!candidate || *candidate != divisor)
      return std::nullopt;
    if (name == "arith.divsi" || name == "arith.divui")
      return MixedRadixFootprint{1, 0, 0, 0};
    if (divisor == 1)
      return MixedRadixFootprint{};
    return MixedRadixFootprint{0, 1, 0, 0};
  }
  if ((name == "arith.addi" || name == "arith.subi") &&
      definition->getNumOperands() == 2) {
    auto lhs = integerMixedRadixFootprint(
        definition->getOperand(0), pid, divisor, active);
    auto rhs = integerMixedRadixFootprint(
        definition->getOperand(1), pid, divisor, active);
    if (!lhs || !rhs)
      return std::nullopt;
    return combineMixedRadixFootprints(
        *lhs, *rhs, name == "arith.subi");
  }
  if (name == "arith.muli" && definition->getNumOperands() == 2) {
    if (auto rhs = splatInteger(definition->getOperand(1))) {
      auto lhs = integerMixedRadixFootprint(
          definition->getOperand(0), pid, divisor, active);
      return lhs ? scaleMixedRadixFootprint(*lhs, *rhs) : std::nullopt;
    }
    if (auto lhs = splatInteger(definition->getOperand(0))) {
      auto rhs = integerMixedRadixFootprint(
          definition->getOperand(1), pid, divisor, active);
      return rhs ? scaleMixedRadixFootprint(*rhs, *lhs) : std::nullopt;
    }
  }
  return std::nullopt;
}

std::optional<MixedRadixFootprint> pointerMixedRadixFootprint(
    Value pointer, Value root, Value pid, int64_t divisor,
    llvm::SmallPtrSetImpl<Value> &active) {
  if (pointer == root)
    return MixedRadixFootprint{};
  if (!active.insert(pointer).second)
    return std::nullopt;
  auto erase = llvm::make_scope_exit([&] { active.erase(pointer); });
  Operation *definition = pointer.getDefiningOp();
  if (!definition)
    return std::nullopt;
  StringRef name = definition->getName().getStringRef();
  if (name == "tt.splat" || name == "tt.broadcast" ||
      name == "tt.expand_dims" || name == "tt.reshape" ||
      name == "tt.trans") {
    if (definition->getNumOperands() != 1)
      return std::nullopt;
    return pointerMixedRadixFootprint(
        definition->getOperand(0), root, pid, divisor, active);
  }
  auto addPtr = dyn_cast<AddPtrOp>(definition);
  if (!addPtr)
    return std::nullopt;
  auto base = pointerMixedRadixFootprint(
      addPtr.getPtr(), root, pid, divisor, active);
  llvm::SmallPtrSet<Value, 32> integerActive;
  auto offset = integerMixedRadixFootprint(
      addPtr.getOffset(), pid, divisor, integerActive);
  if (!base || !offset)
    return std::nullopt;
  return combineMixedRadixFootprints(*base, *offset, false);
}

SmallVector<ProgramMixedRadixAccessFact>
programMixedRadixAccessFacts(FuncOp function) {
  SmallVector<ProgramMixedRadixAccessFact> result;
  SmallVector<GetProgramIdOp> pids;
  function.walk([&](GetProgramIdOp pid) { pids.push_back(pid); });
  auto inspect = [&](Value pointer, Value payload, bool write, bool masked) {
    for (GetProgramIdOp pidOp : pids) {
      Value pid = pidOp.getResult();
      if (!valueDependsOn(pointer, pid))
        continue;
      std::set<int64_t> divisors;
      llvm::SmallPtrSet<Value, 32> visited;
      collectMixedRadixDivisors(pointer, pid, divisors, visited);
      if (divisors.empty())
        continue;
      ProgramMixedRadixAccessFact fact;
      fact.write = write;
      fact.masked = masked;
      fact.programAxis = pidOp.getAxisAsInt();
      fact.laneCount = staticElementCount(payload.getType());
      Type element = getElementTypeOrSelf(payload.getType());
      if (fact.laneCount > 0 && element.isIntOrFloat())
        fact.elementBytes = (element.getIntOrFloatBitWidth() + 7) / 8;
      auto root = uniquePointerRoot(pointer);
      if (root)
        if (auto argument = dyn_cast<BlockArgument>(*root))
          fact.rootArgument = argument.getArgNumber();
      if (divisors.size() == 1 && root && fact.rootArgument >= 0 &&
          fact.elementBytes > 0 && fact.laneCount > 0) {
        fact.divisor = *divisors.begin();
        llvm::SmallPtrSet<Value, 32> active;
        if (auto footprint = pointerMixedRadixFootprint(
                pointer, *root, pid, fact.divisor, active)) {
          fact.quotientStride = footprint->quotientStride;
          fact.remainderStride = footprint->remainderStride;
          fact.localMinimum = footprint->localMinimum;
          fact.localMaximum = footprint->localMaximum;
          fact.complete = true;
        }
      }
      result.push_back(fact);
    }
  };
  function.walk([&](Operation *operation) {
    if (auto load = dyn_cast<LoadOp>(operation))
      inspect(load.getPtr(), load.getResult(), false,
              static_cast<bool>(load.getMask()));
    else if (auto store = dyn_cast<StoreOp>(operation))
      inspect(store.getPtr(), store.getValue(), true,
              static_cast<bool>(store.getMask()));
  });
  return result;
}

SmallVector<IndependentAffineAccessFact>
independentAffineAccessFacts(scf::ForOp loop) {
  SmallVector<IndependentAffineAccessFact> result;
  auto function = loop->getParentOfType<FuncOp>();
  std::array<SmallVector<Value>, 3> pids;
  if (function)
    function.walk([&](GetProgramIdOp pid) {
      int axis = pid.getAxisAsInt();
      if (axis >= 0 && axis < 3)
        pids[axis].push_back(pid.getResult());
    });

  auto inspect = [&](Value pointer, Value payload, bool write, bool masked) {
    IndependentAffineAccessFact fact;
    fact.write = write;
    fact.masked = masked;
    fact.laneCount = staticElementCount(payload.getType());
    Type element = getElementTypeOrSelf(payload.getType());
    if (fact.laneCount < 1 || !element.isIntOrFloat()) {
      result.push_back(fact);
      return;
    }
    fact.elementBytes = (element.getIntOrFloatBitWidth() + 7) / 8;
    auto root = uniquePointerRoot(pointer);
    if (!root) {
      result.push_back(fact);
      return;
    }
    if (auto argument = dyn_cast<BlockArgument>(*root))
      fact.rootArgument = argument.getArgNumber();
    if (fact.rootArgument < 0) {
      result.push_back(fact);
      return;
    }

    bool iterationFound = false;
    for (Value carried : loop.getRegionIterArgs()) {
      auto footprint = pointerFootprint(pointer, *root, carried);
      if (!footprint || footprint->pidStride == 0)
        continue;
      if (iterationFound) {
        result.push_back(fact);
        return;
      }
      iterationFound = true;
      fact.iterationStride = footprint->pidStride;
      fact.localMinimum = footprint->localMin;
      fact.localMaximum = footprint->localMax;
    }
    if (!iterationFound) {
      result.push_back(fact);
      return;
    }

    for (int axis = 0; axis < 3; ++axis) {
      bool axisFound = false;
      int64_t stride = 0;
      for (Value pid : pids[axis]) {
        bool depends = valueDependsOn(pointer, pid);
        fact.programAxisDependencies[axis] |= depends;
        auto footprint = pointerFootprint(pointer, *root, pid);
        if (!footprint) {
          if (depends)
            fact.programAxisAffineComplete[axis] = false;
          continue;
        }
        if (footprint->pidStride == 0)
          continue;
        if (axisFound && stride != footprint->pidStride) {
          fact.programAxisAffineComplete[axis] = false;
          continue;
        }
        axisFound = true;
        stride = footprint->pidStride;
        fact.localMinimum = std::min(
            fact.localMinimum, footprint->localMin);
        fact.localMaximum = std::max(
            fact.localMaximum, footprint->localMax);
      }
      fact.programAxisStrides[axis] = stride;
    }
    fact.complete = llvm::all_of(
        fact.programAxisAffineComplete, [](bool value) { return value; });
    result.push_back(fact);
  };
  loop.walk([&](Operation *operation) {
    if (auto load = dyn_cast<LoadOp>(operation))
      inspect(load.getPtr(), load.getResult(), false,
              static_cast<bool>(load.getMask()));
    else if (auto store = dyn_cast<StoreOp>(operation))
      inspect(store.getPtr(), store.getValue(), true,
              static_cast<bool>(store.getMask()));
  });
  return result;
}

// Complete per-access geometry relative to launch program IDs.  Unlike the
// independent-reduction facts above, this describes the source program before
// Bridge grouping and therefore has no synthetic iteration coordinate.  It is
// an upper-footprint proof input, not a cache-residency or profitability claim.
SmallVector<IndependentAffineAccessFact>
programAffineAccessFacts(FuncOp function) {
  SmallVector<IndependentAffineAccessFact> result;
  std::array<SmallVector<Value>, 3> pids;
  function.walk([&](GetProgramIdOp pid) {
    int axis = pid.getAxisAsInt();
    if (axis >= 0 && axis < 3)
      pids[axis].push_back(pid.getResult());
  });
  auto inspect = [&](Value pointer, Value payload, bool write, bool masked) {
    IndependentAffineAccessFact fact;
    fact.write = write;
    fact.masked = masked;
    fact.iterationStride = 1;
    fact.laneCount = staticElementCount(payload.getType());
    Type element = getElementTypeOrSelf(payload.getType());
    if (fact.laneCount < 1 || !element.isIntOrFloat()) {
      result.push_back(fact);
      return;
    }
    fact.elementBytes = (element.getIntOrFloatBitWidth() + 7) / 8;
    auto root = uniquePointerRoot(pointer);
    if (!root) {
      result.push_back(fact);
      return;
    }
    if (auto argument = dyn_cast<BlockArgument>(*root))
      fact.rootArgument = argument.getArgNumber();
    if (fact.rootArgument < 0) {
      result.push_back(fact);
      return;
    }
    bool sawFootprint = false;
    for (int axis = 0; axis < 3; ++axis) {
      bool axisFound = false;
      int64_t stride = 0;
      for (Value pid : pids[axis]) {
        bool depends = valueDependsOn(pointer, pid);
        fact.programAxisDependencies[axis] |= depends;
        auto footprint = pointerFootprint(pointer, *root, pid);
        if (!footprint) {
          if (depends)
            fact.programAxisAffineComplete[axis] = false;
          continue;
        }
        sawFootprint = true;
        fact.localMinimum =
            std::min(fact.localMinimum, footprint->localMin);
        fact.localMaximum =
            std::max(fact.localMaximum, footprint->localMax);
        if (footprint->pidStride == 0)
          continue;
        if (axisFound && stride != footprint->pidStride) {
          fact.programAxisAffineComplete[axis] = false;
          continue;
        }
        axisFound = true;
        stride = footprint->pidStride;
      }
      fact.programAxisStrides[axis] = stride;
    }
    fact.complete = sawFootprint && llvm::all_of(
        fact.programAxisAffineComplete, [](bool value) { return value; });
    result.push_back(fact);
  };
  function.walk([&](Operation *operation) {
    if (auto load = dyn_cast<LoadOp>(operation))
      inspect(load.getPtr(), load.getResult(), false,
              static_cast<bool>(load.getMask()));
    else if (auto store = dyn_cast<StoreOp>(operation))
      inspect(store.getPtr(), store.getValue(), true,
              static_cast<bool>(store.getMask()));
  });
  return result;
}

bool isEligibleLoop(scf::ForOp loop, SmallVectorImpl<Operation *> &loadSlice,
                    Operation *&compute, Operation *&reduce,
                    bool requireStaticTripCount = true);

bool isNativePipelineScopeProxy(scf::ForOp loop, int64_t defaultNumStages,
                                int64_t &effectiveNumStages) {
  // Source-backed TTIR proxy for the native TTGIR pipeline scope.  Explicit
  // tt.num_stages admits non-dot loads in AssignLatencies; dot loops are in
  // the native latency path without the explicit attribute.  Outer loops are
  // excluded by the native precondition, so reject a loop containing another
  // scf.for here as the conservative TTIR equivalent.
  bool hasNestedLoop = false;
  bool hasLoad = false;
  bool hasDot = false;
  loop.walk([&](Operation *op) {
    if (op != loop.getOperation() && isa<scf::ForOp>(op))
      hasNestedLoop = true;
    hasLoad |= isa<LoadOp, DescriptorLoadOp, DescriptorGatherOp>(op);
    hasDot |= isa<DotOpInterface>(op);
  });
  // Match loopHasDistGreaterThanOne from the native pipeliner at TTIR: a
  // yielded value with no defining op is a loop-carried distance that the
  // pipeline expander does not currently support.
  bool hasUnsupportedDistance = llvm::any_of(
      loop.getBody()->getTerminator()->getOperands(),
      [](Value operand) { return !operand.getDefiningOp(); });
  auto explicitStages = loop->getAttrOfType<IntegerAttr>("tt.num_stages");
  effectiveNumStages = explicitStages ? explicitStages.getInt() : defaultNumStages;
  return !hasNestedLoop && !hasUnsupportedDistance && hasLoad &&
         effectiveNumStages > 1 && (hasDot || explicitStages);
}

// Candidate-side counterpart of the native GPU pipeliner's dynamic-loop
// contract.  The native expander runs with supportDynamicLoops=true and
// predicates its prologue, steady kernel and epilogue against the runtime
// upper bound.  Therefore an unknown trip count is not a rejection reason.
// This planning-cut certificate intentionally proves only the structural
// conditions visible before TTGIR scheduling; final async realization remains
// owned by the native pipeline and post-codegen attestation.
LoopDependenceCertificate certifyNativeDynamicPipelineSubject(
    scf::ForOp loop) {
  bool hasNestedLoop = false;
  bool hasLoad = false;
  loop.walk([&](Operation *operation) {
    if (operation != loop.getOperation() && isa<scf::ForOp>(operation))
      hasNestedLoop = true;
    hasLoad |= isa<LoadOp, DescriptorLoadOp, DescriptorGatherOp>(operation);
  });
  if (hasNestedLoop)
    return {false, "", "nested_loop_requires_independent_pipeline_subject"};
  if (!hasLoad)
    return {false, "", "pipeline_subject_has_no_load_service"};
  bool hasUnsupportedDistance = llvm::any_of(
      loop.getBody()->getTerminator()->getOperands(),
      [](Value operand) { return !operand.getDefiningOp(); });
  if (hasUnsupportedDistance)
    return {false, "", "pipeline_loop_carried_distance_exceeds_one"};
  return {true,
          "native_dynamic_pipeline_predicated_prologue_epilogue_v1", ""};
}

LoopDependenceCertificate
certifyAffineRuntimeGridStrideOrderPreserving(scf::ForOp loop) {
  Value lower = stripShapeOnlyIntegerCasts(loop.getLowerBound());
  Value step = stripShapeOnlyIntegerCasts(loop.getStep());
  auto pid = lower.getDefiningOp<GetProgramIdOp>();
  auto programs = step.getDefiningOp<GetNumProgramsOp>();
  if (!pid || !programs || pid.getAxisAsInt() != programs.getAxisAsInt())
    return {false, "", "bounds_are_not_same_axis_program_id_grid_stride"};
  if (!loop.getRegionIterArgs().empty() || loop.getNumResults() != 0)
    return {false, "", "loop_carried_state_is_not_closed"};
  if (valueDependsOn(loop.getUpperBound(), loop.getInductionVar()) ||
      valueDependsOn(loop.getUpperBound(), lower) ||
      valueDependsOn(loop.getUpperBound(), step))
    return {false, "", "runtime_upper_bound_depends_on_iteration_or_grid"};
  auto upper = splatInteger(loop.getUpperBound());
  if (!upper || *upper < 0 || *upper > std::numeric_limits<int32_t>::max())
    return {false, "", "runtime_upper_bound_is_not_predecision_bounded_i32"};

  unsigned nestedLoops = 0;
  unsigned loads = 0;
  unsigned stores = 0;
  unsigned atomics = 0;
  SmallVector<Value> loadRoots;
  llvm::DenseMap<Value, unsigned> storesPerRoot;
  LoopDependenceCertificate failure{true, "", ""};
  loop.walk([&](Operation *operation) {
    if (!failure.safe)
      return;
    if (operation != loop.getOperation() && isa<scf::ForOp>(operation)) {
      ++nestedLoops;
      failure = {false, "", "nested_loop_requires_a_separate_certificate"};
      return;
    }
    if (isa<AtomicRMWOp, AtomicCASOp>(operation)) {
      ++atomics;
      failure = {false, "", "atomic_effect_is_not_iteration_partitioned"};
      return;
    }
    if (auto load = dyn_cast<LoadOp>(operation)) {
      ++loads;
      auto root = uniquePointerRoot(load.getPtr());
      if (!root) {
        failure = {false, "", "load_pointer_root_is_ambiguous"};
        return;
      }
      loadRoots.push_back(*root);
      return;
    }
    if (auto store = dyn_cast<StoreOp>(operation)) {
      ++stores;
      auto root = uniquePointerRoot(store.getPtr());
      if (!root) {
        failure = {false, "", "store_pointer_root_is_ambiguous"};
        return;
      }
      if (!valueDependsOn(store.getPtr(), loop.getInductionVar())) {
        failure = {false, "", "store_is_not_indexed_by_runtime_ordinal"};
        return;
      }
      if (++storesPerRoot[*root] != 1) {
        failure = {false, "", "multiple_stores_share_one_unpartitioned_root"};
        return;
      }
      return;
    }
    if (operation == loop.getOperation() ||
        operation->hasTrait<OpTrait::IsTerminator>())
      return;
    if (operation->getNumRegions() != 0 || !isMemoryEffectFree(operation))
      failure = {false, "", "body_contains_an_unclosed_effect"};
  });
  if (!failure.safe)
    return failure;
  if (nestedLoops != 0 || loads == 0 || stores == 0)
    return {false, "", "loop_has_no_closed_load_store_tile"};
  for (const auto &item : storesPerRoot)
    if (llvm::is_contained(loadRoots, item.first))
      return {false, "", "input_output_alias_is_not_closed"};
  if (atomics != 0)
    return {false, "", "atomic_effect_is_not_iteration_partitioned"};
  return {true, "affine_runtime_grid_stride_order_preserving_v2", ""};
}

// A second affine-runtime domain partitions one runtime interval into one
// contiguous chunk per physical program.  Unlike the grid-stride domain
// above, this form may carry an arbitrary *closed* SSA recurrence.  The
// materializer preserves the original iteration order, so it does not rely on
// reassociation, floating-point associativity, an operator identity, or a
// benchmark template.
LoopDependenceCertificate
certifyAffineRuntimeProgramPartitionOrderPreserving(scf::ForOp loop) {
  auto function = loop->getParentOfType<FuncOp>();
  if (!function)
    return {false, "", "runtime_partition_has_no_function_scope"};
  SmallVector<GetProgramIdOp> dependentPids;
  function.walk([&](GetProgramIdOp pid) {
    if (valueDependsOn(loop.getLowerBound(), pid.getResult()))
      dependentPids.push_back(pid);
  });
  if (dependentPids.size() != 1)
    return {false, "", "runtime_partition_lower_has_no_unique_program_axis"};
  GetProgramIdOp pid = dependentPids.front();
  llvm::SmallPtrSet<Value, 32> lowerActive;
  auto lower = integerFootprint(
      loop.getLowerBound(), pid.getResult(), lowerActive);
  if (!lower || lower->pidStride <= 0 ||
      lower->localMin != lower->localMax)
    return {false, "", "runtime_partition_lower_is_not_affine_contiguous"};
  auto step = splatInteger(loop.getStep());
  if (!step || *step <= 0)
    return {false, "", "runtime_partition_step_is_not_positive_static"};

  Operation *upperDefinition = loop.getUpperBound().getDefiningOp();
  if (!upperDefinition || upperDefinition->getNumOperands() != 2 ||
      upperDefinition->getName().getStringRef() != "arith.minsi")
    return {false, "", "runtime_partition_upper_is_not_signed_minimum"};
  Value first = upperDefinition->getOperand(0);
  Value second = upperDefinition->getOperand(1);
  bool firstDepends = valueDependsOn(first, pid.getResult());
  bool secondDepends = valueDependsOn(second, pid.getResult());
  if (firstDepends == secondDepends)
    return {false, "", "runtime_partition_upper_has_ambiguous_program_end"};
  Value programEnd = firstDepends ? first : second;
  Value totalExtent = firstDepends ? second : first;
  llvm::SmallPtrSet<Value, 32> endActive;
  auto end = integerFootprint(programEnd, pid.getResult(), endActive);
  auto total = splatInteger(totalExtent);
  if (!end || end->pidStride != lower->pidStride ||
      end->localMin != end->localMax ||
      end->localMin - lower->localMin != lower->pidStride)
    return {false, "", "runtime_partition_end_is_not_next_contiguous_chunk"};
  if (!total || *total < 0 ||
      *total > std::numeric_limits<int32_t>::max())
    return {false, "", "runtime_partition_extent_is_not_predecision_bounded_i32"};
  if (*step > std::numeric_limits<int32_t>::max() - *total)
    return {false, "", "runtime_partition_main_tail_integer_range_is_open"};
  if (valueDependsOn(totalExtent, loop.getInductionVar()))
    return {false, "", "runtime_partition_extent_depends_on_iteration"};

  auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
  if (!yield || yield.getNumOperands() != loop.getRegionIterArgs().size() ||
      loop.getNumResults() != loop.getRegionIterArgs().size())
    return {false, "", "runtime_partition_carried_state_is_not_closed"};
  for (auto [index, carried] : llvm::enumerate(loop.getRegionIterArgs())) {
    Value next = yield.getOperand(index);
    if (next.getType() != carried.getType() || !valueDependsOn(next, carried))
      return {false, "", "runtime_partition_recurrence_is_not_self_closed"};
    for (auto [otherIndex, other] : llvm::enumerate(
             loop.getRegionIterArgs()))
      if (otherIndex != index && valueDependsOn(next, other))
        return {false, "", "runtime_partition_recurrences_are_cross_coupled"};
  }

  unsigned loads = 0;
  unsigned stores = 0;
  SmallVector<Value> loadRoots;
  llvm::DenseMap<Value, unsigned> storesPerRoot;
  LoopDependenceCertificate failure{true, "", ""};
  auto isClosedPureCall = [](CallOp call) {
    auto callee = dyn_cast_or_null<FuncOp>(call.resolveCallable());
    if (!callee || callee.getBody().empty())
      return false;
    bool pure = true;
    callee.walk([&](Operation *operation) {
      if (!pure || operation == callee.getOperation() ||
          operation->hasTrait<OpTrait::IsTerminator>())
        return;
      // Calls hidden behind another callable need their own prospective
      // closure rather than inheriting this one transitively.
      if (isa<CallOp, LoadOp, StoreOp, AtomicRMWOp, AtomicCASOp>(operation) ||
          !isMemoryEffectFree(operation))
        pure = false;
    });
    return pure;
  };
  loop.walk([&](Operation *operation) {
    if (!failure.safe || operation == loop.getOperation() ||
        operation->hasTrait<OpTrait::IsTerminator>())
      return;
    if (isa<scf::ForOp>(operation)) {
      failure = {false, "", "nested_loop_requires_a_separate_certificate"};
      return;
    }
    if (isa<AtomicRMWOp, AtomicCASOp>(operation)) {
      failure = {false, "", "atomic_effect_is_not_iteration_partitioned"};
      return;
    }
    if (auto load = dyn_cast<LoadOp>(operation)) {
      if (load.getIsVolatile()) {
        failure = {false, "", "volatile_load_is_not_order_preservable"};
        return;
      }
      auto root = uniquePointerRoot(load.getPtr());
      if (!root) {
        failure = {false, "", "load_pointer_root_is_ambiguous"};
        return;
      }
      ++loads;
      loadRoots.push_back(*root);
      return;
    }
    if (auto store = dyn_cast<StoreOp>(operation)) {
      auto root = uniquePointerRoot(store.getPtr());
      if (!root) {
        failure = {false, "", "store_pointer_root_is_ambiguous"};
        return;
      }
      if (!valueDependsOn(store.getPtr(), loop.getInductionVar())) {
        failure = {false, "", "store_is_not_indexed_by_partition_ordinal"};
        return;
      }
      ++stores;
      ++storesPerRoot[*root];
      return;
    }
    if (auto call = dyn_cast<CallOp>(operation)) {
      if (!isClosedPureCall(call))
        failure = {false, "", "body_call_is_not_prospectively_pure"};
      return;
    }
    if (!isMemoryEffectFree(operation))
      failure = {false, "",
                 (Twine("body_contains_an_unclosed_effect:") +
                  operation->getName().getStringRef()).str()};
  });
  if (!failure.safe)
    return failure;
  if (loads == 0 || stores == 0)
    return {false, "", "runtime_partition_has_no_closed_load_store_service"};
  for (const auto &item : storesPerRoot)
    if (llvm::is_contained(loadRoots, item.first))
      return {false, "", "input_output_alias_is_not_closed"};
  return {true, "affine_runtime_program_partition_order_preserving_v1", ""};
}

LoopDependenceCertificate certifyAffineRuntimeOrderPreserving(scf::ForOp loop) {
  LoopDependenceCertificate grid =
      certifyAffineRuntimeGridStrideOrderPreserving(loop);
  if (grid.safe)
    return grid;
  LoopDependenceCertificate partition =
      certifyAffineRuntimeProgramPartitionOrderPreserving(loop);
  if (partition.safe)
    return partition;
  // The partition reason is more specific once the lower bound depends on a
  // program ID; otherwise retain the original grid-stride diagnosis.
  return partition.reason == "runtime_partition_lower_has_no_unique_program_axis"
             ? grid
             : partition;
}

bool isAffineRuntimeOrderPreservingCertificate(StringRef certificate) {
  return certificate == "affine_runtime_grid_stride_order_preserving_v2" ||
         certificate ==
             "affine_runtime_program_partition_order_preserving_v1";
}

bool isExistingUnrollOrderCertificate(StringRef certificate) {
  return certificate == "existing_associative_reduction_v1" ||
         certificate == "per_loop_existing_associative_reduction_v1" ||
         certificate == "existing_exact_integer_addition_v2" ||
         certificate == "existing_order_preserved_floating_addition_v2" ||
         certificate == "existing_order_preserving_read_exposure_v1" ||
         certificate ==
             "existing_order_preserving_load_vectorization_v1" ||
         certificate == "existing_affine_pointer_read_exposure_v1" ||
         certificate ==
             "existing_affine_pointer_read_load_vectorization_v1" ||
         certificate ==
             "per_loop_order_preserving_read_exposure_v1" ||
         certificate ==
             "per_loop_order_preserving_load_vectorization_v1";
}

// Canonical capability composition consumed above the compiler ABI.  The
// legacy certificate remains a compatibility projection for frozen plans;
// it is not the semantic shape of the contract.  Geometry-specific proofs
// above establish the facts, while this projection makes their common
// ordered-owned-iteration basis explicit without widening either domain.
llvm::json::Object orderedOwnedIterationCapabilities(
    const LoopDependenceCertificate &certificate) {
  llvm::json::Object result;
  if (!certificate.safe ||
      !isAffineRuntimeOrderPreservingCertificate(certificate.kind))
    return result;
  result["schema"] = "hbv.loop.ordered-owned-iteration.v1";
  result["ownership"] =
      certificate.kind ==
              "affine_runtime_grid_stride_order_preserving_v2"
          ? "same_axis_grid_stride"
          : "contiguous_program_partition";
  result["recurrence"] =
      certificate.kind ==
              "affine_runtime_grid_stride_order_preserving_v2"
          ? "none"
          : "self_closed";
  llvm::json::Array effects;
  effects.push_back("disjoint_output");
  result["effects"] = std::move(effects);
  result["control"] = "straight_or_tail_guarded";
  result["original_order_preserved"] = true;
  result["prospective_extent_closed"] = true;
  result["legacy_projection"] = certificate.kind;
  return result;
}

class LoopBridgeDiscoverPass
    : public impl::TritonLoopBridgeDiscoverBase<LoopBridgeDiscoverPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto entry = findEntryPoint(module);
    if (failed(entry)) {
      module->setAttr(
          kBridgeDiscoveryAttr,
          StringAttr::get(module.getContext(),
                          R"({"schema":"triton.loop-bridge.discovery.v1","extractable":false,"reason":"entry_not_unique"})"));
      return;
    }

    // Runtime integer values are launch facts bound into this compilation,
    // never performance observations.  They let the affine proof evaluate
    // dynamic strides/extents without naming a function or benchmark.
    if (auto raw = module->getAttrOfType<StringAttr>(
            kBridgeRuntimeScalarsAttr)) {
      auto parsed = llvm::json::parse(raw.getValue());
      auto *object = parsed ? parsed->getAsObject() : nullptr;
      auto schema = object ? object->getString("schema") : std::nullopt;
      auto *values = object ? object->getObject("values") : nullptr;
      auto *valuesByName =
          object ? object->getObject("values_by_name") : nullptr;
      auto *grid = object ? object->getArray("grid") : nullptr;
      bool legacy = schema == "triton.loop-bridge.runtime-scalars.v1";
      bool named = schema == "triton.loop-bridge.runtime-scalars.v2";
      if (!object || (!legacy && !named) ||
          (legacy && !values) || (named && !valuesByName) ||
          !grid || grid->empty() || grid->size() > 3) {
        module->setAttr(
            kBridgeDiscoveryAttr,
            StringAttr::get(module.getContext(),
                            R"({"schema":"triton.loop-bridge.discovery.v1","extractable":false,"reason":"runtime_scalar_binding_malformed"})"));
        return;
      }
      if (legacy) {
        for (const auto &item : *values) {
          unsigned index = 0;
          auto integer = item.second.getAsInteger();
          StringRef key = item.first;
          if (key.getAsInteger(10, index) || !integer ||
              index >= (*entry).getNumArguments()) {
            module->setAttr(
                kBridgeDiscoveryAttr,
                StringAttr::get(module.getContext(),
                                R"({"schema":"triton.loop-bridge.discovery.v1","extractable":false,"reason":"runtime_scalar_binding_out_of_domain"})"));
            return;
          }
          (*entry).setArgAttr(
              index, kBridgeBoundScalarAttr,
              IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                               *integer));
        }
      } else {
        // Specialization may remove arbitrary source arguments.  Resolve the
        // remaining arguments by compiler-carried NameLoc rather than by the
        // unstable pre-specialization ordinal.  Extra bindings correspond to
        // removed arguments and are deliberately ignored.
        for (BlockArgument argument : (*entry).getArguments()) {
          auto name = structuralArgumentName(argument);
          if (!name)
            continue;
          auto *value = valuesByName->get(*name);
          auto integer = value ? value->getAsInteger() : std::nullopt;
          if (!integer)
            continue;
          (*entry).setArgAttr(
              argument.getArgNumber(), kBridgeBoundScalarAttr,
              IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                               *integer));
        }
      }
      (*entry).walk([&](GetProgramIdOp pid) {
        unsigned axis = pid.getAxisAsInt();
        if (axis >= grid->size())
          return;
        if (auto extent = (*grid)[axis].getAsInteger();
            extent && *extent > 0)
          pid->setAttr(
              kBridgeAxisExtentAttr,
              IntegerAttr::get(
                  IntegerType::get(module.getContext(), 64), *extent));
      });
    }

    SmallVector<GetProgramIdOp> xPids;
    (*entry).walk([&](GetProgramIdOp pid) {
      if (pid.getAxisAsInt() == 0)
        xPids.push_back(pid);
    });
    LoopDependenceCertificate certificate;
    if (xPids.size() == 1)
      certificate = certifyBridgeProgramIndependence(*entry, xPids.front());
    else
      certificate = {false, "", "axis-zero program ID is not unique"};
    auto directPartition = xPids.size() == 1
                               ? findDirectPidPartition(*entry, xPids.front())
                               : std::nullopt;
    llvm::json::Object facts;
    facts["schema"] = "triton.loop-bridge.discovery.v1";
    facts["extractable"] = true;
    facts["construction_legal"] = certificate.safe;
    facts["dependence_certificate"] = certificate.kind;
    facts["rejection_reason"] = certificate.reason;
    facts["axis_zero_program_id_count"] =
        static_cast<int64_t>(xPids.size());
    facts["partition_recurrence_legal"] = directPartition.has_value();
    facts["partition_recurrence_certificate"] =
        directPartition
            ? "consecutive_pid_signed_quotient_remainder_recurrence_v1"
            : "";
    std::string serialized;
    llvm::raw_string_ostream stream(serialized);
    stream << llvm::json::Value(std::move(facts));
    stream.flush();
    module->setAttr(kBridgeDiscoveryAttr,
                    StringAttr::get(module.getContext(), serialized));
  }
};

class HBVLoopFactsPass
    : public impl::TritonHBVLoopFactsBase<HBVLoopFactsPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto entry = findEntryPoint(module);
    if (failed(entry)) {
      module->setAttr(kStaticFactsAttr,
                      StringAttr::get(module.getContext(),
                                      R"({"schema":"hbv.loop.static-facts.v1","extractable":false,"reason":"entry_not_unique"})"));
      return;
    }

    int64_t loadCount = 0;
    int64_t storeCount = 0;
    int64_t atomicCount = 0;
    int64_t loopCount = 0;
    int64_t reduceCount = 0;
    int64_t maxLoadWidth = 1;
    int64_t inputBytes = 0;
    int64_t outputBytes = 0;
    int64_t arithmeticElements = 0;
    int64_t runtimeActiveExtentArgumentIndex = -1;
    int64_t runtimeActiveExtentContainerWidth = 0;
    int64_t activeInputBytesPerElement = 0;
    int64_t activeOutputBytesPerElement = 0;
    int64_t activeArithmeticElementsPerElement = 0;
    int64_t activeInputFixedBytes = 0;
    int64_t activeOutputFixedBytes = 0;
    int64_t activeArithmeticFixedElements = 0;
    int64_t nativePipelineScopeCount = 0;
    int64_t nativePipelineCandidateCount = 0;
    int64_t affineRuntimeGridStrideCount = 0;
    int64_t affineRuntimeCandidateCount = 0;
    int64_t nativeLoopInputBytesPerIteration = 0;
    int64_t nativeTripCount = 0;
    int64_t nativeDefaultStages = 0;
    int64_t exactPrefixReductionCount = 0;
    int64_t exactPrefixMaximumActiveExtent = 0;
    int64_t exactPrefixMaximumWidth = 0;
    int64_t exactPrefixMaximumElementBytes = 0;
    int64_t stateAxisNormalizationCount = 0;
    int64_t stateAxisMinimumCardinality = 0;
    int64_t stateAxisMaximumCardinality = 0;
    int64_t stateAxisMaximumPaddedCardinality = 0;
    int64_t stateAxisMinimumWidth = 0;
    int64_t stateAxisMaximumWidth = 0;
    int64_t stateAxisMinimumElementBytes = 0;
    int64_t stateAxisMaximumElementBytes = 0;
    int64_t independentInnerReductionCount = 0;
    SmallVector<int64_t> independentInnerReductionTripCounts;
    SmallVector<int64_t> independentInnerReductionInputStrides;
    SmallVector<int64_t> independentInnerReductionOutputStrides;
    SmallVector<int64_t> independentInnerReductionWidths;
    SmallVector<int64_t> independentInnerReductionBodyOperationCounts;
    SmallVector<int64_t> independentInnerReductionBodyTypeVolumes;
    SmallVector<SmallVector<IndependentAffineAccessFact>>
        independentInnerReductionAffineAccesses;
    SmallVector<IndependentAffineAccessFact> programAffineAccesses;
    SmallVector<RuntimeMaskScalarFact> runtimeMaskScalars;
    SmallVector<ProgramMixedRadixAccessFact> programMixedRadixAccesses;
    std::array<int64_t, 8> operationCountByProgramDependencyMask{};
    std::array<int64_t, 8> typeVolumeByProgramDependencyMask{};
    std::array<int64_t, 8> hoistableOperationCountByProgramDependencyMask{};
    std::array<bool, 3> programAxisIndependence{};
    std::array<std::string, 3> programAxisIndependenceReasons{};
    bool programDependencyGraphComplete = true;
    int64_t programSSACriticalPathSteps = 0;
    std::set<std::string> bodyTypeSignature;
    std::map<std::string, int64_t> bodySpecialFunctionCounts;
    auto defaultStagesAttr =
        module->getAttrOfType<IntegerAttr>(kNativeDefaultStagesAttr);
    int64_t compilerDefaultStages =
        defaultStagesAttr ? defaultStagesAttr.getInt() : 0;
    LoopDependenceCertificate nativeCertificate;
    LoopDependenceCertificate affineRuntimeCertificate;
    SmallVector<StateAxisNormalizationRegion, 2> stateAxisSubjects;
    llvm::json::Array loopCensus;
    for (ExactPrefixReduction subject :
         collectExactPrefixReductions(*entry)) {
      ++exactPrefixReductionCount;
      exactPrefixMaximumWidth = std::max(
          exactPrefixMaximumWidth, subject.vectorWidth);
      exactPrefixMaximumActiveExtent = std::max(
          exactPrefixMaximumActiveExtent, subject.activeExtent);
      auto elementType = cast<IntegerType>(
          subject.load.getResult().getType());
      exactPrefixMaximumElementBytes = std::max(
          exactPrefixMaximumElementBytes,
          static_cast<int64_t>((elementType.getWidth() + 7) / 8));
    }
    stateAxisSubjects = collectStateAxisNormalizations(*entry);
    for (const StateAxisNormalizationRegion &subject : stateAxisSubjects) {
      ++stateAxisNormalizationCount;
      if (stateAxisNormalizationCount == 1) {
        stateAxisMinimumCardinality = subject.stateCardinality;
        stateAxisMinimumWidth = subject.stateWidth;
        stateAxisMinimumElementBytes = subject.elementBytes;
      } else {
        stateAxisMinimumCardinality = std::min(
            stateAxisMinimumCardinality, subject.stateCardinality);
        stateAxisMinimumWidth = std::min(
            stateAxisMinimumWidth, subject.stateWidth);
        stateAxisMinimumElementBytes = std::min(
            stateAxisMinimumElementBytes, subject.elementBytes);
      }
      stateAxisMaximumCardinality = std::max(
          stateAxisMaximumCardinality, subject.stateCardinality);
      stateAxisMaximumPaddedCardinality = std::max(
          stateAxisMaximumPaddedCardinality,
          subject.paddedStateCardinality);
      stateAxisMaximumWidth = std::max(
          stateAxisMaximumWidth, subject.stateWidth);
      stateAxisMaximumElementBytes = std::max(
          stateAxisMaximumElementBytes, subject.elementBytes);
    }
    SmallVector<GetProgramIdOp> xPids;
    SmallVector<FuncOp> reachableFunctions{*entry};
    llvm::SmallPtrSet<Operation *, 8> seenFunctions;
    seenFunctions.insert(entry->getOperation());
    for (size_t index = 0; index < reachableFunctions.size(); ++index) {
      reachableFunctions[index].walk([&](CallOp call) {
        auto callee = dyn_cast_or_null<FuncOp>(call.resolveCallable());
        if (callee && seenFunctions.insert(callee.getOperation()).second)
          reachableFunctions.push_back(callee);
      });
    }
    // A planning-cut locator is transport identity, never an admission
    // feature.  Freeze the complete reachable-function walk before emitting
    // any row so the same locator can be bound by a later Plan and rederived
    // by the decision pass without consuming a source/kernel name.
    llvm::DenseMap<Operation *, std::string> planningCutLoopLocators;
    int64_t planningCutLoopOrdinal = 0;
    for (FuncOp function : reachableFunctions)
      function.walk([&](scf::ForOp loop) {
        planningCutLoopLocators[loop.getOperation()] =
            "planning-cut.loop." +
            std::to_string(planningCutLoopOrdinal++);
      });
    programAffineAccesses = programAffineAccessFacts(*entry);
    programMixedRadixAccesses = programMixedRadixAccessFacts(*entry);
    (*entry).walk([&](Operation *operation) {
      if (auto load = dyn_cast<LoadOp>(operation)) {
        if (auto fact = runtimeMaskScalarFromMask(
                load.getMask(), *entry, false))
          runtimeMaskScalars.push_back(*fact);
      } else if (auto store = dyn_cast<StoreOp>(operation)) {
        if (auto fact = runtimeMaskScalarFromMask(
                store.getMask(), *entry, true))
          runtimeMaskScalars.push_back(*fact);
      }
    });

    // Preserve the complete structural distribution of operation dependence
    // on the three launch axes.  The eight masks are not a fitted feature
    // basis: they are a lossless count projection of direct IR dependence and
    // let a later candidate transform derive invariant replication for any
    // grouped-axis subset without using a grid, shape, or operator identity.
    std::array<SmallVector<GetProgramIdOp>, 3> programIdsByAxis;
    entry->walk([&](GetProgramIdOp pid) {
      programIdsByAxis[pid.getAxisAsInt()].push_back(pid);
    });
    for (unsigned axis = 0; axis < 3; ++axis) {
      if (programIdsByAxis[axis].size() != 1) {
        programAxisIndependenceReasons[axis] =
            programIdsByAxis[axis].empty()
                ? "program_axis_unused"
                : "program_axis_id_not_unique";
        continue;
      }
      LoopDependenceCertificate axisCertificate =
          certifyBridgeProgramIndependence(
              *entry, programIdsByAxis[axis].front());
      programAxisIndependence[axis] = axisCertificate.safe;
      programAxisIndependenceReasons[axis] =
          axisCertificate.safe ? axisCertificate.kind
                               : axisCertificate.reason;
    }
    llvm::DenseMap<Operation *, int64_t> operationDependencyDepth;
    entry->walk([&](Operation *operation) {
      if (operation == entry->getOperation() ||
          operation->hasTrait<OpTrait::IsTerminator>() ||
          isa<GetProgramIdOp>(operation))
        return;
      unsigned mask = 0;
      for (unsigned axis = 0; axis < 3; ++axis) {
        bool dependent = llvm::any_of(
            operation->getOperands(), [&](Value operand) {
              return llvm::any_of(
                  programIdsByAxis[axis], [&](GetProgramIdOp pid) {
                    return valueDependsOn(operand, pid.getResult());
                  });
            });
        if (dependent)
          mask |= 1u << axis;
      }
      int64_t typeVolume = 1;
      for (Value operand : operation->getOperands())
        typeVolume = std::max(
            typeVolume, staticElementCount(operand.getType()));
      for (Value result : operation->getResults())
        typeVolume = std::max(
            typeVolume, staticElementCount(result.getType()));
      ++operationCountByProgramDependencyMask[mask];
      typeVolumeByProgramDependencyMask[mask] += typeVolume;
      if (operation->getNumRegions() == 0 &&
          isMemoryEffectFree(operation))
        ++hoistableOperationCountByProgramDependencyMask[mask];
      if (operation->getNumRegions() != 0)
        programDependencyGraphComplete = false;
      int64_t depth = 1;
      for (Value operand : operation->getOperands()) {
        Operation *definition = operand.getDefiningOp();
        if (!definition)
          continue;
        auto found = operationDependencyDepth.find(definition);
        if (found == operationDependencyDepth.end()) {
          // A definition outside the ordered, regionless entry projection
          // prevents an exact SSA critical-path claim.
          if (!isa<GetProgramIdOp>(definition) &&
              !isa<arith::ConstantOp>(definition))
            programDependencyGraphComplete = false;
          continue;
        }
        depth = std::max(depth, found->second + 1);
      }
      operationDependencyDepth[operation] = depth;
      programSSACriticalPathSteps =
          std::max(programSSACriticalPathSteps, depth);
    });

    // Establish one common dynamic active extent from masked entry loads.
    // Multiple masked loads must agree on both the entry argument and the
    // vector container.  Ambiguity leaves the legacy exact-static facts in
    // force and publishes no runtime fact.
    bool activeExtentAmbiguous = false;
    SmallVector<Value> activeLoadResults;
    entry->walk([&](LoadOp load) {
      auto fact = runtimeActiveExtentFromMask(load.getMask(), *entry);
      if (!fact)
        return;
      if (runtimeActiveExtentArgumentIndex < 0) {
        runtimeActiveExtentArgumentIndex = fact->argumentIndex;
        runtimeActiveExtentContainerWidth = fact->containerWidth;
      } else if (runtimeActiveExtentArgumentIndex != fact->argumentIndex ||
                 runtimeActiveExtentContainerWidth != fact->containerWidth) {
        activeExtentAmbiguous = true;
        return;
      }
      activeLoadResults.push_back(load.getResult());
    });
    if (activeExtentAmbiguous || activeLoadResults.empty()) {
      runtimeActiveExtentArgumentIndex = -1;
      runtimeActiveExtentContainerWidth = 0;
      activeLoadResults.clear();
    }

    auto dependsOnActiveLoad = [&](Value value) {
      return llvm::any_of(activeLoadResults, [&](Value source) {
        return valueDependsOn(value, source);
      });
    };
    for (FuncOp function : reachableFunctions)
      function.walk([&](Operation *op) {
      if (!op->hasTrait<OpTrait::IsTerminator>()) {
        auto recordType = [&](Type type) {
          Type scalar = getElementTypeOrSelf(type);
          if (auto pointer = dyn_cast<PointerType>(scalar))
            scalar = getElementTypeOrSelf(pointer.getPointeeType());
          std::string text;
          llvm::raw_string_ostream stream(text);
          scalar.print(stream);
          stream.flush();
          bodyTypeSignature.insert(std::move(text));
        };
        for (Value operand : op->getOperands())
          recordType(operand.getType());
        for (Value result : op->getResults())
          recordType(result.getType());
        StringRef operationName = op->getName().getStringRef();
        if (operationName.starts_with("math."))
          ++bodySpecialFunctionCounts[std::string(operationName)];
      }
      if (function == *entry)
        if (auto pid = dyn_cast<GetProgramIdOp>(op);
            pid && pid.getAxisAsInt() == 0)
          xPids.push_back(pid);
      if (auto load = dyn_cast<LoadOp>(op)) {
        ++loadCount;
        int64_t width = staticElementCount(load.getResult().getType());
        maxLoadWidth = std::max(maxLoadWidth, width);
        inputBytes += staticValueBytes(load.getResult());
        if (runtimeActiveExtentArgumentIndex >= 0) {
          auto fact = runtimeActiveExtentFromMask(load.getMask(), *entry);
          if (fact && fact->argumentIndex == runtimeActiveExtentArgumentIndex &&
              fact->containerWidth == runtimeActiveExtentContainerWidth) {
            Type element = getElementTypeOrSelf(load.getResult().getType());
            activeInputBytesPerElement +=
                (element.getIntOrFloatBitWidth() + 7) / 8;
          }
        }
      } else if (auto store = dyn_cast<StoreOp>(op)) {
        ++storeCount;
        outputBytes += staticValueBytes(store.getValue());
        if (runtimeActiveExtentArgumentIndex >= 0) {
          auto fact = runtimeActiveExtentFromMask(store.getMask(), *entry);
          if (fact && fact->argumentIndex == runtimeActiveExtentArgumentIndex &&
              fact->containerWidth == runtimeActiveExtentContainerWidth) {
            Type element = getElementTypeOrSelf(store.getValue().getType());
            activeOutputBytesPerElement +=
                (element.getIntOrFloatBitWidth() + 7) / 8;
          }
        }
      } else if (isa<AtomicRMWOp, AtomicCASOp>(op)) {
        ++atomicCount;
      } else if (isa<scf::ForOp>(op)) {
        ++loopCount;
      } else if (auto reduce = dyn_cast<ReduceOp>(op)) {
        ++reduceCount;
        for (Value operand : reduce.getSrcs()) {
          arithmeticElements += std::max<int64_t>(1, staticElementCount(operand.getType()) - 1);
          if (runtimeActiveExtentArgumentIndex >= 0 &&
              staticElementCount(operand.getType()) ==
                  runtimeActiveExtentContainerWidth &&
              dependsOnActiveLoad(operand))
            ++activeArithmeticElementsPerElement;
        }
      }
      StringRef name = op->getName().getStringRef();
      if (!isa<arith::ConstantOp>(op) &&
          (name.starts_with("arith.") || name.starts_with("math.")) &&
          op->getNumResults() == 1) {
        Type element = getElementTypeOrSelf(op->getResult(0).getType());
        if (isa<FloatType>(element))
          arithmeticElements += std::max<int64_t>(1, staticElementCount(op->getResult(0).getType()));
        if (runtimeActiveExtentArgumentIndex >= 0 && isa<FloatType>(element) &&
            staticElementCount(op->getResult(0).getType()) ==
                runtimeActiveExtentContainerWidth &&
            dependsOnActiveLoad(op->getResult(0)))
          ++activeArithmeticElementsPerElement;
      }
      });

    // Recognize the source-independent power-of-two scale-format branch from
    // its exact SSA composition.  NameLoc is used only to bind the runtime
    // value to the surviving argument; classification does not consume the
    // argument name, kernel identity, timing, or a backend descendant.
    int64_t scaleFormatBranchCount = 0;
    std::optional<bool> ue8m0WhenConditionTrue;
    std::optional<int64_t> scaleFormatCondition;
    (*entry).walk([&](scf::IfOp ifOp) {
      auto conditionArgument = dyn_cast<BlockArgument>(ifOp.getCondition());
      auto hasPowerOfTwoScaleChain = [&](Region &region) {
        bool found = false;
        region.walk([&](Operation *operation) {
          if (operation->getName().getStringRef() != "math.exp2" ||
              operation->getNumOperands() != 1)
            return;
          Operation *ceil = operation->getOperand(0).getDefiningOp();
          if (!ceil || ceil->getName().getStringRef() != "math.ceil" ||
              ceil->getNumOperands() != 1)
            return;
          Operation *log2 = ceil->getOperand(0).getDefiningOp();
          if (log2 && log2->getName().getStringRef() == "math.log2")
            found = true;
        });
        return found;
      };
      bool thenHasChain = hasPowerOfTwoScaleChain(ifOp.getThenRegion());
      bool elseHasChain = hasPowerOfTwoScaleChain(ifOp.getElseRegion());
      if (thenHasChain == elseHasChain)
        return;
      ++scaleFormatBranchCount;
      ue8m0WhenConditionTrue = thenHasChain;
      if (!conditionArgument)
        return;
      if (auto bound = (*entry).getArgAttrOfType<IntegerAttr>(
              conditionArgument.getArgNumber(), kBridgeBoundScalarAttr))
        scaleFormatCondition = bound.getInt();
    });
    std::string scaleFormatMode = "not_applicable";
    if (scaleFormatBranchCount > 1) {
      scaleFormatMode = "ambiguous";
    } else if (scaleFormatBranchCount == 1) {
      if (!scaleFormatCondition || !ue8m0WhenConditionTrue ||
          (*scaleFormatCondition != 0 && *scaleFormatCondition != 1)) {
        scaleFormatMode = "unbound";
      } else {
        bool selectedUE8M0 =
            ((*scaleFormatCondition == 1) == *ue8m0WhenConditionTrue);
        scaleFormatMode = selectedUE8M0 ? "ue8m0_scale" : "fp32_scale";
      }
    }

    if (runtimeActiveExtentArgumentIndex >= 0) {
      activeInputFixedBytes = inputBytes -
          activeInputBytesPerElement * runtimeActiveExtentContainerWidth;
      activeOutputFixedBytes = outputBytes -
          activeOutputBytesPerElement * runtimeActiveExtentContainerWidth;
      activeArithmeticFixedElements = arithmeticElements -
          activeArithmeticElementsPerElement *
              runtimeActiveExtentContainerWidth;
      if (activeInputFixedBytes < 0 || activeOutputFixedBytes < 0 ||
          activeArithmeticFixedElements < 0) {
        runtimeActiveExtentArgumentIndex = -1;
        runtimeActiveExtentContainerWidth = 0;
        activeInputBytesPerElement = 0;
        activeOutputBytesPerElement = 0;
        activeArithmeticElementsPerElement = 0;
        activeInputFixedBytes = 0;
        activeOutputFixedBytes = 0;
        activeArithmeticFixedElements = 0;
      }
    }

    for (FuncOp function : reachableFunctions)
      function.walk([&](scf::ForOp loop) {
        LoopDependenceCertificate affineRuntime =
            certifyAffineRuntimeOrderPreserving(loop);
        LoopDependenceCertificate pipelineCapability =
            certifyNativeDynamicPipelineSubject(loop);
        SmallVector<Operation *> censusLoadSlice;
        Operation *censusCompute = nullptr;
        Operation *censusReduce = nullptr;
        // Route coverage is deliberately evaluated without a static-trip
        // requirement.  Triton's native unroller owns a factor-wide main
        // loop and ordered remainder for runtime bounds, so rejecting a loop
        // merely because its upper bound is dynamic would be a shared shape-
        // capability gap rather than a route-specific semantic decision.
        bool genericUnrollShape = isEligibleLoop(
            loop, censusLoadSlice, censusCompute, censusReduce,
            /*requireStaticTripCount=*/false);
        LoopDependenceCertificate reorderCapability;
        if (genericUnrollShape) {
          auto load = dyn_cast<LoadOp>(censusLoadSlice.back());
          if (load)
            reorderCapability = certifyExistingUnrollReorder(
                loop, load, censusCompute, censusReduce);
        }
        if (!reorderCapability.safe) {
          SmallVector<Operation *> generalLoads, generalComputes,
              generalState;
          reorderCapability = certifyOrderPreservingReadExposure(
              loop, generalLoads, generalComputes, generalState,
              /*requireVectorizableTensorLoad=*/false);
        }
        bool exactPrefixVectorization =
            static_cast<bool>(matchExactPrefixReduction(loop));
        bool logicalGroupVectorization =
            reorderCapability.safe && censusCompute &&
            censusCompute->getNumResults() == 1 &&
            isRankOneIntegerVector(censusCompute->getResult(0));
        LoopDependenceCertificate loadVectorizationCapability;
        if (!exactPrefixVectorization && !logicalGroupVectorization) {
          SmallVector<Operation *> vectorLoads, vectorComputes, vectorState;
          loadVectorizationCapability = certifyOrderPreservingReadExposure(
              loop, vectorLoads, vectorComputes, vectorState,
              /*requireVectorizableTensorLoad=*/true);
        }
        bool vectorizationCapability =
            exactPrefixVectorization || logicalGroupVectorization ||
            loadVectorizationCapability.safe;
        std::string reorderReason = reorderCapability.safe
            ? ""
            : reorderCapability.reason;
        std::string vectorizationReason = vectorizationCapability
            ? ""
            : loadVectorizationCapability.reason.empty()
                  ? reorderReason
                  : loadVectorizationCapability.reason;
        llvm::json::Object census;
        census["locator"] = planningCutLoopLocators.lookup(
            loop.getOperation());
        census["entry_owned"] = function == *entry;
        std::string location;
        llvm::raw_string_ostream locationStream(location);
        loop.getLoc().print(locationStream);
        locationStream.flush();
        census["source_location"] = location;
        int64_t nestingDepth = 0;
        for (Operation *parent = loop->getParentOp(); parent;
             parent = parent->getParentOp())
          nestingDepth += isa<scf::ForOp>(parent);
        census["nesting_depth"] = nestingDepth;
        auto parentLoop = loop->getParentOfType<scf::ForOp>();
        census["parent_locator"] =
            parentLoop
                ? planningCutLoopLocators.lookup(parentLoop.getOperation())
                : "";
        census["lower_kind"] =
            matchPattern(loop.getLowerBound(), m_Constant())
                ? "static"
                : "runtime";
        census["upper_kind"] =
            matchPattern(loop.getUpperBound(), m_Constant())
                ? "static"
                : "runtime";
        census["step_kind"] =
            matchPattern(loop.getStep(), m_Constant())
                ? "static"
                : "runtime";
        auto censusTripCount = exactStaticTripCount(loop);
        census["exact_static_trip_count"] =
            censusTripCount ? *censusTripCount : 0;
        census["carried_value_count"] =
            static_cast<int64_t>(loop.getInitArgs().size());
        census["native_pipeline_capable"] = pipelineCapability.safe;
        census["native_pipeline_capability_certificate"] =
            pipelineCapability.kind;
        census["native_pipeline_capability_reason"] =
            pipelineCapability.reason;
        census["full_unroll_reorder_capable"] =
            reorderCapability.safe;
        census["full_unroll_reorder_capability_certificate"] =
            reorderCapability.kind;
        census["full_unroll_reorder_capability_reason"] = reorderReason;
        census["full_unroll_vectorization_capable"] =
            vectorizationCapability;
        census["full_unroll_vectorization_capability_certificate"] =
            exactPrefixVectorization
                ? "predicated_exact_prefix_reduction_v1"
                : (logicalGroupVectorization
                       ? "existing_rank1_integer_exact_grouping_v2"
                       : loadVectorizationCapability.safe
                             ? loadVectorizationCapability.kind
                             : "");
        census["full_unroll_vectorization_capability_reason"] =
            vectorizationReason;
        LoopDependenceCertificate nestedCapability =
            certifyNestedInnerDimension(loop);
        census["nested_inner_dimension_capable"] =
            nestedCapability.safe;
        census["nested_inner_dimension_certificate"] =
            nestedCapability.kind;
        census["nested_inner_dimension_reason"] =
            nestedCapability.reason;
        census["runtime_main_tail_certificate"] =
            !censusTripCount &&
                    (reorderCapability.safe || vectorizationCapability)
                ? "native_dynamic_unroll_main_ordered_remainder_v1"
                : "";
        census["affine_runtime_grid_stride"] = affineRuntime.safe;
        census["provider_disposition"] =
            affineRuntime.safe || reorderCapability.safe ||
                    exactPrefixVectorization
                ? "recognized_eligible"
                : "recognized_typed_unsupported";
        census["provider_certificate"] =
            affineRuntime.safe ? affineRuntime.kind
            : reorderCapability.safe ? reorderCapability.kind
            : exactPrefixVectorization
                ? "predicated_exact_prefix_reduction_v1"
                : "";
        census["ordered_contract"] =
            orderedOwnedIterationCapabilities(affineRuntime);
        census["typed_reason"] =
            affineRuntime.safe || reorderCapability.safe ||
                    exactPrefixVectorization
                ? ""
                : (affineRuntime.reason.empty()
                       ? "observable_loop_has_no_supported_owned_iteration_contract"
                       : affineRuntime.reason);
        auto explicitStages =
            loop->getAttrOfType<IntegerAttr>("tt.num_stages");
        census["resolved_num_stages"] =
            explicitStages ? explicitStages.getInt() : compilerDefaultStages;
        loopCensus.push_back(std::move(census));

        if (affineRuntime.safe) {
          ++affineRuntimeCandidateCount;
          if (affineRuntime.kind ==
              "affine_runtime_grid_stride_order_preserving_v2")
            ++affineRuntimeGridStrideCount;
          affineRuntimeCertificate = affineRuntime;
          int64_t loopInputBytes = 0;
          loop.walk([&](LoadOp load) {
            loopInputBytes += staticValueBytes(load.getResult());
          });
          nativeLoopInputBytesPerIteration = loopInputBytes;
        }

        if (!loop->getParentOfType<scf::ForOp>()) {
          auto certificate =
              certifyIndependentIterationExactInnerReduction(loop);
          if (certificate.safe) {
            ++independentInnerReductionCount;
            independentInnerReductionTripCounts.push_back(
                *exactStaticTripCount(loop));
            independentInnerReductionInputStrides.push_back(
                independentIterationAddressStride(loop, false).value_or(0));
            independentInnerReductionOutputStrides.push_back(
                independentIterationAddressStride(loop, true).value_or(0));
            independentInnerReductionWidths.push_back(
                independentInnerReductionWidth(loop));
            int64_t bodyOperationCount = 0;
            int64_t bodyTypeVolume = 0;
            loop.getBody()->walk([&](Operation *operation) {
              if (operation->hasTrait<OpTrait::IsTerminator>())
                return;
              ++bodyOperationCount;
              int64_t operationTypeVolume = 1;
              for (Value operand : operation->getOperands())
                operationTypeVolume = std::max(
                    operationTypeVolume,
                    staticElementCount(operand.getType()));
              for (Value result : operation->getResults())
                operationTypeVolume = std::max(
                    operationTypeVolume,
                    staticElementCount(result.getType()));
              bodyTypeVolume += operationTypeVolume;
            });
            independentInnerReductionBodyOperationCounts.push_back(
                std::max<int64_t>(1, bodyOperationCount));
            independentInnerReductionBodyTypeVolumes.push_back(
                std::max<int64_t>(1, bodyTypeVolume));
            independentInnerReductionAffineAccesses.push_back(
                independentAffineAccessFacts(loop));
          }
        }
        LoopDependenceCertificate pipelineCertificate =
            certifyNativeDynamicPipelineSubject(loop);
        if (!pipelineCertificate.safe)
          return;
        auto selectedPipelineStages =
            loop->getAttrOfType<IntegerAttr>("tt.num_stages");
        int64_t effectiveStages =
            selectedPipelineStages ? selectedPipelineStages.getInt()
                                   : compilerDefaultStages;
        if (effectiveStages <= 1)
          return;
        ++nativePipelineScopeCount;
        if (nativePipelineScopeCount == 1) {
          nativeDefaultStages = effectiveStages;
          APInt lower, upper, step;
          if (matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) &&
              matchPattern(loop.getUpperBound(), m_ConstantInt(&upper)) &&
              matchPattern(loop.getStep(), m_ConstantInt(&step)) &&
              step.getSExtValue() != 0)
            nativeTripCount =
                (upper.getSExtValue() - lower.getSExtValue()) /
                step.getSExtValue();
        } else {
          nativeDefaultStages = 0;
          nativeTripCount = 0;
        }
        ++nativePipelineCandidateCount;
        nativeCertificate = pipelineCertificate;
      });

    bool bridgeLegal = false;
    bool bridgePartitionLegal = false;
    std::string bridgeCertificate;
    std::string bridgeRejection = "bridge discovery is unavailable";
    std::string bridgePartitionCertificate;
    if (auto raw = module->getAttrOfType<StringAttr>(kBridgeDiscoveryAttr)) {
      auto parsed = llvm::json::parse(raw.getValue());
      auto *object = parsed ? parsed->getAsObject() : nullptr;
      if (object && object->getString("schema") ==
                        "triton.loop-bridge.discovery.v1" &&
          object->getBoolean("extractable").value_or(false)) {
        bridgeLegal = object->getBoolean("construction_legal").value_or(false);
        bridgePartitionLegal =
            object->getBoolean("partition_recurrence_legal").value_or(false);
        bridgeCertificate = std::string(
            object->getString("dependence_certificate").value_or(""));
        bridgeRejection = std::string(
            object->getString("rejection_reason").value_or(""));
        bridgePartitionCertificate = std::string(
            object->getString("partition_recurrence_certificate")
                .value_or(""));
      }
    }

    llvm::json::Object facts;
    facts["schema"] = "hbv.loop.static-facts.v12";
    facts["extractable"] = true;
    facts["bridge_legal"] = bridgeLegal;
    facts["bridge_certificate"] = bridgeCertificate;
    facts["bridge_rejection_reason"] = bridgeRejection;
    facts["bridge_partition_recurrence_legal"] =
        bridgePartitionLegal;
    facts["bridge_partition_recurrence_certificate"] =
        bridgePartitionCertificate;
    int64_t bridgeSubjectLoopCount = 0;
    bool bridgePipelineBodyInline = false;
    (*entry).walk([&](scf::ForOp loop) {
      if (!loop->hasAttr(kBridgeSubjectAttr))
        return;
      ++bridgeSubjectLoopCount;
      bool hasDirectMemoryService = false;
      bool hasHiddenCall = false;
      for (Operation &operation : loop.getBody()->without_terminator()) {
        hasDirectMemoryService |= isa<LoadOp, StoreOp>(operation);
        hasHiddenCall |= isa<CallOp>(operation);
      }
      bridgePipelineBodyInline = hasDirectMemoryService && !hasHiddenCall;
    });
    bridgePipelineBodyInline &= bridgeSubjectLoopCount == 1;
    facts["bridge_pipeline_body_inline"] = bridgePipelineBodyInline;
    facts["bridge_pipeline_body_certificate"] =
        bridgePipelineBodyInline
            ? "bridge_program_body_direct_service_in_constructed_loop_v2"
            : "bridge_program_body_hidden_behind_helper_call_v1";
    std::string bridgeCFGPredicationCertificate;
    if (auto certificate =
            (*entry)->getAttrOfType<StringAttr>(kBridgeCFGPredicationAttr))
      bridgeCFGPredicationCertificate = certificate.getValue().str();
    facts["bridge_cfg_predication_certificate"] =
        bridgeCFGPredicationCertificate;
    std::string bridgeCFGPredicationRejection;
    if (auto rejection = (*entry)->getAttrOfType<StringAttr>(
            kBridgeCFGPredicationRejectionAttr))
      bridgeCFGPredicationRejection = rejection.getValue().str();
    facts["bridge_cfg_predication_rejection_reason"] =
        bridgeCFGPredicationRejection;
    facts["axis_zero_program_id_count"] = static_cast<int64_t>(xPids.size());
    facts["load_count"] = loadCount;
    facts["store_count"] = storeCount;
    facts["atomic_count"] = atomicCount;
    facts["native_loop_count"] = loopCount;
    facts["reduction_count"] = reduceCount;
    facts["group_width"] = maxLoadWidth;
    facts["input_bytes_per_logical_program"] = inputBytes;
    facts["output_bytes_per_logical_program"] = outputBytes;
    facts["arithmetic_elements_per_logical_program"] = arithmeticElements;
    facts["scale_format_mode"] = scaleFormatMode;
    llvm::json::Array typeSignature;
    for (const std::string &type : bodyTypeSignature)
      typeSignature.push_back(type);
    facts["body_type_signature"] = std::move(typeSignature);
    llvm::json::Object specialFunctionSignature;
    for (const auto &[name, count] : bodySpecialFunctionCounts)
      specialFunctionSignature[name] = count;
    facts["body_special_function_counts"] =
        std::move(specialFunctionSignature);
    bool tensorLaneFusionEligible = false;
    int64_t exactGroupedValueRoundTripCount = 0;
    (*entry).walk([&](LoadOp load) {
      tensorLaneFusionEligible |=
          isa<RankedTensorType>(load.getPtr().getType()) &&
          load.getBoundaryCheck().empty() && !load.getPadding();
      // This is the exact pre-materialization counterpart of the
      // split-tree recovery in materializeBridgeLogical: one tensor load
      // result is consumed directly and exclusively as a tensor store value.
      // Cloning independent virtual programs therefore creates a split tree
      // whose complete ordered leaves meet at the same grouped terminal.
      // No operator, shape, factor or performance identity is involved.
      if (!isa<RankedTensorType>(load.getResult().getType()) ||
          !load.getBoundaryCheck().empty() || load.getPadding() ||
          !load.getResult().hasOneUse())
        return;
      Operation *user = *load.getResult().getUsers().begin();
      auto store = dyn_cast<StoreOp>(user);
      if (store && store.getValue() == load.getResult() &&
          isa<RankedTensorType>(store.getPtr().getType()) &&
          store.getBoundaryCheck().empty())
        ++exactGroupedValueRoundTripCount;
    });
    facts["tensor_lane_fusion_eligible"] = tensorLaneFusionEligible;
    facts["exact_grouped_value_round_trip_count"] =
        exactGroupedValueRoundTripCount;
    llvm::json::Array dependencyOperationCounts;
    llvm::json::Array dependencyTypeVolumes;
    llvm::json::Array dependencyHoistableCounts;
    for (unsigned mask = 0; mask < 8; ++mask) {
      dependencyOperationCounts.push_back(
          operationCountByProgramDependencyMask[mask]);
      dependencyTypeVolumes.push_back(
          typeVolumeByProgramDependencyMask[mask]);
      dependencyHoistableCounts.push_back(
          hoistableOperationCountByProgramDependencyMask[mask]);
    }
    facts["program_dependency_mask_operation_counts"] =
        std::move(dependencyOperationCounts);
    facts["program_dependency_mask_type_volumes"] =
        std::move(dependencyTypeVolumes);
    facts["program_dependency_mask_hoistable_operation_counts"] =
        std::move(dependencyHoistableCounts);
    facts["program_dependency_graph_complete"] =
        programDependencyGraphComplete;
    facts["program_ssa_critical_path_steps"] =
        programDependencyGraphComplete ? programSSACriticalPathSteps : 0;
    llvm::json::Array axisIndependence;
    llvm::json::Array axisIndependenceReasons;
    for (unsigned axis = 0; axis < 3; ++axis) {
      axisIndependence.push_back(programAxisIndependence[axis]);
      axisIndependenceReasons.push_back(
          programAxisIndependenceReasons[axis]);
    }
    facts["program_axis_independence"] = std::move(axisIndependence);
    facts["program_axis_independence_reasons"] =
        std::move(axisIndependenceReasons);
    facts["runtime_active_extent_argument_index"] =
        runtimeActiveExtentArgumentIndex;
    facts["runtime_active_extent_container_width"] =
        runtimeActiveExtentContainerWidth;
    facts["active_input_bytes_per_element"] = activeInputBytesPerElement;
    facts["active_input_fixed_bytes"] = activeInputFixedBytes;
    facts["active_output_bytes_per_element"] = activeOutputBytesPerElement;
    facts["active_output_fixed_bytes"] = activeOutputFixedBytes;
    facts["active_arithmetic_elements_per_element"] =
        activeArithmeticElementsPerElement;
    facts["active_arithmetic_fixed_elements"] =
        activeArithmeticFixedElements;
    facts["native_pipeline_scope_count"] = nativePipelineScopeCount;
    facts["native_pipeline_candidate_count"] = nativePipelineCandidateCount;
    facts["affine_runtime_grid_stride_count"] =
        affineRuntimeGridStrideCount;
    facts["affine_runtime_candidate_count"] =
        affineRuntimeCandidateCount;
    facts["affine_runtime_certificate"] =
        affineRuntimeCandidateCount == 1
            ? affineRuntimeCertificate.kind
            : "";
    facts["ordered_affine_runtime_contract"] =
        affineRuntimeCandidateCount == 1
            ? llvm::json::Value(
                  orderedOwnedIterationCapabilities(affineRuntimeCertificate))
            : llvm::json::Value(llvm::json::Object());
    facts["native_loop_input_bytes_per_iteration"] =
        nativeLoopInputBytesPerIteration;
    facts["native_pipeline_unroll_legal"] =
        nativePipelineScopeCount == 1 && nativePipelineCandidateCount == 1 &&
        nativeCertificate.safe;
    facts["native_pipeline_certificate"] = nativeCertificate.kind;
    facts["native_trip_count"] = nativeTripCount;
    facts["native_default_stages"] = nativeDefaultStages;
    facts["loop_census"] = std::move(loopCensus);
    facts["exact_prefix_reduction_count"] = exactPrefixReductionCount;
    facts["exact_prefix_reduction_maximum_width"] =
        exactPrefixMaximumWidth;
    facts["exact_prefix_reduction_maximum_active_extent"] =
        exactPrefixMaximumActiveExtent;
    facts["exact_prefix_reduction_maximum_element_bytes"] =
        exactPrefixMaximumElementBytes;
    facts["exact_prefix_reduction_certificate"] =
        exactPrefixReductionCount > 0
            ? "predicated_exact_prefix_reduction_v1"
            : "";
    facts["state_axis_sibling_group_count"] =
        stateAxisNormalizationCount;
    facts["state_axis_sibling_group_minimum_cardinality"] =
        stateAxisMinimumCardinality;
    facts["state_axis_sibling_group_maximum_cardinality"] =
        stateAxisMaximumCardinality;
    facts["state_axis_sibling_group_maximum_padded_cardinality"] =
        stateAxisMaximumPaddedCardinality;
    facts["state_axis_sibling_group_minimum_width"] =
        stateAxisMinimumWidth;
    facts["state_axis_sibling_group_maximum_width"] =
        stateAxisMaximumWidth;
    facts["state_axis_sibling_group_minimum_element_bytes"] =
        stateAxisMinimumElementBytes;
    facts["state_axis_sibling_group_maximum_element_bytes"] =
        stateAxisMaximumElementBytes;
    facts["state_axis_sibling_group_uniform_shape"] =
        stateAxisNormalizationCount > 0 &&
        stateAxisMinimumCardinality == stateAxisMaximumCardinality &&
        stateAxisMinimumWidth == stateAxisMaximumWidth &&
        stateAxisMinimumElementBytes == stateAxisMaximumElementBytes;
    facts["state_axis_graph_schema"] =
        "hbv.loop.state-axis-sibling-graph.v1";
    facts["state_axis_backend_adapter_ref"] =
        stateAxisNormalizationCount > 0
            ? "hbv.loop.state-axis.backend.ttir.v1"
            : "";
    llvm::json::Array stateAxisGroups;
    for (auto [ordinal, subject] : llvm::enumerate(stateAxisSubjects)) {
      llvm::json::Object group;
      group["ordinal"] = static_cast<int64_t>(ordinal);
      group["state_cardinality"] = subject.stateCardinality;
      group["padded_state_cardinality"] =
          subject.paddedStateCardinality;
      group["state_width"] = subject.stateWidth;
      group["element_bytes"] = subject.elementBytes;
      group["lane_operand_index"] = subject.laneOperandIndex;
      group["packable_node_count"] = subject.packableNodeCount;
      group["graph_signature"] = subject.graphSignature;
      llvm::json::Array capabilities;
      for (const std::string &capability : subject.operationCapabilities)
        capabilities.push_back(capability);
      group["operation_capabilities"] = std::move(capabilities);
      stateAxisGroups.push_back(std::move(group));
    }
    facts["state_axis_sibling_groups"] = std::move(stateAxisGroups);
    facts["state_axis_sibling_group_certificate"] =
        stateAxisNormalizationCount > 0
            ? "provider_closed_sibling_state_operation_graph_v1"
            : "";
    facts["independent_inner_reduction_count"] =
        independentInnerReductionCount;
    llvm::json::Array independentTrips;
    for (int64_t trip : independentInnerReductionTripCounts)
      independentTrips.push_back(trip);
    facts["independent_inner_reduction_trip_counts"] =
        std::move(independentTrips);
    llvm::json::Array independentInputStrides;
    for (int64_t stride : independentInnerReductionInputStrides)
      independentInputStrides.push_back(stride);
    facts["independent_inner_reduction_input_iteration_strides"] =
        std::move(independentInputStrides);
    llvm::json::Array independentOutputStrides;
    for (int64_t stride : independentInnerReductionOutputStrides)
      independentOutputStrides.push_back(stride);
    facts["independent_inner_reduction_output_iteration_strides"] =
        std::move(independentOutputStrides);
    llvm::json::Array independentWidths;
    for (int64_t width : independentInnerReductionWidths)
      independentWidths.push_back(width);
    facts["independent_inner_reduction_widths"] =
        std::move(independentWidths);
    llvm::json::Array independentBodyOperationCounts;
    for (int64_t count : independentInnerReductionBodyOperationCounts)
      independentBodyOperationCounts.push_back(count);
    facts["independent_inner_reduction_body_operation_counts"] =
        std::move(independentBodyOperationCounts);
    llvm::json::Array independentBodyTypeVolumes;
    for (int64_t volume : independentInnerReductionBodyTypeVolumes)
      independentBodyTypeVolumes.push_back(volume);
    facts["independent_inner_reduction_body_type_volumes"] =
        std::move(independentBodyTypeVolumes);
    llvm::json::Array independentAffineAccesses;
    for (const auto &[reductionIndex, accesses] :
         llvm::enumerate(independentInnerReductionAffineAccesses)) {
      for (const IndependentAffineAccessFact &access : accesses) {
        llvm::json::Object item;
        item["reduction_index"] = static_cast<int64_t>(reductionIndex);
        item["effect"] = access.write ? "write" : "read";
        item["root_argument"] = access.rootArgument;
        item["element_bytes"] = access.elementBytes;
        item["lane_count"] = access.laneCount;
        item["iteration_stride"] = access.iterationStride;
        llvm::json::Array programStrides;
        for (int64_t stride : access.programAxisStrides)
          programStrides.push_back(stride);
        item["program_axis_strides"] = std::move(programStrides);
        llvm::json::Array programDependencies;
        llvm::json::Array programAffineComplete;
        for (bool value : access.programAxisDependencies)
          programDependencies.push_back(value);
        for (bool value : access.programAxisAffineComplete)
          programAffineComplete.push_back(value);
        item["program_axis_dependencies"] =
            std::move(programDependencies);
        item["program_axis_affine_complete"] =
            std::move(programAffineComplete);
        item["local_minimum"] = access.localMinimum;
        item["local_maximum"] = access.localMaximum;
        item["masked"] = access.masked;
        item["complete"] = access.complete;
        independentAffineAccesses.push_back(std::move(item));
      }
    }
    facts["independent_inner_reduction_affine_accesses"] =
        std::move(independentAffineAccesses);
    facts["independent_inner_reduction_certificate"] =
        independentInnerReductionCount > 0
            ? "independent_iteration_exact_inner_reduction_v1"
            : "";
    llvm::json::Array programAccesses;
    for (const IndependentAffineAccessFact &access : programAffineAccesses) {
      llvm::json::Object item;
      item["effect"] = access.write ? "write" : "read";
      item["root_argument"] = access.rootArgument;
      item["element_bytes"] = access.elementBytes;
      item["lane_count"] = access.laneCount;
      llvm::json::Array programStrides;
      llvm::json::Array programDependencies;
      llvm::json::Array programAffineComplete;
      for (int64_t stride : access.programAxisStrides)
        programStrides.push_back(stride);
      for (bool value : access.programAxisDependencies)
        programDependencies.push_back(value);
      for (bool value : access.programAxisAffineComplete)
        programAffineComplete.push_back(value);
      item["program_axis_strides"] = std::move(programStrides);
      item["program_axis_dependencies"] =
          std::move(programDependencies);
      item["program_axis_affine_complete"] =
          std::move(programAffineComplete);
      item["local_minimum"] = access.localMinimum;
      item["local_maximum"] = access.localMaximum;
      item["masked"] = access.masked;
      item["complete"] = access.complete;
      programAccesses.push_back(std::move(item));
    }
    facts["program_affine_accesses"] = std::move(programAccesses);
    llvm::json::Array runtimeMaskScalarFacts;
    for (const RuntimeMaskScalarFact &fact : runtimeMaskScalars) {
      llvm::json::Object item;
      item["effect"] = fact.write ? "write" : "read";
      item["argument_index"] = fact.argumentIndex;
      item["container_width"] = fact.containerWidth;
      item["mask_element_count"] = fact.maskElementCount;
      item["replication_factor"] = fact.replicationFactor;
      item["bound_value"] = fact.boundValue;
      item["divisibility"] = fact.divisibility;
      item["bound_value_known"] = fact.boundValueKnown;
      item["complete"] = fact.complete;
      runtimeMaskScalarFacts.push_back(std::move(item));
    }
    facts["runtime_mask_scalars"] = std::move(runtimeMaskScalarFacts);
    llvm::json::Array mixedRadixAccessFacts;
    for (const ProgramMixedRadixAccessFact &fact :
         programMixedRadixAccesses) {
      llvm::json::Object item;
      item["effect"] = fact.write ? "write" : "read";
      item["root_argument"] = fact.rootArgument;
      item["element_bytes"] = fact.elementBytes;
      item["lane_count"] = fact.laneCount;
      item["program_axis"] = fact.programAxis;
      item["divisor"] = fact.divisor;
      item["quotient_stride"] = fact.quotientStride;
      item["remainder_stride"] = fact.remainderStride;
      item["local_minimum"] = fact.localMinimum;
      item["local_maximum"] = fact.localMaximum;
      item["masked"] = fact.masked;
      item["complete"] = fact.complete;
      mixedRadixAccessFacts.push_back(std::move(item));
    }
    facts["program_mixed_radix_accesses"] =
        std::move(mixedRadixAccessFacts);
    std::string serialized;
    llvm::raw_string_ostream stream(serialized);
    stream << llvm::json::Value(std::move(facts));
    stream.flush();
    module->setAttr(kStaticFactsAttr,
                    StringAttr::get(module.getContext(), serialized));
    module->removeAttr(kNativeDefaultStagesAttr);
  }
};

class LoopBridgeProgramCoarseningPass
    : public impl::TritonLoopBridgeProgramCoarseningBase<
          LoopBridgeProgramCoarseningPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto factorAttr = module->getAttrOfType<IntegerAttr>(kBridgeFactorAttr);
    if (!factorAttr || factorAttr.getInt() == 1)
      return;
    int64_t factor = factorAttr.getInt();
    if (factor < 2 || !llvm::isPowerOf2_64(factor)) {
      reportFailure(module, "Bridge factor must be a power of two of at least 2");
      return signalPassFailure();
    }
    SmallVector<int64_t, 3> requestedDivisors{factor, 1, 1};
    if (auto raw = module->getAttrOfType<StringAttr>(
            kBridgeRequestedDivisorsAttr)) {
      auto parsed = llvm::json::parse(raw.getValue());
      auto *array = parsed ? parsed->getAsArray() : nullptr;
      if (!array || array->size() != 3) {
        reportFailure(module, "Bridge divisor vector is malformed");
        return signalPassFailure();
      }
      requestedDivisors.clear();
      for (const llvm::json::Value &value : *array) {
        auto divisor = value.getAsInteger();
        if (!divisor || *divisor < 1 ||
            !llvm::isPowerOf2_64(*divisor)) {
          reportFailure(module, "Bridge divisor vector is not positive power-of-two");
          return signalPassFailure();
        }
        requestedDivisors.push_back(*divisor);
      }
    }
    bool staticPartitionRequested = false;
    bool axisVectorContract = false;
    SmallVector<int64_t, 3> contractAxisDivisors;
    if (auto bundle = module->getAttrOfType<StringAttr>(kBundleAttr)) {
      std::string planReason;
      auto plan = parseLoopPlan(bundle.getValue(), planReason);
      if (failed(plan)) {
        reportFailure(module, planReason);
        return signalPassFailure();
      }
      staticPartitionRequested = plan->bridgeStaticPartition;
      axisVectorContract = plan->bridgeAxisVector;
      contractAxisDivisors = plan->bridgeAxisDivisors;
    }
    auto entry = findEntryPoint(module);
    if (failed(entry)) {
      reportFailure(module, "Bridge requires one public kernel");
      return signalPassFailure();
    }
    unsigned requestedGroupedAxes = llvm::count_if(
        requestedDivisors, [](int64_t divisor) { return divisor > 1; });
    bool requestedGenericAxisGrouping =
        requestedGroupedAxes > 1 || requestedDivisors[1] > 1 ||
        requestedDivisors[2] > 1;
    if (requestedGenericAxisGrouping &&
        (!axisVectorContract ||
         contractAxisDivisors != requestedDivisors)) {
      reportFailure(
          module,
          "multi-axis Bridge divisors are not bound by the selected route contract");
      return signalPassFailure();
    }
    // Bridge owns only program-to-loop construction.  Route-local
    // transformations (including exact-prefix vectorization) must be selected
    // and materialized by their own PlanContract.  Performing one here would
    // make Bridge admission depend on an unrelated loop-body fingerprint and
    // would destroy the causal boundary between the two interventions.
    auto setGridDivisor = [&](unsigned axis, int64_t divisor,
                              OpBuilder &builder) {
      StringRef attribute = axis == 0 ? StringRef(kBridgeGridDivisorAttr)
                            : axis == 1 ? StringRef(kBridgeGridDivisorYAttr)
                                        : StringRef(kBridgeGridDivisorZAttr);
      module->setAttr(attribute, builder.getI32IntegerAttr(divisor));
    };
    SmallVector<GetProgramIdOp> sourceProgramIds;
    (*entry).walk([&](GetProgramIdOp pid) {
      sourceProgramIds.push_back(pid);
    });
    SmallVector<GetProgramIdOp> allProgramIds;
    for (GetProgramIdOp pid : sourceProgramIds)
      if (requestedDivisors[pid.getAxisAsInt()] > 1)
        allProgramIds.push_back(pid);
    for (unsigned axis = 0; axis < requestedDivisors.size(); ++axis)
      if (requestedDivisors[axis] > 1 &&
          llvm::none_of(sourceProgramIds, [&](GetProgramIdOp pid) {
            return pid.getAxisAsInt() == axis;
          })) {
        reportFailure(module,
                      "Bridge divisor names an unused program axis");
        return signalPassFailure();
      }
    llvm::sort(allProgramIds, [](GetProgramIdOp lhs, GetProgramIdOp rhs) {
      return lhs.getAxisAsInt() < rhs.getAxisAsInt();
    });
    bool uniqueProgramAxes = !allProgramIds.empty();
    for (auto pair : llvm::zip(allProgramIds, llvm::drop_begin(allProgramIds)))
      uniqueProgramAxes &=
          std::get<0>(pair).getAxisAsInt() !=
          std::get<1>(pair).getAxisAsInt();
    bool genericAxisGrouping =
        uniqueProgramAxes &&
        (allProgramIds.size() > 1 ||
         allProgramIds.front().getAxisAsInt() != 0);
    int64_t groupedProgramCount = 1;
    for (GetProgramIdOp pid : allProgramIds) {
      int64_t divisor = requestedDivisors[pid.getAxisAsInt()];
      if (groupedProgramCount > std::numeric_limits<int64_t>::max() /
                                    divisor) {
        reportFailure(module, "multi-axis Bridge factor product overflows i64");
        return signalPassFailure();
      }
      groupedProgramCount *= divisor;
    }
    if (groupedProgramCount != factor) {
      reportFailure(module,
                    "Bridge divisor product disagrees with grouped-program factor");
      return signalPassFailure();
    }
    if (genericAxisGrouping && staticPartitionRequested) {
      reportFailure(
          module,
          "multi-axis Bridge does not claim signed partition recurrence");
      return signalPassFailure();
    }
    // Use one helper-call materialization for every generic axis vector.  The
    // former flat-body clone path inserted rewritten PID expressions beside
    // still-live ungrouped PID expressions and could violate SSA dominance.
    // A private helper makes the complete program body the unit of cloning:
    // grouped PIDs are explicit arguments, while ungrouped PIDs and all body
    // dependencies retain their original dominance inside the helper.  This
    // is a structural rule for every generic axis vector, not a case branch.
    if (genericAxisGrouping) {
      SmallVector<unsigned> groupedAxes;
      for (GetProgramIdOp pid : allProgramIds)
        groupedAxes.push_back(pid.getAxisAsInt());
      for (GetProgramIdOp pid : allProgramIds) {
        LoopDependenceCertificate axisCertificate =
            certifyBridgeProgramIndependence(*entry, pid);
        if (!axisCertificate.safe) {
          reportFailure(
              module,
              Twine("multi-axis CFG independence proof failed: ") +
                  axisCertificate.reason);
          return signalPassFailure();
        }
      }
      OpBuilder moduleBuilder(module.getContext());
      FuncOp helper = *entry;
      std::string publicName = helper.getName().str();
      FunctionType publicType = helper.getFunctionType();
      NamedAttrList publicAttrs(helper->getAttrs());
      helper.setName(publicName + "__loop_bridge_multiaxis_body");
      helper.setVisibility(SymbolTable::Visibility::Private);
      SmallVector<Type> helperInputs(helper.getArgumentTypes());
      SmallVector<DictionaryAttr> helperArgAttrs;
      for (unsigned index = 0; index < helper.getNumArguments(); ++index) {
        auto attrs = helper.getArgAttrDict(index);
        helperArgAttrs.push_back(
            attrs ? attrs : DictionaryAttr::get(module.getContext()));
      }
      for (size_t index = 0; index < allProgramIds.size(); ++index) {
        helperInputs.push_back(moduleBuilder.getI32Type());
        helperArgAttrs.push_back(DictionaryAttr::get(module.getContext()));
      }
      helper.setFunctionType(FunctionType::get(
          module.getContext(), helperInputs, helper.getResultTypes()));
      SmallVector<BlockArgument> virtualPidArguments;
      for (size_t index = 0; index < allProgramIds.size(); ++index)
        virtualPidArguments.push_back(
            helper.getBody().front().addArgument(
                moduleBuilder.getI32Type(), helper.getLoc()));
      helper.setAllArgAttrs(helperArgAttrs);
      SmallVector<GetProgramIdOp> helperPids;
      helper.walk([&](GetProgramIdOp pid) {
        if (requestedDivisors[pid.getAxisAsInt()] > 1)
          helperPids.push_back(pid);
      });
      llvm::sort(helperPids, [](GetProgramIdOp lhs, GetProgramIdOp rhs) {
        return lhs.getAxisAsInt() < rhs.getAxisAsInt();
      });
      if (helperPids.size() != virtualPidArguments.size()) {
        reportFailure(module, "multi-axis CFG helper lost a program ID");
        return signalPassFailure();
      }
      for (auto [pid, argument] : llvm::zip(helperPids, virtualPidArguments)) {
        pid.getResult().replaceAllUsesWith(argument);
        pid.erase();
      }
      moduleBuilder.setInsertionPoint(helper);
      FuncOp publicEntry = FuncOp::create(
          moduleBuilder, helper.getLoc(), publicName, publicType);
      publicEntry->setAttrs(publicAttrs);
      Block *publicBlock = publicEntry.addEntryBlock();
      OpBuilder builder = OpBuilder::atBlockBegin(publicBlock);
      Location loc = publicEntry.getLoc();
      SmallVector<Value> basePids;
      for (unsigned axis : groupedAxes) {
        int64_t divisor = requestedDivisors[axis];
        Value physical = GetProgramIdOp::create(
            builder, loc, axis);
        Value factorValue = arith::ConstantIntOp::create(
            builder, loc, divisor, 32);
        basePids.push_back(arith::MulIOp::create(
            builder, loc, physical, factorValue));
      }
      for (int64_t ordinal = 0; ordinal < groupedProgramCount; ++ordinal) {
        int64_t remainder = ordinal;
        SmallVector<Value> callArguments(publicBlock->getArguments());
        for (auto [axis, base] : llvm::zip(groupedAxes, basePids)) {
          int64_t divisor = requestedDivisors[axis];
          int64_t digit = remainder % divisor;
          remainder /= divisor;
          Value virtualPid = base;
          if (digit != 0) {
            Value offset = arith::ConstantIntOp::create(
                builder, loc, digit, 32);
            virtualPid = arith::AddIOp::create(
                builder, loc, base, offset);
          }
          callArguments.push_back(virtualPid);
        }
        auto call = CallOp::create(
            builder, loc, helper.getName(), TypeRange{}, callArguments);
        call->setAttr(kBridgeRoleAttr,
                      builder.getStringAttr("multi_axis_helper_call"));
        call->setAttr(kBridgeOrdinalAttr,
                      builder.getI32IntegerAttr(ordinal));
      }
      ReturnOp::create(builder, loc);
      publicEntry->setAttr(
          kBridgeOriginAttr,
          builder.getStringAttr("bridge_multiaxis_constructed"));
      publicEntry->setAttr(kBridgeFactorAttr,
                           builder.getI32IntegerAttr(factor));
      publicEntry->setAttr(kBridgeCardinalityAttr,
                           builder.getI32IntegerAttr(groupedProgramCount));
      publicEntry->setAttr(
          kDependenceAttr,
          builder.getStringAttr("multi_axis_program_disjoint_v1"));
      for (unsigned axis : groupedAxes)
        setGridDivisor(axis, requestedDivisors[axis], builder);
      return;
    }
    if (genericAxisGrouping && (*entry).getBody().getBlocks().size() == 1) {
      for (GetProgramIdOp pid : allProgramIds) {
        LoopDependenceCertificate axisCertificate =
            certifyBridgeProgramIndependence(*entry, pid);
        if (!axisCertificate.safe) {
          reportFailure(
              module,
              Twine("multi-axis independence proof failed: ") +
                  axisCertificate.reason);
          return signalPassFailure();
        }
      }
      Block &block = (*entry).getBody().front();
      auto returnOp = dyn_cast<ReturnOp>(block.getTerminator());
      if (!returnOp || returnOp.getNumOperands() != 0) {
        reportFailure(module, "multi-axis Bridge requires a void return");
        return signalPassFailure();
      }
      SmallVector<Operation *> bodyOps;
      for (Operation &operation : block.without_terminator())
        if (!isa<GetProgramIdOp>(operation))
          bodyOps.push_back(&operation);
      if (bodyOps.empty()) {
        reportFailure(module, "multi-axis Bridge found no program body");
        return signalPassFailure();
      }
      OpBuilder builder(bodyOps.front());
      Location loc = (*entry).getLoc();
      SmallVector<Value> basePids;
      SmallVector<unsigned> groupedAxes;
      for (GetProgramIdOp pid : allProgramIds) {
        unsigned axis = pid.getAxisAsInt();
        groupedAxes.push_back(axis);
        Value physicalPid = GetProgramIdOp::create(builder, loc, axis);
        Value factorValue = arith::ConstantIntOp::create(
            builder, loc, requestedDivisors[axis], 32);
        basePids.push_back(arith::MulIOp::create(
            builder, loc, physicalPid, factorValue));
      }
      int64_t operationOrdinal = 0;
      for (int64_t ordinal = 0; ordinal < groupedProgramCount; ++ordinal) {
        int64_t remainder = ordinal;
        IRMapping mapping;
        for (auto [pid, base] : llvm::zip(allProgramIds, basePids)) {
          int64_t divisor = requestedDivisors[pid.getAxisAsInt()];
          int64_t digit = remainder % divisor;
          remainder /= divisor;
          Value virtualPid = base;
          if (digit != 0) {
            Value offset = arith::ConstantIntOp::create(
                builder, loc, digit, 32);
            virtualPid = arith::AddIOp::create(
                builder, loc, base, offset);
          }
          mapping.map(pid.getResult(), virtualPid);
        }
        for (Operation *operation : bodyOps) {
          Operation *cloned = builder.clone(*operation, mapping);
          cloned->setAttr(
              kBridgeRoleAttr,
              builder.getStringAttr(
                  isa<LoadOp>(cloned)
                      ? "load"
                      : isa<StoreOp>(cloned) ? "store" : "compute"));
          cloned->setAttr(kBridgeOrdinalAttr,
                          builder.getI32IntegerAttr(operationOrdinal++));
        }
      }
      for (Operation *operation : llvm::reverse(bodyOps))
        operation->erase();
      for (GetProgramIdOp pid : allProgramIds)
        pid.erase();
      (*entry)->setAttr(
          kBridgeOriginAttr,
          builder.getStringAttr("bridge_multiaxis_constructed"));
      (*entry)->setAttr(kBridgeFactorAttr,
                        builder.getI32IntegerAttr(factor));
      (*entry)->setAttr(
          kDependenceAttr,
          builder.getStringAttr("multi_axis_program_disjoint_v1"));
      for (unsigned axis : groupedAxes)
        setGridDivisor(axis, requestedDivisors[axis], builder);
      return;
    }
    // A helper call is a faithful fallback for arbitrary CFG, but it hides
    // the program body's memory stages from every route materializer.  First
    // normalize the one CFG family for which moving the continuation is
    // independently certifiable.  Nonmatching CFG remains intact and is
    // reported below as a hidden-body route capability, not guessed through.
    if ((*entry).getBody().getBlocks().size() != 1) {
      BridgeCFGNormalizationResult normalization =
          normalizeSingleEarlyVoidReturnCFG(*entry);
      if (!normalization.normalized)
        (*entry)->setAttr(
            kBridgeCFGPredicationRejectionAttr,
            StringAttr::get(module.getContext(), normalization.reason));
    }
    if ((*entry).getBody().getBlocks().size() != 1) {
      SmallVector<GetProgramIdOp> sourceXPids;
      (*entry).walk([&](GetProgramIdOp pid) {
        if (pid.getAxisAsInt() == 0)
          sourceXPids.push_back(pid);
      });
      if (sourceXPids.size() != 1) {
        reportFailure(module,
                      "CFG Bridge requires one axis-zero program ID");
        return signalPassFailure();
      }
      LoopDependenceCertificate certificate =
          certifyBridgeProgramIndependence(*entry, sourceXPids.front());
      if (!certificate.safe) {
        reportFailure(module,
                      Twine("CFG Bridge independence proof failed: ") +
                          certificate.reason);
        return signalPassFailure();
      }
      auto directPartition =
          findDirectPidPartition(*entry, sourceXPids.front());
      if (staticPartitionRequested && !directPartition) {
        reportFailure(
            module,
            "CFG static partition recurrence requires one direct signed PID quotient/remainder pair");
        return signalPassFailure();
      }
      bool extentIsEntryArgument = false;
      unsigned extentArgumentNumber = 0;
      Operation *extentConstant = nullptr;
      if (staticPartitionRequested) {
        if (auto argument = dyn_cast<BlockArgument>(directPartition->extent)) {
          extentIsEntryArgument =
              argument.getOwner() == &(*entry).getBody().front();
          extentArgumentNumber = argument.getArgNumber();
        } else if (Operation *definition =
                       directPartition->extent.getDefiningOp();
                   definition && isa<arith::ConstantOp>(definition)) {
          extentConstant = definition;
        }
        if (!extentIsEntryArgument && !extentConstant) {
          reportFailure(
              module,
              "CFG static partition recurrence requires an entry-argument or constant partition extent");
          return signalPassFailure();
        }
      }

      OpBuilder moduleBuilder(module.getContext());
      FuncOp helper = *entry;
      std::string publicName = helper.getName().str();
      FunctionType publicType = helper.getFunctionType();
      NamedAttrList publicAttrs(helper->getAttrs());
      helper.setName(publicName + "__loop_bridge_body");
      helper.setVisibility(SymbolTable::Visibility::Private);
      SmallVector<Type> helperInputs(helper.getArgumentTypes());
      SmallVector<DictionaryAttr> helperArgAttrs;
      helperArgAttrs.reserve(helper.getNumArguments() + 1);
      for (unsigned i = 0; i < helper.getNumArguments(); ++i) {
        auto attrs = helper.getArgAttrDict(i);
        helperArgAttrs.push_back(
            attrs ? attrs : DictionaryAttr::get(module.getContext()));
      }
      helperInputs.push_back(moduleBuilder.getI32Type());
      helperArgAttrs.push_back(DictionaryAttr::get(module.getContext()));
      if (staticPartitionRequested) {
        helperInputs.push_back(moduleBuilder.getI32Type());
        helperInputs.push_back(moduleBuilder.getI32Type());
        helperArgAttrs.push_back(DictionaryAttr::get(module.getContext()));
        helperArgAttrs.push_back(DictionaryAttr::get(module.getContext()));
      }
      helper.setFunctionType(FunctionType::get(
          module.getContext(), helperInputs, helper.getResultTypes()));
      BlockArgument virtualPidArgument = helper.getBody().front().addArgument(
          moduleBuilder.getI32Type(), (*entry).getLoc());
      BlockArgument quotientArgument;
      BlockArgument remainderArgument;
      if (staticPartitionRequested) {
        quotientArgument = helper.getBody().front().addArgument(
            moduleBuilder.getI32Type(), (*entry).getLoc());
        remainderArgument = helper.getBody().front().addArgument(
            moduleBuilder.getI32Type(), (*entry).getLoc());
      }
      helper.setAllArgAttrs(helperArgAttrs);
      SmallVector<GetProgramIdOp> helperXPids;
      helper.walk([&](GetProgramIdOp pid) {
        if (pid.getAxisAsInt() == 0)
          helperXPids.push_back(pid);
      });
      if (helperXPids.size() != 1) {
        reportFailure(module,
                      "CFG Bridge helper lost its unique axis-zero PID");
        return signalPassFailure();
      }
      helperXPids.front().getResult().replaceAllUsesWith(
          virtualPidArgument);
      helperXPids.front().erase();
      if (staticPartitionRequested) {
        directPartition->quotient.getResult().replaceAllUsesWith(
            quotientArgument);
        directPartition->remainder.getResult().replaceAllUsesWith(
            remainderArgument);
        directPartition->quotient.erase();
        directPartition->remainder.erase();
      }
      moduleBuilder.setInsertionPoint(helper);
      FuncOp publicEntry = FuncOp::create(
          moduleBuilder, helper.getLoc(), publicName, publicType);
      publicEntry->setAttrs(publicAttrs);
      Block *publicBlock = publicEntry.addEntryBlock();
      OpBuilder builder = OpBuilder::atBlockBegin(publicBlock);
      Location loc = publicEntry.getLoc();
      Value physicalPid = GetProgramIdOp::create(builder, loc, 0);
      Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
      Value upper = arith::ConstantIntOp::create(
          builder, loc, factor, 32);
      Value one = arith::ConstantIntOp::create(builder, loc, 1, 32);
      Value factorValue = arith::ConstantIntOp::create(
          builder, loc, factor, 32);
      Value basePid = arith::MulIOp::create(
          builder, loc, physicalPid, factorValue);
      if (staticPartitionRequested) {
        Value extent;
        if (extentIsEntryArgument) {
          extent = publicBlock->getArgument(extentArgumentNumber);
        } else {
          IRMapping mapping;
          Operation *cloned = builder.clone(*extentConstant, mapping);
          extent = cloned->getResult(0);
        }
        Value quotient = arith::DivSIOp::create(
            builder, loc, basePid, extent);
        Value remainder = arith::RemSIOp::create(
            builder, loc, basePid, extent);
        Value negativeExtent = arith::SubIOp::create(
            builder, loc, zero, extent);
        Value extentPositive = arith::CmpIOp::create(
            builder, loc, arith::CmpIPredicate::sgt, extent, zero);
        Value absoluteExtent = arith::SelectOp::create(
            builder, loc, extentPositive, extent, negativeExtent);
        Value negativeOne = arith::ConstantIntOp::create(
            builder, loc, -1, 32);
        Value quotientStep = arith::SelectOp::create(
            builder, loc, extentPositive, one, negativeOne);
        for (int64_t cloneIndex = 0; cloneIndex < factor; ++cloneIndex) {
          Value virtualPid = basePid;
          if (cloneIndex != 0) {
            Value offset = arith::ConstantIntOp::create(
                builder, loc, cloneIndex, 32);
            virtualPid = arith::AddIOp::create(
                builder, loc, basePid, offset);
          }
          SmallVector<Value> callArguments(publicBlock->getArguments());
          callArguments.append({virtualPid, quotient, remainder});
          auto call = CallOp::create(
              builder, loc, helper.getName(), TypeRange{}, callArguments);
          call->setAttr(kBridgeRoleAttr,
                        builder.getStringAttr("static_helper_call"));
          call->setAttr(kBridgeOrdinalAttr,
                        builder.getI32IntegerAttr(cloneIndex));
          if (cloneIndex + 1 == factor)
            continue;
          Value nextRaw = arith::AddIOp::create(
              builder, loc, remainder, one);
          Value crossesPartition = arith::CmpIOp::create(
              builder, loc, arith::CmpIPredicate::eq,
              nextRaw, absoluteExtent);
          remainder = arith::SelectOp::create(
              builder, loc, crossesPartition, zero, nextRaw);
          Value steppedQuotient = arith::AddIOp::create(
              builder, loc, quotient, quotientStep);
          quotient = arith::SelectOp::create(
              builder, loc, crossesPartition, steppedQuotient, quotient);
        }
        ReturnOp::create(builder, loc);
        publicEntry->setAttr(
            kBridgeOriginAttr,
            builder.getStringAttr("bridge_static_partition_recurrence"));
        publicEntry->setAttr(kBridgeFactorAttr,
                             builder.getI32IntegerAttr(factor));
        publicEntry->setAttr(kDependenceAttr,
                             builder.getStringAttr(certificate.kind));
        publicEntry->setAttr(
            kBridgePartitionRecurrenceAttr,
            builder.getStringAttr(
                "consecutive_pid_signed_quotient_remainder_recurrence_v1"));
        module->setAttr(kBridgeGridDivisorAttr,
                        builder.getI32IntegerAttr(factor));
        return;
      }
      auto loop = scf::ForOp::create(
          builder, loc, zero, upper, one);
      loop->setAttr(kBridgeSubjectAttr, builder.getUnitAttr());
      loop->setAttr(kBridgeFactorAttr,
                    builder.getI32IntegerAttr(factor));
      OpBuilder bodyBuilder = OpBuilder::atBlockBegin(loop.getBody());
      Value virtualPid = arith::AddIOp::create(
          bodyBuilder, loc, basePid, loop.getInductionVar());
      SmallVector<Value> callArguments(publicBlock->getArguments());
      callArguments.push_back(virtualPid);
      CallOp::create(bodyBuilder, loc, helper.getName(), TypeRange{},
                     callArguments);
      ReturnOp::create(builder, loc);
      publicEntry->setAttr(kBridgeOriginAttr,
                           builder.getStringAttr("bridge_constructed"));
      publicEntry->setAttr(kBridgeFactorAttr,
                           builder.getI32IntegerAttr(factor));
      publicEntry->setAttr(kDependenceAttr,
                           builder.getStringAttr(certificate.kind));
      module->setAttr(kBridgeGridDivisorAttr,
                      builder.getI32IntegerAttr(factor));
      return;
    }
    Block &block = (*entry).getBody().front();
    SmallVector<GetProgramIdOp> xProgramIds;
    bool hasNestedTopLevelControl = false;
    for (Operation &op : block.without_terminator()) {
      if (auto pid = dyn_cast<GetProgramIdOp>(op)) {
        if (pid.getAxisAsInt() == 0)
          xProgramIds.push_back(pid);
      }
      // Other launch axes are invariant with respect to the constructed
      // axis-zero loop. Existing structured loops are cloned as part of the
      // already certified program body; Bridge does not rewrite their
      // internal dependence structure. Triton reductions and scans both own
      // region-local combiner blocks; they are dataflow aggregations inside
      // one logical program, not cross-program control flow.
      if (op.getNumRegions() != 0 &&
          !isa<ReduceOp, ScanOp, scf::IfOp, scf::ForOp>(op))
        hasNestedTopLevelControl = true;
    }
    if (xProgramIds.size() != 1 || hasNestedTopLevelControl) {
      reportFailure(module,
                    "Bridge requires one axis-zero program ID and closed structured control");
      return signalPassFailure();
    }
    GetProgramIdOp physicalPid = xProgramIds.front();
    auto returnOp = dyn_cast<ReturnOp>(block.getTerminator());
    if (!returnOp || returnOp.getNumOperands() != 0) {
      reportFailure(module, "Bridge requires a void Triton kernel return");
      return signalPassFailure();
    }
    SmallVector<Operation *> bodyOps;
    for (Operation *op = physicalPid->getNextNode(); op && op != returnOp;
         op = op->getNextNode())
      bodyOps.push_back(op);
    if (bodyOps.empty()) {
      reportFailure(module, "Bridge found no program body after axis-zero program ID");
      return signalPassFailure();
    }
    LoopDependenceCertificate certificate =
        certifyBridgeProgramIndependence(*entry, physicalPid);
    if (!certificate.safe) {
      reportFailure(module,
                    Twine("Bridge program-independence proof failed: ") +
                        certificate.reason);
      return signalPassFailure();
    }

    auto directPartition = findDirectPidPartition(*entry, physicalPid);
    OpBuilder builder(bodyOps.front());
    Location loc = physicalPid.getLoc();
    Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
    Value upper = arith::ConstantIntOp::create(builder, loc, factor, 32);
    Value one = arith::ConstantIntOp::create(builder, loc, 1, 32);
    Value factorValue = arith::ConstantIntOp::create(builder, loc, factor, 32);
    Value basePid = arith::MulIOp::create(
        builder, loc, physicalPid.getResult(), factorValue);
    if (staticPartitionRequested) {
      if (!directPartition) {
        reportFailure(
            module,
            "static partition recurrence requires one direct signed PID quotient/remainder pair");
        return signalPassFailure();
      }
      Value quotient = arith::DivSIOp::create(
          builder, loc, basePid, directPartition->extent);
      Value remainder = arith::RemSIOp::create(
          builder, loc, basePid, directPartition->extent);
      Value negativeExtent = arith::SubIOp::create(
          builder, loc, zero, directPartition->extent);
      Value extentPositive = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::sgt,
          directPartition->extent, zero);
      Value absoluteExtent = arith::SelectOp::create(
          builder, loc, extentPositive, directPartition->extent,
          negativeExtent);
      Value negativeOne = arith::ConstantIntOp::create(
          builder, loc, -1, 32);
      Value quotientStep = arith::SelectOp::create(
          builder, loc, extentPositive, one, negativeOne);
      int64_t ordinal = 0;
      for (int64_t cloneIndex = 0; cloneIndex < factor; ++cloneIndex) {
        Value virtualPid = basePid;
        if (cloneIndex != 0) {
          Value offset = arith::ConstantIntOp::create(
              builder, loc, cloneIndex, 32);
          virtualPid = arith::AddIOp::create(
              builder, loc, basePid, offset);
        }
        IRMapping mapping;
        mapping.map(physicalPid.getResult(), virtualPid);
        mapping.map(directPartition->quotient.getResult(), quotient);
        mapping.map(directPartition->remainder.getResult(), remainder);
        for (Operation *op : bodyOps) {
          if (op == directPartition->quotient.getOperation() ||
              op == directPartition->remainder.getOperation())
            continue;
          Operation *cloned = builder.clone(*op, mapping);
          cloned->setAttr(
              kBridgeRoleAttr,
              builder.getStringAttr(
                  isa<LoadOp>(cloned)
                      ? "load"
                      : isa<StoreOp>(cloned) ? "store" : "compute"));
          cloned->setAttr(kBridgeOrdinalAttr,
                          builder.getI32IntegerAttr(ordinal++));
        }
        if (cloneIndex + 1 == factor)
          continue;
        Value nextRaw = arith::AddIOp::create(
            builder, loc, remainder, one);
        Value crossesPartition = arith::CmpIOp::create(
            builder, loc, arith::CmpIPredicate::eq, nextRaw,
            absoluteExtent);
        remainder = arith::SelectOp::create(
            builder, loc, crossesPartition, zero, nextRaw);
        Value steppedQuotient = arith::AddIOp::create(
            builder, loc, quotient, quotientStep);
        quotient = arith::SelectOp::create(
            builder, loc, crossesPartition, steppedQuotient, quotient);
      }
      for (Operation *op : llvm::reverse(bodyOps))
        op->erase();
      (*entry)->setAttr(
          kBridgeOriginAttr,
          builder.getStringAttr("bridge_static_partition_recurrence"));
      (*entry)->setAttr(kBridgeFactorAttr,
                        builder.getI32IntegerAttr(factor));
      (*entry)->setAttr(kDependenceAttr,
                        builder.getStringAttr(certificate.kind));
      (*entry)->setAttr(
          kBridgePartitionRecurrenceAttr,
          builder.getStringAttr(
              "consecutive_pid_signed_quotient_remainder_recurrence_v1"));
      module->setAttr(kBridgeGridDivisorAttr,
                      builder.getI32IntegerAttr(factor));
      return;
    }
    SmallVector<Value> initialPartition;
    if (directPartition) {
      initialPartition.push_back(arith::DivSIOp::create(
          builder, loc, basePid, directPartition->extent));
      initialPartition.push_back(arith::RemSIOp::create(
          builder, loc, basePid, directPartition->extent));
    }
    auto loop = scf::ForOp::create(
        builder, loc, zero, upper, one, initialPartition);
    loop->setAttr(kBridgeSubjectAttr, builder.getUnitAttr());
    loop->setAttr(kBridgeFactorAttr, builder.getI32IntegerAttr(factor));

    OpBuilder bodyBuilder = OpBuilder::atBlockBegin(loop.getBody());
    Value virtualPid = arith::AddIOp::create(
        bodyBuilder, loc, basePid, loop.getInductionVar());
    IRMapping mapping;
    mapping.map(physicalPid.getResult(), virtualPid);
    if (directPartition) {
      mapping.map(directPartition->quotient.getResult(),
                  loop.getRegionIterArgs()[0]);
      mapping.map(directPartition->remainder.getResult(),
                  loop.getRegionIterArgs()[1]);
    }
    for (Operation *op : bodyOps) {
      if (directPartition &&
          (op == directPartition->quotient.getOperation() ||
           op == directPartition->remainder.getOperation()))
        continue;
      bodyBuilder.clone(*op, mapping);
    }
    if (directPartition) {
      Value quotient = loop.getRegionIterArgs()[0];
      Value remainder = loop.getRegionIterArgs()[1];
      Value nextRaw = arith::AddIOp::create(
          bodyBuilder, loc, remainder, one);
      Value negativeExtent = arith::SubIOp::create(
          bodyBuilder, loc, zero, directPartition->extent);
      Value extentPositive = arith::CmpIOp::create(
          bodyBuilder, loc, arith::CmpIPredicate::sgt,
          directPartition->extent, zero);
      Value absoluteExtent = arith::SelectOp::create(
          bodyBuilder, loc, extentPositive, directPartition->extent,
          negativeExtent);
      Value crossesPartition = arith::CmpIOp::create(
          bodyBuilder, loc, arith::CmpIPredicate::eq, nextRaw,
          absoluteExtent);
      Value nextRemainder = arith::SelectOp::create(
          bodyBuilder, loc, crossesPartition, zero, nextRaw);
      Value negativeOne = arith::ConstantIntOp::create(
          bodyBuilder, loc, -1, 32);
      Value quotientStep = arith::SelectOp::create(
          bodyBuilder, loc, extentPositive, one, negativeOne);
      Value steppedQuotient = arith::AddIOp::create(
          bodyBuilder, loc, quotient, quotientStep);
      Value nextQuotient = arith::SelectOp::create(
          bodyBuilder, loc, crossesPartition, steppedQuotient, quotient);
      if (!loop.getBody()->empty()) {
        if (auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->back()))
          yield->setOperands(ValueRange{nextQuotient, nextRemainder});
        else
          scf::YieldOp::create(bodyBuilder, loc,
                               ValueRange{nextQuotient, nextRemainder});
      }
      (*entry)->setAttr(
          kBridgePartitionRecurrenceAttr,
          builder.getStringAttr(
              "consecutive_pid_signed_quotient_remainder_recurrence_v1"));
    }
    for (Operation *op : llvm::reverse(bodyOps))
      op->erase();

    (*entry)->setAttr(kBridgeOriginAttr,
                      builder.getStringAttr("bridge_constructed"));
    (*entry)->setAttr(kBridgeFactorAttr, builder.getI32IntegerAttr(factor));
    (*entry)->setAttr(kDependenceAttr,
                      builder.getStringAttr(certificate.kind));
    module->setAttr(kBridgeGridDivisorAttr,
                    builder.getI32IntegerAttr(factor));
  }
};

bool isEligibleLoop(scf::ForOp loop, SmallVectorImpl<Operation *> &loadSlice,
                    Operation *&compute, Operation *&reduce,
                    bool requireStaticTripCount) {
  auto trip = exactStaticTripCount(loop);
  auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
  if ((requireStaticTripCount && !trip) || (trip && *trip < 2) ||
      loop.getInitArgs().size() != 1 ||
      !yield || yield.getNumOperands() != 1)
    return false;
  Value accumulator = loop.getRegionIterArgs().front();
  Operation *combiner = yield.getOperand(0).getDefiningOp();
  if (!combiner || !isa<arith::AddIOp, arith::AddFOp>(combiner) ||
      combiner->getNumOperands() != 2 || combiner->getNumResults() != 1)
    return false;
  Value contribution;
  if (combiner->getOperand(0) == accumulator)
    contribution = combiner->getOperand(1);
  else if (combiner->getOperand(1) == accumulator)
    contribution = combiner->getOperand(0);
  else
    return false;
  compute = contribution.getDefiningOp();
  reduce = combiner;
  if (!compute || compute->getBlock() != loop.getBody() ||
      valueDependsOn(contribution, accumulator))
    return false;

  SmallVector<LoadOp> loads;
  bool invalid = false;
  loop.walk([&](Operation *operation) {
    if (operation != loop.getOperation() && isa<scf::ForOp>(operation)) {
      invalid = true;
      return;
    }
    if (auto load = dyn_cast<LoadOp>(operation)) {
      invalid |= load.getIsVolatile() ||
                 valueDependsOn(load.getPtr(), accumulator) ||
                 (load.getMask() &&
                  valueDependsOn(load.getMask(), accumulator)) ||
                 (load.getOther() &&
                  valueDependsOn(load.getOther(), accumulator));
      loads.push_back(load);
      return;
    }
    if (isa<StoreOp, AtomicRMWOp, AtomicCASOp>(operation)) {
      invalid = true;
      return;
    }
    if (operation != loop.getOperation() &&
        !operation->hasTrait<OpTrait::IsTerminator>() &&
        !isMemoryEffectFree(operation))
      invalid = true;
  });
  if (invalid || loads.empty() ||
      llvm::none_of(loads, [&](LoadOp load) {
        return valueDependsOn(contribution, load.getResult());
      }))
    return false;

  llvm::SmallPtrSet<Operation *, 32> slice;
  SmallVector<Operation *> worklist;
  for (LoadOp load : loads) {
    slice.insert(load.getOperation());
    worklist.push_back(load.getOperation());
  }
  while (!worklist.empty()) {
    Operation *operation = worklist.pop_back_val();
    for (Value operand : operation->getOperands()) {
      Operation *definition = operand.getDefiningOp();
      if (!definition || definition->getBlock() != loop.getBody() ||
          slice.contains(definition))
        continue;
      if (!isMemoryEffectFree(definition))
        return false;
      slice.insert(definition);
      worklist.push_back(definition);
    }
  }
  for (Operation &operation : loop.getBody()->without_terminator())
    if (slice.contains(&operation))
      loadSlice.push_back(&operation);
  return !loadSlice.empty();
}

LoopDependenceCertificate
certifyNestedInnerDimension(scf::ForOp loop) {
  auto parent = loop->getParentOfType<scf::ForOp>();
  if (!parent || loop->getBlock() != parent.getBody())
    return {false, "", "eligible inner loop has no direct scf.for parent"};
  APInt lower, upper, step;
  if (!matchPattern(parent.getLowerBound(), m_ConstantInt(&lower)) ||
      !matchPattern(parent.getUpperBound(), m_ConstantInt(&upper)) ||
      !matchPattern(parent.getStep(), m_ConstantInt(&step)) ||
      lower.getSExtValue() != 0 || step.getSExtValue() != 1 ||
      upper.getSExtValue() < 2 ||
      !parent.getInitArgs().empty())
    return {false, "", "outer dimension is not a closed static at-least-two iteration domain"};

  unsigned directLoops = 0;
  unsigned stores = 0;
  StoreOp outputStore;
  for (Operation &op : parent.getBody()->without_terminator()) {
    if (auto nested = dyn_cast<scf::ForOp>(op)) {
      ++directLoops;
      if (nested != loop)
        return {false, "", "outer dimension contains another Loop subject"};
      continue;
    }
    if (auto store = dyn_cast<StoreOp>(op)) {
      ++stores;
      outputStore = store;
      continue;
    }
    if (!isMemoryEffectFree(&op))
      return {false, "", "outer dimension contains an unclosed effect"};
  }
  if (directLoops != 1 || stores != 1 || !outputStore ||
      loop.getNumResults() != 1 ||
      !valueDependsOn(outputStore.getValue(), loop.getResult(0)) ||
      !valueDependsOn(outputStore.getPtr(), parent.getInductionVar()) ||
      valueDependsOn(outputStore.getPtr(), loop.getResult(0)))
    return {false, "", "outer store is not the closed per-iteration sink of the inner result"};
  return {true, "nested_inner_dimension_independent_sink_v1", ""};
}

bool isEligibleRuntimeTopkLoop(scf::ForOp loop, LoadOp &load,
                               arith::AddFOp &reduce,
                               std::string *failureReason = nullptr) {
  auto reject = [&](StringRef reason) {
    if (failureReason)
      *failureReason = reason.str();
    return false;
  };
  APInt upper;
  if (!matchI32ConstantThroughIdentityBitcast(loop.getLowerBound(), 0))
    return reject("lower bound is not identity-bitcast constant i32 zero");
  if (matchPattern(loop.getUpperBound(), m_ConstantInt(&upper)))
    return reject("upper bound is static rather than runtime");
  if (!matchI32ConstantThroughIdentityBitcast(loop.getStep(), 1))
    return reject("step is not identity-bitcast constant i32 one");
  if (!loop.getInductionVar().getType().isInteger(32))
    return reject("induction variable is not i32");
  if (loop.getInitArgs().size() != 1)
    return reject("loop does not have exactly one init argument");
  if (!isVectorF32(loop.getInitArgs().front(), 128))
    return reject("loop init is not tensor<128xf32>");
  if (!isStructurallyZeroTensor(loop.getInitArgs().front()))
    return reject("loop init is not a direct or resolved structural zero");

  unsigned nestedLoops = 0;
  unsigned loadCount = 0;
  unsigned storeCount = 0;
  loop.walk([&](Operation *op) {
    nestedLoops += isa<scf::ForOp>(op);
    loadCount += isa<LoadOp>(op);
    storeCount += isa<StoreOp>(op);
  });
  if (nestedLoops != 1 || loadCount != 1 || storeCount != 0)
    return reject("loop nest/load/store cardinality is not 1/1/0");

  bool sawLoad = false;
  for (Operation &op : loop.getBody()->without_terminator()) {
    if (!sawLoad) {
      if (auto candidate = dyn_cast<LoadOp>(op)) {
        if (!isVectorF32(candidate.getResult(), 128))
          return reject("load result is not tensor<128xf32>");
        if (!candidate.getMask() || !candidate.getOther() ||
            !candidate.getBoundaryCheck().empty() || candidate.getPadding())
          return reject("load is not the exact masked tensor load with other");
        load = candidate;
        sawLoad = true;
      } else if (!isMemoryEffectFree(&op)) {
        return reject("pre-load address slice has a non-free effect");
      }
      continue;
    }
    auto add = dyn_cast<arith::AddFOp>(op);
    if (!add || reduce || !isVectorF32(add.getResult(), 128))
      return reject("post-load body is not one tensor<128xf32> addf");
    bool exactOperands =
        (add.getLhs() == loop.getRegionIterArg(0) &&
         add.getRhs() == load.getResult()) ||
        (add.getRhs() == loop.getRegionIterArg(0) &&
         add.getLhs() == load.getResult());
    if (!exactOperands)
      return reject("addf does not combine the region accumulator and load");
    reduce = add;
  }
  auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
  if (!sawLoad || !reduce)
    return reject("load/addf chain is incomplete");
  if (!yield || yield.getNumOperands() != 1 ||
      yield.getOperand(0) != reduce.getResult())
    return reject("yield does not return the addf result");
  return true;
}

DictionaryAttr subjectLocator(MLIRContext *context,
                              const ParsedLoopPlan &plan) {
  Builder builder(context);
  return builder.getDictionaryAttr({
      builder.getNamedAttr("adapter_version",
                           builder.getI32IntegerAttr(plan.adapterVersion)),
      builder.getNamedAttr("decision_ref", builder.getStringAttr(plan.decisionRef)),
      builder.getNamedAttr("route_ref", builder.getStringAttr(plan.route)),
      builder.getNamedAttr("subject_ref", builder.getStringAttr(plan.subjectRef)),
  });
}

struct ExistingLoopRouteSubject {
  scf::ForOp loop;
  std::string providerLoopLocator;
  std::string parentLoopLocator;
  std::string providerCapabilityCertificate;
  std::string boundMemberRef;
  int64_t nestingDepth = 0;
  int64_t boundRouteFactor = 0;
  SmallVector<Operation *> loads;
  SmallVector<Operation *> computes;
  SmallVector<Operation *> states;
  LoopDependenceCertificate certificate;
  bool coarseOrderPreserving = false;
};

class HBVLoopDecisionPass
    : public impl::TritonHBVLoopDecisionBase<HBVLoopDecisionPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto bundle = module->getAttrOfType<StringAttr>(kBundleAttr);
    std::string reason;
    auto parsed = bundle ? parseLoopPlan(bundle.getValue(), reason)
                         : FailureOr<ParsedLoopPlan>(failure());
    if (!bundle)
      reason = "missing authoritative module Loop PlanBundle";
    if (failed(parsed)) {
      reportFailure(module, reason);
      return signalPassFailure();
    }
    auto entry = findEntryPoint(module);
    if (failed(entry) || (*entry)->hasAttr(kBundleAttr)) {
      reportFailure(module, "expected exactly one unbound public Loop kernel owner");
      return signalPassFailure();
    }
    (*entry)->setAttr(kBundleAttr, bundle);
    module->removeAttr(kBundleAttr);
    if (!parsed->controlled)
      return;

    Builder builder(module.getContext());
    (*entry)->setAttr(
        kMechanismRouteAttr,
        builder.getStringAttr(parsed->mechanismRoute));
    (*entry)->setAttr(
        kRouteSubtypeAttr,
        builder.getStringAttr(parsed->routeSubtype));
    (*entry)->setAttr(
        kArtifactRouteAttr,
        builder.getStringAttr(parsed->artifactRoute));
    if (parsed->composedIntervention) {
      (*entry)->setAttr(
          kCompositionSchemaAttr,
          builder.getStringAttr(parsed->compositionSchema));
      (*entry)->setAttr(
          kCompositionBridgeFactorAttr,
          builder.getI32IntegerAttr(parsed->bridgeFactor));
      (*entry)->setAttr(
          kCompositionRouteFactorAttr,
          builder.getI32IntegerAttr(parsed->routeFactor));
    }
    if (parsed->stateAxisLogical) {
      SmallVector<StateAxisNormalizationRegion, 2> subjects =
          collectStateAxisNormalizations(*entry);
      bool factsMatch =
          subjects.size() ==
              static_cast<size_t>(parsed->stateAxisGroupCount) &&
          llvm::all_of(
              llvm::zip(subjects, parsed->stateAxisGroups),
              [](const auto &pair) {
                return stateAxisSubjectMatchesPlan(
                    std::get<0>(pair), std::get<1>(pair));
              });
      if (!factsMatch) {
        reportFailure(
            module,
            "state-axis sibling-group facts contradict the selected Provider");
        return signalPassFailure();
      }
      for (auto &subject : subjects)
        subject.columnDenominator.getDefiningOp()->setAttr(
            kSubjectAttr, subjectLocator(module.getContext(), *parsed));
      (*entry)->setAttr(kRouteAttr,
                        builder.getStringAttr(parsed->route));
      (*entry)->setAttr(kSubjectRefAttr,
                        builder.getStringAttr(parsed->subjectRef));
      (*entry)->setAttr(
          kDependenceAttr,
          builder.getStringAttr(
              "provider_closed_sibling_state_operation_graph_v1"));
      (*entry)->setAttr(
          kStateAxisGroupCountAttr,
          builder.getI64IntegerAttr(parsed->stateAxisGroupCount));
      return;
    }
    if (parsed->exactPrefixReduction) {
      SmallVector<ExactPrefixReduction> subjects =
          collectExactPrefixReductions(*entry);
      if (subjects.size() != 1) {
        reportFailure(
            module,
            "exact-prefix route requires one provider-closed recurrence");
        return signalPassFailure();
      }
      if (parsed->adapterVersion == 6) {
        auto elementType = dyn_cast<IntegerType>(
            subjects.front().load.getResult().getType());
        int64_t elementBytes = elementType
                                   ? (elementType.getWidth() + 7) / 8
                                   : 0;
        if (parsed->exactPrefixActiveExtent != subjects.front().activeExtent ||
            parsed->exactPrefixContainerWidth != subjects.front().vectorWidth ||
            parsed->exactPrefixElementBytes != elementBytes) {
          reportFailure(
              module,
              "logical exact-prefix subtype facts do not match its provider subject");
          return signalPassFailure();
        }
      }
      (*entry)->setAttr(kRouteAttr, builder.getStringAttr(parsed->route));
      (*entry)->setAttr(kSubjectRefAttr,
                        builder.getStringAttr(parsed->subjectRef));
      (*entry)->setAttr(
          kDependenceAttr,
          builder.getStringAttr("predicated_exact_prefix_reduction_v1"));
      return;
    }
    if (parsed->bridgeConstructed) {
      SmallVector<scf::ForOp> bridgeLoops;
      (*entry).walk([&](scf::ForOp loop) {
        if (loop->hasAttr(kBridgeSubjectAttr))
          bridgeLoops.push_back(loop);
      });
      auto origin = (*entry)->getAttrOfType<StringAttr>(kBridgeOriginAttr);
      auto factor = (*entry)->getAttrOfType<IntegerAttr>(kBridgeFactorAttr);
      auto dependence = (*entry)->getAttrOfType<StringAttr>(kDependenceAttr);
      bool multiAxis = origin && dependence &&
          origin.getValue() == "bridge_multiaxis_constructed" &&
          dependence.getValue() == "multi_axis_program_disjoint_v1";
      if (!origin || !factor ||
          factor.getInt() != parsed->bridgeFactor || !dependence ||
          (!multiAxis && dependence.getValue() !=
                             "bridge_pid_partitioned_disjoint_v1")) {
        reportFailure(module,
                      "selected Bridge Loop route does not match certified origin/factor");
        return signalPassFailure();
      }
      if (parsed->bridgeStaticPartition) {
        SmallVector<Operation *> staticSubjects;
        (*entry).walk([&](Operation *op) {
          if (op->hasAttr(kBridgeRoleAttr))
            staticSubjects.push_back(op);
        });
        auto recurrence = (*entry)->getAttrOfType<StringAttr>(
            kBridgePartitionRecurrenceAttr);
        if (!bridgeLoops.empty() || staticSubjects.empty() ||
            origin.getValue() != "bridge_static_partition_recurrence" ||
            !recurrence ||
            recurrence.getValue() !=
                "consecutive_pid_signed_quotient_remainder_recurrence_v1") {
          reportFailure(
              module,
              "static partition route does not match its recurrence materializer");
          return signalPassFailure();
        }
        staticSubjects.front()->setAttr(
            kSubjectAttr, subjectLocator(module.getContext(), *parsed));
        (*entry)->setAttr(kRouteAttr,
                          builder.getStringAttr(parsed->route));
        (*entry)->setAttr(kSubjectRefAttr,
                          builder.getStringAttr(parsed->subjectRef));
        return;
      }
      if (multiAxis) {
        SmallVector<Operation *> staticSubjects;
        (*entry).walk([&](Operation *op) {
          if (op->hasAttr(kBridgeRoleAttr))
            staticSubjects.push_back(op);
        });
        auto cardinality = (*entry)->getAttrOfType<IntegerAttr>(
            kBridgeCardinalityAttr);
        if (!parsed->bridgeAxisVector || !bridgeLoops.empty() ||
            staticSubjects.size() !=
                static_cast<size_t>(parsed->bridgeFactor) ||
            !llvm::all_of(staticSubjects,
                          [](Operation *op) { return isa<CallOp>(op); }) ||
            !cardinality || cardinality.getInt() != parsed->bridgeFactor) {
          reportFailure(
              module,
              parsed->bridgeAxisVector
                  ? "multi-axis Bridge has no complete Plan-bound route subject"
                  : "multi-axis Bridge requires an axis-vector-bound route contract");
          return signalPassFailure();
        }
        for (Operation *subject : staticSubjects)
          subject->setAttr(
              kSubjectAttr, subjectLocator(module.getContext(), *parsed));
        (*entry)->setAttr(kRouteAttr,
                          builder.getStringAttr(parsed->route));
        (*entry)->setAttr(kSubjectRefAttr,
                          builder.getStringAttr(parsed->subjectRef));
        (*entry)->setAttr(
            kCompositionBridgeAxisDivisorsAttr,
            builder.getStringAttr(llvm::join(
                llvm::map_range(
                    parsed->bridgeAxisDivisors,
                    [](int64_t divisor) { return std::to_string(divisor); }),
                ",")));
        return;
      }
      if (bridgeLoops.size() != 1 ||
          origin.getValue() != "bridge_constructed") {
        reportFailure(module,
                      "selected Bridge Loop route has no unique constructed loop");
        return signalPassFailure();
      }
      scf::ForOp loop = bridgeLoops.front();
      loop->setAttr(kSubjectAttr, subjectLocator(module.getContext(), *parsed));
      (*entry)->setAttr(kRouteAttr, builder.getStringAttr(parsed->route));
      (*entry)->setAttr(kSubjectRefAttr,
                        builder.getStringAttr(parsed->subjectRef));
      if (parsed->route == kPipelineRoute) {
        loop->setAttr("tt.num_stages",
                      builder.getI32IntegerAttr(parsed->stageCount));
      } else {
        loop->setAttr("tt.loop_unroll_factor",
                      builder.getI32IntegerAttr(parsed->unrollFactor));
        if (parsed->compositionV2 &&
            parsed->routeFactor != parsed->bridgeFactor) {
          loop->setAttr(
              kSourceExactTripCountAttr,
              builder.getI64IntegerAttr(parsed->bridgeFactor));
          loop->setAttr(
              kMainTailAttr,
              builder.getStringAttr("native_factor_requested"));
        }
        int64_t ordinal = 0;
        for (Operation &op : loop.getBody()->without_terminator()) {
          op.setAttr(kBridgeRoleAttr, builder.getStringAttr(
              isa<LoadOp>(op) ? "load" : isa<StoreOp>(op) ? "store" : "compute"));
          op.setAttr(kBridgeOrdinalAttr, builder.getI32IntegerAttr(ordinal++));
        }
      }
      return;
    }
    if (parsed->providerClosedStatic) {
      struct ClosedNest {
        scf::ForOp root;
        SmallVector<scf::ForOp> loops;
        SmallVector<scf::ForOp> leaves;
        std::string certificate;
      };
      SmallVector<ClosedNest> eligible;
      std::string lastReason = "no static scf.for nest found";
      (*entry).walk([&](scf::ForOp loop) {
        if (loop->getParentOfType<scf::ForOp>())
          return;
        ClosedNest nest{loop, {}, {}, ""};
        auto certificate = certifyProviderClosedStaticNest(
            loop, nest.loops, nest.leaves);
        if (certificate.safe) {
          nest.certificate = certificate.kind;
          eligible.push_back(std::move(nest));
        } else
          lastReason = certificate.reason;
      });
      if (eligible.empty()) {
        reportFailure(
            module,
            Twine("provider-closed phase route has no legal static nest: ") +
                lastReason);
        return signalPassFailure();
      }
      SmallVector<int64_t> observedFactors;
      for (const ClosedNest &nest : eligible)
        for (scf::ForOp loop : nest.loops)
          observedFactors.push_back(*exactStaticTripCount(loop));
      if (observedFactors.size() != parsed->unrollFactors.size()) {
        reportFailure(
            module,
            Twine("provider-closed factor count ") +
                Twine(observedFactors.size()) + " does not match contract " +
                Twine(parsed->unrollFactors.size()));
        return signalPassFailure();
      }
      for (size_t ordinal = 0; ordinal < observedFactors.size(); ++ordinal) {
        int64_t observed = observedFactors[ordinal];
        int64_t expected = parsed->unrollFactors[ordinal];
        if (observed == expected)
          continue;
        reportFailure(
            module, Twine("provider-closed factor mismatch at structural ordinal ") +
                        Twine(ordinal) + ": observed " + Twine(observed) +
                        ", contract " + Twine(expected));
        return signalPassFailure();
      }
      int64_t subjectOrdinal = 0;
      for (ClosedNest &nest : eligible) {
        for (scf::ForOp loop : nest.loops) {
          if (loop->hasAttr(kSubjectAttr) ||
              loop->hasAttr("tt.loop_unroll_factor")) {
            reportFailure(module,
                          "provider-closed static nest carries stale guidance");
            return signalPassFailure();
          }
          loop->setAttr(kSubjectAttr,
                        subjectLocator(module.getContext(), *parsed));
          loop->setAttr(
              "tt.loop_unroll_factor",
              builder.getI32IntegerAttr(*exactStaticTripCount(loop)));
        }
        for (scf::ForOp leaf : nest.leaves) {
          std::string roleSubject =
              parsed->subjectRef + ".static." +
              std::to_string(subjectOrdinal++);
          if (failed(tagProviderClosedLeaf(leaf, roleSubject, builder))) {
            reportFailure(
                module,
                "provider-closed static leaf cannot form load/compute/reduce phases");
            return signalPassFailure();
          }
        }
      }
      (*entry)->setAttr(kRouteAttr, builder.getStringAttr(parsed->route));
      (*entry)->setAttr(kSubjectRefAttr,
                        builder.getStringAttr(parsed->subjectRef));
      bool independentReduction = llvm::all_of(
          eligible, [](const ClosedNest &nest) {
            return nest.certificate ==
                   "independent_iteration_exact_inner_reduction_v1";
          });
      (*entry)->setAttr(
          kDependenceAttr,
          builder.getStringAttr(
              independentReduction
                  ? "independent_iteration_exact_inner_reduction_v1"
                  : "provider_closed_complete_static_nest_v1"));
      return;
    }
    if (parsed->runtimeGuardedLogical) {
      SmallVector<std::tuple<scf::ForOp, LoadOp, arith::AddFOp>> eligible;
      std::string eligibilityReason = "no scf.for found";
      (*entry).walk([&](scf::ForOp loop) {
        LoadOp load;
        arith::AddFOp reduce;
        std::string loopReason;
        if (isEligibleRuntimeTopkLoop(loop, load, reduce, &loopReason))
          eligible.emplace_back(loop, load, reduce);
        else
          eligibilityReason = std::move(loopReason);
      });
      if (eligible.size() != 1) {
        reportFailure(
            module,
            Twine("selected runtime-topk Loop route requires one fresh eligible scf.for: ") +
                eligibilityReason);
        return signalPassFailure();
      }
      auto loop = std::get<0>(eligible.front());
      if (loop->hasAttr(kSubjectAttr) ||
          loop->hasAttr("tt.loop_unroll_factor")) {
        reportFailure(module,
                      "selected runtime-topk Loop subject carries stale guidance");
        return signalPassFailure();
      }
      loop->setAttr(kSubjectAttr, subjectLocator(module.getContext(), *parsed));
      (*entry)->setAttr(kRouteAttr, builder.getStringAttr(parsed->route));
      (*entry)->setAttr(kSubjectRefAttr,
                        builder.getStringAttr(parsed->subjectRef));
      return;
    }
    if (parsed->affineRuntimePartial) {
      SmallVector<scf::ForOp> eligible;
      std::string lastReason = "no scf.for found";
      (*entry).walk([&](scf::ForOp loop) {
        auto certificate = certifyAffineRuntimeOrderPreserving(loop);
        if (certificate.safe)
          eligible.push_back(loop);
        else
          lastReason = certificate.reason;
      });
      if (eligible.size() != 1) {
        reportFailure(
            module,
            Twine("affine-runtime route requires one provider-closed loop: ") +
                lastReason);
        return signalPassFailure();
      }
      scf::ForOp loop = eligible.front();
      if (loop->hasAttr(kSubjectAttr) ||
          loop->hasAttr("tt.loop_unroll_factor")) {
        reportFailure(module,
                      "affine-runtime subject carries stale guidance");
        return signalPassFailure();
      }
      loop->setAttr(kSubjectAttr,
                    subjectLocator(module.getContext(), *parsed));
      LoopDependenceCertificate certificate =
          certifyAffineRuntimeOrderPreserving(loop);
      // Contiguous program partitions use the HBV-owned main/tail
      // materializer after the generic unroll pass.  Do not publish the
      // native unroll marker, which would consume the loop first and recreate
      // the per-lane guards that this capability exists to remove.
      if (certificate.kind ==
          "affine_runtime_program_partition_order_preserving_v1")
        loop->setAttr(kMainTailAttr,
                      builder.getStringAttr("requested"));
      else
        loop->setAttr("tt.loop_unroll_factor",
                      builder.getI32IntegerAttr(parsed->unrollFactor));
      (*entry)->setAttr(kRouteAttr,
                        builder.getStringAttr(parsed->route));
      (*entry)->setAttr(kSubjectRefAttr,
                        builder.getStringAttr(parsed->subjectRef));
      (*entry)->setAttr(
          kDependenceAttr,
          builder.getStringAttr(certificate.kind));
      return;
    }

    if (parsed->route == kPipelineRoute) {
      SmallVector<std::pair<scf::ForOp, LoopDependenceCertificate>> eligible;
      std::string lastReason = "no scf.for found";
      (*entry).walk([&](scf::ForOp loop) {
        LoopDependenceCertificate certificate =
            certifyNativeDynamicPipelineSubject(loop);
        if (certificate.safe)
          eligible.emplace_back(loop, std::move(certificate));
        else
          lastReason = std::move(certificate.reason);
      });
      if (eligible.size() != 1) {
        reportFailure(
            module,
            Twine("selected native pipeline route requires one compatible "
                  "dynamic-or-static scf.for: ") +
                lastReason);
        return signalPassFailure();
      }
      scf::ForOp loop = eligible.front().first;
      if (loop->hasAttr(kSubjectAttr) ||
          loop->hasAttr("tt.loop_unroll_factor")) {
        reportFailure(
            module,
            "selected pipeline subject carries stale or conflicting guidance");
        return signalPassFailure();
      }
      loop->setAttr(kSubjectAttr,
                    subjectLocator(module.getContext(), *parsed));
      loop->setAttr("tt.num_stages",
                    builder.getI32IntegerAttr(parsed->stageCount));
      (*entry)->setAttr(kRouteAttr, builder.getStringAttr(parsed->route));
      (*entry)->setAttr(kSubjectRefAttr,
                        builder.getStringAttr(parsed->subjectRef));
      (*entry)->setAttr(
          kDependenceAttr,
          builder.getStringAttr(eligible.front().second.kind));
      return;
    }

    llvm::DenseMap<Operation *, std::string> planningCutLocators;
    int64_t planningCutOrdinal = 0;
    (*entry).walk([&](scf::ForOp loop) {
      planningCutLocators[loop.getOperation()] =
          "planning-cut.loop." + std::to_string(planningCutOrdinal++);
    });
    SmallVector<ExistingLoopRouteSubject, 2> eligible;
    (*entry).walk([&](scf::ForOp loop) {
      int64_t nestingDepth = 0;
      for (Operation *parent = loop->getParentOp(); parent;
           parent = parent->getParentOp())
        nestingDepth += isa<scf::ForOp>(parent);
      auto parentLoop = loop->getParentOfType<scf::ForOp>();
      std::string parentLocator =
          parentLoop
              ? planningCutLocators.lookup(parentLoop.getOperation())
              : "";
      SmallVector<Operation *> loadSlice;
      Operation *compute = nullptr;
      Operation *reduce = nullptr;
      // Triton's native unroller preserves runtime trip counts with a
      // factor-wide main loop plus an ordered scalar remainder loop.  Dynamic
      // bounds therefore enter the same dependence proof as static bounds;
      // the bound kind itself is never a route-family rejection reason.
      bool narrowShape = isEligibleLoop(
          loop, loadSlice, compute, reduce,
          /*requireStaticTripCount=*/false);
      LoopDependenceCertificate narrowCertificate;
      if (narrowShape) {
        auto load = dyn_cast<LoadOp>(loadSlice.back());
        if (load)
          narrowCertificate = certifyExistingUnrollReorder(
              loop, load, compute, reduce);
      }
      bool narrowLogical =
          parsed->route != kLogicalRoute ||
          (narrowCertificate.safe && compute &&
           compute->getNumResults() == 1 &&
           isRankOneIntegerVector(compute->getResult(0)));
      if (narrowCertificate.safe && narrowLogical) {
        eligible.push_back(ExistingLoopRouteSubject{
            loop, planningCutLocators.lookup(loop.getOperation()),
            parentLocator,
            parsed->route == kLogicalRoute
                ? "existing_rank1_integer_exact_grouping_v2"
                : narrowCertificate.kind,
            "", nestingDepth, 0, std::move(loadSlice), {compute}, {reduce},
            std::move(narrowCertificate), false});
        return;
      }

      SmallVector<Operation *> generalLoads, generalComputes, generalStates;
      LoopDependenceCertificate generalCertificate =
          certifyOrderPreservingReadExposure(
              loop, generalLoads, generalComputes, generalStates,
              /*requireVectorizableTensorLoad=*/
                  parsed->route == kLogicalRoute);
      if (generalCertificate.safe)
        eligible.push_back(ExistingLoopRouteSubject{
            loop, planningCutLocators.lookup(loop.getOperation()),
            parentLocator, generalCertificate.kind, "", nestingDepth, 0,
            std::move(generalLoads), std::move(generalComputes),
            std::move(generalStates), std::move(generalCertificate), true});
    });
    if (parsed->providerBoundSubjectSet) {
      SmallVector<ExistingLoopRouteSubject, 2> selected;
      for (const auto &member : parsed->providerBoundMembers) {
        auto found = llvm::find_if(
            eligible, [&](const ExistingLoopRouteSubject &subject) {
              return subject.providerLoopLocator ==
                     member.providerLoopLocator;
            });
        if (found == eligible.end()) {
          reportFailure(
              module,
              Twine("Plan-bound existing Loop locator is not route-capable: ") +
                  member.providerLoopLocator);
          return signalPassFailure();
        }
        auto exactTrip = exactStaticTripCount(found->loop);
        bool tripMatches =
            member.exactStaticTripCount > 0
                ? exactTrip && *exactTrip == member.exactStaticTripCount
                : !exactTrip &&
                      member.runtimeMainTailCertificateRef ==
                          "native_dynamic_unroll_main_ordered_remainder_v1";
        if (!tripMatches ||
            found->providerCapabilityCertificate !=
                member.routeCapabilityCertificateRef ||
            found->nestingDepth != member.nestingDepth ||
            found->parentLoopLocator != member.parentLoopLocator) {
          reportFailure(
              module,
              Twine("Plan-bound existing Loop facts changed at locator: ") +
                  member.providerLoopLocator);
          return signalPassFailure();
        }
        selected.push_back(*found);
        selected.back().boundMemberRef = member.memberRef;
        selected.back().boundRouteFactor = member.routeFactor;
      }
      eligible = std::move(selected);
    }
    if (eligible.empty() ||
        (!parsed->providerBoundSubjectSet &&
         !parsed->multiSubject && eligible.size() != 1)) {
      reportFailure(module,
                    "selected HBV-owned Loop route requires at least one "
                    "fresh eligible existing scf.for");
      return signalPassFailure();
    }
    LoopDependenceCertificate nestedCertificate;
    if (parsed->nestedSubject) {
      if (eligible.size() != 1) {
        reportFailure(module,
                      "nested inner-dimension route requires exactly one eligible leaf subject");
        return signalPassFailure();
      }
      nestedCertificate =
          certifyNestedInnerDimension(eligible.front().loop);
      if (!nestedCertificate.safe) {
        reportFailure(module,
                      Twine("nested inner-dimension proof failed: ") +
                          nestedCertificate.reason);
        return signalPassFailure();
      }
      if (parsed->providerBoundSubjectSet &&
          nestedCertificate.kind !=
              parsed->providerBoundMembers.front()
                  .nestedContextCertificateRef) {
        reportFailure(
            module,
            "Plan-bound nested context certificate changed before mutation");
        return signalPassFailure();
      }
    }
    (*entry)->setAttr(kRouteAttr, builder.getStringAttr(parsed->route));
    (*entry)->setAttr(kSubjectRefAttr, builder.getStringAttr(parsed->subjectRef));
    int64_t subjectOrdinal = 0;
    for (ExistingLoopRouteSubject &subject : eligible) {
      scf::ForOp loop = subject.loop;
      if (loop->hasAttr(kSubjectAttr) ||
          loop->hasAttr("tt.loop_unroll_factor")) {
        reportFailure(module, "selected Loop subject carries stale or conflicting guidance");
        return signalPassFailure();
      }
      std::string roleSubject =
          parsed->providerBoundSubjectSet
              ? subject.boundMemberRef
              : parsed->subjectRef + "." +
                    std::to_string(subjectOrdinal++);
      auto roleSubjectAttr = builder.getStringAttr(roleSubject);
      loop->setAttr(kSubjectAttr, subjectLocator(module.getContext(), *parsed));
      loop->setAttr("tt.loop_unroll_factor",
                    builder.getI32IntegerAttr(
                        parsed->providerBoundSubjectSet
                            ? subject.boundRouteFactor
                            : parsed->unrollFactor));
      // A factor route owns a factor-wide main body plus an optional ordered
      // remainder whenever the factor is not the complete trip count.  This
      // is equally necessary for a runtime bound and for a static trip count
      // larger than (or indivisible by) the selected factor.
      auto exactTrip = exactStaticTripCount(loop);
      if (exactTrip)
        loop->setAttr(
            kSourceExactTripCountAttr,
            builder.getI64IntegerAttr(*exactTrip));
      int64_t requestedFactor =
          parsed->providerBoundSubjectSet
              ? subject.boundRouteFactor
              : parsed->unrollFactor;
      if (!exactTrip || *exactTrip != requestedFactor)
        loop->setAttr(kMainTailAttr,
                      builder.getStringAttr("native_factor_requested"));
      int64_t loadIndex = 0;
      for (Operation *op : subject.loads) {
        op->setAttr(kRoleAttr, builder.getStringAttr("load"));
        op->setAttr(kRoleSubjectAttr, roleSubjectAttr);
        if (isa<LoadOp>(op))
          op->setAttr(kRoleIndexAttr,
                      builder.getI32IntegerAttr(loadIndex++));
      }
      for (Operation *op : subject.computes) {
        op->setAttr(kRoleAttr, builder.getStringAttr("compute"));
        op->setAttr(kRoleSubjectAttr, roleSubjectAttr);
      }
      for (Operation *op : subject.states) {
        op->setAttr(kRoleAttr, builder.getStringAttr("reduce"));
        op->setAttr(kRoleSubjectAttr, roleSubjectAttr);
      }
    }
    StringRef aggregateCertificate =
        parsed->route == kLogicalRoute
            ? "per_loop_order_preserving_load_vectorization_v1"
            : "per_loop_order_preserving_read_exposure_v1";
    (*entry)->setAttr(
        kDependenceAttr,
        builder.getStringAttr(
            parsed->nestedSubject
                ? nestedCertificate.kind
            : eligible.size() == 1
                ? eligible.front().certificate.kind
                : aggregateCertificate));
  }
};

FailureOr<ParsedLoopPlan> parseFunctionPlan(FuncOp func, std::string &reason) {
  auto bundle = func->getAttrOfType<StringAttr>(kBundleAttr);
  if (!bundle) {
    reason = "missing function-owned Loop PlanBundle";
    return failure();
  }
  return parseLoopPlan(bundle.getValue(), reason);
}

bool collectRoles(FuncOp func, SmallVectorImpl<Operation *> &loads,
                  SmallVectorImpl<Operation *> &computes,
                  SmallVectorImpl<Operation *> &reduces,
                  std::optional<StringRef> subject = std::nullopt) {
  bool unknown = false;
  func.walk([&](Operation *op) {
    auto role = op->getAttrOfType<StringAttr>(kRoleAttr);
    if (!role)
      return;
    auto roleSubject = op->getAttrOfType<StringAttr>(kRoleSubjectAttr);
    if (subject && (!roleSubject || roleSubject.getValue() != *subject))
      return;
    if (role.getValue() == "load")
      loads.push_back(op);
    else if (role.getValue() == "compute")
      computes.push_back(op);
    else if (role.getValue() == "reduce")
      reduces.push_back(op);
    else
      unknown = true;
  });
  return !unknown;
}

void clearRoles(ArrayRef<Operation *> operations) {
  for (Operation *op : operations) {
    op->removeAttr(kRoleAttr);
    op->removeAttr(kRoleSubjectAttr);
    op->removeAttr(kRoleIndexAttr);
    op->removeAttr(kUnrollPartitionLineageAttr);
  }
}

std::optional<StringRef>
roleBlockPartition(ArrayRef<Operation *> operations, Block *block) {
  std::optional<StringRef> result;
  for (Operation *operation : operations) {
    if (operation->getBlock() != block)
      continue;
    auto value = operation->getAttrOfType<StringAttr>(
        kUnrollPartitionLineageAttr);
    if (!value)
      continue;
    if (value.getValue() != "main" && value.getValue() != "tail")
      return std::nullopt;
    if (result && *result != value.getValue())
      return std::nullopt;
    result = value.getValue();
  }
  return result;
}

SmallVector<std::string> collectRoleSubjects(FuncOp func) {
  std::set<std::string> seen;
  SmallVector<std::string> result;
  bool hasUnscoped = false;
  func.walk([&](Operation *op) {
    if (!op->hasAttr(kRoleAttr))
      return;
    auto subject = op->getAttrOfType<StringAttr>(kRoleSubjectAttr);
    if (!subject) {
      hasUnscoped = true;
      return;
    }
    if (seen.insert(subject.getValue().str()).second)
      result.push_back(subject.getValue().str());
  });
  if (hasUnscoped && result.empty())
    result.push_back("");
  llvm::sort(result);
  return result;
}

LogicalResult dependencyClosure(ArrayRef<Operation *> roots,
                                const llvm::SmallPtrSetImpl<Operation *> &stop,
                                llvm::SmallPtrSetImpl<Operation *> &closure) {
  SmallVector<Operation *> worklist(roots.begin(), roots.end());
  for (Operation *root : roots)
    closure.insert(root);
  while (!worklist.empty()) {
    Operation *operation = worklist.pop_back_val();
    for (Value operand : operation->getOperands()) {
      Operation *definition = operand.getDefiningOp();
      if (!definition || definition->getBlock() != operation->getBlock() ||
          stop.contains(definition) || closure.contains(definition))
        continue;
      if (!isMemoryEffectFree(definition))
        return failure();
      closure.insert(definition);
      worklist.push_back(definition);
    }
  }
  return success();
}

SmallVector<Operation *> blockOrder(const llvm::SmallPtrSetImpl<Operation *> &set,
                                    Block *block) {
  SmallVector<Operation *> result;
  for (Operation &operation : *block)
    if (set.contains(&operation))
      result.push_back(&operation);
  return result;
}

LogicalResult materializePhase(FuncOp func,
                               std::optional<StringRef> subject = std::nullopt,
                               bool generalCardinality = false,
                               int64_t unrollFactor = 4,
                               Block *onlyBlock = nullptr,
                               bool clearLineage = true) {
  SmallVector<Operation *> loads, computes, reduces;
  if (!collectRoles(func, loads, computes, reduces, subject))
    return failure();
  auto retainBlock = [onlyBlock](SmallVectorImpl<Operation *> &operations) {
    if (onlyBlock)
      llvm::erase_if(operations, [onlyBlock](Operation *operation) {
        return operation->getBlock() != onlyBlock;
      });
  };
  retainBlock(loads);
  retainBlock(computes);
  retainBlock(reduces);

  // A statically exhausted factor-wide main loop and its scalar epilogue can
  // both be inlined into the same parent block.  In that form block identity
  // no longer separates the factor group from the ordered remainder, but the
  // operation-owned partition lineage emitted by LoopUnroll still does.  Do
  // not count or reorder tail roles as part of the factor-wide main group.
  // Clearing only the tail roles preserves their original scalar order while
  // allowing the main roles in the same block to be materialized normally.
  if (!onlyBlock) {
    SmallVector<Operation *> allRoles;
    llvm::append_range(allRoles, loads);
    llvm::append_range(allRoles, computes);
    llvm::append_range(allRoles, reduces);
    bool hasMainLineage = false;
    bool hasTailLineage = false;
    bool hasInvalidLineage = false;
    for (Operation *operation : allRoles) {
      auto partition = operation->getAttrOfType<StringAttr>(
          kUnrollPartitionLineageAttr);
      if (!partition)
        continue;
      if (partition.getValue() == "main")
        hasMainLineage = true;
      else if (partition.getValue() == "tail")
        hasTailLineage = true;
      else
        hasInvalidLineage = true;
    }
    if (hasInvalidLineage)
      return failure();
    if (hasMainLineage && hasTailLineage) {
      auto isTail = [](Operation *operation) {
        auto partition = operation->getAttrOfType<StringAttr>(
            kUnrollPartitionLineageAttr);
        return partition && partition.getValue() == "tail";
      };
      SmallVector<Operation *> scalarTail;
      llvm::copy_if(allRoles, std::back_inserter(scalarTail), isTail);
      llvm::erase_if(loads, isTail);
      llvm::erase_if(computes, isTail);
      llvm::erase_if(reduces, isTail);
      clearRoles(scalarTail);
    }
  }

  // Native dynamic unrolling creates one factor-wide main-loop block and one
  // scalar ordered remainder-loop block.  Reorder only the main block; the
  // scalar tail is already the exact source order and must remain untouched.
  if (!onlyBlock) {
    SmallVector<Block *> roleBlocks;
    auto rememberBlock = [&](Operation *operation) {
      if (!llvm::is_contained(roleBlocks, operation->getBlock()))
        roleBlocks.push_back(operation->getBlock());
    };
    llvm::for_each(loads, rememberBlock);
    llvm::for_each(computes, rememberBlock);
    llvm::for_each(reduces, rememberBlock);
    if (roleBlocks.size() > 1) {
      for (Block *block : roleBlocks) {
        auto parentLoop = dyn_cast_or_null<scf::ForOp>(block->getParentOp());
        auto mainTail = parentLoop
                            ? parentLoop->getAttrOfType<StringAttr>(
                                  kMainTailAttr)
                            : StringAttr();
        SmallVector<Operation *> allRoles;
        llvm::append_range(allRoles, loads);
        llvm::append_range(allRoles, computes);
        llvm::append_range(allRoles, reduces);
        auto partition = roleBlockPartition(allRoles, block);
        bool isTail =
            (mainTail && mainTail.getValue() == "tail") ||
            (partition && *partition == "tail");
        bool isMain =
            (mainTail && mainTail.getValue() == "main") ||
            (partition && *partition == "main");
        if (isTail) {
          SmallVector<Operation *> scalarTail;
          llvm::copy_if(loads, std::back_inserter(scalarTail),
                        [block](Operation *op) {
                          return op->getBlock() == block;
                        });
          llvm::copy_if(computes, std::back_inserter(scalarTail),
                        [block](Operation *op) {
                          return op->getBlock() == block;
                        });
          llvm::copy_if(reduces, std::back_inserter(scalarTail),
                        [block](Operation *op) {
                          return op->getBlock() == block;
                        });
          clearRoles(scalarTail);
          continue;
        }
        if (isMain) {
          if (failed(materializePhase(func, subject, generalCardinality,
                                      unrollFactor, block,
                                      clearLineage)))
            return failure();
          continue;
        }
        unsigned blockComputes = llvm::count_if(
            computes, [block](Operation *op) { return op->getBlock() == block; });
        unsigned blockReduces = llvm::count_if(
            reduces, [block](Operation *op) { return op->getBlock() == block; });
        if (blockComputes == 1 && blockReduces == 1) {
          SmallVector<Operation *> scalarTail;
          llvm::copy_if(loads, std::back_inserter(scalarTail),
                        [block](Operation *op) {
                          return op->getBlock() == block;
                        });
          llvm::copy_if(computes, std::back_inserter(scalarTail),
                        [block](Operation *op) {
                          return op->getBlock() == block;
                        });
          llvm::copy_if(reduces, std::back_inserter(scalarTail),
                        [block](Operation *op) {
                          return op->getBlock() == block;
                        });
          clearRoles(scalarTail);
          continue;
        }
        if (failed(materializePhase(func, subject, generalCardinality,
                                    unrollFactor, block,
                                    clearLineage)))
          return failure();
      }
      return success();
    }
  }

  if (loads.empty() ||
      (!generalCardinality && reduces.empty()) ||
      (generalCardinality && computes.empty() && reduces.empty()) ||
      (!generalCardinality &&
       (computes.size() != static_cast<size_t>(unrollFactor) ||
        reduces.size() != static_cast<size_t>(unrollFactor))) ||
      llvm::any_of(loads, [&](Operation *op) {
        return op->getBlock() != loads.front()->getBlock();
      }) || llvm::any_of(computes, [&](Operation *op) {
        return op->getBlock() != loads.front()->getBlock();
      }) || llvm::any_of(reduces, [&](Operation *op) {
        return op->getBlock() != loads.front()->getBlock();
      })) {
    return failure();
  }

  llvm::SmallPtrSet<Operation *, 32> loadPhase;
  llvm::SmallPtrSet<Operation *, 32> computePhase;
  llvm::SmallPtrSet<Operation *, 32> reducePhase;
  llvm::SmallPtrSet<Operation *, 32> fixedPrefix;
  llvm::SmallPtrSet<Operation *, 32> tagged;
  tagged.insert(loads.begin(), loads.end());
  tagged.insert(computes.begin(), computes.end());
  tagged.insert(reduces.begin(), reduces.end());
  Operation *firstTagged = nullptr;
  for (Operation &op : *loads.front()->getBlock()) {
    if (!firstTagged && tagged.contains(&op))
      firstTagged = &op;
    if (!firstTagged)
      fixedPrefix.insert(&op);
  }
  if (!firstTagged ||
      failed(dependencyClosure(loads, fixedPrefix, loadPhase))) {
    return failure();
  }
  if (generalCardinality) {
    // Once an independent induction recurrence is fully unrolled, the clone
    // that advances iteration i becomes an address dependency of load i+1.
    // Its pre-unroll role was "reduce" because it carried loop state, but its
    // post-unroll causal service is address generation.  Reclassify only the
    // pure operations proven by the actual SSA backward slice; stores and
    // other effects can never migrate into the load phase.
    llvm::erase_if(computes,
                   [&](Operation *op) { return loadPhase.contains(op); });
    bool effectfulOverlap = llvm::any_of(reduces, [&](Operation *op) {
      return loadPhase.contains(op) && !isMemoryEffectFree(op);
    });
    if (effectfulOverlap)
      return failure();
    llvm::erase_if(reduces,
                   [&](Operation *op) { return loadPhase.contains(op); });
  } else if (llvm::any_of(computes, [&](Operation *op) {
               return loadPhase.contains(op);
             }) ||
             llvm::any_of(reduces, [&](Operation *op) {
               return loadPhase.contains(op);
             })) {
    return failure();
  }
  llvm::SmallPtrSet<Operation *, 32> loadStops;
  loadStops.insert(loadPhase.begin(), loadPhase.end());
  loadStops.insert(fixedPrefix.begin(), fixedPrefix.end());
  if (failed(dependencyClosure(computes, loadStops, computePhase))) {
    return failure();
  }
  llvm::SmallPtrSet<Operation *, 32> priorPhases;
  priorPhases.insert(fixedPrefix.begin(), fixedPrefix.end());
  priorPhases.insert(loadPhase.begin(), loadPhase.end());
  priorPhases.insert(computePhase.begin(), computePhase.end());
  if (llvm::any_of(reduces, [&](Operation *op) { return priorPhases.contains(op); }) ||
      failed(dependencyClosure(reduces, priorPhases, reducePhase)))
    return failure();

  Block *block = loads.front()->getBlock();
  SmallVector<Operation *> orderedLoads = blockOrder(loadPhase, block);
  SmallVector<Operation *> orderedComputes = blockOrder(computePhase, block);
  SmallVector<Operation *> orderedReduces = blockOrder(reducePhase, block);
  if (orderedLoads.empty() ||
      (!generalCardinality && orderedReduces.empty()) ||
      (generalCardinality && orderedComputes.empty() &&
       orderedReduces.empty())) {
    return failure();
  }
  Operation *last = firstTagged->getPrevNode();
  auto appendPhase = [&](ArrayRef<Operation *> operations) {
    for (Operation *op : operations) {
      if (!last) {
        if (op != &block->front())
          op->moveBefore(&block->front());
      } else if (op != last) {
        op->moveAfter(last);
      }
      last = op;
    }
  };
  appendPhase(orderedLoads);
  appendPhase(orderedComputes);
  appendPhase(orderedReduces);
  if (clearLineage) {
    // Clear every clone in this subject, including pure induction operations
    // whose service role was reclassified from reduce to load after unrolling.
    func.walk([&](Operation *op) {
      auto role = op->getAttrOfType<StringAttr>(kRoleAttr);
      if (!role)
        return;
      auto roleSubject = op->getAttrOfType<StringAttr>(kRoleSubjectAttr);
      if (subject && (!roleSubject || roleSubject.getValue() != *subject))
        return;
      if (onlyBlock && op->getBlock() != onlyBlock)
        return;
      op->removeAttr(kRoleAttr);
      op->removeAttr(kRoleSubjectAttr);
      op->removeAttr(kRoleIndexAttr);
      op->removeAttr(kUnrollPartitionLineageAttr);
    });
  }
  return success();
}

LogicalResult inlineBridgeHelpers(FuncOp func, bool invariantHoisting);

LogicalResult materializeBridgePhase(FuncOp func, bool clearLineage = true) {
  SmallVector<Operation *> tagged;
  SmallVector<Operation *> loads;
  SmallVector<Operation *> stores;
  SmallVector<Operation *> helperCalls;
  func.walk([&](Operation *op) {
    auto role = op->getAttrOfType<StringAttr>(kBridgeRoleAttr);
    if (!role)
      return;
    tagged.push_back(op);
    if (role.getValue() == "load")
      loads.push_back(op);
    else if (role.getValue() == "store")
      stores.push_back(op);
    else if (role.getValue() == "multi_axis_helper_call")
      helperCalls.push_back(op);
  });
  // A helper-call sequence is only program-major.  It is not phase-major
  // evidence.  Inline the complete one-block helper first so load, compute
  // and store operations become directly reorderable in the public wrapper;
  // arbitrary CFG helpers remain a typed capability failure.
  if (!helperCalls.empty()) {
    Block *block = helperCalls.front()->getBlock();
    if (helperCalls.size() != tagged.size() ||
        llvm::any_of(helperCalls,
                     [&](Operation *op) { return op->getBlock() != block; }))
      return failure();
    if (failed(inlineBridgeHelpers(func, /*invariantHoisting=*/false)))
      return failure();
    return materializeBridgePhase(func, clearLineage);
  }
  if (tagged.empty() || loads.empty() || stores.empty())
    return failure();
  Block *block = tagged.front()->getBlock();
  if (llvm::any_of(tagged, [&](Operation *op) { return op->getBlock() != block; }))
    return failure();

  llvm::SmallPtrSet<Operation *, 32> taggedSet(tagged.begin(), tagged.end());
  Operation *first = nullptr;
  llvm::SmallPtrSet<Operation *, 32> fixedPrefix;
  for (Operation &op : *block) {
    if (!first && taggedSet.contains(&op))
      first = &op;
    if (!first)
      fixedPrefix.insert(&op);
  }
  if (!first)
    return failure();
  llvm::SmallPtrSet<Operation *, 32> loadPhase;
  if (failed(dependencyClosure(loads, fixedPrefix, loadPhase)))
    return failure();
  llvm::SmallPtrSet<Operation *, 32> storeSet(stores.begin(), stores.end());
  if (llvm::any_of(stores, [&](Operation *op) { return loadPhase.contains(op); }))
    return failure();

  SmallVector<Operation *> computeRoots;
  for (Operation *op : tagged)
    if (!loadPhase.contains(op) && !storeSet.contains(op))
      computeRoots.push_back(op);
  llvm::SmallPtrSet<Operation *, 32> loadStops;
  loadStops.insert(fixedPrefix.begin(), fixedPrefix.end());
  loadStops.insert(loadPhase.begin(), loadPhase.end());
  llvm::SmallPtrSet<Operation *, 32> computePhase;
  if (failed(dependencyClosure(computeRoots, loadStops, computePhase)))
    return failure();
  if (llvm::any_of(stores,
                   [&](Operation *op) { return computePhase.contains(op); }))
    return failure();

  llvm::SmallPtrSet<Operation *, 32> priorPhases;
  priorPhases.insert(loadStops.begin(), loadStops.end());
  priorPhases.insert(computePhase.begin(), computePhase.end());
  llvm::SmallPtrSet<Operation *, 32> storePhase;
  if (failed(dependencyClosure(stores, priorPhases, storePhase)))
    return failure();
  if (llvm::any_of(storePhase, [&](Operation *op) {
        return op->getBlock() != block ||
               (!storeSet.contains(op) && !isMemoryEffectFree(op));
      }))
    return failure();

  llvm::SmallPtrSet<Operation *, 32> accounted;
  accounted.insert(loadPhase.begin(), loadPhase.end());
  accounted.insert(computePhase.begin(), computePhase.end());
  accounted.insert(storePhase.begin(), storePhase.end());
  if (llvm::any_of(tagged,
                   [&](Operation *op) { return !accounted.contains(op); }))
    return failure();

  SmallVector<Operation *> orderedLoads = blockOrder(loadPhase, block);
  SmallVector<Operation *> orderedCompute = blockOrder(computePhase, block);
  SmallVector<Operation *> orderedStores = blockOrder(storePhase, block);
  Operation *anchor = first->getPrevNode();
  if (orderedLoads.empty() || orderedCompute.empty() || orderedStores.empty())
    return failure();
  Operation *last = anchor;
  auto movePhase = [&](ArrayRef<Operation *> phase) {
    for (Operation *op : phase) {
      if (!last) {
        if (op != &block->front())
          op->moveBefore(&block->front());
      } else if (op != last) {
        op->moveAfter(last);
      }
      last = op;
    }
  };
  movePhase(orderedLoads);
  movePhase(orderedCompute);
  movePhase(orderedStores);
  if (clearLineage)
    for (Operation *op : tagged) {
      op->removeAttr(kBridgeRoleAttr);
      op->removeAttr(kBridgeOrdinalAttr);
    }
  return success();
}

// Expand the complete-program helper only after the provider has proved every
// grouped program axis independent.  Unlike the removed flat cloning path,
// every helper argument has an explicit mapping, so SSA dominance is closed.
// When H is requested, a pure operation whose complete operand slice is
// independent of all virtual PID arguments is cloned only for the first call
// and reused by later calls.  This is an axis-dependence rule, not an
// operation-name or workload rule.
LogicalResult inlineBridgeHelpers(FuncOp func, bool invariantHoisting) {
  SmallVector<CallOp> calls;
  func.walk([&](CallOp call) {
    auto role = call->getAttrOfType<StringAttr>(kBridgeRoleAttr);
    if (role && role.getValue() == "multi_axis_helper_call")
      calls.push_back(call);
  });
  if (calls.empty())
    return success();
  ModuleOp module = func->getParentOfType<ModuleOp>();
  FuncOp helper = module.lookupSymbol<FuncOp>(calls.front().getCallee());
  if (!helper || helper.getBody().getBlocks().size() != 1 ||
      helper.getNumResults() != 0 ||
      helper.getNumArguments() <= func.getNumArguments())
    return failure();
  if (llvm::any_of(calls, [&](CallOp call) {
        return call.getCallee() != helper.getName() ||
               call.getNumResults() != 0 ||
               call.getNumOperands() != helper.getNumArguments();
      }))
    return failure();

  Block &helperBlock = helper.getBody().front();
  auto helperReturn = dyn_cast<ReturnOp>(helperBlock.getTerminator());
  if (!helperReturn || helperReturn.getNumOperands() != 0)
    return failure();
  SmallVector<BlockArgument> virtualArguments;
  for (unsigned index = func.getNumArguments();
       index < helper.getNumArguments(); ++index)
    virtualArguments.push_back(helper.getArgument(index));

  llvm::DenseMap<Operation *, SmallVector<Value>> sharedResults;
  llvm::DenseMap<Operation *, int64_t> operationOrdinals;
  int64_t nextOrdinal = 0;
  for (Operation &operation : helperBlock.without_terminator())
    operationOrdinals[&operation] = nextOrdinal++;

  bool hoistedAny = false;
  for (CallOp call : calls) {
    IRMapping mapping;
    for (auto [argument, operand] :
         llvm::zip(helper.getArguments(), call.getOperands()))
      mapping.map(argument, operand);
    OpBuilder builder(call);
    for (Operation &operation : helperBlock.without_terminator()) {
      bool dependsOnVirtual = false;
      for (Value operand : operation.getOperands()) {
        for (BlockArgument argument : virtualArguments) {
          llvm::SmallPtrSet<Value, 32> visited;
          dependsOnVirtual |= valueDependsOn(operand, argument, visited);
        }
      }
      bool shareable = invariantHoisting && !dependsOnVirtual &&
                       operation.getNumResults() != 0 &&
                       operation.getNumRegions() == 0 &&
                       isMemoryEffectFree(&operation);
      auto shared = sharedResults.find(&operation);
      if (shareable && shared != sharedResults.end()) {
        for (auto [result, replacement] :
             llvm::zip(operation.getResults(), shared->second))
          mapping.map(result, replacement);
        hoistedAny = true;
        continue;
      }
      Operation *clone = builder.clone(operation, mapping);
      if (shareable) {
        SmallVector<Value> results(clone->getResults());
        sharedResults[&operation] = results;
        // A shared operation is H output, not an F lane member.
        clone->removeAttr(kBridgeRoleAttr);
        clone->removeAttr(kBridgeOrdinalAttr);
      } else {
        StringRef role = isa<LoadOp>(clone) ? "load"
                         : isa<StoreOp>(clone) ? "store"
                                              : "compute";
        clone->setAttr(kBridgeRoleAttr, builder.getStringAttr(role));
        clone->setAttr(kBridgeOrdinalAttr,
                       builder.getI64IntegerAttr(
                           operationOrdinals.lookup(&operation)));
      }
    }
    call.erase();
  }
  helper.erase();
  if (invariantHoisting && !hoistedAny)
    return failure();
  return success();
}

LogicalResult materializeBridgeLogical(FuncOp func, int64_t routeFactor,
                                      bool invariantHoisting,
                                      bool tensorLaneFusion,
                                      bool exactSplitElision) {
  if (failed(inlineBridgeHelpers(func, invariantHoisting)))
    return failure();
  if (!tensorLaneFusion)
    return materializeBridgePhase(func);
  if (failed(materializeBridgePhase(func, /*clearLineage=*/false)))
    return failure();
  // Bridge owns the number of logical programs in the constructed subject;
  // the logical route owns the number of lanes grouped in each factorized
  // main iteration.  They coincide only on diagonal compositions.  Reading
  // Bridge cardinality here makes every legal off-diagonal composition ask a
  // route-local clone group to contain too many lanes.
  if (routeFactor < 2 || !llvm::isPowerOf2_64(routeFactor))
    return failure();
  int64_t factor = routeFactor;
  llvm::DenseMap<int64_t, SmallVector<LoadOp>> loadGroups;
  llvm::DenseMap<int64_t, SmallVector<StoreOp>> storeGroups;
  func.walk([&](Operation *op) {
    auto ordinal = op->getAttrOfType<IntegerAttr>(kBridgeOrdinalAttr);
    if (!ordinal)
      return;
    if (auto load = dyn_cast<LoadOp>(op))
      loadGroups[ordinal.getInt()].push_back(load);
    else if (auto store = dyn_cast<StoreOp>(op))
      storeGroups[ordinal.getInt()].push_back(store);
  });
  int64_t splitElisionCount = 0;
  auto joinValues = [&](OpBuilder &builder, Location loc,
                        ArrayRef<Value> inputs) -> FailureOr<Value> {
    if (inputs.empty() || !llvm::isPowerOf2_64(inputs.size()))
      return failure();
    // A grouped load is split only to preserve the existing per-program SSA
    // graph.  When a later grouped terminal consumes every ordered leaf of
    // that exact split tree, rebuilding the same join tree is mechanically
    // redundant.  Recover the root using only SSA producer/port identity;
    // no operation, shape, factor-profit or workload identity participates.
    SmallVector<Value> recovered(inputs.begin(), inputs.end());
    bool exactSplitTree = exactSplitElision && inputs.size() > 1;
    while (exactSplitTree && recovered.size() > 1) {
      SmallVector<Value> parents;
      for (size_t index = 0; index < recovered.size(); index += 2) {
        auto left = recovered[index].getDefiningOp<SplitOp>();
        auto right = recovered[index + 1].getDefiningOp<SplitOp>();
        if (!left || left != right || recovered[index] != left.getOutLHS() ||
            recovered[index + 1] != left.getOutRHS()) {
          exactSplitTree = false;
          break;
        }
        parents.push_back(left.getSrc());
      }
      recovered = std::move(parents);
    }
    if (exactSplitTree && recovered.size() == 1) {
      ++splitElisionCount;
      return recovered.front();
    }
    SmallVector<Value> current(inputs.begin(), inputs.end());
    while (current.size() > 1) {
      SmallVector<Value> next;
      for (size_t i = 0; i < current.size(); i += 2)
        next.push_back(JoinOp::create(builder, loc, current[i], current[i + 1]));
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

  bool groupedLoad = false;
  SmallVector<int64_t> loadOrdinals;
  for (auto &entry : loadGroups)
    loadOrdinals.push_back(entry.first);
  llvm::sort(loadOrdinals);
  for (int64_t ordinal : loadOrdinals) {
    auto &loads = loadGroups[ordinal];
    if (loads.size() != static_cast<size_t>(factor))
      return failure();
    // Scalar service (for example one length or reduction value per virtual
    // program) has no logical tensor dimension to join.  Keep those loads in
    // the already phase-major schedule; at least one tensor load must still
    // be grouped below for this route to count as realized.
    if (!isa<RankedTensorType>(loads.front().getPtr().getType()))
      continue;
    bool masked = static_cast<bool>(loads.front().getMask());
    bool hasOther = static_cast<bool>(loads.front().getOther());
    if (llvm::any_of(loads, [&](LoadOp load) {
          return static_cast<bool>(load.getMask()) != masked ||
                 static_cast<bool>(load.getOther()) != hasOther ||
                 !load.getBoundaryCheck().empty() || load.getPadding();
        }))
      return failure();

    // Clone-major order can put a later virtual program's pointer/mask slice
    // after the first virtual program's consumer.  Grouping the loads at that
    // point would make the replacement split fail SSA dominance.  Hoist only
    // the exact backward slice needed by this same-origin load group before
    // its earliest consumer.  The Bridge certificate has already proved that
    // virtual programs are disjoint; writes and unknown effects remain
    // forbidden here.  Non-volatile input reads may move with the slice.
    Operation *earliestUser = nullptr;
    for (LoadOp load : loads) {
      for (Operation *user : load.getResult().getUsers()) {
        if (user->getBlock() != load->getBlock())
          return failure();
        if (!earliestUser || user->isBeforeInBlock(earliestUser))
          earliestUser = user;
      }
    }
    if (!earliestUser)
      return failure();
    llvm::SmallPtrSet<Operation *, 32> moveSet;
    SmallVector<Operation *> worklist;
    for (LoadOp load : loads) {
      moveSet.insert(load.getOperation());
      worklist.push_back(load.getOperation());
    }
    while (!worklist.empty()) {
      Operation *operation = worklist.pop_back_val();
      for (Value operand : operation->getOperands()) {
        Operation *definition = operand.getDefiningOp();
        if (!definition || definition->getBlock() != operation->getBlock() ||
            definition->isBeforeInBlock(earliestUser) ||
            moveSet.contains(definition))
          continue;
        auto dependencyLoad = dyn_cast<LoadOp>(definition);
        if ((!dependencyLoad && !isMemoryEffectFree(definition)) ||
            (dependencyLoad && dependencyLoad.getIsVolatile()))
          return failure();
        moveSet.insert(definition);
        worklist.push_back(definition);
      }
    }
    SmallVector<Operation *> orderedMove =
        blockOrder(moveSet, earliestUser->getBlock());
    for (Operation *operation : orderedMove)
      operation->moveBefore(earliestUser);
    llvm::sort(loads, [](LoadOp left, LoadOp right) {
      return left->isBeforeInBlock(right);
    });
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
    auto pointer = joinValues(builder, loads.front().getLoc(), pointers);
    if (failed(pointer))
      return failure();
    Value mask;
    if (masked) {
      auto joined = joinValues(builder, loads.front().getLoc(), masks);
      if (failed(joined))
        return failure();
      mask = *joined;
    }
    Value other;
    if (hasOther) {
      auto joined = joinValues(builder, loads.front().getLoc(), others);
      if (failed(joined))
        return failure();
      other = *joined;
    }
    auto grouped = LoadOp::create(
        builder, loads.front().getLoc(), *pointer, mask, other,
        loads.front().getCache(), loads.front().getEvict(),
        loads.front().getIsVolatile());
    SmallVector<Value> pieces = splitValue(
        builder, loads.front().getLoc(), grouped.getResult(), factor);
    for (auto [load, piece] : llvm::zip(loads, pieces))
      load.getResult().replaceAllUsesWith(piece);
    for (LoadOp load : llvm::reverse(loads))
      load.erase();
    groupedLoad = true;
  }

  for (auto &entry : storeGroups) {
    auto &stores = entry.second;
    if (stores.size() != static_cast<size_t>(factor) ||
        !isa<RankedTensorType>(stores.front().getPtr().getType()))
      continue;
    bool masked = static_cast<bool>(stores.front().getMask());
    if (llvm::any_of(stores, [&](StoreOp store) {
          return static_cast<bool>(store.getMask()) != masked ||
                 !store.getBoundaryCheck().empty();
        }))
      return failure();
    // Every grouped value/mask must dominate the replacement store.  The
    // phase-major expansion keeps the old stores ordered after their value
    // producers, so place the grouped store after the final old store before
    // erasing the originals.  Inserting before the first store would make the
    // later virtual-program values violate SSA dominance.
    OpBuilder builder(stores.back());
    builder.setInsertionPointAfter(stores.back());
    SmallVector<Value> pointers, values, masks;
    for (StoreOp store : stores) {
      pointers.push_back(store.getPtr());
      values.push_back(store.getValue());
      if (masked)
        masks.push_back(store.getMask());
    }
    auto pointer = joinValues(builder, stores.front().getLoc(), pointers);
    auto value = joinValues(builder, stores.front().getLoc(), values);
    if (failed(pointer) || failed(value))
      return failure();
    if (masked) {
      auto mask = joinValues(builder, stores.front().getLoc(), masks);
      if (failed(mask))
        return failure();
      StoreOp::create(builder, stores.front().getLoc(), *pointer, *value, *mask,
                      stores.front().getCache(), stores.front().getEvict());
    } else {
      StoreOp::create(builder, stores.front().getLoc(), *pointer, *value,
                      stores.front().getCache(), stores.front().getEvict());
    }
    for (StoreOp store : llvm::reverse(stores))
      store.erase();
  }
  // Removing the grouped terminals above can leave the now redundant split
  // tree dead.  Erase from leaves to root; live boundary splits are retained.
  SmallVector<SplitOp> splits;
  func.walk([&](SplitOp split) { splits.push_back(split); });
  for (SplitOp split : llvm::reverse(splits))
    if (split->use_empty())
      split.erase();
  func.walk([&](Operation *op) {
    op->removeAttr(kBridgeRoleAttr);
    op->removeAttr(kBridgeOrdinalAttr);
  });
  func->setAttr(kSplitElisionAttr,
                IntegerAttr::get(IntegerType::get(func.getContext(), 64),
                                 splitElisionCount));
  // A scalar reduction result cannot be joined with tensor values.  In that
  // case the route still performs real grouped tensor loads while retaining the
  // already phase-major scalar stores.  Success continues to require a material
  // load grouping, so this cannot degrade into plain unroll/reorder.
  return success(groupedLoad);
}

// Coarse logical vectorization for an existing loop.  After the general
// phase-major proof has exposed every read, corresponding clones of the same
// source load are joined into one wider tensor load and immediately split
// back into the original SSA leaves.  All compute and loop-carried state
// operations retain their exact source order; this is memory-lane
// vectorization, not floating-point reassociation.
LogicalResult materializeOrderPreservingLoadVectorization(
    FuncOp func, std::optional<StringRef> subject, int64_t unrollFactor) {
  if (unrollFactor < 2 || !llvm::isPowerOf2_64(unrollFactor) ||
      failed(materializePhase(
          func, subject, /*generalCardinality=*/true, unrollFactor,
          /*onlyBlock=*/nullptr, /*clearLineage=*/false)))
    return failure();

  llvm::DenseMap<Block *, llvm::DenseMap<int64_t, SmallVector<LoadOp>>>
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
    auto index = op->getAttrOfType<IntegerAttr>(kRoleIndexAttr);
    if (!index) {
      malformed = true;
      return;
    }
    Block *block = op->getBlock();
    if (!groupsByBlock.contains(block))
      blocks.push_back(block);
    groupsByBlock[block][index.getInt()].push_back(load);
  });
  if (malformed || blocks.empty())
    return failure();

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

  int64_t groupedLoadCount = 0;
  for (Block *block : blocks) {
    auto &groups = groupsByBlock[block];
    SmallVector<int64_t> ordinals;
    for (auto &entry : groups)
      ordinals.push_back(entry.first);
    llvm::sort(ordinals);
    for (int64_t ordinal : ordinals) {
      SmallVector<LoadOp> &loads = groups[ordinal];
      if (loads.size() != static_cast<size_t>(unrollFactor))
        return failure();
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
        return failure();

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
        return failure();
      Value mask;
      if (masked) {
        FailureOr<Value> joined = joinValues(
            builder, first.getLoc(), masks);
        if (failed(joined))
          return failure();
        mask = *joined;
      }
      Value other;
      if (hasOther) {
        FailureOr<Value> joined = joinValues(
            builder, first.getLoc(), others);
        if (failed(joined))
          return failure();
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
  if (groupedLoadCount == 0)
    return failure();

  SmallVector<Operation *> lineage;
  func.walk([&](Operation *op) {
    auto role = op->getAttrOfType<StringAttr>(kRoleAttr);
    if (!role)
      return;
    auto roleSubject = op->getAttrOfType<StringAttr>(kRoleSubjectAttr);
    if (!subject || (roleSubject && roleSubject.getValue() == *subject))
      lineage.push_back(op);
  });
  clearRoles(lineage);
  int64_t prior = 0;
  if (auto count =
          func->getAttrOfType<IntegerAttr>(kVectorizedLoadGroupCountAttr))
    prior = count.getInt();
  func->setAttr(kVectorizedLoadGroupCountAttr,
                IntegerAttr::get(IntegerType::get(func.getContext(), 64),
                                 prior + groupedLoadCount));
  return success();
}

LogicalResult materializeLogical(FuncOp func,
                                 std::optional<StringRef> subject = std::nullopt,
                                 int64_t unrollFactor = 4,
                                 Block *onlyBlock = nullptr) {
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
    return materializeOrderPreservingLoadVectorization(
        func, subject, unrollFactor);

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
        if (failed(materializeLogical(func, subject, unrollFactor, block)))
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
  reductions.back()->getResult(0).replaceAllUsesWith(reduced.getResult().front());
  for (Operation *op : llvm::reverse(reductions))
    op->erase();
  clearRoles(loads);
  clearRoles(computes);
  return success();
}

LogicalResult materializeRuntimeGuardedLogical(FuncOp func) {
  SmallVector<scf::ForOp> subjects;
  func.walk([&](scf::ForOp loop) {
    if (loop->hasAttr(kSubjectAttr))
      subjects.push_back(loop);
  });
  if (subjects.size() != 1)
    return failure();
  scf::ForOp loop = subjects.front();
  LoadOp sourceLoad;
  arith::AddFOp sourceReduce;
  if (!isEligibleRuntimeTopkLoop(loop, sourceLoad, sourceReduce))
    return failure();

  Location loc = loop.getLoc();
  OpBuilder builder(loop);
  Value four = arith::ConstantIntOp::create(builder, loc, 4, 32);
  Value condition = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::eq, loop.getUpperBound(), four);
  auto ifOp = scf::IfOp::create(builder, loc, loop.getResultTypes(), condition,
                                /*addThenBlock=*/true,
                                /*addElseBlock=*/true);
  ifOp->setAttr(kRuntimeGuardAttr,
                builder.getStringAttr("topk_eq_4_original_else"));

  OpBuilder thenBuilder = ifOp.getThenBodyBuilder();
  SmallVector<Value> loaded;
  for (int64_t iteration = 0; iteration < 4; ++iteration) {
    IRMapping mapping;
    Value iv = arith::ConstantIntOp::create(thenBuilder, loc, iteration, 32);
    mapping.map(loop.getInductionVar(), iv);
    mapping.map(loop.getRegionIterArg(0), loop.getInitArgs().front());
    bool foundLoad = false;
    for (Operation &operation : loop.getBody()->without_terminator()) {
      if (auto load = dyn_cast<LoadOp>(operation)) {
        auto clonedLoad = LoadOp::create(
            thenBuilder, load.getLoc(), mapping.lookupOrDefault(load.getPtr()),
            mapping.lookupOrDefault(load.getMask()),
            mapping.lookupOrDefault(load.getOther()), load.getCache(),
            load.getEvict(), load.getIsVolatile());
        loaded.push_back(clonedLoad.getResult());
        foundLoad = true;
        break;
      }
      thenBuilder.clone(operation, mapping);
    }
    if (!foundLoad)
      return failure();
  }
  if (loaded.size() != 4)
    return failure();
  Value sum01 = arith::AddFOp::create(thenBuilder, loc, loaded[0], loaded[1]);
  Value sum23 = arith::AddFOp::create(thenBuilder, loc, loaded[2], loaded[3]);
  Value sum = arith::AddFOp::create(thenBuilder, loc, sum01, sum23);
  scf::YieldOp::create(thenBuilder, loc, sum);

  OpBuilder elseBuilder = ifOp.getElseBodyBuilder();
  auto elseLoop = cast<scf::ForOp>(elseBuilder.clone(*loop.getOperation()));
  elseLoop->removeAttr(kSubjectAttr);
  scf::YieldOp::create(elseBuilder, loc, elseLoop.getResults());

  loop.getResults().replaceAllUsesWith(ifOp.getResults());
  loop.erase();
  return success();
}

// Partially unroll a provider-certified runtime loop without assuming a
// compile-time trip count.  The original first lane is always valid because
// it is still guarded by scf.for.  Additional lanes retain the exact body and
// are guarded against the runtime upper bound.  A closed carried recurrence
// is threaded through each guarded lane in the original sequential order;
// there is no reassociation or floating-point identity assumption.
LogicalResult materializeAffineRuntimePartialUnroll(scf::ForOp loop,
                                                    int64_t factor) {
  if (factor < 2 ||
      loop.getRegionIterArgs().size() != loop.getResults().size())
    return failure();
  Location loc = loop.getLoc();
  Value originalStep = loop.getStep();
  OpBuilder before(loop);
  Value factorValue = arith::ConstantIntOp::create(
      before, loc, originalStep.getType(), factor);
  Value widenedStep = arith::MulIOp::create(
      before, loc, originalStep, factorValue);

  Operation *terminator = loop.getBody()->getTerminator();
  OpBuilder bodyBuilder(terminator);
  SmallVector<Operation *> sourceBody;
  for (Operation &operation : loop.getBody()->without_terminator())
    sourceBody.push_back(&operation);
  auto originalYield = dyn_cast<scf::YieldOp>(terminator);
  if (!originalYield ||
      originalYield.getNumOperands() != loop.getRegionIterArgs().size())
    return failure();
  SmallVector<Value> carried(originalYield.getOperands());
  for (int64_t lane = 1; lane < factor; ++lane) {
    Value laneValue = arith::ConstantIntOp::create(
        bodyBuilder, loc, originalStep.getType(), lane);
    Value offset = arith::MulIOp::create(
        bodyBuilder, loc, originalStep, laneValue);
    Value laneIv = arith::AddIOp::create(
        bodyBuilder, loc, loop.getInductionVar(), offset);
    Value inBounds = arith::CmpIOp::create(
        bodyBuilder, loc, arith::CmpIPredicate::slt, laneIv,
        loop.getUpperBound());
    SmallVector<Type> resultTypes;
    llvm::append_range(resultTypes, loop.getResultTypes());
    auto guard = scf::IfOp::create(
        bodyBuilder, loc, resultTypes, inBounds,
        /*addThenBlock=*/true,
        /*addElseBlock=*/!resultTypes.empty());
    OpBuilder thenBuilder = guard.getThenBodyBuilder();
    IRMapping mapping;
    mapping.map(loop.getInductionVar(), laneIv);
    for (auto [regionValue, currentValue] :
         llvm::zip(loop.getRegionIterArgs(), carried))
      mapping.map(regionValue, currentValue);
    for (Operation *operation : sourceBody)
      thenBuilder.clone(*operation, mapping);
    SmallVector<Value> next;
    for (Value value : originalYield.getOperands())
      next.push_back(mapping.lookupOrDefault(value));
    scf::YieldOp::create(thenBuilder, loc, next);
    if (!resultTypes.empty()) {
      OpBuilder elseBuilder = guard.getElseBodyBuilder();
      scf::YieldOp::create(elseBuilder, loc, carried);
      carried.assign(guard.getResults().begin(), guard.getResults().end());
    }
  }
  if (!carried.empty())
    originalYield->setOperands(carried);
  loop.getStepMutable().assign(widenedStep);
  loop->removeAttr("tt.loop_unroll_factor");
  return success();
}

// Materialize a provider-certified ordered runtime partition as two loops:
// an unconditional factor-wide main loop and one scalar tail.  Unlike the
// guarded partial-unroll path, this pays no per-lane control cost in complete
// groups.  The provider has already proved a positive static step, bounded
// signed extent, disjoint output service and closed carried recurrences.
LogicalResult materializeAffineRuntimeMainTail(scf::ForOp loop,
                                               int64_t factor) {
  if (factor < 2 ||
      loop.getRegionIterArgs().size() != loop.getResults().size())
    return failure();
  auto step = splatInteger(loop.getStep());
  if (!step || *step <= 0)
    return failure();
  Type inductionType = loop.getStep().getType();
  unsigned bitWidth = 64;
  if (auto integer = dyn_cast<IntegerType>(inductionType))
    bitWidth = integer.getWidth();
  else if (!isa<IndexType>(inductionType))
    return failure();
  int64_t signedMaximum =
      bitWidth >= 64 ? std::numeric_limits<int64_t>::max()
                     : (int64_t{1} << (bitWidth - 1)) - 1;
  if (*step > signedMaximum / factor)
    return failure();

  Location loc = loop.getLoc();
  OpBuilder before(loop);
  Value factorValue = arith::ConstantIntOp::create(
      before, loc, inductionType, factor);
  Value widenedStep = arith::MulIOp::create(
      before, loc, loop.getStep(), factorValue);
  Value zero = arith::ConstantIntOp::create(
      before, loc, inductionType, 0);
  Value hasWork = arith::CmpIOp::create(
      before, loc, arith::CmpIPredicate::slt, loop.getLowerBound(),
      loop.getUpperBound());
  Value rawDistance = arith::SubIOp::create(
      before, loc, loop.getUpperBound(), loop.getLowerBound());
  Value distance = arith::SelectOp::create(
      before, loc, hasWork, rawDistance, zero);
  Value stepQuotient = arith::DivSIOp::create(
      before, loc, distance, loop.getStep());
  Value stepRemainder = arith::RemSIOp::create(
      before, loc, distance, loop.getStep());
  Value hasPartialIteration = arith::CmpIOp::create(
      before, loc, arith::CmpIPredicate::ne, stepRemainder, zero);
  Value one = arith::ConstantIntOp::create(
      before, loc, inductionType, 1);
  Value partialIteration = arith::SelectOp::create(
      before, loc, hasPartialIteration, one, zero);
  Value tripCount = arith::AddIOp::create(
      before, loc, stepQuotient, partialIteration);
  Value fullGroupCount = arith::DivSIOp::create(
      before, loc, tripCount, factorValue);
  Value mainIterationCount = arith::MulIOp::create(
      before, loc, fullGroupCount, factorValue);
  Value mainDistance = arith::MulIOp::create(
      before, loc, mainIterationCount, loop.getStep());
  Value mainEnd = arith::AddIOp::create(
      before, loc, loop.getLowerBound(), mainDistance);

  auto mainLoop = scf::ForOp::create(
      before, loc, loop.getLowerBound(), mainEnd, widenedStep,
      loop.getInitArgs());
  mainLoop->setAttrs(loop->getAttrs());
  mainLoop->setAttr(kMainTailAttr, before.getStringAttr("main"));
  mainLoop->removeAttr("tt.loop_unroll_factor");
  auto sourceYield = loop.getBody()->empty()
                         ? scf::YieldOp()
                         : dyn_cast<scf::YieldOp>(loop.getBody()->back());
  if (!sourceYield ||
      sourceYield.getNumOperands() != loop.getRegionIterArgs().size())
    return failure();
  SmallVector<Operation *> sourceBody;
  for (Operation &operation : loop.getBody()->without_terminator())
    sourceBody.push_back(&operation);

  auto mainYield = mainLoop.getBody()->empty()
                       ? scf::YieldOp()
                       : dyn_cast<scf::YieldOp>(mainLoop.getBody()->back());
  OpBuilder mainBuilder = mainYield
      ? OpBuilder(mainYield)
      : OpBuilder::atBlockEnd(mainLoop.getBody());
  SmallVector<Value> carried(mainLoop.getRegionIterArgs().begin(),
                             mainLoop.getRegionIterArgs().end());
  for (int64_t lane = 0; lane < factor; ++lane) {
    Value laneIv = mainLoop.getInductionVar();
    if (lane != 0) {
      Value laneValue = arith::ConstantIntOp::create(
          mainBuilder, loc, inductionType, lane);
      Value offset = arith::MulIOp::create(
          mainBuilder, loc, loop.getStep(), laneValue);
      laneIv = arith::AddIOp::create(
          mainBuilder, loc, mainLoop.getInductionVar(), offset);
    }
    IRMapping mapping;
    mapping.map(loop.getInductionVar(), laneIv);
    for (auto [original, current] :
         llvm::zip(loop.getRegionIterArgs(), carried))
      mapping.map(original, current);
    for (Operation *operation : sourceBody)
      mainBuilder.clone(*operation, mapping);
    SmallVector<Value> next;
    for (Value value : sourceYield.getOperands())
      next.push_back(mapping.lookupOrDefault(value));
    carried = std::move(next);
  }
  if (mainYield)
    mainYield->setOperands(carried);
  else
    mainYield = scf::YieldOp::create(mainBuilder, loc, carried);

  OpBuilder after(mainLoop);
  after.setInsertionPointAfter(mainLoop);
  auto tailLoop = scf::ForOp::create(
      after, loc, mainEnd, loop.getUpperBound(), loop.getStep(),
      mainLoop.getResults());
  tailLoop->setAttrs(loop->getAttrs());
  tailLoop->setAttr(kMainTailAttr, after.getStringAttr("tail"));
  tailLoop->removeAttr("tt.loop_unroll_factor");
  auto tailYield = tailLoop.getBody()->empty()
                       ? scf::YieldOp()
                       : dyn_cast<scf::YieldOp>(tailLoop.getBody()->back());
  OpBuilder tailBuilder = tailYield
      ? OpBuilder(tailYield)
      : OpBuilder::atBlockEnd(tailLoop.getBody());
  IRMapping tailMapping;
  tailMapping.map(loop.getInductionVar(), tailLoop.getInductionVar());
  for (auto [original, current] :
       llvm::zip(loop.getRegionIterArgs(), tailLoop.getRegionIterArgs()))
    tailMapping.map(original, current);
  for (Operation *operation : sourceBody)
    tailBuilder.clone(*operation, tailMapping);
  SmallVector<Value> tailResults;
  for (Value value : sourceYield.getOperands())
    tailResults.push_back(tailMapping.lookupOrDefault(value));
  if (tailYield)
    tailYield->setOperands(tailResults);
  else
    scf::YieldOp::create(tailBuilder, loc, tailResults);

  loop.getResults().replaceAllUsesWith(tailLoop.getResults());
  loop.erase();
  return success();
}

class HBVLoopMaterializePass
    : public impl::TritonHBVLoopMaterializeBase<HBVLoopMaterializePass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto entry = findEntryPoint(module);
    std::string reason;
    auto parsed = succeeded(entry) ? parseFunctionPlan(*entry, reason)
                                   : FailureOr<ParsedLoopPlan>(failure());
    if (failed(entry))
      reason = "missing unique public Loop kernel";
    if (failed(parsed)) {
      reportFailure(module, reason);
      return signalPassFailure();
    }
    if (!parsed->controlled)
      return;
    Builder builder(module.getContext());
    auto dependence = (*entry)->getAttrOfType<StringAttr>(kDependenceAttr);
    if (parsed->stateAxisLogical) {
      SmallVector<StateAxisNormalizationRegion, 2> subjects =
          collectStateAxisNormalizations(*entry);
      if (!dependence ||
          dependence.getValue() !=
              "provider_closed_sibling_state_operation_graph_v1" ||
          subjects.size() !=
              static_cast<size_t>(parsed->stateAxisGroupCount)) {
        reportFailure(
            module,
            "state-axis materialization lost its Provider operation graph");
        return signalPassFailure();
      }
      for (auto [subject, group] :
           llvm::zip(subjects, parsed->stateAxisGroups)) {
        if (!stateAxisSubjectMatchesPlan(subject, group)) {
          reportFailure(
              module,
              "state-axis materialization subject contradicts its Plan group");
          return signalPassFailure();
        }
      }
      for (StateAxisNormalizationRegion &subject :
           llvm::reverse(subjects)) {
        if (failed(materializeStateAxisNormalization(subject))) {
          reportFailure(
              module,
              "state-axis sibling-group materialization failed");
          return signalPassFailure();
        }
      }
      if (!collectStateAxisNormalizations(*entry).empty()) {
        reportFailure(
            module,
            "state-axis route retained an unmaterialized sibling group");
        return signalPassFailure();
      }
      (*entry)->setAttr(
          kRealizedAttr,
          builder.getStringAttr("logical_group_state_axis_slp"));
      (*entry)->setAttr(kPostconditionAttr,
                        builder.getStringAttr("pass"));
      return;
    }
    if (parsed->exactPrefixReduction) {
      SmallVector<ExactPrefixReduction> subjects =
          collectExactPrefixReductions(*entry);
      auto elementBudget =
          module->getAttrOfType<IntegerAttr>(kExactPrefixVectorElementBudgetAttr);
      if (!elementBudget || elementBudget.getInt() < 1) {
        reportFailure(
            module,
            "exact-prefix materializer lacks its target-provider element budget");
        return signalPassFailure();
      }
      if (subjects.size() == 1 && subjects.front().vectorWidth >
              elementBudget.getInt()) {
        reportFailure(
            module,
            "exact-prefix subject exceeds the explicit vector materialization acquisition budget");
        return signalPassFailure();
      }
      if (subjects.size() != 1 || failed(materializeExactPrefixReduction(
                                      subjects.front(), elementBudget.getInt()))) {
        reportFailure(module,
                      "exact-prefix reduction materialization failed");
        return signalPassFailure();
      }
      (*entry)->setAttr(
          kPrefixReductionAttr,
          builder.getStringAttr("predicated_exact_prefix_reduction_v1"));
      (*entry)->setAttr(
          kRealizedAttr,
          builder.getStringAttr("predicated_exact_prefix_reduction"));
      (*entry)->setAttr(kPostconditionAttr,
                        builder.getStringAttr("pass"));
      return;
    }
    if (parsed->bridgeConstructed) {
      if (!dependence ||
          (dependence.getValue() != "bridge_pid_partitioned_disjoint_v1" &&
           dependence.getValue() != "multi_axis_program_disjoint_v1")) {
        reportFailure(module,
                      "Bridge materialization lost its program-independence certificate");
        return signalPassFailure();
      }
    } else if (parsed->route != kPipelineRoute &&
               !parsed->runtimeGuardedLogical) {
      if (!dependence ||
          (!isExistingUnrollOrderCertificate(dependence.getValue()) &&
           dependence.getValue() !=
               "nested_inner_dimension_independent_sink_v1" &&
           dependence.getValue() !=
               "provider_closed_complete_static_nest_v1" &&
           dependence.getValue() !=
               "independent_iteration_exact_inner_reduction_v1" &&
           !isAffineRuntimeOrderPreservingCertificate(
               dependence.getValue()))) {
        reportFailure(module,
                      "unroll materialization lost its iteration-dependence certificate");
        return signalPassFailure();
      }
    }
    if (parsed->affineRuntimePartial) {
      SmallVector<scf::ForOp> subjects;
      (*entry).walk([&](scf::ForOp loop) {
        if (loop->hasAttr(kSubjectAttr))
          subjects.push_back(loop);
      });
      // Triton's generic loop-unroll pass may rebuild a dynamic loop even
      // when it cannot consume its factor, dropping transient operation
      // attributes.  Re-identification is safe only through the same unique
      // provider certificate; no location or workload identity is used.
      if (subjects.empty()) {
        (*entry).walk([&](scf::ForOp loop) {
          if (certifyAffineRuntimeOrderPreserving(loop).safe)
            subjects.push_back(loop);
        });
        if (subjects.size() == 1)
          subjects.front()->setAttr(
              kSubjectAttr,
              subjectLocator(module.getContext(), *parsed));
      }
      if (subjects.empty() || subjects.size() > 2) {
        reportFailure(
            module,
            Twine("guarded affine runtime subject count is ") +
                Twine(subjects.size()));
        return signalPassFailure();
      }
      bool mainTailEligible = dependence &&
          dependence.getValue() ==
              "affine_runtime_program_partition_order_preserving_v1";
      bool forceGuarded =
          parsed->affineRuntimeMaterializationPolicy == "guarded_lanes";
      bool forceMainTail =
          parsed->affineRuntimeMaterializationPolicy == "main_tail";
      if (forceMainTail && !mainTailEligible) {
        reportFailure(
            module,
            "explicit main-tail requires contiguous ordered ownership");
        return signalPassFailure();
      }
      bool mainTail = false;
      if (subjects.size() == 1) {
        unsigned existingStores = 0;
        subjects.front().walk([&](StoreOp) { ++existingStores; });
        LogicalResult result = success();
        if (existingStores < static_cast<unsigned>(parsed->unrollFactor))
          result = (mainTailEligible && !forceGuarded) || forceMainTail
                       ? materializeAffineRuntimeMainTail(
                             subjects.front(), parsed->unrollFactor)
                       : materializeAffineRuntimePartialUnroll(
                             subjects.front(), parsed->unrollFactor);
        if (failed(result)) {
          reportFailure(
              module,
              Twine("ordered affine runtime materialization rejected factor=") +
                  Twine(parsed->unrollFactor) + " iter_args=" +
                  Twine(subjects.front().getRegionIterArgs().size()) +
                  " results=" + Twine(subjects.front().getNumResults()));
          return signalPassFailure();
        }
        mainTail = ((mainTailEligible && !forceGuarded) || forceMainTail) &&
                   existingStores <
                       static_cast<unsigned>(parsed->unrollFactor);
      }
      // Main-tail replaces the original subject with two proved descendants;
      // never retain handles into the erased source loop.
      subjects.clear();
      (*entry).walk([&](scf::ForOp loop) {
        if (loop->hasAttr(kSubjectAttr))
          subjects.push_back(loop);
      });
      unsigned stores = 0;
      unsigned atomics = 0;
      for (scf::ForOp subject : subjects) {
        subject.walk([&](Operation *operation) {
          stores += isa<StoreOp>(operation);
          atomics += isa<AtomicRMWOp, AtomicCASOp>(operation);
        });
        subject->removeAttr("tt.loop_unroll_factor");
      }
      if (stores < static_cast<unsigned>(parsed->unrollFactor) ||
          atomics != 0) {
        reportFailure(module,
                      "guarded affine runtime postcondition is incomplete");
        return signalPassFailure();
      }
      (*entry)->setAttr(
          kRealizedAttr,
          builder.getStringAttr(
              mainTail ? "affine_runtime_main_tail"
                       : "affine_runtime_partial_unroll"));
    } else if (parsed->route == kPipelineRoute) {
      SmallVector<scf::ForOp> subjects;
      (*entry).walk([&](scf::ForOp loop) {
        if (loop->hasAttr(kSubjectAttr))
          subjects.push_back(loop);
      });
      auto stages = subjects.size() == 1
                        ? subjects.front()->getAttrOfType<IntegerAttr>("tt.num_stages")
                        : IntegerAttr();
      if (subjects.size() != 1 || !stages ||
          stages.getInt() != parsed->stageCount) {
        reportFailure(module,
                      "native pipeline subject or resolved stage request was not preserved");
        return signalPassFailure();
      }
      (*entry)->setAttr(kRealizedAttr, builder.getStringAttr("software_pipeline"));
    } else {
      bool hadStaticMainPartitionLineage = false;
      bool hadStaticTailPartitionLineage = false;
      (*entry).walk([&](Operation *operation) {
        auto partition = operation->getAttrOfType<StringAttr>(
            kUnrollPartitionLineageAttr);
        if (!partition)
          return;
        hadStaticMainPartitionLineage |= partition.getValue() == "main";
        hadStaticTailPartitionLineage |= partition.getValue() == "tail";
      });
      bool materialized = parsed->runtimeGuardedLogical
                              ? succeeded(materializeRuntimeGuardedLogical(*entry))
                          : parsed->bridgeConstructed && parsed->route == kPhaseRoute
                              ? succeeded(materializeBridgePhase(*entry))
                          : parsed->bridgeConstructed && parsed->route == kLogicalRoute
                              ? succeeded(materializeBridgeLogical(
                                    *entry, parsed->unrollFactor,
                                    parsed->bridgeInvariantHoisting,
                                    parsed->bridgeTensorLaneFusion,
                                    parsed->bridgeExactSplitElision))
                          : [&]() {
                              auto subjects = collectRoleSubjects(*entry);
                              if (subjects.empty())
                                return false;
                              for (const std::string &subject : subjects) {
                                std::optional<StringRef> filter = subject.empty()
                                    ? std::nullopt
                                    : std::optional<StringRef>(subject);
                                int64_t subjectFactor = parsed->unrollFactor;
                                if (parsed->providerBoundSubjectSet) {
                                  auto member = llvm::find_if(
                                      parsed->providerBoundMembers,
                                      [&](const auto &item) {
                                        return item.memberRef == subject;
                                      });
                                  if (member ==
                                      parsed->providerBoundMembers.end())
                                    return false;
                                  subjectFactor = member->routeFactor;
                                }
                                bool coarseOrderPreserving =
                                    dependence &&
                                    (dependence.getValue() ==
                                         "existing_order_preserving_read_exposure_v1" ||
                                     dependence.getValue() ==
                                         "existing_affine_pointer_read_exposure_v1" ||
                                     dependence.getValue() ==
                                         "per_loop_order_preserving_read_exposure_v1");
                                LogicalResult result = parsed->route == kPhaseRoute
                                    ? materializePhase(
                                          *entry, filter,
                                          parsed->providerClosedStatic ||
                                              coarseOrderPreserving,
                                          subjectFactor)
                                    : materializeLogical(
                                          *entry, filter, subjectFactor);
                                if (failed(result))
                                  return false;
                              }
                              return true;
                            }();
      if (!materialized) {
        reportFailure(module, "full-unroll clone lineage cannot realize the selected Loop route");
        return signalPassFailure();
      }
      if (parsed->providerBoundSubjectSet)
        (*entry)->setAttr(
            kProviderBoundMembersAttr,
            builder.getStringAttr(
                parsed->providerBoundMemberSignature));
      unsigned dynamicMainLoops = 0;
      unsigned dynamicTailLoops = 0;
      (*entry).walk([&](scf::ForOp loop) {
        auto role = loop->getAttrOfType<StringAttr>(kMainTailAttr);
        dynamicMainLoops += role && role.getValue() == "main";
        dynamicTailLoops += role && role.getValue() == "tail";
      });
      // Canonicalization may erase a statically exhausted main loop after
      // unrolling while retaining only its ordered epilogue loop.  A surviving
      // `tail` marker is still direct evidence that LoopUnroll consumed the
      // requested factor; requiring a live main loop incorrectly rejects that
      // valid static main-plus-remainder materialization.
      // A nondivisible exact-static factor can consume both SCF loops and
      // inline the ordered epilogue into the parent block.  In that case the
      // operation-owned main/tail lineage is the proof that materialization
      // consumed a factor-wide main group and preserved its scalar tail.
      bool factorMainTail = dynamicMainLoops + dynamicTailLoops > 0 ||
                            (hadStaticMainPartitionLineage &&
                             hadStaticTailPartitionLineage);
      StringRef realized = factorMainTail
                               ? (parsed->route == kPhaseRoute
                                      ? "phase_major_factor_main_tail"
                                      : "logical_group_factor_main_tail")
                           : parsed->runtimeGuardedLogical
                               ? "logical_group_runtime_guarded"
                           : parsed->bridgeConstructed && parsed->route == kPhaseRoute
                               ? "phase_major_bridge"
                           : parsed->bridgeConstructed && parsed->route == kLogicalRoute
                               ? (!parsed->bridgeInvariantHoisting &&
                                          !parsed->bridgeTensorLaneFusion
                                      ? "logical_group_bridge"
                                  : parsed->bridgeInvariantHoisting &&
                                          parsed->bridgeTensorLaneFusion
                                      ? "factorized_bridge_GHF"
                                  : parsed->bridgeInvariantHoisting
                                      ? "factorized_bridge_GH"
                                      : "logical_group_bridge_GF")
                           : parsed->providerClosedStatic
                               ? "phase_major_provider_closed"
                           : parsed->route == kPhaseRoute ? "phase_major"
                                                         : "logical_group";
      (*entry)->setAttr(kRealizedAttr, builder.getStringAttr(realized));
    }
    (*entry)->setAttr(kPostconditionAttr, builder.getStringAttr("pass"));
    if (parsed->composedIntervention)
      (*entry)->setAttr(
          kCompositionPostconditionAttr,
          builder.getStringAttr("bridge_then_route_pass"));
  }
};

class HBVValidateLoopPlanPass
    : public impl::TritonHBVValidateLoopPlanBase<HBVValidateLoopPlanPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (module->hasAttr(kBundleAttr)) {
      reportFailure(module, "module Loop PlanBundle survived focal ownership closure");
      return signalPassFailure();
    }
    auto entry = findEntryPoint(module);
    std::string reason;
    auto parsed = succeeded(entry) ? parseFunctionPlan(*entry, reason)
                                   : FailureOr<ParsedLoopPlan>(failure());
    if (failed(entry))
      reason = "missing unique public Loop kernel";
    if (failed(parsed)) {
      reportFailure(module, reason);
      return signalPassFailure();
    }
    SmallVector<Operation *> roleOps;
    (*entry).walk([&](Operation *op) {
      if (op->hasAttr(kRoleAttr) || op->hasAttr(kRoleSubjectAttr) ||
          op->hasAttr(kRoleIndexAttr) ||
          op->hasAttr(kUnrollPartitionLineageAttr))
        roleOps.push_back(op);
    });
    if (!roleOps.empty()) {
      reportFailure(module, "transient Loop role lineage survived materialization");
      return signalPassFailure();
    }
    if (!parsed->controlled)
      return;
    auto route = (*entry)->getAttrOfType<StringAttr>(kRouteAttr);
    auto mechanismRoute =
        (*entry)->getAttrOfType<StringAttr>(kMechanismRouteAttr);
    auto routeSubtype =
        (*entry)->getAttrOfType<StringAttr>(kRouteSubtypeAttr);
    auto artifactRoute =
        (*entry)->getAttrOfType<StringAttr>(kArtifactRouteAttr);
    auto subject = (*entry)->getAttrOfType<StringAttr>(kSubjectRefAttr);
    auto realized = (*entry)->getAttrOfType<StringAttr>(kRealizedAttr);
    auto postcondition = (*entry)->getAttrOfType<StringAttr>(kPostconditionAttr);
    auto dependence = (*entry)->getAttrOfType<StringAttr>(kDependenceAttr);
    auto compositionSchema =
        (*entry)->getAttrOfType<StringAttr>(kCompositionSchemaAttr);
    auto compositionBridgeFactor =
        (*entry)->getAttrOfType<IntegerAttr>(kCompositionBridgeFactorAttr);
    auto compositionRouteFactor =
        (*entry)->getAttrOfType<IntegerAttr>(kCompositionRouteFactorAttr);
    auto compositionBridgeAxisDivisors =
        (*entry)->getAttrOfType<StringAttr>(
            kCompositionBridgeAxisDivisorsAttr);
    auto compositionPostcondition =
        (*entry)->getAttrOfType<StringAttr>(kCompositionPostconditionAttr);
    if (!route || route.getValue() != parsed->route ||
        !mechanismRoute ||
        mechanismRoute.getValue() != parsed->mechanismRoute ||
        !routeSubtype || routeSubtype.getValue() != parsed->routeSubtype ||
        !artifactRoute || artifactRoute.getValue() != parsed->artifactRoute ||
        !subject ||
        subject.getValue() != parsed->subjectRef || !realized ||
        !postcondition || postcondition.getValue() != "pass") {
      reportFailure(module, "Loop contract, subject, route, and postcondition do not correspond");
      return signalPassFailure();
    }
    auto providerBoundMembers =
        (*entry)->getAttrOfType<StringAttr>(kProviderBoundMembersAttr);
    if (parsed->providerBoundSubjectSet &&
        (!providerBoundMembers ||
         providerBoundMembers.getValue() !=
             parsed->providerBoundMemberSignature)) {
      reportFailure(
          module,
          "provider-bound Loop member/factor artifact identity is missing");
      return signalPassFailure();
    }
    if (!parsed->providerBoundSubjectSet && providerBoundMembers) {
      reportFailure(
          module,
          "unselected provider-bound Loop member identity survived");
      return signalPassFailure();
    }
    if (parsed->compositionV3) {
      std::string expectedDivisors = llvm::join(
          llvm::map_range(
              parsed->bridgeAxisDivisors,
              [](int64_t divisor) { return std::to_string(divisor); }),
          ",");
      if (!compositionBridgeAxisDivisors ||
          compositionBridgeAxisDivisors.getValue() != expectedDivisors) {
        reportFailure(
            module,
            "multi-axis Bridge postcondition lost its Plan-bound divisor vector");
        return signalPassFailure();
      }
    }
    if (parsed->composedIntervention &&
        (!compositionSchema ||
         compositionSchema.getValue() != parsed->compositionSchema ||
         !compositionBridgeFactor ||
         compositionBridgeFactor.getInt() != parsed->bridgeFactor ||
         !compositionRouteFactor ||
         compositionRouteFactor.getInt() != parsed->routeFactor ||
         !compositionPostcondition ||
         compositionPostcondition.getValue() != "bridge_then_route_pass")) {
      reportFailure(
          module,
          "Bridge then route composition lost independent factor lineage");
      return signalPassFailure();
    }
    if (parsed->bridgeConstructed &&
        (!dependence ||
         (dependence.getValue() != "bridge_pid_partitioned_disjoint_v1" &&
          dependence.getValue() != "multi_axis_program_disjoint_v1"))) {
      reportFailure(module,
                    "Bridge postcondition lost its program-independence certificate");
      return signalPassFailure();
    }
    if (!parsed->bridgeConstructed && parsed->route != kPipelineRoute &&
        !parsed->runtimeGuardedLogical && !parsed->exactPrefixReduction &&
        !parsed->stateAxisLogical &&
        (!dependence ||
         (!isExistingUnrollOrderCertificate(dependence.getValue()) &&
          dependence.getValue() !=
              "nested_inner_dimension_independent_sink_v1" &&
          dependence.getValue() !=
              "provider_closed_complete_static_nest_v1" &&
          dependence.getValue() !=
              "independent_iteration_exact_inner_reduction_v1" &&
          !isAffineRuntimeOrderPreservingCertificate(
              dependence.getValue())))) {
      reportFailure(module,
                    "unroll postcondition lost its iteration-dependence certificate");
      return signalPassFailure();
    }
    if (parsed->stateAxisLogical) {
      auto groupCount = (*entry)->getAttrOfType<IntegerAttr>(
          kStateAxisGroupCountAttr);
      unsigned subjectMarkers = 0;
      unsigned packedRoots = 0;
      unsigned packedNodes = 0;
      unsigned crossStateConsumers = 0;
      bool malformedArtifact = false;
      (*entry).walk([&](Operation *operation) {
        subjectMarkers += operation->hasAttr(kSubjectAttr);
        packedNodes += operation->hasAttr(kStateAxisPackableNodeAttr);
        auto role = operation->getAttrOfType<StringAttr>(
            kStateAxisArtifactAttr);
        if (!role)
          return;
        if (role.getValue() == "packed_root")
          ++packedRoots;
        else if (role.getValue() == "packed_node")
          malformedArtifact |=
              !operation->hasAttr(kStateAxisPackableNodeAttr);
        else if (role.getValue() == "cross_state_consumer")
          ++crossStateConsumers;
        else
          malformedArtifact = true;
      });
      int64_t expectedPackedNodes = 0;
      for (const ParsedLoopPlan::StateAxisGroup &group :
           parsed->stateAxisGroups)
        expectedPackedNodes += group.packableNodeCount;
      bool exactCounts =
          packedRoots == static_cast<unsigned>(parsed->stateAxisGroupCount) &&
          crossStateConsumers == packedRoots &&
          packedNodes == static_cast<unsigned>(expectedPackedNodes);
      if (!dependence ||
          dependence.getValue() !=
              "provider_closed_sibling_state_operation_graph_v1" ||
          realized.getValue() != "logical_group_state_axis_slp" ||
          !groupCount ||
          groupCount.getInt() != parsed->stateAxisGroupCount ||
          subjectMarkers !=
              static_cast<unsigned>(parsed->stateAxisGroupCount) ||
          !exactCounts || malformedArtifact ||
          !collectStateAxisNormalizations(*entry).empty()) {
        reportFailure(
            module,
            Twine("state-axis postcondition lost Provider/artifact correspondence: ") +
                "groups=" + Twine(parsed->stateAxisGroupCount) +
                " roots=" + Twine(packedRoots) +
                " nodes=" + Twine(packedNodes) + "/" +
                Twine(expectedPackedNodes) + " cross=" +
                Twine(crossStateConsumers) + " subjects=" +
                Twine(subjectMarkers));
        return signalPassFailure();
      }
      return;
    }
    if (parsed->exactPrefixReduction) {
      auto prefix = (*entry)->getAttrOfType<StringAttr>(kPrefixReductionAttr);
      if (!dependence ||
          dependence.getValue() !=
              "predicated_exact_prefix_reduction_v1" ||
          !prefix || prefix.getValue() !=
                         "predicated_exact_prefix_reduction_v1" ||
          realized.getValue() != "predicated_exact_prefix_reduction") {
        reportFailure(module,
                      "exact-prefix postcondition lost its provider proof");
        return signalPassFailure();
      }
      return;
    }
    if (parsed->route == kPipelineRoute) {
      SmallVector<scf::ForOp> subjects;
      (*entry).walk([&](scf::ForOp loop) {
        if (loop->hasAttr(kSubjectAttr))
          subjects.push_back(loop);
      });
      if (subjects.size() != 1 || realized.getValue() != "software_pipeline") {
        reportFailure(module, "pipeline route lost its live focal subject");
        return signalPassFailure();
      }
      return;
    }
    if (parsed->affineRuntimePartial) {
      unsigned subjects = 0;
      unsigned mainLoops = 0;
      unsigned tailLoops = 0;
      (*entry).walk([&](scf::ForOp loop) {
        subjects += loop->hasAttr(kSubjectAttr);
        auto role = loop->getAttrOfType<StringAttr>(kMainTailAttr);
        mainLoops += role && role.getValue() == "main";
        tailLoops += role && role.getValue() == "tail";
      });
      bool expectedMainTail = mainLoops != 0 || tailLoops != 0;
      bool realizedMatches = expectedMainTail
          ? realized.getValue() == "affine_runtime_main_tail" &&
                subjects == 2 && mainLoops == 1 && tailLoops == 1
          : realized.getValue() == "affine_runtime_partial_unroll" &&
                subjects >= 1 && subjects <= 2 && mainLoops == 0 &&
                tailLoops == 0;
      if (subjects < 1 || subjects > 2 ||
          !realizedMatches ||
          !dependence || !isAffineRuntimeOrderPreservingCertificate(
              dependence.getValue())) {
        reportFailure(
            module,
            "affine-runtime partial-unroll postcondition failed");
        return signalPassFailure();
      }
      return;
    }
    bool residualSubjectLoop = false;
    bool unclassifiedResidualSubjectLoop = false;
    unsigned dynamicMainLoops = 0;
    unsigned dynamicTailLoops = 0;
    (*entry).walk([&](scf::ForOp loop) {
      bool residualSubject = loop->hasAttr(kSubjectAttr);
      residualSubjectLoop |= residualSubject;
      auto role = loop->getAttrOfType<StringAttr>(kMainTailAttr);
      dynamicMainLoops += role && role.getValue() == "main";
      dynamicTailLoops += role && role.getValue() == "tail";
      unclassifiedResidualSubjectLoop |=
          residualSubject &&
          (!role || (role.getValue() != "main" &&
                     role.getValue() != "tail"));
    });
    StringRef expectedFactorRealized =
        parsed->route == kPhaseRoute
            ? "phase_major_factor_main_tail"
            : "logical_group_factor_main_tail";
    bool factorMainTail =
        (dynamicMainLoops + dynamicTailLoops > 0 &&
         !unclassifiedResidualSubjectLoop) ||
        (!residualSubjectLoop &&
         realized.getValue() == expectedFactorRealized);
    if (residualSubjectLoop &&
        (!factorMainTail || realized.getValue() != expectedFactorRealized)) {
      reportFailure(module, "full-unroll route retained the focal loop");
      return signalPassFailure();
    }
    if (parsed->route == kLogicalRoute) {
      unsigned logicalReductions = 0;
      unsigned runtimeGuards = 0;
      unsigned runtimeElseLoops = 0;
      unsigned runtimeThenLoads = 0;
      unsigned runtimeThenAdds = 0;
      unsigned bridgeJoins = 0;
      auto vectorizedLoadGroups = (*entry)->getAttrOfType<IntegerAttr>(
          kVectorizedLoadGroupCountAttr);
      (*entry).walk([&](JoinOp) { ++bridgeJoins; });
      (*entry).walk([&](ReduceOp reduce) {
        if (parsed->runtimeGuardedLogical)
          return;
        if (parsed->providerBoundSubjectSet) {
          logicalReductions += llvm::any_of(
              parsed->providerBoundMembers,
              [&](const ParsedLoopPlan::ProviderBoundMember &member) {
                return isLogicalIntegerGroupingReduction(
                    reduce, member.routeFactor);
              });
          return;
        }
        logicalReductions += isLogicalIntegerGroupingReduction(
            reduce, parsed->unrollFactor);
      });
      if (parsed->runtimeGuardedLogical) {
        (*entry).walk([&](scf::IfOp ifOp) {
          if (!ifOp->hasAttr(kRuntimeGuardAttr))
            return;
          ++runtimeGuards;
          ifOp.getThenRegion().walk([&](LoadOp load) {
            runtimeThenLoads += isVectorF32(load.getResult(), 128);
          });
          ifOp.getThenRegion().walk([&](arith::AddFOp add) {
            runtimeThenAdds += isVectorF32(add.getResult(), 128);
          });
          if (ifOp.elseBlock())
            runtimeElseLoops += llvm::count_if(
                ifOp.elseBlock()->getOperations(),
                [](Operation &op) { return isa<scf::ForOp>(op); });
        });
      }
      if (parsed->bridgeConstructed && parsed->bridgeInvariantHoisting &&
          !parsed->bridgeTensorLaneFusion) {
        if (!parsed->bridgeInvariantHoisting || bridgeJoins != 0 ||
            realized.getValue() != "factorized_bridge_GH") {
          reportFailure(module,
                        "factorized Bridge H route lost its intervention postcondition");
          return signalPassFailure();
        }
        return;
      }
      StringRef expectedRealized = factorMainTail
                                       ? "logical_group_factor_main_tail"
                                   : parsed->runtimeGuardedLogical
                                       ? "logical_group_runtime_guarded"
                                   : parsed->bridgeConstructed
                                       ? (!parsed->bridgeInvariantHoisting &&
                                                  !parsed->bridgeTensorLaneFusion
                                              ? "logical_group_bridge"
                                          : parsed->bridgeInvariantHoisting
                                              ? "factorized_bridge_GHF"
                                              : "logical_group_bridge_GF")
                                       : "logical_group";
      bool logicalPostcondition =
          parsed->bridgeConstructed
              ? bridgeJoins > 0
          : parsed->runtimeGuardedLogical
              ? runtimeGuards == 1 && runtimeElseLoops == 1 &&
                    runtimeThenLoads == 4 && runtimeThenAdds == 3 &&
                    logicalReductions == 0
          : parsed->providerBoundSubjectSet
              ? logicalReductions >= parsed->providerBoundMembers.size() ||
                    (vectorizedLoadGroups &&
                     vectorizedLoadGroups.getInt() >=
                         static_cast<int64_t>(
                             parsed->providerBoundMembers.size()) &&
                     bridgeJoins > 0)
              : logicalReductions >= 1 ||
                    (vectorizedLoadGroups &&
                     vectorizedLoadGroups.getInt() > 0 && bridgeJoins > 0);
      if (!logicalPostcondition || realized.getValue() != expectedRealized) {
        reportFailure(module, "logical-group materialization postcondition failed");
        return signalPassFailure();
      }
    } else if (realized.getValue() !=
               (factorMainTail
                    ? "phase_major_factor_main_tail"
                : parsed->bridgeConstructed
                    ? "phase_major_bridge"
                : parsed->providerClosedStatic
                    ? "phase_major_provider_closed"
                    : "phase_major")) {
      reportFailure(module, "phase-major postcondition failed");
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::triton
