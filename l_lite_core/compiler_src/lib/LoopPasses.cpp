// L-core production pass implementation. Only native compiler dependencies.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "LCore/Passes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Analysis/AxisInfo.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MathExtras.h"
#include "LCore/Attributes.h"
#include "LCore/AffineInterval.h"
#include "LCore/Certificate.h"
#include "LCore/Plan.h"
#include "LCore/GuardedMemoryPairs.h"
#include <array>
#include <map>
#include <set>
#include <tuple>
#include <iostream>
#include <iterator>
namespace mlir::triton::lcore {
#define GEN_PASS_DEF_TRITONLOOPBRIDGEDISCOVER
#define GEN_PASS_DEF_TRITONLOOPBRIDGEPROGRAMCOARSENING
#define GEN_PASS_DEF_TRITONHBVLOOPFACTS
#define GEN_PASS_DEF_TRITONHBVLOOPDECISION
#define GEN_PASS_DEF_TRITONHBVLOOPMATERIALIZE
#define GEN_PASS_DEF_TRITONHBVVALIDATELOOPPLAN
#include "LCore/Passes.h.inc"
}
using namespace mlir;
using namespace mlir::triton;
namespace {
namespace impl = mlir::triton::lcore::impl;
using namespace mlir::triton::lcore;
#include "LCore/Analysis.inc"
#include "LCore/BridgeControlFlow.inc"
#include "LCore/BridgeDiscovery.inc"
#include "LCore/BridgeConstruction.inc"
#include "LCore/OperationLineage.inc"
#include "LCore/PhaseScheduling.inc"
#include "LCore/ElementwisePacking.inc"
#include "LCore/PackingMemoryProof.inc"
#include "LCore/LogicalVectorization.inc"
#include "LCore/OrdinaryMaterialization.inc"
#include "LCore/OrdinaryValidation.inc"
#include "LCore/OrdinaryPasses.inc"
}

namespace mlir::triton::lcore {
std::unique_ptr<Pass> createBridgeDiscoverPass() {
  return std::make_unique<LoopBridgeDiscoverPass>();
}
std::unique_ptr<Pass> createBridgeConstructionPass() {
  return std::make_unique<LoopBridgeProgramCoarseningPass>();
}
std::unique_ptr<Pass> createFactsPass() {
  return std::make_unique<CoreFactsPass>();
}
std::unique_ptr<Pass> createDecisionPass() {
  return std::make_unique<CoreDecisionPass>();
}
std::unique_ptr<Pass> createMaterializePass() {
  return std::make_unique<CoreMaterializePass>();
}
std::unique_ptr<Pass> createValidatePass() {
  return std::make_unique<CoreValidatePass>();
}
void registerPasses() {
  mlir::registerPass([] { return createBridgeDiscoverPass(); });
  mlir::registerPass([] { return createBridgeConstructionPass(); });
  mlir::registerPass([] { return createFactsPass(); });
  mlir::registerPass([] { return createDecisionPass(); });
  mlir::registerPass([] { return createMaterializePass(); });
  mlir::registerPass([] { return createValidatePass(); });
}
} // namespace mlir::triton::lcore
