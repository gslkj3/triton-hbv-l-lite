// Keep native registration entry points; delegate to the single shared L core.
// No historical alternative implementation is linked into this build.
#include "LCore/Passes.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

namespace mlir::triton {
std::unique_ptr<Pass> createTritonHBVLoopDecision() { return lcore::createDecisionPass(); }
std::unique_ptr<Pass> createTritonHBVLoopFacts() { return lcore::createFactsPass(); }
std::unique_ptr<Pass> createTritonLoopBridgeDiscover() { return lcore::createBridgeDiscoverPass(); }
std::unique_ptr<Pass> createTritonLoopBridgeProgramCoarsening() { return lcore::createBridgeConstructionPass(); }
std::unique_ptr<Pass> createTritonHBVLoopMaterialize() { return lcore::createMaterializePass(); }
std::unique_ptr<Pass> createTritonHBVValidateLoopPlan() { return lcore::createValidatePass(); }
}
