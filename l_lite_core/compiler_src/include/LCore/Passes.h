#pragma once
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::triton::lcore {
std::unique_ptr<mlir::Pass> createBridgeDiscoverPass();
std::unique_ptr<mlir::Pass> createBridgeConstructionPass();
std::unique_ptr<mlir::Pass> createFactsPass();
std::unique_ptr<mlir::Pass> createDecisionPass();
std::unique_ptr<mlir::Pass> createMaterializePass();
std::unique_ptr<mlir::Pass> createValidatePass();
// Register once when constructing a compiler/tool; not alongside legacy L passes.
void registerPasses();
} // namespace mlir::triton::lcore
