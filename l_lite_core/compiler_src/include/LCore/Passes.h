#pragma once
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/JSON.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include <memory>

namespace mlir::triton::lcore {
// Read-only analysis result. Never serialized into an IR attribute.
llvm::json::Object queryPlanningFacts(mlir::ModuleOp module);
llvm::json::Object queryBridgeDiscovery(mlir::ModuleOp module);
mlir::DictionaryAttr makeDecisionAttribute(mlir::ModuleOp module,
    const llvm::json::Object &bundle, std::string &reason);
std::unique_ptr<mlir::Pass> createBridgeDiscoverPass();
std::unique_ptr<mlir::Pass> createBridgeConstructionPass();
std::unique_ptr<mlir::Pass> createFactsPass();
std::unique_ptr<mlir::Pass> createDecisionPass();
std::unique_ptr<mlir::Pass> createMaterializePass();
std::unique_ptr<mlir::Pass> createValidatePass();
// Register once when constructing a compiler/tool; not alongside legacy L passes.
void registerPasses();
} // namespace mlir::triton::lcore
