#ifndef L_CORE_PASSES
#define L_CORE_PASSES

include "mlir/Pass/PassBase.td"

def TritonHBVLoopDecision : Pass<"triton-hbv-loop-decision", "mlir::ModuleOp"> {
  let summary = "Close one validated HBV Loop plan before native LoopUnroll";
  let dependentDialects = ["mlir::triton::TritonDialect", "mlir::scf::SCFDialect"];
}

def TritonHBVLoopFacts : Pass<"triton-hbv-loop-facts", "mlir::ModuleOp"> {
  let summary = "Extract immutable static facts for a later HBV Loop decision";
  let dependentDialects = ["mlir::triton::TritonDialect", "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];
}

def TritonLoopBridgeDiscover : Pass<"triton-loop-bridge-discover", "mlir::ModuleOp"> {
  let summary = "Discover and certify Loop Bridge candidates before HBV analysis";
  let dependentDialects = ["mlir::triton::TritonDialect", "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];
}

def TritonLoopBridgeProgramCoarsening : Pass<"triton-loop-bridge-program-coarsening", "mlir::ModuleOp"> {
  let summary = "Materialize an independently certified Loop Bridge candidate";
  let dependentDialects = ["mlir::triton::TritonDialect", "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];
}

def TritonHBVLoopMaterialize : Pass<"triton-hbv-loop-materialize", "mlir::ModuleOp"> {
  let summary = "Materialize one closed HBV Loop route after native LoopUnroll";
  let dependentDialects = ["mlir::triton::TritonDialect", "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];
}

def TritonHBVValidateLoopPlan : Pass<"triton-hbv-validate-loop-plan", "mlir::ModuleOp"> {
  let summary = "Validate HBV Loop route lineage and TTIR postcondition";
  let dependentDialects = ["mlir::triton::TritonDialect", "mlir::scf::SCFDialect"];
}

#endif
