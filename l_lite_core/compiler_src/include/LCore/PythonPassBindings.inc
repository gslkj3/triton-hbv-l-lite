// Include inside the native TTIR pybind module initializer, with
// LCore/Passes.h and ADD_PASS_WRAPPER_0 in scope.
// These are compiler entry points, not six optimization mechanisms.
// Prediction/ranking stays in Python; decision closes an already selected plan.
ADD_PASS_WRAPPER_0("add_hbv_loop_decision", mlir::triton::lcore::createDecisionPass);
ADD_PASS_WRAPPER_0("add_hbv_loop_facts", mlir::triton::lcore::createFactsPass);
ADD_PASS_WRAPPER_0("add_loop_bridge_discover", mlir::triton::lcore::createBridgeDiscoverPass);
ADD_PASS_WRAPPER_0("add_loop_bridge_program_coarsening",
                   mlir::triton::lcore::createBridgeConstructionPass);
ADD_PASS_WRAPPER_0("add_hbv_loop_materialize", mlir::triton::lcore::createMaterializePass);
ADD_PASS_WRAPPER_0("add_hbv_validate_loop_plan", mlir::triton::lcore::createValidatePass);
