// RUN: triton-opt %s -triton-loop-bridge-program-coarsening -triton-hbv-loop-facts | FileCheck %s

// A scan owns a region-local combiner but introduces no cross-program control.
// Bridge may clone it with the rest of each independently certified program.

// CHECK: tt.hbv.l.static_facts
// CHECK-SAME: bridge_pipeline_body_certificate\22:\22bridge_program_body_direct_service_in_constructed_loop_v2
// CHECK: tt.loop_bridge.factor = 2
// CHECK: tt.loop_bridge.origin = "bridge_constructed"
// CHECK: scf.for
// CHECK: "tt.scan"

module attributes {tt.loop_bridge.factor = 2 : i32} {
  tt.func public @scan_body(%output: !tt.ptr<f32>) {
    %pid = tt.get_program_id x : i32
    %input = arith.constant dense<1.0> : tensor<4xf32>
    %scan = "tt.scan"(%input) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      tt.scan.return %sum : f32
    }) : (tensor<4xf32>) -> tensor<4xf32>
    %pointer = tt.addptr %output, %pid : !tt.ptr<f32>, i32
    %value = arith.constant 1.0 : f32
    tt.store %pointer, %value : !tt.ptr<f32>
    tt.return
  }
}
