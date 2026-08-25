// RUN: triton-opt %s --split-input-file -triton-loop-bridge-discover -triton-loop-bridge-program-coarsening -triton-hbv-loop-facts | FileCheck %s

// Bridge proves program-axis independence.  A read-only body is vacuously
// free of cross-program write conflicts; absence of an output-store category
// is not a legality failure.

// CHECK-LABEL: tt.func public @read_only
// CHECK-SAME: tt.hbv.l.dependence_certificate = "bridge_pid_partitioned_disjoint_v1"
// CHECK-SAME: tt.loop_bridge.factor = 2
// CHECK-SAME: tt.loop_bridge.origin = "bridge_constructed"
// CHECK: scf.for
// CHECK: tt.load

module attributes {tt.loop_bridge.factor = 2 : i32} {
  tt.func public @read_only(%input: !tt.ptr<f32>) {
    %pid = tt.get_program_id x : i32
    %pointer = tt.addptr %input, %pid : !tt.ptr<f32>, i32
    %value = tt.load %pointer : !tt.ptr<f32>
    tt.return
  }
}

// -----

// A store-only body is equally independent when its PID footprint is
// disjoint.  The route never requires a load category.

// CHECK-LABEL: tt.func public @store_only
// CHECK-SAME: tt.hbv.l.dependence_certificate = "bridge_pid_partitioned_disjoint_v1"
// CHECK-SAME: tt.loop_bridge.factor = 2
// CHECK-SAME: tt.loop_bridge.origin = "bridge_constructed"
// CHECK: scf.for
// CHECK-NOT: tt.load
// CHECK: tt.store

module attributes {tt.loop_bridge.factor = 2 : i32} {
  tt.func public @store_only(%output: !tt.ptr<i32>) {
    %pid = tt.get_program_id x : i32
    %pointer = tt.addptr %output, %pid : !tt.ptr<i32>, i32
    tt.store %pointer, %pid : !tt.ptr<i32>
    tt.return
  }
}
