// RUN: triton-opt %s --split-input-file -triton-loop-bridge-discover -triton-loop-bridge-program-coarsening -triton-hbv-loop-facts | FileCheck %s

// CHECK-LABEL: tt.func public @row_major_cross_axis_scalar
// CHECK-SAME: tt.hbv.l.dependence_certificate = "bridge_pid_partitioned_disjoint_v1"
// CHECK-SAME: tt.loop_bridge.factor = 2
// CHECK-SAME: tt.loop_bridge.origin = "bridge_constructed"
// CHECK: scf.for
// CHECK: tt.get_program_id y
// CHECK: tt.store
module attributes {tt.loop_bridge.factor = 2 : i32} {
  tt.func public @row_major_cross_axis_scalar(%output: !tt.ptr<i32>) {
    %pid_x = tt.get_program_id x {tt.loop_bridge.axis_extent = 1940 : i64} : i32
    %pid_y = tt.get_program_id y {tt.loop_bridge.axis_extent = 2 : i64} : i32
    %c1940 = arith.constant 1940 : i32
    %row = arith.muli %pid_y, %c1940 : i32
    %offset = arith.addi %pid_x, %row : i32
    %address = tt.addptr %output, %offset : !tt.ptr<i32>, i32
    tt.store %address, %offset : !tt.ptr<i32>
    tt.return
  }
}

// -----

// CHECK-LABEL: tt.func public @cross_axis_with_safe_local_range
// CHECK-SAME: tt.hbv.l.dependence_certificate = "bridge_pid_partitioned_disjoint_v1"
// CHECK-SAME: tt.loop_bridge.factor = 2
// CHECK-SAME: tt.loop_bridge.origin = "bridge_constructed"
// CHECK: scf.for
// CHECK: tt.get_program_id y
// CHECK: tt.make_range
// CHECK: tt.store
module attributes {tt.loop_bridge.factor = 2 : i32} {
  tt.func public @cross_axis_with_safe_local_range(%output: !tt.ptr<i32>) {
    %pid_x = tt.get_program_id x {tt.loop_bridge.axis_extent = 1940 : i64} : i32
    %pid_y = tt.get_program_id y {tt.loop_bridge.axis_extent = 2 : i64} : i32
    %c4 = arith.constant 4 : i32
    %c1940 = arith.constant 1940 : i32
    %x_base = arith.muli %pid_x, %c4 : i32
    %y_base = arith.muli %pid_y, %c1940 : i32
    %base = arith.addi %x_base, %y_base : i32
    %bases = tt.splat %base : i32 -> tensor<4xi32>
    %lanes = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %offsets = arith.addi %bases, %lanes : tensor<4xi32>
    %pointers = tt.splat %output : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %addresses = tt.addptr %pointers, %offsets : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    tt.store %addresses, %offsets : tensor<4x!tt.ptr<i32>>
    tt.return
  }
}
