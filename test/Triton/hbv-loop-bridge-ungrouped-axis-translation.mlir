// RUN: triton-opt %s --split-input-file -triton-loop-bridge-discover | FileCheck %s --check-prefixes=ROW-MAJOR,SAFE-RANGE,OVERLAP-RANGE

// Certification is conditional on the grouped x axis.  The y program ID is
// fixed across those virtual x iterations and is therefore a common address
// translation, not a local interval.

// ROW-MAJOR: tt.loop_bridge.discovery
// ROW-MAJOR-SAME: construction_legal\22:true
// ROW-MAJOR-SAME: dependence_certificate\22:\22bridge_pid_partitioned_disjoint_v1
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

// SAFE-RANGE: tt.loop_bridge.discovery
// SAFE-RANGE-SAME: construction_legal\22:true
// SAFE-RANGE-SAME: dependence_certificate\22:\22bridge_pid_partitioned_disjoint_v1
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

// -----

// OVERLAP-RANGE: tt.loop_bridge.discovery
// OVERLAP-RANGE-SAME: construction_legal\22:false
// OVERLAP-RANGE-SAME: affine interval overlaps adjacent program
// OVERLAP-RANGE-SAME: stride=1, local_min=0, local_max=3
module attributes {tt.loop_bridge.factor = 2 : i32} {
  tt.func public @cross_axis_with_overlapping_local_range(%output: !tt.ptr<i32>) {
    %pid_x = tt.get_program_id x {tt.loop_bridge.axis_extent = 1940 : i64} : i32
    %pid_y = tt.get_program_id y {tt.loop_bridge.axis_extent = 2 : i64} : i32
    %c1940 = arith.constant 1940 : i32
    %y_base = arith.muli %pid_y, %c1940 : i32
    %base = arith.addi %pid_x, %y_base : i32
    %bases = tt.splat %base : i32 -> tensor<4xi32>
    %lanes = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %offsets = arith.addi %bases, %lanes : tensor<4xi32>
    %pointers = tt.splat %output : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %addresses = tt.addptr %pointers, %offsets : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    tt.store %addresses, %offsets : tensor<4x!tt.ptr<i32>>
    tt.return
  }
}
