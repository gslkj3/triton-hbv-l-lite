// RUN: triton-opt %s --split-input-file -triton-loop-bridge-discover | FileCheck %s --check-prefixes=POSITIVE,NEGATIVE-STRIDE,NEGATIVE-LOCAL,NEGATIVE-INNER,NEGATIVE-DIRECTION,NEGATIVE-EXTENT

// POSITIVE: tt.loop_bridge.discovery
// POSITIVE-SAME: construction_legal\22:true
// POSITIVE-SAME: dependence_certificate\22:\22bridge_pid_partitioned_disjoint_v1
// POSITIVE-SAME: partition_recurrence_legal\22:true
// NEGATIVE-STRIDE: tt.loop_bridge.discovery
// NEGATIVE-STRIDE-SAME: construction_legal\22:false
// NEGATIVE-STRIDE-SAME: rejection_reason\22:\22store 0 affine pid footprint is unavailable
// NEGATIVE-LOCAL: tt.loop_bridge.discovery
// NEGATIVE-LOCAL-SAME: construction_legal\22:false
// NEGATIVE-INNER: tt.loop_bridge.discovery
// NEGATIVE-INNER-SAME: construction_legal\22:false
// NEGATIVE-DIRECTION: tt.loop_bridge.discovery
// NEGATIVE-DIRECTION-SAME: construction_legal\22:false
// NEGATIVE-EXTENT: tt.loop_bridge.discovery
// NEGATIVE-EXTENT-SAME: construction_legal\22:false

module attributes {tt.loop_bridge.runtime_scalars = "{\22grid\22:[1280],\22schema\22:\22triton.loop-bridge.runtime-scalars.v1\22,\22values\22:{\221\22:5,\222\22:256}}"} {
  tt.func public @positive_transpose(
      %output: !tt.ptr<f32>,
      %extent: i32 {tt.loop_bridge.bound_scalar = 5 : i64},
      %stride: i32 {tt.loop_bridge.bound_scalar = 256 : i64}) {
    %pid = tt.get_program_id x {tt.loop_bridge.axis_extent = 1280 : i64} : i32
    %q = arith.divsi %pid, %extent : i32
    %r = arith.remsi %pid, %extent : i32
    %major = arith.muli %r, %stride : i32
    %offset = arith.addi %major, %q : i32
    %pointer = tt.addptr %output, %offset : !tt.ptr<f32>, i32
    %value = arith.constant 0.0 : f32
    tt.store %pointer, %value : !tt.ptr<f32>
    tt.return
  }
}

// -----

module {
  tt.func public @negative_stride(
      %output: !tt.ptr<f32>,
      %extent: i32 {tt.loop_bridge.bound_scalar = 5 : i64},
      %stride: i32 {tt.loop_bridge.bound_scalar = 255 : i64}) {
    %pid = tt.get_program_id x {tt.loop_bridge.axis_extent = 1280 : i64} : i32
    %q = arith.divsi %pid, %extent : i32
    %r = arith.remsi %pid, %extent : i32
    %major = arith.muli %r, %stride : i32
    %offset = arith.addi %major, %q : i32
    %pointer = tt.addptr %output, %offset : !tt.ptr<f32>, i32
    %value = arith.constant 0.0 : f32
    tt.store %pointer, %value : !tt.ptr<f32>
    tt.return
  }
}

// -----

module {
  tt.func public @negative_local_interval(
      %output: !tt.ptr<f32>,
      %extent: i32 {tt.loop_bridge.bound_scalar = 5 : i64}) {
    %pid = tt.get_program_id x {tt.loop_bridge.axis_extent = 1280 : i64} : i32
    %q = arith.divsi %pid, %extent : i32
    %r = arith.remsi %pid, %extent : i32
    %c250 = arith.constant 250 : i32
    %c50 = arith.constant 50 : i32
    %qbase = arith.muli %q, %c250 : i32
    %rbase = arith.muli %r, %c50 : i32
    %base = arith.addi %qbase, %rbase : i32
    %lanes = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %base_tensor = tt.splat %base : i32 -> tensor<64xi32>
    %offsets = arith.addi %base_tensor, %lanes : tensor<64xi32>
    %pointers = tt.splat %output : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %addresses = tt.addptr %pointers, %offsets : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %value = arith.constant dense<0.0> : tensor<64xf32>
    tt.store %addresses, %value : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
  tt.func public @negative_inner_recombination(
      %output: !tt.ptr<f32>,
      %extent: i32 {tt.loop_bridge.bound_scalar = 3 : i64}) {
    %pid = tt.get_program_id x {tt.loop_bridge.axis_extent = 1536 : i64} : i32
    %q = arith.divsi %pid, %extent : i32
    %r = arith.remsi %pid, %extent : i32
    %c768 = arith.constant 768 : i32
    %c192 = arith.constant 192 : i32
    %c64 = arith.constant 64 : i32
    %qbase = arith.muli %q, %c768 : i32
    %rbase = arith.muli %r, %c192 : i32
    %base = arith.addi %qbase, %rbase : i32
    %groups = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %c64_tensor = tt.splat %c64 : i32 -> tensor<4xi32>
    %group_offsets = arith.muli %groups, %c64_tensor : tensor<4xi32>
    %base_tensor = tt.splat %base : i32 -> tensor<4xi32>
    %offsets = arith.addi %base_tensor, %group_offsets : tensor<4xi32>
    %pointers = tt.splat %output : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %addresses = tt.addptr %pointers, %offsets : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %value = arith.constant dense<0.0> : tensor<4xf32>
    tt.store %addresses, %value : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
  tt.func public @negative_direction(
      %output: !tt.ptr<f32>,
      %extent: i32 {tt.loop_bridge.bound_scalar = 5 : i64}) {
    %pid = tt.get_program_id x {tt.loop_bridge.axis_extent = 1280 : i64} : i32
    %q = arith.divsi %pid, %extent : i32
    %r = arith.remsi %pid, %extent : i32
    %c250 = arith.constant 250 : i32
    %c50 = arith.constant 50 : i32
    %qbase = arith.muli %q, %c250 : i32
    %rbase = arith.muli %r, %c50 : i32
    %offset = arith.subi %qbase, %rbase : i32
    %pointer = tt.addptr %output, %offset : !tt.ptr<f32>, i32
    %value = arith.constant 0.0 : f32
    tt.store %pointer, %value : !tt.ptr<f32>
    tt.return
  }
}

// -----

module {
  tt.func public @negative_extent(
      %output: !tt.ptr<f32>,
      %extent: i32 {tt.loop_bridge.bound_scalar = 0 : i64}) {
    %pid = tt.get_program_id x {tt.loop_bridge.axis_extent = 1280 : i64} : i32
    %q = arith.divsi %pid, %extent : i32
    %r = arith.remsi %pid, %extent : i32
    %c250 = arith.constant 250 : i32
    %c50 = arith.constant 50 : i32
    %qbase = arith.muli %q, %c250 : i32
    %rbase = arith.muli %r, %c50 : i32
    %offset = arith.addi %qbase, %rbase : i32
    %pointer = tt.addptr %output, %offset : !tt.ptr<f32>, i32
    %value = arith.constant 0.0 : f32
    tt.store %pointer, %value : !tt.ptr<f32>
    tt.return
  }
}
