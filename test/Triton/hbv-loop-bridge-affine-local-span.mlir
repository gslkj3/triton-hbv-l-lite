// RUN: triton-opt %s --split-input-file -triton-loop-bridge-discover | FileCheck %s --check-prefixes=SAFE-RANGE,OVERLAP-RANGE,SAFE-IV,OVERLAP-IV,DYNAMIC-IV,UNCHANGED-CARRIED,MUTATING-CARRIED

// The Strong footprint is pid_stride plus one closed local interval.  These
// cases freeze the soundness boundary before natural-development replay.

// SAFE-RANGE: tt.loop_bridge.discovery
// SAFE-RANGE-SAME: construction_legal\22:true
// SAFE-RANGE-SAME: dependence_certificate\22:\22bridge_pid_partitioned_disjoint_v1
module attributes {tt.loop_bridge.factor = 2 : i32} {
  tt.func public @safe_make_range(%output: !tt.ptr<i32>) {
    %pid = tt.get_program_id x : i32
    %c4 = arith.constant 4 : i32
    %base_offset = arith.muli %pid, %c4 : i32
    %bases = tt.splat %base_offset : i32 -> tensor<4xi32>
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
  tt.func public @overlap_make_range(%output: !tt.ptr<i32>) {
    %pid = tt.get_program_id x : i32
    %bases = tt.splat %pid : i32 -> tensor<4xi32>
    %lanes = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %offsets = arith.addi %bases, %lanes : tensor<4xi32>
    %pointers = tt.splat %output : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %addresses = tt.addptr %pointers, %offsets : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    tt.store %addresses, %offsets : tensor<4x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// SAFE-IV: tt.loop_bridge.discovery
// SAFE-IV-SAME: construction_legal\22:true
// SAFE-IV-SAME: dependence_certificate\22:\22bridge_pid_partitioned_disjoint_v1
module attributes {tt.loop_bridge.factor = 2 : i32} {
  tt.func public @safe_static_iv(%output: !tt.ptr<i32>) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c4 = arith.constant 4 : i32
    %c1 = arith.constant 1 : i32
    %program_base = arith.muli %pid, %c4 : i32
    scf.for %iv = %c0 to %c4 step %c1 : i32 {
      %offset = arith.addi %program_base, %iv : i32
      %address = tt.addptr %output, %offset : !tt.ptr<i32>, i32
      tt.store %address, %offset : !tt.ptr<i32>
    }
    tt.return
  }
}

// -----

// OVERLAP-IV: tt.loop_bridge.discovery
// OVERLAP-IV-SAME: construction_legal\22:false
// OVERLAP-IV-SAME: affine interval overlaps adjacent program
// OVERLAP-IV-SAME: stride=1, local_min=0, local_max=3
module attributes {tt.loop_bridge.factor = 2 : i32} {
  tt.func public @overlap_static_iv(%output: !tt.ptr<i32>) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c4 = arith.constant 4 : i32
    %c1 = arith.constant 1 : i32
    scf.for %iv = %c0 to %c4 step %c1 : i32 {
      %offset = arith.addi %pid, %iv : i32
      %address = tt.addptr %output, %offset : !tt.ptr<i32>, i32
      tt.store %address, %offset : !tt.ptr<i32>
    }
    tt.return
  }
}

// -----

// DYNAMIC-IV: tt.loop_bridge.discovery
// DYNAMIC-IV-SAME: construction_legal\22:false
// DYNAMIC-IV-SAME: affine pid footprint is unavailable
module attributes {tt.loop_bridge.factor = 2 : i32} {
  tt.func public @dynamic_iv(%output: !tt.ptr<i32>, %extent: i32) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c4 = arith.constant 4 : i32
    %c1 = arith.constant 1 : i32
    %program_base = arith.muli %pid, %c4 : i32
    scf.for %iv = %c0 to %extent step %c1 : i32 {
      %offset = arith.addi %program_base, %iv : i32
      %address = tt.addptr %output, %offset : !tt.ptr<i32>, i32
      tt.store %address, %offset : !tt.ptr<i32>
    }
    tt.return
  }
}

// -----

// UNCHANGED-CARRIED: tt.loop_bridge.discovery
// UNCHANGED-CARRIED-SAME: construction_legal\22:true
// UNCHANGED-CARRIED-SAME: dependence_certificate\22:\22bridge_pid_partitioned_disjoint_v1
module attributes {tt.loop_bridge.factor = 2 : i32} {
  tt.func public @unchanged_carried(%output: !tt.ptr<i32>) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c4 = arith.constant 4 : i32
    %c1 = arith.constant 1 : i32
    %program_base = arith.muli %pid, %c4 : i32
    %final = scf.for %iv = %c0 to %c4 step %c1
        iter_args(%carried = %program_base) -> (i32) : i32 {
      scf.yield %carried : i32
    }
    %address = tt.addptr %output, %final : !tt.ptr<i32>, i32
    tt.store %address, %final : !tt.ptr<i32>
    tt.return
  }
}

// -----

// MUTATING-CARRIED: tt.loop_bridge.discovery
// MUTATING-CARRIED-SAME: construction_legal\22:false
// MUTATING-CARRIED-SAME: affine pid footprint is unavailable
module attributes {tt.loop_bridge.factor = 2 : i32} {
  tt.func public @mutating_carried(%output: !tt.ptr<i32>) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c4 = arith.constant 4 : i32
    %c1 = arith.constant 1 : i32
    %program_base = arith.muli %pid, %c4 : i32
    %final = scf.for %iv = %c0 to %c4 step %c1
        iter_args(%carried = %program_base) -> (i32) : i32 {
      %address = tt.addptr %output, %carried : !tt.ptr<i32>, i32
      tt.store %address, %carried : !tt.ptr<i32>
      %next = arith.addi %carried, %c1 : i32
      scf.yield %next : i32
    }
    tt.return
  }
}
