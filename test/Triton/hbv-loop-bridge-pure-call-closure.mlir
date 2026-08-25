// RUN: triton-opt %s --split-input-file -triton-loop-bridge-discover | FileCheck %s --check-prefixes=PURE,EFFECT,NESTED

// A resolved, closed and memory-effect-free helper is an ordinary pure SSA
// operation for Bridge program-independence purposes.

// PURE: tt.loop_bridge.discovery
// PURE-SAME: construction_legal\22:true
// PURE-SAME: dependence_certificate\22:\22bridge_pid_partitioned_disjoint_v1

module attributes {tt.loop_bridge.factor = 2 : i32} {
  tt.func private @pure_increment(%value: i32) -> i32 {
    %one = arith.constant 1 : i32
    %next = arith.addi %value, %one : i32
    tt.return %next : i32
  }
  tt.func public @pure_call(%output: !tt.ptr<i32>) {
    %pid = tt.get_program_id x : i32
    %value = tt.call @pure_increment(%pid) : (i32) -> i32
    %pointer = tt.addptr %output, %pid : !tt.ptr<i32>, i32
    tt.store %pointer, %value : !tt.ptr<i32>
    tt.return
  }
}

// -----

// A helper containing a load is not made pure merely by hiding the effect
// behind tt.call.

// EFFECT: tt.loop_bridge.discovery
// EFFECT-SAME: construction_legal\22:false
// EFFECT-SAME: rejection_reason\22:\22program call is not prospectively pure

module attributes {tt.loop_bridge.factor = 2 : i32} {
  tt.func private @effectful_load(%input: !tt.ptr<i32>) -> i32 {
    %value = tt.load %input : !tt.ptr<i32>
    tt.return %value : i32
  }
  tt.func public @effect_call(
      %input: !tt.ptr<i32>, %output: !tt.ptr<i32>) {
    %pid = tt.get_program_id x : i32
    %value = tt.call @effectful_load(%input) : (!tt.ptr<i32>) -> i32
    %pointer = tt.addptr %output, %pid : !tt.ptr<i32>, i32
    tt.store %pointer, %value : !tt.ptr<i32>
    tt.return
  }
}

// -----

// A nested call requires a separate transitive certificate and is therefore
// rejected by this deliberately one-level prospective proof.

// NESTED: tt.loop_bridge.discovery
// NESTED-SAME: construction_legal\22:false
// NESTED-SAME: rejection_reason\22:\22program call is not prospectively pure

module attributes {tt.loop_bridge.factor = 2 : i32} {
  tt.func private @leaf(%value: i32) -> i32 {
    %one = arith.constant 1 : i32
    %next = arith.addi %value, %one : i32
    tt.return %next : i32
  }
  tt.func private @nested(%value: i32) -> i32 {
    %next = tt.call @leaf(%value) : (i32) -> i32
    tt.return %next : i32
  }
  tt.func public @nested_call(%output: !tt.ptr<i32>) {
    %pid = tt.get_program_id x : i32
    %value = tt.call @nested(%pid) : (i32) -> i32
    %pointer = tt.addptr %output, %pid : !tt.ptr<i32>, i32
    tt.store %pointer, %value : !tt.ptr<i32>
    tt.return
  }
}
