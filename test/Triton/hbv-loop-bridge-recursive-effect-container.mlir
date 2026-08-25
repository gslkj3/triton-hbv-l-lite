// RUN: triton-opt %s --split-input-file -triton-loop-bridge-discover | FileCheck %s --check-prefixes=POSITIVE,ALIAS,ATOMIC,VOLATILE

// scf.for has RecursiveMemoryEffects.  The container is transparent to the
// effect proof while its leaf operations remain recursively audited.

// POSITIVE: tt.loop_bridge.discovery
// POSITIVE-SAME: construction_legal\22:true
// POSITIVE-SAME: dependence_certificate\22:\22bridge_pid_partitioned_disjoint_v1
module attributes {tt.loop_bridge.factor = 2 : i32} {
  tt.func public @disjoint_store_after_local_loop(%output: !tt.ptr<i32>) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %sum = scf.for %iv = %c0 to %c4 step %c1
        iter_args(%acc = %pid) -> (i32) {
      %next = arith.addi %acc, %pid : i32
      scf.yield %next : i32
    }
    %pointer = tt.addptr %output, %pid : !tt.ptr<i32>, i32
    tt.store %pointer, %sum : !tt.ptr<i32>
    tt.return
  }
}

// -----

// A store to one shared address remains illegal even though it is nested in a
// recognized recursive-effect container.

// ALIAS: tt.loop_bridge.discovery
// ALIAS-SAME: construction_legal\22:false
// ALIAS-SAME: rejection_reason

module attributes {tt.loop_bridge.factor = 2 : i32} {
  tt.func public @nonpartitioned_store_inside_loop(%output: !tt.ptr<i32>) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    scf.for %iv = %c0 to %c2 step %c1 {
      tt.store %output, %pid : !tt.ptr<i32>
    }
    tt.return
  }
}

// -----

// Atomic leaves retain their unconditional Strong rejection.

// ATOMIC: tt.loop_bridge.discovery
// ATOMIC-SAME: construction_legal\22:false
// ATOMIC-SAME: rejection_reason\22:\22atomic memory operation

module attributes {tt.loop_bridge.factor = 2 : i32} {
  tt.func public @atomic_inside_loop(%output: !tt.ptr<i32>) {
    %pid = tt.get_program_id x : i32
    %true = arith.constant true
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    scf.for %iv = %c0 to %c2 step %c1 {
      %unused = tt.atomic_rmw add, relaxed, gpu, %output, %pid, %true :
          (!tt.ptr<i32>, i32, i1) -> i32
    }
    tt.return
  }
}

// -----

// Volatile-load leaves retain their unconditional Strong rejection.

// VOLATILE: tt.loop_bridge.discovery
// VOLATILE-SAME: construction_legal\22:false
// VOLATILE-SAME: rejection_reason\22:\22volatile load

module attributes {tt.loop_bridge.factor = 2 : i32} {
  tt.func public @volatile_load_inside_loop(%input: !tt.ptr<i32>) {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    scf.for %iv = %c0 to %c2 step %c1 {
      %unused = tt.load %input {isVolatile = true} : !tt.ptr<i32>
    }
    tt.return
  }
}
