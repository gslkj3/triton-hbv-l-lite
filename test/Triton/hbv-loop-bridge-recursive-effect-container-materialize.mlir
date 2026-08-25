// RUN: triton-opt %s -triton-loop-bridge-discover -triton-loop-bridge-program-coarsening -triton-hbv-loop-facts | FileCheck %s

// A recursively certified local loop survives inside the new Bridge loop.
// The two scf.for operations prove that Bridge materialized an outer virtual-
// program loop without deleting or flattening the source-local loop.

// CHECK-LABEL: tt.func public @disjoint_store_after_local_loop
// CHECK-SAME: tt.hbv.l.dependence_certificate = "bridge_pid_partitioned_disjoint_v1"
// CHECK-SAME: tt.loop_bridge.factor = 2
// CHECK-SAME: tt.loop_bridge.origin = "bridge_constructed"
// CHECK: scf.for
// CHECK: scf.for

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
