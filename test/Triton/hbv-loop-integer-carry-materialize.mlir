// RUN: triton-opt %s | FileCheck %s --check-prefix=SOURCE
// SOURCE-LABEL: tt.func public @integer_carry
// Transformation assertions: python/test/unit/l_lite/test_materialization_numeric.py
// Decisions are generated from current facts, not archived serialized contracts.
//
// Four contributions must be reduced AND added to the nonzero incoming state.
// This is a fixed serialized transformation contract, not a predictor dependency.
// HISTORICAL-CHECK-LABEL: tt.func public @integer_carry
// HISTORICAL-CHECK: %[[INITIAL:.*]] = arith.constant dense<7>
// HISTORICAL-CHECK: %[[SUM:.*]] = "tt.reduce"
// HISTORICAL-CHECK: arith.addi %[[INITIAL]], %[[SUM]]
// HISTORICAL-CHECK: tt.store
module {
  tt.func public @integer_carry(%x: !tt.ptr<i32>, %y: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %c16 = arith.constant 16 : i32
    %three = arith.constant dense<3> : tensor<16xi32>
    %initial = arith.constant dense<7> : tensor<16xi32>
    %range = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %base = tt.splat %x : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %out = tt.splat %y : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %outptr = tt.addptr %out, %range : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %initial) -> tensor<16xi32> : i32 {
      %offset = arith.muli %i, %c16 : i32
      %offsets = tt.splat %offset : i32 -> tensor<16xi32>
      %indices = arith.addi %offsets, %range : tensor<16xi32>
      %ptr = tt.addptr %base, %indices : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
      %value = tt.load %ptr : tensor<16x!tt.ptr<i32>>
      %product = arith.muli %value, %three : tensor<16xi32>
      %next = arith.addi %acc, %product : tensor<16xi32>
      scf.yield %next : tensor<16xi32>
    }
    tt.store %outptr, %result : tensor<16x!tt.ptr<i32>>
    tt.return
  }
}
