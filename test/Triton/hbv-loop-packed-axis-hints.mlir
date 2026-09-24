// RUN: triton-opt %s | FileCheck %s --check-prefix=SOURCE
// SOURCE-LABEL: tt.func public @rank_hint_packing
// Transformation assertions: python/test/unit/l_lite/test_materialization_numeric.py
// Decisions are generated from current facts, not archived serialized contracts.
//
// Rank-one axis hints cannot be copied onto newly packed higher-rank operations.
// HISTORICAL-CHECK-LABEL: tt.func public @rank_hint_packing
// HISTORICAL-CHECK: arith.mulf
// HISTORICAL-CHECK-SAME: tensor<{{[0-9]+x([0-9]+x)+}}f32>
// HISTORICAL-CHECK: tt.store
module {
  tt.func public @rank_hint_packing(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %c16 = arith.constant 16 : i32
    %three = arith.constant dense<3.0> : tensor<16xf32>
    %initial = arith.constant dense<7.0> : tensor<16xf32>
    %range = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %base = tt.splat %x : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %out = tt.splat %y : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %outptr = tt.addptr %out, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %initial) -> tensor<16xf32> : i32 {
      %offset = arith.muli %i, %c16 : i32
      %offsets = tt.splat %offset : i32 -> tensor<16xi32>
      %indices = arith.addi %offsets, %range : tensor<16xi32>
      %ptr = tt.addptr %base, %indices : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      %value = tt.load %ptr : tensor<16x!tt.ptr<f32>>
      %product = arith.mulf %value, %three {tt.contiguity = dense<16> : tensor<1xi32>, tt.divisibility = dense<1> : tensor<1xi32>, tt.constancy = dense<1> : tensor<1xi32>} : tensor<16xf32>
      %next = arith.addf %acc, %product : tensor<16xf32>
      scf.yield %next : tensor<16xf32>
    }
    tt.store %outptr, %result : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}
