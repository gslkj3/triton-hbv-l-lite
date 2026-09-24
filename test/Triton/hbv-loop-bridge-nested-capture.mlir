// RUN: triton-opt %s | FileCheck %s --check-prefix=SOURCE
// SOURCE-LABEL: tt.func public @_partition_recurrence_captured_by_inner_for
// Transformation assertions: python/test/unit/l_lite/test_materialization_numeric.py
// Decisions are generated from current facts, not archived serialized contracts.
//
// Nested loop bodies capture parent-block quotient/remainder definitions.
// The phase scheduler must include those definitions in each moved closure.
// The verifier checks SSA dominance after every pass.
// HISTORICAL-CHECK-LABEL: tt.func public @_partition_recurrence_captured_by_inner_for
// HISTORICAL-CHECK-SAME: tt.hbv.l.realized = "phase_major_bridge"
// HISTORICAL-CHECK: scf.for
// HISTORICAL-CHECK: scf.for
// HISTORICAL-CHECK-NOT: scf.for
// HISTORICAL-CHECK: tt.return
#loc = loc("nested-capture-fixture.py":205:0)
#loc24 = loc("x"(#loc))
#loc25 = loc("y"(#loc))
module {
  tt.func public @_partition_recurrence_captured_by_inner_for(%x: !tt.ptr<f32> loc("x"(#loc)), %y: !tt.ptr<f32> loc("y"(#loc))) attributes {noinline = false} {
    %accumulator = arith.constant dense<0.000000e+00> : tensor<16xf32> loc(#loc41)
    %c1_i32 = arith.constant 1 : i32 loc(#loc3)
    %c0_i32 = arith.constant 0 : i32 loc(#loc3)
    %c16_i32 = arith.constant 16 : i32 loc(#loc4)
    %c4_i32 = arith.constant 4 : i32 loc(#loc4)
    %pid = tt.get_program_id x : i32 loc(#loc27)
    %batch = arith.divsi %pid, %c4_i32 : i32 loc(#loc28)
    %head = arith.remsi %pid, %c4_i32 : i32 loc(#loc29)
    %lanes = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32> loc(#loc30)
    %accumulator_0 = scf.for %iteration = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%accumulator_1 = %accumulator) -> (tensor<16xf32>)  : i32 {
      %logical = arith.muli %batch, %c4_i32 : i32 loc(#loc32)
      %logical_2 = arith.addi %logical, %head : i32 loc(#loc33)
      %logical_3 = arith.muli %logical_2, %c4_i32 : i32 loc(#loc34)
      %logical_4 = arith.addi %logical_3, %iteration : i32 loc(#loc35)
      %value = arith.muli %logical_4, %c16_i32 : i32 loc(#loc36)
      %value_5 = tt.addptr %x, %value : !tt.ptr<f32>, i32 loc(#loc37)
      %value_6 = tt.splat %value_5 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>> loc(#loc38)
      %value_7 = tt.addptr %value_6, %lanes : tensor<16x!tt.ptr<f32>>, tensor<16xi32> loc(#loc38)
      %value_8 = tt.load %value_7 : tensor<16x!tt.ptr<f32>> loc(#loc39)
      %accumulator_9 = arith.addf %accumulator_1, %value_8 : tensor<16xf32> loc(#loc40)
      scf.yield %accumulator_9 : tensor<16xf32> loc(#loc18)
    } loc(#loc31)
    %0 = arith.muli %pid, %c16_i32 : i32 loc(#loc19)
    %1 = tt.addptr %y, %0 : !tt.ptr<f32>, i32 loc(#loc20)
    %2 = tt.splat %1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>> loc(#loc21)
    %3 = tt.addptr %2, %lanes : tensor<16x!tt.ptr<f32>>, tensor<16xi32> loc(#loc21)
    tt.store %3, %accumulator_0 : tensor<16x!tt.ptr<f32>> loc(#loc22)
    tt.return loc(#loc23)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("nested-capture-fixture.py":129:31)
#loc2 = loc("nested-capture-fixture.py":212:37)
#loc3 = loc("nested-capture-fixture.py":213:33)
#loc4 = loc(unknown)
#loc5 = loc("nested-capture-fixture.py":208:24)
#loc6 = loc("nested-capture-fixture.py":209:19)
#loc7 = loc("nested-capture-fixture.py":210:17)
#loc8 = loc("nested-capture-fixture.py":211:25)
#loc9 = loc("nested-capture-fixture.py":214:27)
#loc10 = loc("nested-capture-fixture.py":214:35)
#loc11 = loc("nested-capture-fixture.py":214:43)
#loc12 = loc("nested-capture-fixture.py":214:47)
#loc13 = loc("nested-capture-fixture.py":215:38)
#loc14 = loc("nested-capture-fixture.py":215:28)
#loc15 = loc("nested-capture-fixture.py":215:46)
#loc16 = loc("nested-capture-fixture.py":215:24)
#loc17 = loc("nested-capture-fixture.py":216:23)
#loc18 = loc("nested-capture-fixture.py":216:8)
#loc19 = loc("nested-capture-fixture.py":217:23)
#loc20 = loc("nested-capture-fixture.py":217:17)
#loc21 = loc("nested-capture-fixture.py":217:31)
#loc22 = loc("nested-capture-fixture.py":217:38)
#loc23 = loc("nested-capture-fixture.py":217:4)
#loc26 = loc("accumulator"(#loc2))
#loc27 = loc("pid"(#loc5))
#loc28 = loc("batch"(#loc6))
#loc29 = loc("head"(#loc7))
#loc30 = loc("lanes"(#loc8))
#loc31 = loc("accumulator"(#loc3))
#loc32 = loc("logical"(#loc9))
#loc33 = loc("logical"(#loc10))
#loc34 = loc("logical"(#loc11))
#loc35 = loc("logical"(#loc12))
#loc36 = loc("value"(#loc13))
#loc37 = loc("value"(#loc14))
#loc38 = loc("value"(#loc15))
#loc39 = loc("value"(#loc16))
#loc40 = loc("accumulator"(#loc17))
#loc41 = loc(callsite(#loc1 at #loc26))
