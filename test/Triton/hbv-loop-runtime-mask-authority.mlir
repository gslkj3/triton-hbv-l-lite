// RUN: triton-opt %s --split-input-file -triton-hbv-loop-facts | FileCheck %s --check-prefixes=MIXED,INCOMPLETE

// An entry scalar that exceeds a narrow local container is only a syntactic
// mask candidate for that operation.  It cannot be published as an exact
// active-extent relation.  Complete relations over the wider container remain
// authoritative; no kernel, operator, dtype, or shape label participates.

// MIXED: tt.hbv.l.static_facts
// MIXED-SAME: runtime_mask_scalars\22:[{\22argument_index\22:2,\22bound_value\22:128,\22bound_value_known\22:true,\22complete\22:true,\22container_width\22:128
// MIXED-SAME: effect\22:\22read
// MIXED-SAME: {\22argument_index\22:2,\22bound_value\22:128,\22bound_value_known\22:true,\22complete\22:true,\22container_width\22:128
// MIXED-SAME: effect\22:\22write

module {
  tt.func public @mixed_complete_and_incomplete_masks(
      %input: !tt.ptr<f32>,
      %output: !tt.ptr<f32>,
      %bound: i32 {tt.loop_bridge.bound_scalar = 128 : i64}) {
    %zero = arith.constant 0.0 : f32

    %lanes32 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %bound32 = tt.splat %bound : i32 -> tensor<32xi32>
    %mask32 = arith.cmpi slt, %lanes32, %bound32 : tensor<32xi32>
    %input32 = tt.splat %input : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %pointers32 = tt.addptr %input32, %lanes32 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %other32 = tt.splat %zero : f32 -> tensor<32xf32>
    %unused = tt.load %pointers32, %mask32, %other32 : tensor<32x!tt.ptr<f32>>

    %lanes128 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %bound128 = tt.splat %bound : i32 -> tensor<128xi32>
    %mask128 = arith.cmpi slt, %lanes128, %bound128 : tensor<128xi32>
    %input128 = tt.splat %input : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %input_pointers128 = tt.addptr %input128, %lanes128 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %other128 = tt.splat %zero : f32 -> tensor<128xf32>
    %values128 = tt.load %input_pointers128, %mask128, %other128 : tensor<128x!tt.ptr<f32>>
    %output128 = tt.splat %output : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %output_pointers128 = tt.addptr %output128, %lanes128 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %output_pointers128, %values128, %mask128 : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// When every syntactic mask candidate exceeds its local container, the
// Provider emits no active-extent relation.  Downstream masked-memory logic
// can then refuse the missing relation; it cannot consume a fabricated fact.

// INCOMPLETE: tt.hbv.l.static_facts
// INCOMPLETE-SAME: runtime_mask_scalars\22:[]

module {
  tt.func public @incomplete_only_mask(
      %input: !tt.ptr<f32>,
      %output: !tt.ptr<f32>,
      %bound: i32 {tt.loop_bridge.bound_scalar = 128 : i64}) {
    %zero = arith.constant 0.0 : f32
    %lanes = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %bound_tensor = tt.splat %bound : i32 -> tensor<32xi32>
    %mask = arith.cmpi slt, %lanes, %bound_tensor : tensor<32xi32>
    %input_base = tt.splat %input : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %input_pointers = tt.addptr %input_base, %lanes : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %other = tt.splat %zero : f32 -> tensor<32xf32>
    %values = tt.load %input_pointers, %mask, %other : tensor<32x!tt.ptr<f32>>
    %output_base = tt.splat %output : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %output_pointers = tt.addptr %output_base, %lanes : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %output_pointers, %values, %mask : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}
