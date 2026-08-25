// RUN: triton-opt %s --split-input-file -triton-loop-bridge-discover -triton-hbv-loop-facts | FileCheck %s --check-prefix=SOURCE
// RUN: triton-opt %s --split-input-file -triton-loop-bridge-discover -triton-loop-bridge-program-coarsening -triton-hbv-loop-facts | FileCheck %s --check-prefixes=TRUE,FALSE

// SOURCE: tt.hbv.l.static_facts
// SOURCE-SAME: bridge_pipeline_body_certificate\22:\22bridge_program_body_hidden_behind_helper_call_v1
// SOURCE-SAME: program_axis_independence\22:[true,false,false]

// TRUE: tt.hbv.l.static_facts
// TRUE-SAME: body_special_function_counts
// TRUE-SAME: math.ceil\22:1
// TRUE-SAME: math.exp2\22:1
// TRUE-SAME: math.log2\22:1
// TRUE-SAME: bridge_pipeline_body_certificate\22:\22bridge_program_body_direct_service_in_constructed_loop_v2
// TRUE-SAME: scale_format_mode\22:\22ue8m0_scale
// TRUE-SAME: schema\22:\22hbv.loop.static-facts.v12

module attributes {tt.loop_bridge.factor = 2 : i32, tt.loop_bridge.runtime_scalars = "{\22grid\22:[1],\22schema\22:\22triton.loop-bridge.runtime-scalars.v2\22,\22values_by_name\22:{\22scale_mode\22:1}}"} {
  tt.func public @scale_true(
      %output: !tt.ptr<f32>,
      %scale_mode: i1 loc("scale_mode")) {
    %pid = tt.get_program_id x : i32
    %base = arith.constant 1.5 : f32
    %scale = scf.if %scale_mode -> (f32) {
      %log = math.log2 %base : f32
      %ceil = math.ceil %log : f32
      %exp = math.exp2 %ceil : f32
      scf.yield %exp : f32
    } else {
      scf.yield %base : f32
    }
    %pointer = tt.addptr %output, %pid : !tt.ptr<f32>, i32
    tt.store %pointer, %scale : !tt.ptr<f32>
    tt.return
  }
}

// -----

// FALSE: tt.hbv.l.static_facts
// FALSE-SAME: body_type_signature
// FALSE-SAME: f32
// FALSE-SAME: i1
// FALSE-SAME: bridge_pipeline_body_certificate\22:\22bridge_program_body_direct_service_in_constructed_loop_v2
// FALSE-SAME: scale_format_mode\22:\22fp32_scale
// FALSE-SAME: schema\22:\22hbv.loop.static-facts.v12

module attributes {tt.loop_bridge.factor = 2 : i32, tt.loop_bridge.runtime_scalars = "{\22grid\22:[1],\22schema\22:\22triton.loop-bridge.runtime-scalars.v2\22,\22values_by_name\22:{\22scale_mode\22:0}}"} {
  tt.func public @scale_false(
      %output: !tt.ptr<f32>,
      %scale_mode: i1 loc("scale_mode")) {
    %pid = tt.get_program_id x : i32
    %base = arith.constant 1.5 : f32
    %scale = scf.if %scale_mode -> (f32) {
      %log = math.log2 %base : f32
      %ceil = math.ceil %log : f32
      %exp = math.exp2 %ceil : f32
      scf.yield %exp : f32
    } else {
      scf.yield %base : f32
    }
    %pointer = tt.addptr %output, %pid : !tt.ptr<f32>, i32
    tt.store %pointer, %scale : !tt.ptr<f32>
    tt.return
  }
}
