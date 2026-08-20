// RUN: triton-opt %s --split-input-file -triton-loop-bridge-program-coarsening -triton-hbv-loop-facts | FileCheck %s

// CHECK: bridge_cfg_predication_certificate\22:\22single_early_void_return_predicated_v1
// CHECK: bridge_pipeline_body_certificate\22:\22bridge_program_body_direct_service_in_constructed_loop_v2
// CHECK: bridge_pipeline_body_inline\22:true
// CHECK: tt.loop_bridge.cfg_predication = "single_early_void_return_predicated_v1"
// CHECK: scf.for
// CHECK-NOT: tt.call
// CHECK: tt.load {{.*}}, %{{.*}} : !tt.ptr<f32>
// CHECK: tt.store {{.*}}, %{{.*}}, %{{.*}} : !tt.ptr<f32>
// CHECK: bridge_cfg_predication_rejection_reason\22:\22continuation_contains_nonpredicatable_or_unbounded_arith.divsi
// CHECK: bridge_pipeline_body_certificate\22:\22bridge_program_body_hidden_behind_helper_call_v1
// CHECK: bridge_pipeline_body_inline\22:false
// CHECK: tt.loop_bridge.cfg_predication_rejection = "continuation_contains_nonpredicatable_or_unbounded_arith.divsi"
// CHECK: scf.for
// CHECK: tt.call @unbounded_divisor__loop_bridge_body

module attributes {tt.loop_bridge.factor = 2 : i32} {
  tt.func public @single_early_return(
      %input: !tt.ptr<f32>,
      %output: !tt.ptr<f32>) {
    %pid = tt.get_program_id x : i32
    %zero = arith.constant 0 : i32
    %skip = arith.cmpi slt, %pid, %zero : i32
    cf.cond_br %skip, ^return, ^service
  ^return:
    tt.return
  ^service:
    %input_pointer = tt.addptr %input, %pid : !tt.ptr<f32>, i32
    %value = tt.load %input_pointer : !tt.ptr<f32>
    %output_pointer = tt.addptr %output, %pid : !tt.ptr<f32>, i32
    tt.store %output_pointer, %value : !tt.ptr<f32>
    tt.return
  }
}

// -----

module attributes {tt.loop_bridge.factor = 2 : i32} {
  tt.func public @unbounded_divisor(
      %input: !tt.ptr<f32>,
      %output: !tt.ptr<f32>,
      %divisor: i32) {
    %pid = tt.get_program_id x : i32
    %zero = arith.constant 0 : i32
    %skip = arith.cmpi slt, %pid, %zero : i32
    cf.cond_br %skip, ^return, ^service
  ^return:
    tt.return
  ^service:
    %unused = arith.divsi %pid, %divisor : i32
    %input_pointer = tt.addptr %input, %pid : !tt.ptr<f32>, i32
    %value = tt.load %input_pointer : !tt.ptr<f32>
    %output_pointer = tt.addptr %output, %pid : !tt.ptr<f32>, i32
    tt.store %output_pointer, %value : !tt.ptr<f32>
    tt.return
  }
}
