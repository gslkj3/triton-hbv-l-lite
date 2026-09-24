"""Default route at the actual native make_ttir cut, before any TTIR unroll."""
from .ttir import PreparationConfig, prepare_at_analysis_point
from .compiler import CompilerBinding, build_candidate
from .factors import UnsupportedCandidate
from .state import Route


def bind_default_decision(module, options, capability):
    from triton._C.libtriton import passes
    prepared=prepare_at_analysis_point(module,PreparationConfig(capability,options.num_warps,
                                                     options.num_stages))
    route='native_no_pipeline_candidate'
    if options.num_stages>=2:
        try:
            candidate=build_candidate(provider=prepared.provider,route=Route.PIPELINE,
                factor=options.num_stages,identity=prepared.route_input_ir_sha256,
                binding=CompilerBinding(prepared.source_ir_sha256,
                    '7c56a5e40f7fd928dfd5c72902d5def0097db73a','native-default-pipeline-v1'))
        except UnsupportedCandidate:
            candidate=None
        if candidate is not None:
            module.set_attr('tt.hbv.plan_bundle',passes.ttir.make_l_decision_attribute(
                module,candidate.plan.to_dict()))
            route=Route.PIPELINE.value
    # The route description is diagnostic output, not a compiler attribute.
    return route
