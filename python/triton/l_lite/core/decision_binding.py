"""Nonpredictive compiler consumer for an externally selected route plan.

Run at the post-cleanup analysis cut. This applies Bridge and binds the plan;
native make_ttir still owns Decision -> LoopUnroll -> Materialize -> Validate.
"""
from .ir_identity import ir_digest
from .ttir import prepare_at_analysis_point


def bind_selected_decision(module, *, config, plan_json, source_sha256,
                           route_input_sha256):
    import json
    from triton._C.libtriton import passes
    if ir_digest(module) != source_sha256:
        raise ValueError('analysis decision belongs to a different input IR')
    if not plan_json or not route_input_sha256:
        raise ValueError('selected decision requires a plan and projected input identity')
    prepared = prepare_at_analysis_point(module, config)
    if prepared.route_input_ir_sha256 != route_input_sha256:
        raise ValueError('actual Bridge output differs from analyzed candidate')
    # The full record stays outside IR; only validated typed decision fields
    # are passed to the materialization suffix.
    module.set_attr('tt.hbv.plan_bundle', passes.ttir.make_l_decision_attribute(
        module, json.loads(plan_json)))
    return prepared.grid_divisors
