"""Single current L plan envelope; no historical contracts or model imports."""
from dataclasses import dataclass
import json
from .factors import content_ref
from .state import Route


@dataclass(frozen=True)
class Plan:
    # Immutable canonical payload; callers receive fresh decoded dictionaries.
    payload: str

    def canonical_json(self):
        return self.payload

    def to_dict(self):
        return json.loads(self.payload)


def _binding(role, owner, key, value):
    return dict(semantic_role=role, native_owner_or_binding_kind=owner,
                native_key_or_adapter_id=key, typed_value_or_typed_reference=value,
                required=True, binding_schema_version=1)


def build_plan(*, route, members, provider_ref, identity, binding):
    route = Route(route)
    if any(not isinstance(value, str) or not value for value in (
            provider_ref, identity, binding.source_ref, binding.compiler_commit,
            binding.materialization_binding_ref)):
        raise ValueError('missing plan identity or target binding')
    if not members:
        raise ValueError('empty route member set')
    pipeline = route == Route.PIPELINE
    nested = not pipeline and any(m.parent_loop_locator for m in members)
    focal = len(members) == 1 and not nested
    ordinals = [int(m.provider_loop_locator.rsplit('.', 1)[-1]) for m in members]
    if ordinals != sorted(set(ordinals)):
        raise ValueError('members must have unique ordered compiler locators')
    kind = {Route.PIPELINE: 'pipeline_stage_count', Route.REORDER: 'phase_reorder_grouping_width',
            Route.VECTORIZE: 'logical_vector_grouping_width'}[route]
    if any(m.route_factor_kind != kind for m in members):
        raise ValueError('member belongs to a different route')
    if pipeline and any(m.nested_context_certificate_ref for m in members):
        raise ValueError('pipeline uses ordinary member binding, not nested unroll context')
    if not pipeline and ((nested and len(members) != 1)
                         or (not nested and any(m.nesting_depth for m in members))):
        raise ValueError('member set lacks the required nested context scope')
    policy = ('provider_bound_nested_inner_dimension_scf_for' if nested else
              'provider_bound_focal_existing_scf_for' if focal else
              'provider_bound_independent_existing_scf_for_set')
    version = {Route.REORDER: 11, Route.VECTORIZE: 12, Route.PIPELINE: 13}[route]
    schema = {Route.REORDER: 'hbv.loop-provider.bound-subject-set.v1',
              Route.VECTORIZE: 'hbv.loop-provider.bound-logical-subject-set.v2',
              Route.PIPELINE: 'hbv.loop-provider.bound-pipeline-subject-set.v1'}[route]
    params = dict(provider_schema=schema, subject_policy=policy,
                  provider_ref=provider_ref, kind=route.value)
    if pipeline:
        stages = {m.route_factor for m in members}
        if len(stages) != 1:
            raise ValueError('pipeline members must share the kernel stage option')
        params.update(shared_stage_count=next(iter(stages)), stage_scope='whole_kernel_shared_option')
    elif route == Route.VECTORIZE:
        params['packing_policy'] = 'registered_exact_operation_capability'
    subject = content_ref('l.provider-loop-pipeline-subject-set.' if pipeline else
                          'l.provider-loop-subject-set.',
                          dict(params, member_refs=[m.member_ref for m in members]))
    params.update(subject_ref=subject, adapter_version=version,
                  members=[m.to_dict() for m in members])
    bindings = []
    if pipeline:
        bindings.append(_binding('loop_pipeline_stage_count', 'native_triton_pipeline_option',
                                 'tt.num_stages@whole_kernel', f'stage_count:{params["shared_stage_count"]}'))
        bindings.extend(_binding('loop_pipeline_member', 'native_triton_software_pipeline',
                                 f'hbv.loop.pipeline.provider_bound.v13@{m.provider_loop_locator}', m.member_ref)
                        for m in members)
    else:
        adapter = ('hbv.loop.phase_major.provider_bound.v11' if route == Route.REORDER else
                   'hbv.loop.logical_group.provider_bound.v12')
        for m in members:
            bindings.extend((
                _binding('loop_full_unroll_member', 'native_triton_loop_unroll',
                         f'tt.loop_unroll_factor@{m.provider_loop_locator}',
                         f'factor:{m.route_factor};member:{m.member_ref}'),
                _binding('loop_route_member', 'triton_hbv_loop_adapter',
                         f'{adapter}@{m.provider_loop_locator}', m.member_ref)))
    before = 'pre_pipeline' if pipeline else 'pre_loop_unroll'
    after = 'post_pipeline' if pipeline else 'post_route'
    subject_guard = ('l.subject.provider_bound_pipeline_set' if pipeline else
                     'l.subject.provider_bound_nested_inner' if nested else
                     'l.subject.provider_bound_focal' if focal else 'l.subject.provider_bound_independent_set')
    guard_specs = (
        ('l.bundle.schema', 'python_preflight+ttir_focal', 'LCorePlan+TritonHBVLoopDecisionPass'),
        ('l.target.binding', 'python_preflight', binding.materialization_binding_ref),
        (subject_guard, before, 'LoopCensusProvider+TritonHBVLoopDecisionPass'),
        ('l.subject.structure', before, 'OrdinaryInnermostLoopScope'),
        ('l.effects_dependencies', before, 'PerMemberRouteLegality+ParentContext'),
        ('l.parameters.closed', before, 'LCoreRouteMember'),
        ('l.route.mutual_exclusion', before, 'TritonHBVLoopDecisionPass'),
        ('l.pipeline.member_lineage' if pipeline else 'l.unroll.lineage',
         'post_pipeline+final_static' if pipeline else 'post_loop_unroll',
         'PerMemberAsyncCopyCommitWaitLineage' if pipeline else 'TritonHBVLoopMaterializePass'),
        ('l.route.postcondition', after+'+final_static', 'TritonHBVValidateLoopPlanPass'),
        ('l.ir.verify', 'route_suffix', 'MLIR+native_target_verifiers'),
        ('l.observation.correspondence', after+'+final_static', 'LCoreMaterializationObservation'))
    guards = [dict(guard_id=g, responsible_stage=s, verifier_or_legality_binding=v, required=True)
              for g, s, v in guard_specs]
    feedback_ids = (('loop.route.realized', 'loop.route.parameters',
                     'loop.pipeline.member_artifact_lineage', 'loop.route.postcondition') if pipeline else
                    ('loop.route.realized', 'loop.route.parameters', 'loop.route.postcondition',
                     'codegen.instruction_family_counts'))
    feedback = []
    for field in feedback_ids:
        static = field in ('loop.pipeline.member_artifact_lineage', 'codegen.instruction_family_counts')
        feedback.append(dict(field_id=field, availability_stage='final_static_code' if static else after,
                             source_kind='codegen_realized' if static else 'route_realized', required=True,
                             evidence_sink='ObservationRecord' if static else 'RealizationRecord'))
    contract = dict(schema_version=1, decision_ref='decision.'+identity, project_kind='loop',
                    route_ref=route.value, candidate_parameters=params,
                    dynamic_materialization_bindings=bindings, dynamic_guard_assumptions=guards,
                    requested_feedback_fields=feedback,
                    fallback_binding=dict(original_route_ref='l.original.default', max_original_route_retries=1,
                                          retry_loop_guard='hbv_disable_decision'),
                    minimal_provenance=dict(producer_schema='hbv.plan_contract.v1', source_ref=binding.source_ref,
                                            compiler_commit=binding.compiler_commit, adapter_version=version),
                    subject_locator=dict(kind='hbv_typed_marker', subject_ref=subject,
                                         anchor_or_marker_ref='tt.hbv.l.subject'))
    return Plan(json.dumps(dict(schema_version=1, bundle_id='bundle.'+identity, contract=contract),
                           sort_keys=True, separators=(',', ':'), allow_nan=False))
