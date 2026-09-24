"""Current compiler facts, preserved without importing models or old Providers."""
from dataclasses import dataclass
from hashlib import sha256
import json
from .provider import LoopFacts
from .state import Loop, innermost
from .load_service import describe_loop_loads


@dataclass(frozen=True)
class CompilerFacts:
    existing_loop_subjects: tuple[LoopFacts, ...]
    raw_payload: str
    resolved_num_warps: int
    resolved_num_stages: int
    semantic_ref: str

    def whole_kernel_payload(self):
        """Fresh full snapshot, not just the loop census or selected features."""
        return json.loads(self.raw_payload)


def decode_facts(raw, *, num_warps, num_stages):
    if any(type(n) is not int or n < 1 for n in (num_warps, num_stages)):
        raise ValueError('invalid resolved launch configuration')
    # A compiler query returns a mapping directly. Text is accepted only for
    # archived, chain-external snapshots, never as an IR transport.
    p = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(p, dict):
        raise ValueError('compiler facts must be a structured mapping')
    if p.get('schema') not in ('hbv.loop.static-facts.v16', 'l.core.planning-facts.v1') or p.get('extractable') is not True:
        raise ValueError('current extractable compiler facts required')
    rows = p['loop_census']
    if type(p['native_loop_count']) is not int or p['native_loop_count'] != len(rows):
        raise ValueError('loop census count mismatch')
    subjects = []
    for row in rows:
        trip = row['exact_static_trip_count']
        known = row['exact_static_trip_count_known']
        if (type(known) is not bool or type(trip) is not int or trip < 0
                or (not known and trip != 0)):
            raise ValueError('invalid compiler trip observation')
        # Availability is authoritative; zero is also a valid known count.
        # Missing availability must fail, not fall back to legacy inference.
        kwargs = dict(exact_static_trip_count=trip if known else None)
        if 'launch_trip_count_known' in row:
            launch_known = row['launch_trip_count_known']
            launch_trip = row.get('launch_trip_count')
            if (type(launch_known) is not bool or type(launch_trip) is not int
                    or launch_trip < 0 or (not launch_known and launch_trip != 0)
                    or row.get('launch_trip_count_scope') != 'per_loop_invocation_if_reached'):
                raise ValueError('invalid launch trip observation')
            kwargs['launch_trip_count'] = launch_trip if launch_known else None
        for field in ('locator', 'lower_kind', 'upper_kind', 'step_kind', 'carried_value_count',
                      'entry_owned', 'nesting_depth', 'parent_locator',
                      'nested_inner_dimension_capable', 'nested_inner_dimension_certificate',
                      'nested_inner_dimension_reason', 'runtime_main_tail_certificate',
                      'preexisting_pipeline_stage_count', 'source_operation_group_count',
                      'phase_barrier_operation_count', 'native_visible_async_load_count',
                      'logical_elementwise_packable_operation_count',
                      'logical_load_adapter_opportunity_count', 'logical_store_adapter_opportunity_count'):
            kwargs[field] = row[field]
        for name, prefix in (('pipeline', 'native_pipeline'), ('phase', 'full_unroll_reorder'),
                             ('logical', 'full_unroll_vectorization')):
            kwargs[name+'_capable'] = row[prefix+'_capable']
            kwargs[name+'_certificate'] = row[prefix+'_capability_certificate']
            kwargs[name+'_reason'] = row[prefix+'_capability_reason']
        observed = row['direct_memory_order_constraints_observed']
        if type(observed) is not bool:
            raise ValueError('invalid memory observation availability')
        for field in ('iteration_invariant_store_address_count', 'explicit_synchronization_barrier_count'):
            kwargs[field] = row[field] if observed else None
        work_observed = row.get('direct_container_work_observed', False)
        if type(work_observed) is not bool:
            raise ValueError('invalid direct work availability')
        for field in ('direct_load_container_bytes', 'direct_store_container_bytes',
                      'opaque_body_region_count', 'unconditional_entry_loop_ancestry'):
            kwargs[field] = row[field] if work_observed else None
        kwargs['memory_opaque_body_region_count'] = row.get('memory_opaque_body_region_count') if work_observed else None
        if 'structured_body_container_work' in row:
            from .structured_work import structured_work_from_fact, structured_container_bounds
            reads, writes = structured_container_bounds(structured_work_from_fact(
                row['structured_body_container_work']))
            kwargs['body_container_work_observed'] = True
            kwargs['body_load_container_bytes'] = reads.lower if reads.lower == reads.upper else None
            kwargs['body_store_container_bytes'] = writes.lower if writes.lower == writes.upper else None
        accesses=p.get('native_pointer_axis_facts')
        # Old snapshots and helper-owned loops remain unknown. Entry-only
        # observations cannot silently turn a helper's memory work into zero.
        if (row['entry_owned'] and isinstance(accesses,list) and
                all('loop_ancestry_innermost_first' in a for a in accesses)):
            service=describe_loop_loads(accesses,row['locator'])
            if service['all_direct']:
                kwargs['source_load_service_counts']=tuple(service['counts'][k] for k in
                    ('scalar','tensor_source_wide','tensor_source_narrow','unknown'))
        subjects.append(LoopFacts(**kwargs))
    # Validate structure before checking route capability.
    innermost(tuple(Loop(s.locator, s.parent_locator or None, s.exact_static_trip_count) for s in subjects))
    by_id = {s.locator:s for s in subjects}
    for s in subjects:
        if s.parent_locator and s.nesting_depth != by_id[s.parent_locator].nesting_depth + 1:
            raise ValueError('loop nesting depth differs from parent chain')
    encoded = json.dumps(p, sort_keys=True, separators=(',', ':'), allow_nan=False)
    identity = json.dumps(dict(payload=p, num_warps=num_warps, num_stages=num_stages),
                          sort_keys=True, separators=(',', ':'), allow_nan=False)
    return CompilerFacts(tuple(subjects), encoded, num_warps, num_stages,
                         'l.core.compiler-facts.'+sha256(identity.encode()).hexdigest())
