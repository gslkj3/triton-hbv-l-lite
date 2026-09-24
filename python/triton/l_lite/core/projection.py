"""Direct route-state projection, not backend allocation or a time prediction.

Consumes the actual ordinary-loop input after any upstream construction. Loop
identities bind evidence only; neither identity nor construction origin is a
numeric coordinate. Keep per-loop state rather than summing heterogeneous loops.
"""
from dataclasses import dataclass
import math
from .state import Loop, Route, innermost, transition
from .transport import bind_member


@dataclass(frozen=True)
class DirectProjection:
    provider_ref: str
    candidate_ref: str
    # Named state remains per member; None is not encoded as zero.
    members: tuple[tuple[tuple[str, float | None], ...], ...]
    # Source loop identities align sparse route actions to the whole-kernel
    # phase tree. They are not numeric timing features or Bridge-origin tags.
    member_locators: tuple[str, ...]

    def __post_init__(self):
        if (not self.members or len(self.members) != len(self.member_locators) or
                len(set(self.member_locators)) != len(self.member_locators) or
                any(not locator for locator in self.member_locators)):
            raise ValueError('every direct member needs one unique loop locator')


def project_direct_state(provider, planned):
    subjects = {s.locator: s for s in provider.existing_loop_subjects}
    loops = tuple(Loop(s.locator, s.parent_locator or None, s.exact_static_trip_count)
                  for s in provider.existing_loop_subjects)
    leaves = {s.locator for s in innermost(loops)}
    if len(planned.members) != len(planned.candidate.state) or not planned.members:
        raise ValueError('member/state count mismatch')
    rows, locators = [], []
    seen = set()
    for member, state in zip(planned.members, planned.candidate.state):
        locator = member.provider_loop_locator
        if locator not in leaves or locator in seen:
            raise ValueError('projection must bind unique innermost loops')
        seen.add(locator)
        locators.append(locator)
        subject = subjects[locator]
        if bind_member(subject, route=state.route, factor=member.route_factor) != member:
            raise ValueError('projection member differs from current facts')
        expected = transition(Loop(locator, subject.parent_locator or None,
                                   subject.exact_static_trip_count),
                              state.route, member.route_factor)
        if state != expected:
            raise ValueError('projection state differs from direct transition')
        # Preserve the statically certified candidate above. Only the service
        # projection may use an observed launch count; it grants no new route
        # capability and does not specialize the compiled loop.
        n = subject.launch_trip_count if subject.launch_trip_count is not None else state.iterations
        service_state = transition(Loop(locator, subject.parent_locator or None, n),
                                   state.route, member.route_factor)
        invocation_count = 1 if subject.unconditional_entry_loop_ancestry is True else None
        parent = subject.parent_locator
        while parent:
            ancestor = subjects[parent]
            count = (ancestor.launch_trip_count if ancestor.launch_trip_count is not None
                     else ancestor.exact_static_trip_count)
            if invocation_count == 0 or count == 0:
                invocation_count = 0
            elif invocation_count is None or count is None:
                invocation_count = None
            else:
                invocation_count *= count
            parent = ancestor.parent_locator
        numeric = dict(
            source_iterations=n,
            invocations_per_program=invocation_count,
            direct_load_container_bytes=subject.direct_load_container_bytes,
            direct_store_container_bytes=subject.direct_store_container_bytes,
            body_container_work_observed=subject.body_container_work_observed,
            body_load_container_bytes=subject.body_load_container_bytes,
            body_store_container_bytes=subject.body_store_container_bytes,
            opaque_body_regions=subject.opaque_body_region_count,
            memory_opaque_body_regions=subject.memory_opaque_body_region_count,
            requested_stages=state.requested_stages,
            grouping_width=state.grouping_width,
            phase_reorder_width=(state.grouping_width if state.route == Route.REORDER else 1),
            logical_pack_width=(state.grouping_width if state.route == Route.VECTORIZE else 1),
            main_groups=service_state.main_groups,
            tail_iterations=service_state.tail,
            carried_values=subject.carried_value_count,
            source_operation_groups=subject.source_operation_group_count,
            phase_barriers=subject.phase_barrier_operation_count,
            visible_async_loads=subject.native_visible_async_load_count,
            packable_elementwise_ops=subject.logical_elementwise_packable_operation_count,
            packable_load_opportunities=subject.logical_load_adapter_opportunity_count,
            packable_store_opportunities=subject.logical_store_adapter_opportunity_count,
            invariant_store_addresses=subject.iteration_invariant_store_address_count,
            explicit_barriers=subject.explicit_synchronization_barrier_count)
        # Pre-transform source services, not final async instructions. Keep
        # them separate from requested route/factor and preserve missingness.
        service=subject.source_load_service_counts
        for i,name in enumerate(('scalar','wide_tensor','narrow_tensor','unknown')):
            numeric['source_'+name+'_loads']=None if service is None else service[i]
        # Retained semantics: factor changes stage depth OR unrolled group width.
        # These are requested/structural quantities, never realized vector width,
        # allocated registers, occupancy, or overlapping memory transactions.
        if state.route != Route.PIPELINE:
            numeric['tail_fraction'] = None if n is None else (service_state.tail / n if n else 0.0)
        rows.append(tuple(numeric.items()))
    return DirectProjection(provider.semantic_ref, planned.candidate.identity,
                            tuple(rows), tuple(locators))


def require_coordinates(member, names):
    """Select declared coordinates; absence is an explicit missing-state error.

    Model selection and field admission belong above this projection. No
    training-range check, factor allowlist, or missing-value imputation here.
    """
    values = dict(member)
    if len(values) != len(member) or len(set(names)) != len(names):
        raise ValueError('duplicate coordinate')
    result = {}
    for name in names:
        value = values.get(name)
        if value is None:
            raise ValueError('required state unavailable: ' + name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('invalid coordinate: ' + name)
        result[name] = float(value)
    return result
