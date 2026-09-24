"""Origin-neutral, content-bound loop member transport for the L compiler.

The surrounding compiler plan envelope is still being migrated. This module
does not import or serialize historical plan variants.
"""
from dataclasses import asdict, dataclass
from .factors import admit_factor, content_ref, UnsupportedCandidate
from .state import Route


@dataclass(frozen=True)
class RouteMember:
    provider_loop_locator: str
    nesting_depth: int
    exact_static_trip_count: int | None
    runtime_main_tail_certificate_ref: str
    route_capability_certificate_ref: str
    route_factor: int
    route_factor_kind: str
    factor_admission_ref: str
    parent_loop_locator: str
    nested_context_certificate_ref: str
    schema: str = 'hbv.loop.provider-bound-route-member.v1'

    def __post_init__(self):
        pipeline = self.route_factor_kind == 'pipeline_stage_count'
        for locator in (self.provider_loop_locator, self.parent_loop_locator):
            if locator == '' and locator == self.parent_loop_locator:
                continue
            prefix = 'planning-cut.loop.'
            if not isinstance(locator, str) or not locator.startswith(prefix) or not locator[len(prefix):].isdigit():
                raise ValueError('invalid member locator')
        if (not self.provider_loop_locator or type(self.nesting_depth) is not int or self.nesting_depth < 0
                or not self.route_capability_certificate_ref
                or bool(self.parent_loop_locator) != bool(self.nesting_depth)
                or (pipeline and bool(self.nested_context_certificate_ref))
                or (not pipeline and bool(self.parent_loop_locator) != bool(self.nested_context_certificate_ref))
                or self.schema != 'hbv.loop.provider-bound-route-member.v1'):
            raise ValueError('invalid member proof or scope')
        if self.parent_loop_locator and not pipeline and (self.nesting_depth < 1 or self.nested_context_certificate_ref not in {
                'nested_inner_local_equivalence_v2'}):
            raise ValueError('unrecognized nested context proof')
        kinds = {'pipeline_stage_count': Route.PIPELINE,
                 'phase_reorder_grouping_width': Route.REORDER,
                 'logical_vector_grouping_width': Route.VECTORIZE}
        if self.route_factor_kind not in kinds:
            raise ValueError('unknown factor semantics')
        admission = admit_factor(route=kinds[self.route_factor_kind], factor=self.route_factor,
                                 iterations=self.exact_static_trip_count,
                                 runtime_certificate=self.runtime_main_tail_certificate_ref)
        if admission.ref != self.factor_admission_ref:
            raise ValueError('member factor admission does not match its state')

    @property
    def member_ref(self):
        return content_ref('l.provider-loop-member.', asdict(self))

    def to_dict(self):
        return dict(asdict(self), member_ref=self.member_ref)


def bind_member(loop, *, route, factor):
    route = Route(route)
    if not loop.entry_owned:
        raise UnsupportedCandidate('helper_loop_not_exposed_to_materializer')
    certificate = loop.route_certificate(route)
    if not certificate:
        raise UnsupportedCandidate('loop_has_no_route_capability_or_opportunity')
    nested = bool(loop.parent_locator) and route != Route.PIPELINE
    if nested and not loop.nested_inner_dimension_capable:
        raise UnsupportedCandidate('nested_loop_lacks_parent_context_proof')
    admission = admit_factor(route=route, factor=factor,
                             iterations=loop.exact_static_trip_count,
                             runtime_certificate=loop.runtime_main_tail_certificate)
    return RouteMember(loop.locator, loop.nesting_depth, loop.exact_static_trip_count,
                       loop.runtime_main_tail_certificate, certificate, factor,
                       admission.factor_kind, admission.ref,
                       loop.parent_locator,
                       loop.nested_inner_dimension_certificate if nested else '')
