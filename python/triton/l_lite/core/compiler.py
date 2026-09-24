"""Current ordinary-loop plan construction, independent of historical packages.

No historical iteration executors or time models are imported. The upstream
Bridge choice has already been applied to the Provider supplied here.
"""
from dataclasses import dataclass
from .state import Loop, Route, innermost, transition
from .candidate_contract import Candidate
from .transport import bind_member, RouteMember
from .factors import UnsupportedCandidate
from .plan import build_plan


@dataclass(frozen=True)
class CompilerBinding:
    source_ref: str
    compiler_commit: str
    materialization_binding_ref: str


@dataclass(frozen=True)
class PlannedCandidate:
    candidate: Candidate
    plan: object
    members: tuple[RouteMember, ...]


def build_candidate(*, provider, route, factor, identity, binding):
    route = Route(route)
    # Validate the entire structural census BEFORE filtering capability.
    loops = tuple(Loop(s.locator, s.parent_locator or None, s.exact_static_trip_count)
                  for s in provider.existing_loop_subjects)
    leaves = {loop.locator: loop for loop in innermost(loops)}
    selected = [s for s in provider.existing_loop_subjects
                if s.locator in leaves and s.entry_owned and s.route_certificate(route)]
    if not selected:
        raise UnsupportedCandidate('provider_has_no_route_capable_loop')
    if route != Route.PIPELINE and any(s.nesting_depth for s in selected):
        # This is the existing materializer's subject-set limit, not a model
        # support restriction. Do not silently target an outer loop instead.
        if len(selected) != 1 or not selected[0].nested_inner_dimension_capable:
            raise UnsupportedCandidate('nested_route_lacks_unique_certified_inner_subject')
    def ordinal(subject):
        prefix = 'planning-cut.loop.'
        if not subject.locator.startswith(prefix) or not subject.locator[len(prefix):].isdigit():
            raise ValueError('invalid compiler loop locator')
        return int(subject.locator[len(prefix):])
    selected.sort(key=ordinal)
    members = tuple(bind_member(s, route=route, factor=factor) for s in selected)
    plan = build_plan(route=route, members=members, provider_ref=provider.semantic_ref,
                      identity=identity, binding=binding)
    states = tuple(transition(leaves[s.locator], route, factor) for s in selected)
    return PlannedCandidate(Candidate(identity, states, plan.canonical_json()), plan, members)
