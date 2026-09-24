"""Ordinary-loop state at the peer-route boundary.

Legality is supplied by the compiler, not inferred from training coverage.
None denotes unavailable iteration count, never zero work.
"""
from dataclasses import dataclass
from enum import Enum


class Route(str, Enum):
    PIPELINE = 'l.nvidia.software_pipeline.v1'
    REORDER = 'l.ttir.full_unroll_phase_major.v1'
    VECTORIZE = 'l.ttir.full_unroll_logical_group.v1'


@dataclass(frozen=True)
class Loop:
    locator: str
    parent: str | None
    iterations: int | None

    def __post_init__(self):
        if not self.locator or (self.iterations is not None and
                (type(self.iterations) is not int or self.iterations < 0)):
            raise ValueError('invalid ordinary loop state')


def innermost(loops):
    """Select structurally, before examining any route capability."""
    by_id = {loop.locator: loop for loop in loops}
    if len(by_id) != len(loops):
        raise ValueError('duplicate loop locator')
    for loop in loops:
        seen = {loop.locator}
        parent = loop.parent
        while parent is not None:
            if parent not in by_id or parent in seen:
                raise ValueError('incomplete or cyclic loop structure')
            seen.add(parent)
            parent = by_id[parent].parent
    parents = {loop.parent for loop in loops}
    return tuple(loop for loop in loops if loop.locator not in parents)


@dataclass(frozen=True)
class ResponseState:
    route: Route
    iterations: int | None
    requested_stages: int
    grouping_width: int
    main_groups: int | None
    tail: int | None


def transition(loop, route, factor):
    """Structural effect only: no legality, occupancy or speedup claim."""
    route = Route(route)
    if type(factor) is not int or factor < 1:
        raise ValueError('invalid route factor')
    if route == Route.PIPELINE:
        return ResponseState(route, loop.iterations, factor, 1, None, None)
    main, tail = (None, None) if loop.iterations is None else divmod(loop.iterations, factor)
    return ResponseState(route, loop.iterations, 1, factor, main, tail)
