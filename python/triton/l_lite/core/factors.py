"""Route dose semantics independent of source origin and fitted models.

These are the current materializer's rules, not a training support filter.
Hardware scheduling and final realization must still be checked by the compiler.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from .state import Route


class UnsupportedCandidate(ValueError):
    """Well-formed candidate not supported by the declared transformation."""


def content_ref(prefix, values):
    return prefix + sha256(json.dumps(values, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()[:24]


@dataclass(frozen=True)
class FactorAdmission:
    route: Route
    factor: int
    factor_kind: str
    iterations: int | None
    main_iterations: int | None
    tail_iterations: int | None
    groups: int | None
    disposition: str
    runtime_certificate: str

    @property
    def ref(self):
        return content_ref('l.core.factor.', asdict(self))


def admit_factor(*, route, factor, iterations, runtime_certificate=''):
    route = Route(route)
    if type(factor) is not int or factor < 1:
        raise ValueError('route factor must be a positive integer')
    if iterations is not None and (type(iterations) is not int or iterations < 0):
        raise ValueError('invalid exact iteration count')
    if iterations is not None and runtime_certificate:
        raise ValueError('exact and runtime iteration evidence conflict')
    if iterations == 0:
        raise UnsupportedCandidate('empty_loop_has_no_route_intervention')
    # The current member transport requires >=2 for all three routes. A stage
    # count of one can be meaningful natively but is not this pipeline candidate.
    if factor < 2:
        raise UnsupportedCandidate('materializer_requires_factor_at_least_two')
    if iterations is None and not runtime_certificate:
        raise UnsupportedCandidate('runtime_loop_requires_iteration_certificate')
    kind = {Route.PIPELINE: 'pipeline_stage_count',
            Route.REORDER: 'phase_reorder_grouping_width',
            Route.VECTORIZE: 'logical_vector_grouping_width'}[route]
    if route == Route.PIPELINE:
        return FactorAdmission(route, factor, kind, iterations, iterations,
                               0 if iterations is not None else None, None,
                               'retained_live_focal_loop', runtime_certificate)
    if factor & (factor - 1):
        raise UnsupportedCandidate('full_unroll_grouping_factor_requires_power_of_two_at_least_two')
    if iterations is None:
        return FactorAdmission(route, factor, kind, None, None, None, None,
                               'runtime_factorized_main_tail_focal_loop', runtime_certificate)
    if factor > iterations:
        raise UnsupportedCandidate('route_grouping_factor_exceeds_exact_subject_trip_count')
    groups, tail = divmod(iterations, factor)
    disposition = ('eliminated_focal_loop' if groups == 1 and tail == 0 else
                   'factorized_main_tail_focal_loop' if tail else 'retained_factor_group_focal_loop')
    return FactorAdmission(route, factor, kind, iterations, groups * factor, tail,
                           groups, disposition, '')
