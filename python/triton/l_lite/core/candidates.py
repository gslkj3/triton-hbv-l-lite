"""Cartesian candidate planning. Shared by predictive and measured selection."""
from dataclasses import dataclass
from hashlib import sha256
import json
from .compiler import build_candidate, PlannedCandidate
from .state import Route
from .factors import UnsupportedCandidate


@dataclass(frozen=True)
class BoundCandidate:
    upstream_factor: int
    upstream_input: object
    planned: PlannedCandidate


@dataclass(frozen=True)
class Rejection:
    upstream_factor: int
    route: Route
    factor: int
    reason: str


@dataclass(frozen=True)
class CandidateBatch:
    candidates: tuple[BoundCandidate, ...]
    rejections: tuple[Rejection, ...]


def enumerate_candidates(prepared, route_factors, *, binding):
    # Configuration is a finite search space, not a learned factor allowlist.
    choices = {Route(route): tuple(sorted(set(factors))) for route, factors in route_factors.items()}
    if any(type(f) is not int or f < 1 for factors in choices.values() for f in factors):
        raise ValueError('invalid route factor configuration')
    candidates, rejected = [], []
    for upstream_factor, item in prepared.inputs:
        for route in sorted(choices, key=lambda r: r.value):
            for factor in choices[route]:
                identity = sha256(json.dumps(dict(
                    source=binding.source_ref, compiler=binding.compiler_commit,
                    materializer=binding.materialization_binding_ref,
                    route_input=item.route_input_ir_sha256,
                    upstream_factor=upstream_factor, route=route.value, factor=factor),
                    sort_keys=True).encode()).hexdigest()
                try:
                    plan = build_candidate(provider=item.provider, route=route, factor=factor,
                                           identity=identity, binding=binding)
                except UnsupportedCandidate as error:
                    rejected.append(Rejection(upstream_factor, route, factor, str(error)))
                    continue
                candidates.append(BoundCandidate(upstream_factor, item, plan))
    return CandidateBatch(tuple(candidates), tuple(rejected))
