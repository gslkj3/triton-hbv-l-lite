from pathlib import Path
import sys

import pytest
import triton


# This also lets the test run against an already-built Triton 3.6 tree while
# the public snapshot itself has not been rebuilt yet.
PUBLIC_TRITON = Path(__file__).resolve().parents[4] / "python" / "triton"
if str(PUBLIC_TRITON) not in triton.__path__:
    triton.__path__.append(str(PUBLIC_TRITON))

from triton.l_lite import (  # noqa: E402
    LOOP_AUTOTUNE_CANDIDATE_META_PARAMETER,
    LOOP_COMPOSITION_ROUTES,
    build_loop_exhaustive_autotune_domain_v1,
    build_loop_intervention_cartesian_graph_v1,
    build_loop_native_autotune_control_v1,
)
from triton.l_lite.contract import (  # noqa: E402
    LLiteSchemaError,
    LoopBridgeRouteCompositionLegalityV2,
)


ROUTE_FACTORS = {route: (2, 4) for route in LOOP_COMPOSITION_ROUTES}


def _domain():
    graph = build_loop_intervention_cartesian_graph_v1(
        bridge_factors=(1, 2, 4),
        route_factors=ROUTE_FACTORS,
        existing_loop_subject_available=False,
    )
    return graph, build_loop_exhaustive_autotune_domain_v1(graph)


class _FakeKernel:
    arg_names = ("x",)
    fn = staticmethod(lambda x: x)

    def __init__(self, calls, candidate_ref):
        self.calls = calls
        self.candidate_ref = candidate_ref

    def run(self, *args, grid, warmup, **kwargs):
        assert LOOP_AUTOTUNE_CANDIDATE_META_PARAMETER not in kwargs
        self.calls.append((self.candidate_ref, args[0], warmup))
        return self.candidate_ref


def test_cartesian_domain_keeps_every_requested_route_cell():
    graph, domain = _domain()
    requested = tuple(
        arm for arm in graph.arms if arm.route_ref in LOOP_COMPOSITION_ROUTES)

    assert len(domain.candidates) == 1 + len(requested) == 19
    assert sum(candidate.original for candidate in domain.candidates) == 1
    assert not domain.prediction_pruning_permitted
    assert not domain.early_stopping_permitted


def test_full_unroll_equality_is_a_typed_materializer_limit():
    graph, _ = _domain()
    rejected = tuple(
        arm for arm in graph.arms
        if arm.route_ref.endswith((
            "full_unroll_phase_major.v1",
            "full_unroll_logical_group.v1"))
        and arm.bridge_factor > 1
        and arm.route_factor != arm.bridge_factor)

    assert rejected
    assert {arm.typed_reason for arm in rejected} == {
        "current_full_unroll_requires_route_factor_equal_trip_count"}


def test_runtime_route_subject_requires_main_tail_certificate():
    with pytest.raises(LLiteSchemaError):
        LoopBridgeRouteCompositionLegalityV2(
            bridge_factor=1,
            route_ref="l.ttir.full_unroll_phase_major.v1",
            route_factor=2,
            bridge_constructed=False,
            route_subject_available=True,
            route_subject_ref="l.route-subject.runtime",
            route_subject_exact_trip_count=None,
            runtime_main_tail_certificate_ref="",
            route_factor_kind="unroll_grouping_factor",
            factor_admission_ref="l.route-factor-admission.test",
            legal=True,
            typed_reason="",
            proof_refs=("proof:test",),
        )


def test_native_autotuner_measures_all_candidates_and_selects_minimum():
    _, domain = _domain()
    calls = []
    winner = domain.candidates[-1]
    original = domain.candidates[0]
    kernel = _FakeKernel(calls, original.candidate_ref)
    bindings = {
        candidate.candidate_ref: _FakeKernel(calls, candidate.candidate_ref)
        for candidate in domain.candidates
    }
    bindings[original.candidate_ref] = kernel

    def do_bench(kernel_call, quantiles):
        assert quantiles == (0.5, 0.2, 0.8)
        kernel_call()
        latency = 80.0 if calls[-1][0] == winner.candidate_ref else 100.0
        return [latency, latency - 1.0, latency + 1.0]

    control = build_loop_native_autotune_control_v1(
        kernel=kernel,
        domain=domain,
        bindings=bindings,
        key=("x",),
        do_bench=do_bench,
    )

    assert control.run(7, grid=(1,), warmup=False) == winner.candidate_ref
    assert tuple(ref for ref, _, _ in calls[:-1]) == tuple(
        candidate.candidate_ref for candidate in domain.candidates)
    assert control.native_autotuner.early_config_prune is None
    assert control.native_autotuner.perf_model is None


def test_native_key_cache_reuses_the_winner():
    _, domain = _domain()
    calls = []
    original = domain.candidates[0]
    kernel = _FakeKernel(calls, original.candidate_ref)
    bindings = {
        candidate.candidate_ref: _FakeKernel(calls, candidate.candidate_ref)
        for candidate in domain.candidates
    }
    bindings[original.candidate_ref] = kernel

    def do_bench(kernel_call, quantiles):
        kernel_call()
        return [100.0, 99.0, 101.0]

    control = build_loop_native_autotune_control_v1(
        kernel=kernel,
        domain=domain,
        bindings=bindings,
        key=("x",),
        do_bench=do_bench,
    )
    control.run(7, grid=(1,), warmup=False)
    first_count = len(calls)
    control.run(7, grid=(1,), warmup=False)

    assert len(calls) == first_count + 1
    assert calls[-1][0] == control.best_candidate_ref
