from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import math
from dataclasses import replace
from contextlib import contextmanager
import os
from types import MappingProxyType
import sys
import time
from typing import Mapping, Sequence, Tuple

from ..runtime.autotuner import Autotuner, Config
from ..runtime.jit import KernelInterface
from .contract import LLiteSchemaError as HBVSchemaError
from .composition import (
    LOOP_COMPOSITION_ROUTES,
    LOOP_ORIGINAL_ROUTE,
    LoopInterventionCartesianGraphV1,
    LoopInterventionCompositionArmV1,
    build_loop_intervention_cartesian_graph_v1,
)


LOOP_AUTOTUNE_DOMAIN_SCHEMA_V1 = "hbv.loop-autotune-domain.v1"
LOOP_AUTOTUNE_NATIVE_CONTROL_SCHEMA_V1 = (
    "hbv.loop-native-autotune-control.v1")
LOOP_AUTOTUNE_CANDIDATE_META_PARAMETER = (
    "_l_lite_candidate_ref")
LOOP_EXACT_PREFIX_ROUTE = (
    "l.ttir.predicated_exact_prefix_reduction.v1")
LOOP_FULL_UNROLL_VECTORIZATION_MECHANISM = (
    "l.ttir.full_unroll_logical_group.v1")
LOOP_AUTOTUNE_CARTESIAN_CANDIDATE = "bridge_route_cartesian"
LOOP_AUTOTUNE_DIRECT_CANDIDATE = "direct_route"


def _route_gap_class(*, pipeline_capable: bool, capable: bool,
                     reason: str) -> str:
    if capable:
        return "supported_at_planning_cut"
    if not pipeline_capable:
        return "outside_native_pipeline_reference_scope"
    if reason in {
        "reduction_contribution_is_not_tensor_32xi32",
    }:
        return "shared_materializer_capability_gap"
    return "route_specific_semantic_or_proof_rejection"


def _build_loop_mechanism_coverage_v1(artifact) -> dict:
    """Project passive compiler facts into a per-loop coverage matrix.

    This is read-only test evidence.  It is neither an autotune key nor a
    pruning input.  ``actual`` means the scf.for operations present at initial
    TTIR; ``observable`` means those still enumerated at HBV's planning cut.
    The distinction prevents source-loop elimination from being misreported
    as an HBV discovery failure.
    """
    raw = getattr(artifact.metadata, "hbv_loop_census_json", None)
    if not raw:
        return {
            "schema": "hbv.l-lite.loop-mechanism-coverage.v1",
            "actual_ttir_loop_count": 0,
            "observable_planning_cut_loop_count": 0,
            "observable_to_actual_ratio": 1.0,
            "loop_matrix": (),
            "native_pipeline_capable_loop_count": 0,
            "pipeline_capable_reorder_supported_count": 0,
            "pipeline_capable_vectorization_supported_count": 0,
        }
    census = json.loads(raw)
    actual = int(census.get("initial_ttir_loop_count", 0))
    loops = tuple(census.get("planning_cut_loops", ()))
    matrix = []
    for item in loops:
        pipeline = bool(item.get("native_pipeline_capable", False))
        reorder = bool(item.get("full_unroll_reorder_capable", False))
        vector = bool(item.get(
            "full_unroll_vectorization_capable", False))
        reorder_reason = str(item.get(
            "full_unroll_reorder_capability_reason", ""))
        vector_reason = str(item.get(
            "full_unroll_vectorization_capability_reason", ""))
        matrix.append({
            "locator": item["locator"],
            "source_location": item["source_location"],
            "nesting_depth": int(item["nesting_depth"]),
            "exact_static_trip_count": int(
                item.get("exact_static_trip_count", 0)),
            "bound_kinds": {
                "lower": item["lower_kind"],
                "upper": item["upper_kind"],
                "step": item["step_kind"],
            },
            "native_pipeline": {
                "planning_cut_capable": pipeline,
                "certificate": str(item.get(
                    "native_pipeline_capability_certificate", "")),
                "reason": str(item.get(
                    "native_pipeline_capability_reason", "")),
            },
            "full_unroll_and_reorder": {
                "planning_cut_capable": reorder,
                "certificate": str(item.get(
                    "full_unroll_reorder_capability_certificate", "")),
                "reason": reorder_reason,
                "relative_to_pipeline": _route_gap_class(
                    pipeline_capable=pipeline, capable=reorder,
                    reason=reorder_reason),
            },
            "full_unroll_and_vectorization": {
                "planning_cut_capable": vector,
                "certificate": str(item.get(
                    "full_unroll_vectorization_capability_certificate", "")),
                "reason": vector_reason,
                "relative_to_pipeline": _route_gap_class(
                    pipeline_capable=pipeline, capable=vector,
                    reason=vector_reason),
            },
        })
    pipeline_rows = tuple(
        row for row in matrix
        if row["native_pipeline"]["planning_cut_capable"])
    observable = len(matrix)
    return {
        "schema": "hbv.l-lite.loop-mechanism-coverage.v1",
        "actual_ttir_loop_count": actual,
        "observable_planning_cut_loop_count": observable,
        "observable_to_actual_ratio": (
            1.0 if actual == 0 else observable / actual),
        "loop_matrix": tuple(matrix),
        "native_pipeline_capable_loop_count": len(pipeline_rows),
        "pipeline_capable_reorder_supported_count": sum(
            row["full_unroll_and_reorder"]["planning_cut_capable"]
            for row in pipeline_rows),
        "pipeline_capable_vectorization_supported_count": sum(
            row["full_unroll_and_vectorization"]["planning_cut_capable"]
            for row in pipeline_rows),
    }


@contextmanager
def _qualification_diagnostics():
    """Suppress expected compiler rejection dumps for broad L-lite runs."""
    if os.environ.get("L_LITE_SHOW_QUALIFICATION_ERRORS") == "1":
        yield
        return
    sys.stderr.flush()
    saved_stderr = os.dup(2)
    null_stderr = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_stderr, 2)
        yield
    finally:
        os.dup2(saved_stderr, 2)
        os.close(null_stderr)
        os.close(saved_stderr)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(
        value, allow_nan=False, separators=(",", ":"),
        sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class LoopAutotuneCandidateV1:
    candidate_ref: str
    composition_arm_ref: str
    bridge_factor: int
    route_ref: str
    route_factor: int
    original: bool
    legality_proof_refs: Tuple[str, ...]
    candidate_kind: str = LOOP_AUTOTUNE_CARTESIAN_CANDIDATE

    def __post_init__(self):
        payload = {
            key: value for key, value in self.__dict__.items()
            if key != "candidate_ref"
        }
        if (
            self.candidate_kind not in (
                LOOP_AUTOTUNE_CARTESIAN_CANDIDATE,
                LOOP_AUTOTUNE_DIRECT_CANDIDATE)
            or (
                self.candidate_kind == LOOP_AUTOTUNE_CARTESIAN_CANDIDATE
                and not self.composition_arm_ref.startswith(
                    "l.composition-arm."))
            or (
                self.candidate_kind == LOOP_AUTOTUNE_DIRECT_CANDIDATE
                and not self.composition_arm_ref.startswith("l.direct-arm."))
            or self.bridge_factor < 1
            or self.route_factor < 1
            or self.original != (self.route_ref == LOOP_ORIGINAL_ROUTE)
            or (not self.original
                and self.route_ref not in (
                    *LOOP_COMPOSITION_ROUTES, LOOP_EXACT_PREFIX_ROUTE))
            or (self.original and (
                self.bridge_factor != 1 or self.route_factor != 1
                or self.candidate_kind
                != LOOP_AUTOTUNE_CARTESIAN_CANDIDATE))
            or (
                self.candidate_kind == LOOP_AUTOTUNE_DIRECT_CANDIDATE
                and (self.route_ref != LOOP_EXACT_PREFIX_ROUTE
                     or self.bridge_factor != 1 or self.route_factor != 1))
            or not self.legality_proof_refs
            or tuple(sorted(set(self.legality_proof_refs)))
            != self.legality_proof_refs
            or self.candidate_ref
            != "l.autotune-candidate." + _digest(payload)[:24]
        ):
            raise HBVSchemaError("Loop autotune candidate is malformed")

    @property
    def mechanism_route_ref(self) -> str:
        """Return the top-level causal mechanism, not its materializer."""
        if self.route_ref == LOOP_EXACT_PREFIX_ROUTE:
            return LOOP_FULL_UNROLL_VECTORIZATION_MECHANISM
        return self.route_ref


@dataclass(frozen=True)
class LoopAutotuneDomainV1:
    schema: str
    candidates: Tuple[LoopAutotuneCandidateV1, ...]
    prediction_pruning_permitted: bool
    early_stopping_permitted: bool
    domain_ref: str

    def __post_init__(self):
        payload = {
            **{key: value for key, value in self.__dict__.items()
               if key not in {"domain_ref", "candidates"}},
            "candidates": [item.__dict__ for item in self.candidates],
        }
        arm_refs = tuple(
            item.composition_arm_ref for item in self.candidates)
        if (
            self.schema != LOOP_AUTOTUNE_DOMAIN_SCHEMA_V1
            or not self.candidates
            or sum(item.original for item in self.candidates) != 1
            or len(set(arm_refs)) != len(arm_refs)
            or len({item.candidate_ref for item in self.candidates})
            != len(self.candidates)
            or self.prediction_pruning_permitted
            or self.early_stopping_permitted
            or self.domain_ref != "l.autotune-domain." + _digest(payload)[:24]
        ):
            raise HBVSchemaError("Loop autotune domain is malformed")


def _candidate(arm: LoopInterventionCompositionArmV1) -> LoopAutotuneCandidateV1:
    values = {
        "composition_arm_ref": arm.arm_ref,
        "bridge_factor": arm.bridge_factor,
        "route_ref": arm.route_ref,
        "route_factor": arm.route_factor,
        "original": arm.route_ref == LOOP_ORIGINAL_ROUTE,
        "legality_proof_refs": arm.proof_refs,
        "candidate_kind": LOOP_AUTOTUNE_CARTESIAN_CANDIDATE,
    }
    return LoopAutotuneCandidateV1(
        **values,
        candidate_ref="l.autotune-candidate." + _digest(values)[:24],
    )

def add_loop_exact_prefix_autotune_candidate_v1(
    domain: LoopAutotuneDomainV1,
    *,
    legality_proof_refs: Sequence[str],
) -> LoopAutotuneDomainV1:
    """Add the independently proven dynamic exact-prefix route.

    Exact-prefix replacement is the dynamic-recurrence materializer of the
    full-unroll-plus-vectorization mechanism.  It is not a Bridge-by-route
    composition: the factor is a predicated static container width, so the
    source recurrence is replaced directly.  A distinct candidate kind keeps
    this materialization lineage honest without inventing a fourth causal
    mechanism or a fake Bridge attribution.
    """
    if not isinstance(domain, LoopAutotuneDomainV1):
        raise HBVSchemaError("exact-prefix extension requires a domain")
    proofs = tuple(sorted(set(legality_proof_refs)))
    if not proofs or any(
        not isinstance(value, str) or not value for value in proofs
    ):
        raise HBVSchemaError("exact-prefix legality proof is absent")
    if any(
        item.route_ref == LOOP_EXACT_PREFIX_ROUTE
        for item in domain.candidates
    ):
        raise HBVSchemaError("exact-prefix candidate already exists")
    arm_payload = {
        "route_ref": LOOP_EXACT_PREFIX_ROUTE,
        "bridge_factor": 1,
        "route_factor": 1,
        "legality_proof_refs": proofs,
    }
    values = {
        "composition_arm_ref": (
            "l.direct-arm." + _digest(arm_payload)[:24]),
        "bridge_factor": 1,
        "route_ref": LOOP_EXACT_PREFIX_ROUTE,
        "route_factor": 1,
        "original": False,
        "legality_proof_refs": proofs,
        "candidate_kind": LOOP_AUTOTUNE_DIRECT_CANDIDATE,
    }
    direct = LoopAutotuneCandidateV1(
        **values,
        candidate_ref="l.autotune-candidate." + _digest(values)[:24],
    )
    candidates = domain.candidates + (direct,)
    domain_values = {
        "schema": LOOP_AUTOTUNE_DOMAIN_SCHEMA_V1,
        "candidates": candidates,
        "prediction_pruning_permitted": False,
        "early_stopping_permitted": False,
    }
    payload = {
        **{key: value for key, value in domain_values.items()
           if key != "candidates"},
        "candidates": [item.__dict__ for item in candidates],
    }
    return LoopAutotuneDomainV1(
        **domain_values,
        domain_ref="l.autotune-domain." + _digest(payload)[:24],
    )


def build_loop_autotune_domain_v1(
    graph: LoopInterventionCartesianGraphV1,
) -> LoopAutotuneDomainV1:
    """Return Original plus every admitted two-pass product candidate.

    The graph's rejected cells are intentionally discarded.  This control
    retains neither their identity nor their reasons.  Bridge-only arms are
    outside the user's Bridge-by-route Cartesian comparator.
    """
    if not isinstance(graph, LoopInterventionCartesianGraphV1):
        raise HBVSchemaError("Loop autotune requires a Cartesian graph")
    admitted = tuple(
        arm for arm in graph.arms
        if arm.route_ref in LOOP_COMPOSITION_ROUTES
        and arm.composition_legal)
    candidates = tuple(
        [_candidate(graph.arms[0])]
        + [_candidate(arm) for arm in admitted])
    values = {
        "schema": LOOP_AUTOTUNE_DOMAIN_SCHEMA_V1,
        "candidates": candidates,
        "prediction_pruning_permitted": False,
        "early_stopping_permitted": False,
    }
    payload = {
        **{key: value for key, value in values.items()
           if key != "candidates"},
        "candidates": [item.__dict__ for item in candidates],
    }
    return LoopAutotuneDomainV1(
        **values,
        domain_ref="l.autotune-domain." + _digest(payload)[:24],
    )


def build_loop_exhaustive_autotune_domain_v1(
    graph: LoopInterventionCartesianGraphV1,
) -> LoopAutotuneDomainV1:
    """Return Original plus every requested Bridge-by-route product cell.

    Unlike the production/admitted-domain constructor above, this comparator
    domain retains statically rejected cells.  Their bindings fail inside
    native autotune and receive ``inf`` timings.  This is the L-lite fairness
    contract: no requested Cartesian cell disappears before autotune pays for
    its validation attempt.
    """
    if not isinstance(graph, LoopInterventionCartesianGraphV1):
        raise HBVSchemaError(
            "Exhaustive Loop autotune requires a Cartesian graph")
    requested = tuple(
        arm for arm in graph.arms
        if arm.route_ref in LOOP_COMPOSITION_ROUTES)
    candidates = tuple(
        [_candidate(graph.arms[0])]
        + [_candidate(arm) for arm in requested])
    values = {
        "schema": LOOP_AUTOTUNE_DOMAIN_SCHEMA_V1,
        "candidates": candidates,
        "prediction_pruning_permitted": False,
        "early_stopping_permitted": False,
    }
    payload = {
        **{key: value for key, value in values.items()
           if key != "candidates"},
        "candidates": [item.__dict__ for item in candidates],
    }
    return LoopAutotuneDomainV1(
        **values,
        domain_ref="l.autotune-domain." + _digest(payload)[:24],
    )


def project_loop_autotune_domain_v1(
    domain: LoopAutotuneDomainV1,
    admitted_candidate_refs: Sequence[str],
) -> LoopAutotuneDomainV1:
    """Project a shared qualification result into the control domain.

    Qualification remains outside L-lite and may prove that a structurally
    plausible arm is not materializable for one concrete loop subject.  The
    control receives only the admitted identities: no rejected identity,
    reason, graph, model score, or timing label crosses this boundary.
    """
    if not isinstance(domain, LoopAutotuneDomainV1):
        raise HBVSchemaError("Loop autotune projection requires a domain")
    refs = tuple(admitted_candidate_refs)
    available_refs = {
        item.candidate_ref for item in domain.candidates}
    if (
        not refs
        or len(set(refs)) != len(refs)
        or not set(refs).issubset(available_refs)
    ):
        raise HBVSchemaError("Loop autotune projection refs are malformed")
    candidates = tuple(
        candidate for candidate in domain.candidates
        if candidate.candidate_ref in set(refs))
    if sum(candidate.original for candidate in candidates) != 1:
        raise HBVSchemaError(
            "Loop autotune projection must retain Original")
    values = {
        "schema": LOOP_AUTOTUNE_DOMAIN_SCHEMA_V1,
        "candidates": candidates,
        "prediction_pruning_permitted": False,
        "early_stopping_permitted": False,
    }
    payload = {
        **{key: value for key, value in values.items()
           if key != "candidates"},
        "candidates": [item.__dict__ for item in candidates],
    }
    return LoopAutotuneDomainV1(
        **values,
        domain_ref="l.autotune-domain." + _digest(payload)[:24],
    )


def _candidate_config(candidate_ref: str) -> Config:
    # None prevents the control from adding num_warps/num_stages/num_ctas to
    # the user's launch.  The only tuned meta-parameter is candidate identity.
    # Candidate refs also make Triton's disk cache sensitive to domain changes.
    return Config(
        {LOOP_AUTOTUNE_CANDIDATE_META_PARAMETER: candidate_ref},
        num_warps=None, num_stages=None, num_ctas=None,
    )


class _LoopAutotuneKernelMuxV1(KernelInterface):
    """Dispatch one native autotune config to one prebuilt kernel binding."""

    def __init__(self, kernel, bindings: Mapping[str, KernelInterface]):
        self.fn = kernel
        self._bindings = MappingProxyType(dict(bindings))
        self._last_candidate_ref = None

    def run(self, *args, grid, warmup, **kwargs):
        candidate_ref = kwargs.pop(
            LOOP_AUTOTUNE_CANDIDATE_META_PARAMETER, None)
        self._last_candidate_ref = candidate_ref
        binding = self._bindings.get(candidate_ref)
        if binding is None:
            raise HBVSchemaError(
                "Native Loop autotune candidate binding is absent")
        return binding.run(
            *args, grid=grid, warmup=warmup, **kwargs)

    @property
    def last_candidate_ref(self) -> str | None:
        return self._last_candidate_ref


class LoopNativeAutotuneControlV1(KernelInterface):
    """Thin facade over Triton's unmodified ``Autotuner`` selection path."""

    schema = LOOP_AUTOTUNE_NATIVE_CONTROL_SCHEMA_V1

    def __init__(
        self, *, kernel, domain: LoopAutotuneDomainV1,
        bindings: Mapping[str, KernelInterface], key: Sequence[str],
        reset_to_zero=None, restore_value=None, pre_hook=None,
        post_hook=None, do_bench=None, cache_results=False,
    ):
        candidate_refs = tuple(
            item.candidate_ref for item in domain.candidates)
        if (
            not hasattr(kernel, "arg_names")
            or not isinstance(key, (tuple, list))
            or not key
            or any(not isinstance(name, str) or not name for name in key)
            or set(bindings) != set(candidate_refs)
        ):
            raise HBVSchemaError(
                "Native Loop autotune construction is malformed")
        original_ref = next(
            item.candidate_ref for item in domain.candidates
            if item.original)
        if bindings[original_ref] is not kernel:
            raise HBVSchemaError(
                "Native Loop autotune Original must be the unbound kernel")
        if any(
            not callable(getattr(binding, "run", None))
            for binding in bindings.values()
        ):
            raise HBVSchemaError(
                "Native Loop autotune binding cannot run")
        self.fn = kernel
        self.domain = domain
        self._candidate_refs = candidate_refs
        self._mux = _LoopAutotuneKernelMuxV1(kernel, bindings)
        configs = [_candidate_config(ref) for ref in candidate_refs]
        self._config_by_ref = MappingProxyType(dict(zip(
            candidate_refs, configs)))
        self._autotuner = Autotuner(
            self._mux,
            kernel.arg_names,
            configs,
            list(key),
            reset_to_zero,
            restore_value,
            pre_hook=pre_hook,
            post_hook=post_hook,
            prune_configs_by=None,
            do_bench=do_bench,
            cache_results=cache_results,
        )
        # The comparator requires a fresh, fully charged run even when a
        # host application globally enables TRITON_CACHE_AUTOTUNING.  Triton's
        # constructor lets that environment knob override ``False``; restore
        # the explicit control contract without changing its benchmark or
        # winner algorithm.
        if not cache_results:
            self._autotuner.cache_results = False

    def run(self, *args, grid, warmup, **kwargs):
        return self._autotuner.run(
            *args, grid=grid, warmup=warmup, **kwargs)

    @property
    def native_autotuner(self) -> Autotuner:
        return self._autotuner

    @property
    def candidate_refs(self) -> Tuple[str, ...]:
        return self._candidate_refs

    @property
    def best_candidate_ref(self) -> str | None:
        config = getattr(self._autotuner, "best_config", None)
        if config is None:
            return None
        return config.kwargs.get(
            LOOP_AUTOTUNE_CANDIDATE_META_PARAMETER)

    @property
    def last_attempted_candidate_ref(self) -> str | None:
        return self._mux.last_candidate_ref

    @property
    def config_timings(self) -> Mapping[str, Tuple[float, ...]]:
        timings = getattr(self._autotuner, "configs_timings", None)
        if timings is None:
            return MappingProxyType({})
        by_ref = {
            config.kwargs[LOOP_AUTOTUNE_CANDIDATE_META_PARAMETER]:
            tuple(float(value) for value in values)
            for config, values in timings.items()
        }
        if set(by_ref) != set(self._candidate_refs):
            raise HBVSchemaError(
                "Native Loop autotune did not time its complete domain")
        return MappingProxyType(by_ref)

    @property
    def bench_time_seconds(self) -> float | None:
        value = getattr(self._autotuner, "bench_time", None)
        return None if value is None else float(value)


def build_loop_native_autotune_control_v1(
    *, kernel, domain: LoopAutotuneDomainV1,
    bindings: Mapping[str, KernelInterface], key: Sequence[str],
    reset_to_zero=None, restore_value=None, pre_hook=None,
    post_hook=None, do_bench=None, cache_results=False,
) -> LoopNativeAutotuneControlV1:
    """Bind the complete admitted domain to Triton's native autotuner.

    This function supplies no prune callback, performance model, early stop,
    custom winner algorithm, or custom result cache.
    """
    return LoopNativeAutotuneControlV1(
        kernel=kernel,
        domain=domain,
        bindings=bindings,
        key=key,
        reset_to_zero=reset_to_zero,
        restore_value=restore_value,
        pre_hook=pre_hook,
        post_hook=post_hook,
        do_bench=do_bench,
        cache_results=cache_results,
    )
