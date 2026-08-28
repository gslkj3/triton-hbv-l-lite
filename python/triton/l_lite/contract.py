"""Typed legality contracts shared by the L-lite control and materializers.

This module decides only whether two sequential interventions have a
well-formed, currently supported subject.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


class LLiteContractError(ValueError):
    """Base error for malformed or unsupported L-lite contracts."""


class LLiteSchemaError(LLiteContractError):
    """A serialized or in-memory contract violates its schema."""


class LLiteUnsupportedContractError(LLiteContractError):
    """The request is valid in principle but unsupported by this snapshot."""


LOOP_BRIDGE_ROUTE_COMPOSITION_SCHEMA_V1 = (
    "l-lite.loop.bridge-route-composition.v1")
LOOP_BRIDGE_ROUTE_COMPOSITION_SCHEMA_V2 = (
    "l-lite.loop.bridge-route-composition.v2")

_ROUTES = {
    "l.nvidia.software_pipeline.v1",
    "l.ttir.full_unroll_phase_major.v1",
    "l.ttir.full_unroll_logical_group.v1",
}


def _validate_bridge_factor(value: int, path: str) -> int:
    if (not isinstance(value, int) or isinstance(value, bool) or value < 1
            or value & (value - 1)):
        raise LLiteUnsupportedContractError(
            f"{path} must be a positive power of two")
    return value


def _validate_route_factor(value: int, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise LLiteUnsupportedContractError(
            f"{path} must be a positive integer")
    return value


@dataclass(frozen=True)
class LoopBridgeRouteCompositionLegalityV1:
    bridge_factor: int
    route_ref: str
    route_factor: int
    bridge_constructed: bool
    route_subject_available: bool
    legal: bool
    typed_reason: str
    proof_refs: Tuple[str, ...]
    schema: str = LOOP_BRIDGE_ROUTE_COMPOSITION_SCHEMA_V1

    def __post_init__(self):
        _validate_bridge_factor(self.bridge_factor, "bridge_factor")
        _validate_route_factor(self.route_factor, "route_factor")
        if (
            self.schema != LOOP_BRIDGE_ROUTE_COMPOSITION_SCHEMA_V1
            or self.route_ref not in _ROUTES
            or self.bridge_constructed != (self.bridge_factor > 1)
            or self.legal == bool(self.typed_reason)
            or not self.proof_refs
            or tuple(sorted(set(self.proof_refs))) != self.proof_refs
        ):
            raise LLiteSchemaError(
                "Loop Bridge/route composition legality is malformed")


def certify_loop_bridge_route_composition_v1(
    *,
    bridge_factor: int,
    route_ref: str,
    route_factor: int,
    route_subject_available: bool,
    static_partition_recurrence: bool = False,
) -> LoopBridgeRouteCompositionLegalityV1:
    """Certify the materializable subset without using timing information."""
    _validate_bridge_factor(bridge_factor, "bridge_factor")
    _validate_route_factor(route_factor, "route_factor")
    if route_ref not in _ROUTES:
        raise LLiteUnsupportedContractError(
            f"Loop composition route {route_ref!r} is not implemented")
    constructed = bridge_factor > 1
    reason = ""
    proofs = [
        "proof:bridge-and-route-are-sequential-independent-interventions",
        "proof:route-factor-retains-route-local-meaning",
    ]
    if not constructed and not route_subject_available:
        reason = "route_subject_absent_after_unchanged_bridge"
    elif static_partition_recurrence and (
            not constructed
            or route_ref != "l.ttir.full_unroll_phase_major.v1"):
        reason = "static_partition_requires_constructed_phase_subject"
    elif route_ref in {
            "l.ttir.full_unroll_phase_major.v1",
            "l.ttir.full_unroll_logical_group.v1"}:
        if constructed and route_factor != bridge_factor:
            reason = "current_full_unroll_requires_route_factor_equal_trip_count"
        else:
            proofs.append("proof:current-full-unroll-materializer-capability")
    else:
        proofs.append(
            "proof:pipeline-stage-factor-independent-of-bridge-factor")
    return LoopBridgeRouteCompositionLegalityV1(
        bridge_factor=bridge_factor,
        route_ref=route_ref,
        route_factor=route_factor,
        bridge_constructed=constructed,
        route_subject_available=route_subject_available,
        legal=not reason,
        typed_reason=reason,
        proof_refs=tuple(sorted(proofs)),
    )


@dataclass(frozen=True)
class LoopBridgeRouteCompositionLegalityV2:
    bridge_factor: int
    route_ref: str
    route_factor: int
    bridge_constructed: bool
    route_subject_available: bool
    route_subject_ref: str
    route_subject_exact_trip_count: Optional[int]
    runtime_main_tail_certificate_ref: str
    route_factor_kind: str
    factor_admission_ref: str
    legal: bool
    typed_reason: str
    proof_refs: Tuple[str, ...]
    schema: str = LOOP_BRIDGE_ROUTE_COMPOSITION_SCHEMA_V2

    def __post_init__(self):
        _validate_bridge_factor(self.bridge_factor, "bridge_factor")
        if (
            self.schema != LOOP_BRIDGE_ROUTE_COMPOSITION_SCHEMA_V2
            or self.route_ref not in _ROUTES
            or not isinstance(self.route_factor, int)
            or isinstance(self.route_factor, bool)
            or self.route_factor < 1
            or self.bridge_constructed != (self.bridge_factor > 1)
            or not self.route_subject_ref
            or (not self.route_subject_available and (
                self.route_subject_exact_trip_count is not None
                or bool(self.runtime_main_tail_certificate_ref)))
            or (self.route_subject_exact_trip_count is not None and (
                not isinstance(self.route_subject_exact_trip_count, int)
                or isinstance(self.route_subject_exact_trip_count, bool)
                or self.route_subject_exact_trip_count < 1))
            or (self.route_subject_exact_trip_count is not None
                and self.runtime_main_tail_certificate_ref)
            or (self.route_subject_available
                and self.route_subject_exact_trip_count is None
                and not self.runtime_main_tail_certificate_ref)
            or not self.route_factor_kind
            or not self.factor_admission_ref
            or self.legal == bool(self.typed_reason)
            or not self.proof_refs
            or tuple(sorted(set(self.proof_refs))) != self.proof_refs
        ):
            raise LLiteSchemaError(
                "Loop Bridge/route V2 composition legality is malformed")


def certify_loop_bridge_route_composition_v2(
    *,
    bridge_factor: int,
    route_ref: str,
    route_factor: int,
    existing_loop_subject_available: bool,
    existing_loop_exact_trip_count: int = 0,
    runtime_main_tail_certificate_ref: str = "",
    static_partition_recurrence: bool = False,
) -> LoopBridgeRouteCompositionLegalityV2:
    """Certify route-factor meaning after Bridge establishes the subject."""
    from .factor_ontology import (
        LoopRouteSubjectV1,
        decide_loop_route_factor_admission_v1,
        route_subject_after_bridge_v1,
    )

    _validate_bridge_factor(bridge_factor, "bridge_factor")
    if route_ref not in _ROUTES:
        raise LLiteUnsupportedContractError(
            f"Loop composition route {route_ref!r} is not implemented")
    if (not isinstance(existing_loop_exact_trip_count, int)
            or isinstance(existing_loop_exact_trip_count, bool)
            or existing_loop_exact_trip_count < 0):
        raise LLiteSchemaError("existing loop exact trip count is malformed")
    if existing_loop_exact_trip_count and not existing_loop_subject_available:
        raise LLiteSchemaError(
            "exact existing loop trip requires an available subject")
    if runtime_main_tail_certificate_ref and (
            not existing_loop_subject_available
            or existing_loop_exact_trip_count):
        raise LLiteSchemaError(
            "runtime main-tail certificate has no runtime existing subject")

    if not existing_loop_subject_available:
        existing = LoopRouteSubjectV1.absent()
    elif existing_loop_exact_trip_count:
        existing = LoopRouteSubjectV1.exact(
            origin="existing_loop", trip_count=existing_loop_exact_trip_count)
    else:
        existing = LoopRouteSubjectV1.runtime(
            certificate_ref=runtime_main_tail_certificate_ref)
    subject = route_subject_after_bridge_v1(
        bridge_factor=bridge_factor, existing_subject=existing)
    admission = decide_loop_route_factor_admission_v1(
        subject, route_ref=route_ref, route_factor=route_factor)

    reason = admission.typed_reason
    proofs = set(admission.proof_refs)
    proofs.add("proof:Bridge-then-route-sequential-composition-v2")
    if static_partition_recurrence and (
            bridge_factor == 1
            or route_ref != "l.ttir.full_unroll_phase_major.v1"):
        reason = "static_partition_requires_constructed_phase_subject"
    elif not reason:
        proofs.add("proof:semantic-factor-admission-precedes-materialization")
        if route_ref == "l.nvidia.software_pipeline.v1":
            proofs.add("proof:pipeline-stage-factor-independent-of-Bridge")
        else:
            proofs.add("proof:full-route-grouping-does-not-own-source-trip")

    return LoopBridgeRouteCompositionLegalityV2(
        bridge_factor=bridge_factor,
        route_ref=route_ref,
        route_factor=route_factor,
        bridge_constructed=bridge_factor > 1,
        route_subject_available=subject.origin != "absent",
        route_subject_ref=subject.subject_ref,
        route_subject_exact_trip_count=subject.exact_trip_count,
        runtime_main_tail_certificate_ref=(
            subject.runtime_main_tail_certificate_ref),
        route_factor_kind=admission.factor_kind,
        factor_admission_ref=admission.admission_ref,
        legal=not reason,
        typed_reason=reason,
        proof_refs=tuple(sorted(proofs)),
    )
