from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Tuple

from .contract import (
    LLiteSchemaError as HBVSchemaError,
    LOOP_BRIDGE_ROUTE_COMPOSITION_SCHEMA_V1,
    LOOP_BRIDGE_ROUTE_COMPOSITION_SCHEMA_V2,
    LoopBridgeRouteCompositionLegalityV1,
    LoopBridgeRouteCompositionLegalityV2,
    certify_loop_bridge_route_composition_v1,
    certify_loop_bridge_route_composition_v2,
)


LOOP_INTERVENTION_CARTESIAN_GRAPH_SCHEMA_V1 = (
    "hbv.loop.intervention-cartesian-graph.v1")
LOOP_INTERVENTION_CARTESIAN_GRAPH_SCHEMA_V2 = (
    "hbv.loop.intervention-cartesian-graph.v2")
LOOP_COMPOSITION_ROUTES = (
    "l.nvidia.software_pipeline.v1",
    "l.ttir.full_unroll_phase_major.v1",
    "l.ttir.full_unroll_logical_group.v1",
)
LOOP_BRIDGE_ONLY_ROUTE = "l.bridge-only.default-route"
LOOP_ORIGINAL_ROUTE = "l.original.default"
LOOP_BRIDGE_INTERVENTION_ATTESTATION_SCHEMA_V1 = (
    "hbv.loop.bridge-intervention-attestation.v1")
LOOP_ROUTE_INTERVENTION_ATTESTATION_SCHEMA_V1 = (
    "hbv.loop.route-intervention-attestation.v1")
LOOP_BRIDGE_ROUTE_ATTESTATION_SCHEMA_V1 = (
    "hbv.loop.bridge-route-attestation.v1")


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(
        value, allow_nan=False, separators=(",", ":"),
        sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class LoopInterventionCompositionArmV1:
    arm_ref: str
    bridge_factor: int
    route_ref: str
    route_factor: int
    bridge_action: str
    route_action: str
    bridge_constructed: bool
    route_applied: bool
    composition_legal: bool
    typed_reason: str
    proof_refs: Tuple[str, ...]

    def __post_init__(self):
        payload = {
            key: value for key, value in self.__dict__.items()
            if key != "arm_ref"
        }
        if (
            self.bridge_factor < 1
            or self.route_factor < 1
            or self.bridge_constructed != (self.bridge_factor > 1)
            or self.route_applied != (
                self.route_ref in LOOP_COMPOSITION_ROUTES)
            or self.composition_legal == bool(self.typed_reason)
            or not self.proof_refs
            or tuple(sorted(set(self.proof_refs))) != self.proof_refs
            or self.arm_ref != "l.composition-arm." + _digest(payload)[:24]
        ):
            raise HBVSchemaError(
                "Loop intervention composition arm is malformed")


@dataclass(frozen=True)
class LoopInterventionCartesianGraphV1:
    schema: str
    bridge_factors: Tuple[int, ...]
    route_factors: Tuple[Tuple[str, Tuple[int, ...]], ...]
    existing_loop_subject_available: bool
    arms: Tuple[LoopInterventionCompositionArmV1, ...]
    graph_ref: str

    def __post_init__(self):
        payload = {
            "schema": self.schema,
            "bridge_factors": self.bridge_factors,
            "route_factors": self.route_factors,
            "existing_loop_subject_available": (
                self.existing_loop_subject_available),
            "arms": [arm.__dict__ for arm in self.arms],
        }
        expected_count = (
            1 + max(0, len(self.bridge_factors) - 1)
            + len(self.bridge_factors) * sum(
                len(factors) for _, factors in self.route_factors))
        if (
            self.schema != LOOP_INTERVENTION_CARTESIAN_GRAPH_SCHEMA_V1
            or not self.bridge_factors
            or self.bridge_factors[0] != 1
            or tuple(sorted(set(self.bridge_factors)))
            != self.bridge_factors
            or tuple(route for route, _ in self.route_factors)
            != LOOP_COMPOSITION_ROUTES
            or any(not factors for _, factors in self.route_factors)
            or len(self.arms) != expected_count
            or len({arm.arm_ref for arm in self.arms}) != len(self.arms)
            or self.arms[0].route_ref != LOOP_ORIGINAL_ROUTE
            or self.graph_ref != "l.composition-graph." + _digest(payload)[:24]
        ):
            raise HBVSchemaError(
                "Loop intervention Cartesian graph is malformed")

    def arm(self, bridge_factor: int, route_ref: str,
            route_factor: int) -> LoopInterventionCompositionArmV1:
        matches = tuple(
            arm for arm in self.arms
            if (arm.bridge_factor, arm.route_ref, arm.route_factor)
            == (bridge_factor, route_ref, route_factor))
        if len(matches) != 1:
            raise HBVSchemaError(
                "Loop composition arm lookup is not unique")
        return matches[0]


def _arm(
    *, bridge_factor: int, route_ref: str, route_factor: int,
    bridge_action: str, route_action: str,
    legality: LoopBridgeRouteCompositionLegalityV1 | None = None,
    legal: bool | None = None, typed_reason: str = "",
    proof_refs: Tuple[str, ...] = (),
) -> LoopInterventionCompositionArmV1:
    if legality is not None:
        legal = legality.legal
        typed_reason = legality.typed_reason
        proof_refs = legality.proof_refs
    if legal is None:
        raise HBVSchemaError("Loop composition arm legality is absent")
    values = {
        "bridge_factor": bridge_factor,
        "route_ref": route_ref,
        "route_factor": route_factor,
        "bridge_action": bridge_action,
        "route_action": route_action,
        "bridge_constructed": bridge_factor > 1,
        "route_applied": route_ref in LOOP_COMPOSITION_ROUTES,
        "composition_legal": legal,
        "typed_reason": typed_reason,
        "proof_refs": tuple(sorted(set(proof_refs))),
    }
    return LoopInterventionCompositionArmV1(
        arm_ref="l.composition-arm." + _digest(values)[:24], **values)


def build_loop_intervention_cartesian_graph_v1(
    *,
    bridge_factors: Tuple[int, ...] = (1, 2, 4, 8),
    route_factors: Mapping[str, Tuple[int, ...]],
    existing_loop_subject_available: bool,
    static_partition_recurrence: bool = False,
) -> LoopInterventionCartesianGraphV1:
    """Enumerate all requested arms before capability/legal pruning.

    No timing, profitability, workload identity or backend descendant is
    accepted by this constructor.  Unsupported pairs remain typed arms.
    """
    bridge_factors = tuple(sorted(set(bridge_factors)))
    if not bridge_factors or bridge_factors[0] != 1:
        raise HBVSchemaError(
            "Loop composition graph must include unchanged Bridge factor one")
    ordered_route_factors = tuple(
        (route, tuple(sorted(set(route_factors[route]))))
        for route in LOOP_COMPOSITION_ROUTES)
    arms = [_arm(
        bridge_factor=1,
        route_ref=LOOP_ORIGINAL_ROUTE,
        route_factor=1,
        bridge_action="retain_input_ir",
        route_action="retain_original_route",
        legal=True,
        proof_refs=("proof:unchanged-ir-and-route-identity",),
    )]
    for factor in bridge_factors[1:]:
        arms.append(_arm(
            bridge_factor=factor,
            route_ref=LOOP_BRIDGE_ONLY_ROUTE,
            route_factor=1,
            bridge_action=f"construct_program_loop:factor={factor}",
            route_action="retain_default_route",
            legal=True,
            proof_refs=("proof:bridge-legality-owned-before-hbv-route",),
        ))
    for bridge_factor in bridge_factors:
        for route_ref, factors in ordered_route_factors:
            for route_factor in factors:
                legality = certify_loop_bridge_route_composition_v1(
                    bridge_factor=bridge_factor,
                    route_ref=route_ref,
                    route_factor=route_factor,
                    route_subject_available=(
                        bridge_factor > 1
                        or existing_loop_subject_available),
                    static_partition_recurrence=(
                        static_partition_recurrence
                        and bridge_factor > 1
                        and route_ref
                        == "l.ttir.full_unroll_phase_major.v1"),
                )
                arms.append(_arm(
                    bridge_factor=bridge_factor,
                    route_ref=route_ref,
                    route_factor=route_factor,
                    bridge_action=(
                        f"construct_program_loop:factor={bridge_factor}"
                        if bridge_factor > 1 else "retain_input_ir"),
                    route_action=(
                        f"apply_route:{route_ref}:factor={route_factor}"),
                    legality=legality,
                ))
    values = {
        "schema": LOOP_INTERVENTION_CARTESIAN_GRAPH_SCHEMA_V1,
        "bridge_factors": bridge_factors,
        "route_factors": ordered_route_factors,
        "existing_loop_subject_available": existing_loop_subject_available,
        "arms": tuple(arms),
    }
    payload = {
        **{key: value for key, value in values.items() if key != "arms"},
        "arms": [arm.__dict__ for arm in arms],
    }
    return LoopInterventionCartesianGraphV1(
        **values,
        graph_ref="l.composition-graph." + _digest(payload)[:24],
    )


@dataclass(frozen=True)
class LoopInterventionCompositionArmV2:
    arm_ref: str
    bridge_factor: int
    route_ref: str
    route_factor: int
    bridge_action: str
    route_action: str
    bridge_constructed: bool
    route_applied: bool
    route_subject_ref: str
    factor_admission_ref: str
    composition_legal: bool
    typed_reason: str
    proof_refs: Tuple[str, ...]

    def __post_init__(self):
        payload = {
            key: value for key, value in self.__dict__.items()
            if key != "arm_ref"
        }
        if (
            self.bridge_factor < 1
            or self.route_factor < 1
            or self.bridge_constructed != (self.bridge_factor > 1)
            or self.route_applied != (
                self.route_ref in LOOP_COMPOSITION_ROUTES)
            or not self.route_subject_ref
            or not self.factor_admission_ref
            or self.composition_legal == bool(self.typed_reason)
            or not self.proof_refs
            or tuple(sorted(set(self.proof_refs))) != self.proof_refs
            or self.arm_ref
            != "l.composition-arm.v2." + _digest(payload)[:24]
        ):
            raise HBVSchemaError(
                "Loop V2 intervention composition arm is malformed")


@dataclass(frozen=True)
class LoopInterventionCartesianGraphV2:
    schema: str
    bridge_factors: Tuple[int, ...]
    route_factors: Tuple[Tuple[str, Tuple[int, ...]], ...]
    existing_loop_subject_available: bool
    existing_loop_exact_trip_count: int
    runtime_main_tail_certificate_ref: str
    arms: Tuple[LoopInterventionCompositionArmV2, ...]
    graph_ref: str

    def __post_init__(self):
        payload = {
            "schema": self.schema,
            "bridge_factors": self.bridge_factors,
            "route_factors": self.route_factors,
            "existing_loop_subject_available": (
                self.existing_loop_subject_available),
            "existing_loop_exact_trip_count": (
                self.existing_loop_exact_trip_count),
            "runtime_main_tail_certificate_ref": (
                self.runtime_main_tail_certificate_ref),
            "arms": [arm.__dict__ for arm in self.arms],
        }
        expected_count = (
            1 + max(0, len(self.bridge_factors) - 1)
            + len(self.bridge_factors) * sum(
                len(factors) for _, factors in self.route_factors))
        if (
            self.schema != LOOP_INTERVENTION_CARTESIAN_GRAPH_SCHEMA_V2
            or not self.bridge_factors
            or self.bridge_factors[0] != 1
            or tuple(sorted(set(self.bridge_factors)))
            != self.bridge_factors
            or tuple(route for route, _ in self.route_factors)
            != LOOP_COMPOSITION_ROUTES
            or any(not factors for _, factors in self.route_factors)
            or not isinstance(self.existing_loop_exact_trip_count, int)
            or isinstance(self.existing_loop_exact_trip_count, bool)
            or self.existing_loop_exact_trip_count < 0
            or self.existing_loop_exact_trip_count
            and not self.existing_loop_subject_available
            or self.runtime_main_tail_certificate_ref
            and (
                not self.existing_loop_subject_available
                or self.existing_loop_exact_trip_count)
            or len(self.arms) != expected_count
            or len({arm.arm_ref for arm in self.arms}) != len(self.arms)
            or self.arms[0].route_ref != LOOP_ORIGINAL_ROUTE
            or self.graph_ref
            != "l.composition-graph.v2." + _digest(payload)[:24]
        ):
            raise HBVSchemaError(
                "Loop intervention Cartesian graph V2 is malformed")

    def arm(self, bridge_factor: int, route_ref: str,
            route_factor: int) -> LoopInterventionCompositionArmV2:
        matches = tuple(
            arm for arm in self.arms
            if (arm.bridge_factor, arm.route_ref, arm.route_factor)
            == (bridge_factor, route_ref, route_factor))
        if len(matches) != 1:
            raise HBVSchemaError(
                "Loop V2 composition arm lookup is not unique")
        return matches[0]


def _arm_v2(
    *, bridge_factor: int, route_ref: str, route_factor: int,
    bridge_action: str, route_action: str,
    route_subject_ref: str, factor_admission_ref: str,
    legality: LoopBridgeRouteCompositionLegalityV2 | None = None,
    legal: bool | None = None, typed_reason: str = "",
    proof_refs: Tuple[str, ...] = (),
) -> LoopInterventionCompositionArmV2:
    if legality is not None:
        legal = legality.legal
        typed_reason = legality.typed_reason
        proof_refs = legality.proof_refs
        route_subject_ref = legality.route_subject_ref
        factor_admission_ref = legality.factor_admission_ref
    if legal is None:
        raise HBVSchemaError("Loop V2 composition arm legality is absent")
    values = {
        "bridge_factor": bridge_factor,
        "route_ref": route_ref,
        "route_factor": route_factor,
        "bridge_action": bridge_action,
        "route_action": route_action,
        "bridge_constructed": bridge_factor > 1,
        "route_applied": route_ref in LOOP_COMPOSITION_ROUTES,
        "route_subject_ref": route_subject_ref,
        "factor_admission_ref": factor_admission_ref,
        "composition_legal": legal,
        "typed_reason": typed_reason,
        "proof_refs": tuple(sorted(set(proof_refs))),
    }
    return LoopInterventionCompositionArmV2(
        arm_ref="l.composition-arm.v2." + _digest(values)[:24], **values)


def build_loop_intervention_cartesian_graph_v2(
    *,
    bridge_factors: Tuple[int, ...] = (1, 2, 4, 8),
    route_factors: Mapping[str, Tuple[int, ...]],
    existing_loop_subject_available: bool,
    existing_loop_exact_trip_count: int = 0,
    runtime_main_tail_certificate_ref: str = "",
    static_partition_recurrence: bool = False,
) -> LoopInterventionCartesianGraphV2:
    """Enumerate every intervention before materialization or timing."""
    bridge_factors = tuple(sorted(set(bridge_factors)))
    if not bridge_factors or bridge_factors[0] != 1:
        raise HBVSchemaError(
            "Loop V2 composition graph must include Bridge factor one")
    ordered_route_factors = tuple(
        (route, tuple(sorted(set(route_factors[route]))))
        for route in LOOP_COMPOSITION_ROUTES)
    arms = [_arm_v2(
        bridge_factor=1,
        route_ref=LOOP_ORIGINAL_ROUTE,
        route_factor=1,
        bridge_action="retain_input_ir",
        route_action="retain_original_route",
        route_subject_ref="l.route-subject.not-applicable.original",
        factor_admission_ref="l.factor-admission.not-applicable.original",
        legal=True,
        proof_refs=("proof:unchanged-ir-and-route-identity",),
    )]
    for factor in bridge_factors[1:]:
        arms.append(_arm_v2(
            bridge_factor=factor,
            route_ref=LOOP_BRIDGE_ONLY_ROUTE,
            route_factor=1,
            bridge_action=f"construct_program_loop:factor={factor}",
            route_action="retain_default_route",
            route_subject_ref=f"l.route-subject.bridge-only.factor-{factor}",
            factor_admission_ref=(
                "l.factor-admission.not-applicable.bridge-only"),
            legal=True,
            proof_refs=("proof:bridge-legality-owned-before-hbv-route",),
        ))
    for bridge_factor in bridge_factors:
        for route_ref, factors in ordered_route_factors:
            for route_factor in factors:
                legality = certify_loop_bridge_route_composition_v2(
                    bridge_factor=bridge_factor,
                    route_ref=route_ref,
                    route_factor=route_factor,
                    existing_loop_subject_available=(
                        existing_loop_subject_available),
                    existing_loop_exact_trip_count=(
                        existing_loop_exact_trip_count),
                    runtime_main_tail_certificate_ref=(
                        runtime_main_tail_certificate_ref),
                    static_partition_recurrence=(
                        static_partition_recurrence
                        and bridge_factor > 1
                        and route_ref
                        == "l.ttir.full_unroll_phase_major.v1"),
                )
                arms.append(_arm_v2(
                    bridge_factor=bridge_factor,
                    route_ref=route_ref,
                    route_factor=route_factor,
                    bridge_action=(
                        f"construct_program_loop:factor={bridge_factor}"
                        if bridge_factor > 1 else "retain_input_ir"),
                    route_action=(
                        f"apply_route:{route_ref}:factor={route_factor}"),
                    route_subject_ref=legality.route_subject_ref,
                    factor_admission_ref=legality.factor_admission_ref,
                    legality=legality,
                ))
    values = {
        "schema": LOOP_INTERVENTION_CARTESIAN_GRAPH_SCHEMA_V2,
        "bridge_factors": bridge_factors,
        "route_factors": ordered_route_factors,
        "existing_loop_subject_available": existing_loop_subject_available,
        "existing_loop_exact_trip_count": existing_loop_exact_trip_count,
        "runtime_main_tail_certificate_ref": (
            runtime_main_tail_certificate_ref),
        "arms": tuple(arms),
    }
    payload = {
        **{key: value for key, value in values.items() if key != "arms"},
        "arms": [arm.__dict__ for arm in arms],
    }
    return LoopInterventionCartesianGraphV2(
        **values,
        graph_ref="l.composition-graph.v2." + _digest(payload)[:24],
    )


@dataclass(frozen=True)
class LoopBridgeInterventionAttestationV1:
    schema: str
    composition_arm_ref: str
    bridge_factor: int
    bridge_constructed: bool
    requested_factor_lineage_count: int
    realized_origin_count: int
    realized_dependence_count: int
    passed: bool
    typed_reason: str
    proof_refs: Tuple[str, ...]
    attestation_ref: str

    def __post_init__(self):
        payload = {
            key: value for key, value in self.__dict__.items()
            if key != "attestation_ref"
        }
        if (
            self.schema != LOOP_BRIDGE_INTERVENTION_ATTESTATION_SCHEMA_V1
            or not self.composition_arm_ref.startswith("l.composition-arm.")
            or self.bridge_factor < 1
            or self.bridge_constructed != (self.bridge_factor > 1)
            or min(
                self.requested_factor_lineage_count,
                self.realized_origin_count,
                self.realized_dependence_count) < 0
            or self.passed == bool(self.typed_reason)
            or not self.proof_refs
            or tuple(sorted(set(self.proof_refs))) != self.proof_refs
            or self.attestation_ref != (
                "l.bridge-attestation." + _digest(payload)[:24])
        ):
            raise HBVSchemaError(
                "Loop Bridge intervention attestation is malformed")


@dataclass(frozen=True)
class LoopRouteInterventionAttestationV1:
    schema: str
    composition_arm_ref: str
    route_ref: str
    route_factor: int
    requested_factor_lineage_count: int
    realized_route_count: int
    realized_postcondition_count: int
    passed: bool
    typed_reason: str
    proof_refs: Tuple[str, ...]
    attestation_ref: str

    def __post_init__(self):
        payload = {
            key: value for key, value in self.__dict__.items()
            if key != "attestation_ref"
        }
        if (
            self.schema != LOOP_ROUTE_INTERVENTION_ATTESTATION_SCHEMA_V1
            or not self.composition_arm_ref.startswith("l.composition-arm.")
            or self.route_ref not in LOOP_COMPOSITION_ROUTES
            or self.route_factor < 1
            or min(
                self.requested_factor_lineage_count,
                self.realized_route_count,
                self.realized_postcondition_count) < 0
            or self.passed == bool(self.typed_reason)
            or not self.proof_refs
            or tuple(sorted(set(self.proof_refs))) != self.proof_refs
            or self.attestation_ref != (
                "l.route-attestation." + _digest(payload)[:24])
        ):
            raise HBVSchemaError(
                "Loop route intervention attestation is malformed")


@dataclass(frozen=True)
class LoopBridgeRouteCompositionAttestationV1:
    schema: str
    composition_arm_ref: str
    intervention_order: Tuple[str, str]
    bridge_attestation_ref: str
    route_attestation_ref: str
    passed: bool
    typed_reason: str
    proof_refs: Tuple[str, ...]
    attestation_ref: str

    def __post_init__(self):
        payload = {
            key: value for key, value in self.__dict__.items()
            if key != "attestation_ref"
        }
        if (
            self.schema != LOOP_BRIDGE_ROUTE_ATTESTATION_SCHEMA_V1
            or not self.composition_arm_ref.startswith("l.composition-arm.")
            or self.intervention_order != ("bridge", "route")
            or not self.bridge_attestation_ref.startswith(
                "l.bridge-attestation.")
            or not self.route_attestation_ref.startswith(
                "l.route-attestation.")
            or self.passed == bool(self.typed_reason)
            or not self.proof_refs
            or tuple(sorted(set(self.proof_refs))) != self.proof_refs
            or self.attestation_ref != (
                "l.composition-attestation." + _digest(payload)[:24])
        ):
            raise HBVSchemaError(
                "Loop Bridge then route attestation is malformed")


def attest_loop_bridge_route_materialization_v1(
    *, composition_arm_ref: str, bridge_factor: int, route_ref: str,
    route_factor: int, signatures: Mapping[str, object],
) -> Tuple[
    LoopBridgeInterventionAttestationV1,
    LoopRouteInterventionAttestationV1,
    LoopBridgeRouteCompositionAttestationV1,
]:
    """Close the two intervention postconditions without timing inputs."""
    if route_ref not in LOOP_COMPOSITION_ROUTES:
        raise HBVSchemaError("Loop route attestation route is unknown")
    if (
        not isinstance(bridge_factor, int)
        or isinstance(bridge_factor, bool)
        or bridge_factor < 1
        or bridge_factor & (bridge_factor - 1)
        or not isinstance(route_factor, int)
        or isinstance(route_factor, bool)
        or route_factor < 1
    ):
        raise HBVSchemaError(
            "Loop materialization attestation factors are malformed")

    def count(key: str) -> int:
        value = signatures.get(key, 0)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise HBVSchemaError(
                "Loop materialization signature count is malformed")
        return value

    bridge_constructed = bridge_factor > 1
    bridge_request = count("ttir_composition_bridge_factor_count")
    bridge_origin = (
        count("ttir_bridge_origin_count")
        + count("ttir_bridge_multiaxis_origin_count")
        + count("ttir_bridge_static_origin_count"))
    bridge_dependence = (
        count("ttir_bridge_dependence_count")
        + count("ttir_bridge_multiaxis_dependence_count"))
    bridge_passed = (
        bridge_request == 1
        and ((bridge_origin == 1 and bridge_dependence == 1)
             if bridge_constructed
             else (bridge_origin == 0 and bridge_dependence == 0)))
    bridge_values = {
        "schema": LOOP_BRIDGE_INTERVENTION_ATTESTATION_SCHEMA_V1,
        "composition_arm_ref": composition_arm_ref,
        "bridge_factor": bridge_factor,
        "bridge_constructed": bridge_constructed,
        "requested_factor_lineage_count": bridge_request,
        "realized_origin_count": bridge_origin,
        "realized_dependence_count": bridge_dependence,
        "passed": bridge_passed,
        "typed_reason": (
            "" if bridge_passed
            else "bridge_intervention_lineage_or_postcondition_failed"),
        "proof_refs": tuple(sorted((
            "proof:bridge-factor-request-lineage",
            "proof:bridge-origin-and-independence-postcondition",
        ))),
    }
    bridge = LoopBridgeInterventionAttestationV1(
        **bridge_values,
        attestation_ref=(
            "l.bridge-attestation." + _digest(bridge_values)[:24]))

    route_request = count("ttir_composition_route_factor_count")
    realized_route = count("ttir_expected_realized_count")
    route_postcondition = count("ttir_composition_postcondition_count")
    route_passed = (
        route_request == 1
        and realized_route == 1
        and route_postcondition == 1)
    route_values = {
        "schema": LOOP_ROUTE_INTERVENTION_ATTESTATION_SCHEMA_V1,
        "composition_arm_ref": composition_arm_ref,
        "route_ref": route_ref,
        "route_factor": route_factor,
        "requested_factor_lineage_count": route_request,
        "realized_route_count": realized_route,
        "realized_postcondition_count": route_postcondition,
        "passed": route_passed,
        "typed_reason": (
            "" if route_passed
            else "route_intervention_lineage_or_postcondition_failed"),
        "proof_refs": tuple(sorted((
            "proof:route-factor-request-lineage",
            "proof:route-realization-and-postcondition",
        ))),
    }
    route = LoopRouteInterventionAttestationV1(
        **route_values,
        attestation_ref=(
            "l.route-attestation." + _digest(route_values)[:24]))

    composed_passed = (
        bridge.passed and route.passed
        and count("ttir_composition_schema_count") == 1)
    composition_values = {
        "schema": LOOP_BRIDGE_ROUTE_ATTESTATION_SCHEMA_V1,
        "composition_arm_ref": composition_arm_ref,
        "intervention_order": ("bridge", "route"),
        "bridge_attestation_ref": bridge.attestation_ref,
        "route_attestation_ref": route.attestation_ref,
        "passed": composed_passed,
        "typed_reason": (
            "" if composed_passed
            else "ordered_bridge_route_composition_attestation_failed"),
        "proof_refs": tuple(sorted((
            "proof:bridge-attestation-precedes-route-attestation",
            "proof:composition-schema-lineage",
        ))),
    }
    composition = LoopBridgeRouteCompositionAttestationV1(
        **composition_values,
        attestation_ref=(
            "l.composition-attestation."
            + _digest(composition_values)[:24]))
    return bridge, route, composition
