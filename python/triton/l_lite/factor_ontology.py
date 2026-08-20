"""Causal ontology for two sequential loop-optimization factors.

This module owns meanings only.  It deliberately does not decide target
capability, profitability or backend response.  Bridge constructs (or keeps)
the route subject; the route factor then acts on that subject with a
route-specific meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Optional, Tuple

from .contract import LLiteSchemaError as HBVSchemaError


LOOP_FACTOR_ONTOLOGY_SCHEMA_V1 = "hbv.loop-factor-ontology.v1"
LOOP_ROUTE_SUBJECT_SCHEMA_V1 = "hbv.loop-route-subject.v1"
LOOP_ROUTE_FACTOR_ADMISSION_SCHEMA_V1 = (
    "hbv.loop-route-factor-admission.v1")
LOOP_NEST_ROUTE_SUBJECT_SCHEMA_V1 = "hbv.loop-nest-route-subject.v1"
LOOP_NEST_STRUCTURAL_SCOPE_SCHEMA_V1 = (
    "hbv.loop-nest-structural-scope.v1")
LOOP_NEST_ROUTE_ADMISSION_SCHEMA_V1 = (
    "hbv.loop-nest-route-admission.v1")

SOFTWARE_PIPELINE_ROUTE = "l.nvidia.software_pipeline.v1"
FULL_UNROLL_REORDER_ROUTE = "l.ttir.full_unroll_phase_major.v1"
FULL_UNROLL_VECTOR_ROUTE = "l.ttir.full_unroll_logical_group.v1"
EXACT_PREFIX_ARTIFACT_ROUTE = (
    "l.ttir.predicated_exact_prefix_reduction.v1")
LOGICAL_GROUPED_LOAD_SUBTYPE = "grouped_load_vectorization"
LOGICAL_EXACT_PREFIX_SUBTYPE = "predicated_exact_prefix_vectorization"
ROUTES = (
    SOFTWARE_PIPELINE_ROUTE,
    FULL_UNROLL_REORDER_ROUTE,
    FULL_UNROLL_VECTOR_ROUTE,
)


def _ref(prefix: str, payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()
    return prefix + sha256(encoded).hexdigest()[:24]


def _power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


@dataclass(frozen=True)
class LoopRouteFactorMeaningV1:
    route_ref: str
    factor_kind: str
    source_trip_owner: str
    subject_terminal_disposition: str
    factor_controls_complete_trip_count: bool
    schema: str = LOOP_FACTOR_ONTOLOGY_SCHEMA_V1

    def __post_init__(self):
        expected = {
            SOFTWARE_PIPELINE_ROUTE: (
                "pipeline_stage_count", "retained_live_focal_loop"),
            FULL_UNROLL_REORDER_ROUTE: (
                "phase_reorder_grouping_width",
                "factor_dependent_focal_loop_disposition"),
            FULL_UNROLL_VECTOR_ROUTE: (
                "logical_vector_grouping_width",
                "factor_dependent_focal_loop_disposition"),
        }
        if (
            self.schema != LOOP_FACTOR_ONTOLOGY_SCHEMA_V1
            or self.route_ref not in expected
            or (self.factor_kind, self.subject_terminal_disposition)
            != expected[self.route_ref]
            or self.source_trip_owner != "Bridge_or_existing_source_subject"
            or self.factor_controls_complete_trip_count is not False
        ):
            raise HBVSchemaError("loop route factor meaning is malformed")


ROUTE_FACTOR_MEANINGS_V1 = {
    route_ref: LoopRouteFactorMeaningV1(
        route_ref=route_ref,
        factor_kind=factor_kind,
        source_trip_owner="Bridge_or_existing_source_subject",
        subject_terminal_disposition=disposition,
        factor_controls_complete_trip_count=False,
    )
    for route_ref, factor_kind, disposition in (
        (SOFTWARE_PIPELINE_ROUTE, "pipeline_stage_count",
         "retained_live_focal_loop"),
        (FULL_UNROLL_REORDER_ROUTE, "phase_reorder_grouping_width",
         "factor_dependent_focal_loop_disposition"),
        (FULL_UNROLL_VECTOR_ROUTE, "logical_vector_grouping_width",
         "factor_dependent_focal_loop_disposition"),
    )
}


@dataclass(frozen=True)
class LoopLogicalVectorSubtypeMeaningV1:
    """Typed input/materializer variants under one logical-vector cause.

    A subtype is not a fourth route.  It prevents the exact-prefix adapter's
    semantic factor one from being mistaken for a one-wide grouped-load
    unroll while preserving one causal mechanism identity downstream.
    """

    subtype: str
    mechanism_route_ref: str
    factor_kind: str
    source_subject_kind: str
    artifact_route_ref: str
    schema: str = LOOP_FACTOR_ONTOLOGY_SCHEMA_V1

    def __post_init__(self):
        expected = {
            LOGICAL_GROUPED_LOAD_SUBTYPE: (
                "logical_vector_grouping_width", "scf_for",
                FULL_UNROLL_VECTOR_ROUTE),
            LOGICAL_EXACT_PREFIX_SUBTYPE: (
                "exact_prefix_vectorization_adapter",
                "provider_closed_dynamic_integer_prefix_recurrence",
                EXACT_PREFIX_ARTIFACT_ROUTE),
        }
        if (
            self.schema != LOOP_FACTOR_ONTOLOGY_SCHEMA_V1
            or self.subtype not in expected
            or self.mechanism_route_ref != FULL_UNROLL_VECTOR_ROUTE
            or (self.factor_kind, self.source_subject_kind,
                self.artifact_route_ref) != expected[self.subtype]
        ):
            raise HBVSchemaError(
                "logical-vector subtype meaning is malformed")


LOGICAL_VECTOR_SUBTYPE_MEANINGS_V1 = {
    subtype: LoopLogicalVectorSubtypeMeaningV1(
        subtype=subtype,
        mechanism_route_ref=FULL_UNROLL_VECTOR_ROUTE,
        factor_kind=factor_kind,
        source_subject_kind=source_kind,
        artifact_route_ref=artifact_route,
    )
    for subtype, factor_kind, source_kind, artifact_route in (
        (LOGICAL_GROUPED_LOAD_SUBTYPE,
         "logical_vector_grouping_width", "scf_for",
         FULL_UNROLL_VECTOR_ROUTE),
        (LOGICAL_EXACT_PREFIX_SUBTYPE,
         "exact_prefix_vectorization_adapter",
         "provider_closed_dynamic_integer_prefix_recurrence",
         EXACT_PREFIX_ARTIFACT_ROUTE),
    )
}


@dataclass(frozen=True)
class LoopExactPrefixSubjectV1:
    active_extent: int
    reduction_container_width: int
    element_bytes: int
    exact_reduction_certificate: str
    source_fact_refs: Tuple[str, ...]
    subject_ref: str
    schema: str = LOOP_ROUTE_SUBJECT_SCHEMA_V1

    def __post_init__(self):
        payload = {
            key: value for key, value in self.__dict__.items()
            if key != "subject_ref"
        }
        if (
            self.schema != LOOP_ROUTE_SUBJECT_SCHEMA_V1
            or not isinstance(self.active_extent, int)
            or isinstance(self.active_extent, bool)
            or self.active_extent < 1
            or not isinstance(self.reduction_container_width, int)
            or isinstance(self.reduction_container_width, bool)
            or self.reduction_container_width < self.active_extent
            or not _power_of_two(self.reduction_container_width)
            or not isinstance(self.element_bytes, int)
            or isinstance(self.element_bytes, bool)
            or self.element_bytes < 1
            or self.exact_reduction_certificate
            != "predicated_exact_prefix_reduction_v1"
            or not self.source_fact_refs
            or self.source_fact_refs
            != tuple(sorted(set(self.source_fact_refs)))
            or self.subject_ref != _ref("l.exact-prefix-subject.", payload)
        ):
            raise HBVSchemaError("exact-prefix route subject is malformed")

    @classmethod
    def provider_closed(
        cls, *, active_extent: int, reduction_container_width: int,
        element_bytes: int, exact_reduction_certificate: str,
        source_fact_refs: Tuple[str, ...],
    ) -> "LoopExactPrefixSubjectV1":
        values = {
            "schema": LOOP_ROUTE_SUBJECT_SCHEMA_V1,
            "active_extent": active_extent,
            "reduction_container_width": reduction_container_width,
            "element_bytes": element_bytes,
            "exact_reduction_certificate": exact_reduction_certificate,
            "source_fact_refs": tuple(sorted(set(source_fact_refs))),
        }
        return cls(
            **values,
            subject_ref=_ref("l.exact-prefix-subject.", values))


@dataclass(frozen=True)
class LoopLogicalExactPrefixAdmissionV1:
    mechanism_route_ref: str
    route_subtype: str
    route_factor: int
    factor_kind: str
    artifact_route_ref: str
    subject_ref: str
    legal: bool
    typed_reason: str
    proof_refs: Tuple[str, ...]
    admission_ref: str
    schema: str = LOOP_ROUTE_FACTOR_ADMISSION_SCHEMA_V1

    def __post_init__(self):
        payload = {
            key: value for key, value in self.__dict__.items()
            if key != "admission_ref"
        }
        meaning = LOGICAL_VECTOR_SUBTYPE_MEANINGS_V1[
            LOGICAL_EXACT_PREFIX_SUBTYPE]
        if (
            self.schema != LOOP_ROUTE_FACTOR_ADMISSION_SCHEMA_V1
            or self.mechanism_route_ref != meaning.mechanism_route_ref
            or self.route_subtype != meaning.subtype
            or not isinstance(self.route_factor, int)
            or isinstance(self.route_factor, bool)
            or self.route_factor < 1
            or self.legal != (self.route_factor == 1)
            or self.factor_kind != meaning.factor_kind
            or self.artifact_route_ref != meaning.artifact_route_ref
            or not self.subject_ref
            or self.legal == bool(self.typed_reason)
            or not self.proof_refs
            or self.proof_refs != tuple(sorted(set(self.proof_refs)))
            or self.admission_ref != _ref(
                "l.logical-exact-prefix-admission.", payload)
        ):
            raise HBVSchemaError(
                "logical exact-prefix subtype admission is malformed")


def decide_loop_logical_exact_prefix_admission_v1(
    subject: LoopExactPrefixSubjectV1,
    *, route_factor: int = 1,
) -> LoopLogicalExactPrefixAdmissionV1:
    """Admit the provider-closed exact-prefix logical-vector subcase."""
    meaning = LOGICAL_VECTOR_SUBTYPE_MEANINGS_V1[
        LOGICAL_EXACT_PREFIX_SUBTYPE]
    reason = ""
    if route_factor != 1:
        reason = "exact_prefix_vectorization_requires_semantic_factor_one"
    values = {
        "schema": LOOP_ROUTE_FACTOR_ADMISSION_SCHEMA_V1,
        "mechanism_route_ref": meaning.mechanism_route_ref,
        "route_subtype": meaning.subtype,
        "route_factor": route_factor,
        "factor_kind": meaning.factor_kind,
        "artifact_route_ref": meaning.artifact_route_ref,
        "subject_ref": subject.subject_ref,
        "legal": not reason,
        "typed_reason": reason,
        "proof_refs": tuple(sorted((
            "proof:exact-prefix-is-logical-vector-subtype-not-fourth-route",
            "proof:provider-closed-dynamic-integer-prefix-recurrence",
            "proof:semantic-factor-one-is-not-grouped-load-width",
            *subject.source_fact_refs,
        ))),
    }
    return LoopLogicalExactPrefixAdmissionV1(
        **values,
        admission_ref=_ref("l.logical-exact-prefix-admission.", values))


@dataclass(frozen=True)
class LoopRouteSubjectV1:
    origin: str
    exact_trip_count: Optional[int]
    runtime_main_tail_certificate_ref: str
    subject_ref: str
    schema: str = LOOP_ROUTE_SUBJECT_SCHEMA_V1

    def __post_init__(self):
        values = {
            "schema": self.schema,
            "origin": self.origin,
            "exact_trip_count": self.exact_trip_count,
            "runtime_main_tail_certificate_ref": (
                self.runtime_main_tail_certificate_ref),
        }
        static = self.exact_trip_count is not None
        if (
            self.schema != LOOP_ROUTE_SUBJECT_SCHEMA_V1
            or self.origin not in {
                "absent", "existing_loop", "bridge_constructed"}
            or self.exact_trip_count is not None
            and (
                not isinstance(self.exact_trip_count, int)
                or isinstance(self.exact_trip_count, bool)
                or self.exact_trip_count < 1)
            or self.origin == "absent"
            and (static or self.runtime_main_tail_certificate_ref)
            or self.origin == "bridge_constructed" and not static
            or static and self.runtime_main_tail_certificate_ref
            or self.subject_ref != _ref("l.route-subject.", values)
        ):
            raise HBVSchemaError("loop route subject is malformed")

    @classmethod
    def absent(cls) -> "LoopRouteSubjectV1":
        values = {
            "schema": LOOP_ROUTE_SUBJECT_SCHEMA_V1,
            "origin": "absent",
            "exact_trip_count": None,
            "runtime_main_tail_certificate_ref": "",
        }
        return cls(**values, subject_ref=_ref("l.route-subject.", values))

    @classmethod
    def exact(cls, *, origin: str, trip_count: int) -> "LoopRouteSubjectV1":
        values = {
            "schema": LOOP_ROUTE_SUBJECT_SCHEMA_V1,
            "origin": origin,
            "exact_trip_count": trip_count,
            "runtime_main_tail_certificate_ref": "",
        }
        return cls(**values, subject_ref=_ref("l.route-subject.", values))

    @classmethod
    def runtime(cls, *, certificate_ref: str = "") -> "LoopRouteSubjectV1":
        values = {
            "schema": LOOP_ROUTE_SUBJECT_SCHEMA_V1,
            "origin": "existing_loop",
            "exact_trip_count": None,
            "runtime_main_tail_certificate_ref": certificate_ref,
        }
        return cls(**values, subject_ref=_ref("l.route-subject.", values))


@dataclass(frozen=True)
class LoopBridgeAxisVectorSubjectV1:
    """Provider-closed program region constructed from multiple grid axes."""

    axis_divisors: Tuple[int, int, int]
    exact_trip_count: int
    origin: str
    runtime_main_tail_certificate_ref: str
    subject_ref: str
    schema: str = LOOP_ROUTE_SUBJECT_SCHEMA_V1

    def __post_init__(self):
        values = {
            "schema": self.schema,
            "axis_divisors": self.axis_divisors,
            "exact_trip_count": self.exact_trip_count,
            "origin": self.origin,
            "runtime_main_tail_certificate_ref": (
                self.runtime_main_tail_certificate_ref),
        }
        product = 1
        for divisor in self.axis_divisors:
            product *= divisor
        if (
            self.schema != LOOP_ROUTE_SUBJECT_SCHEMA_V1
            or len(self.axis_divisors) != 3
            or sum(divisor > 1 for divisor in self.axis_divisors) < 2
            or any(not isinstance(divisor, int)
                   or isinstance(divisor, bool)
                   or not _power_of_two(divisor)
                   for divisor in self.axis_divisors)
            or self.exact_trip_count != product
            or self.origin != "bridge_axis_vector_constructed"
            or self.runtime_main_tail_certificate_ref
            or self.subject_ref != _ref(
                "l.route-subject.axis-vector.", values)
        ):
            raise HBVSchemaError(
                "multi-axis Bridge route subject is malformed")

    @classmethod
    def provider_closed(
            cls, *, axis_divisors: Tuple[int, int, int]
            ) -> "LoopBridgeAxisVectorSubjectV1":
        product = 1
        for divisor in axis_divisors:
            product *= divisor
        values = {
            "schema": LOOP_ROUTE_SUBJECT_SCHEMA_V1,
            "axis_divisors": tuple(axis_divisors),
            "exact_trip_count": product,
            "origin": "bridge_axis_vector_constructed",
            "runtime_main_tail_certificate_ref": "",
        }
        return cls(
            **values,
            subject_ref=_ref("l.route-subject.axis-vector.", values))


def route_subject_after_bridge_v1(
    *, bridge_factor: int, existing_subject: LoopRouteSubjectV1,
) -> LoopRouteSubjectV1:
    """Derive the subject after Bridge without importing route semantics."""
    if (
        not isinstance(bridge_factor, int)
        or isinstance(bridge_factor, bool)
        or bridge_factor < 1
        or not _power_of_two(bridge_factor)
    ):
        raise HBVSchemaError("Bridge factor is outside the ontology")
    if bridge_factor == 1:
        return existing_subject
    return LoopRouteSubjectV1.exact(
        origin="bridge_constructed", trip_count=bridge_factor)


@dataclass(frozen=True)
class LoopRouteFactorAdmissionV1:
    route_ref: str
    route_factor: int
    factor_kind: str
    subject_ref: str
    source_trip_count: Optional[int]
    main_iteration_extent: Optional[int]
    tail_iteration_extent: Optional[int]
    main_group_count: Optional[int]
    subject_terminal_disposition: str
    full_source_iteration_elimination: bool
    legal: bool
    typed_reason: str
    proof_refs: Tuple[str, ...]
    admission_ref: str
    schema: str = LOOP_ROUTE_FACTOR_ADMISSION_SCHEMA_V1

    def __post_init__(self):
        payload = {
            key: value for key, value in self.__dict__.items()
            if key != "admission_ref"
        }
        exact = self.source_trip_count is not None
        extents = (self.main_iteration_extent, self.tail_iteration_extent)
        pipeline = self.route_ref == SOFTWARE_PIPELINE_ROUTE
        if (
            self.schema != LOOP_ROUTE_FACTOR_ADMISSION_SCHEMA_V1
            or self.route_ref not in ROUTE_FACTOR_MEANINGS_V1
            or not isinstance(self.route_factor, int)
            or isinstance(self.route_factor, bool)
            or self.route_factor < 1
            or self.factor_kind
            != ROUTE_FACTOR_MEANINGS_V1[self.route_ref].factor_kind
            or not self.subject_ref
            or not self.subject_terminal_disposition
            or not isinstance(self.full_source_iteration_elimination, bool)
            or self.legal == bool(self.typed_reason)
            or not self.proof_refs
            or tuple(sorted(set(self.proof_refs))) != self.proof_refs
            or exact != all(value is not None for value in extents)
            or exact and (
                self.source_trip_count < 1
                or any(not isinstance(value, int) or isinstance(value, bool)
                       or value < 0 for value in extents)
                or sum(extents) != self.source_trip_count)
            or self.main_group_count is not None
            and (
                not isinstance(self.main_group_count, int)
                or isinstance(self.main_group_count, bool)
                or self.main_group_count < 1)
            or pipeline and self.main_group_count is not None
            or self.full_source_iteration_elimination
            and (
                pipeline or self.main_group_count != 1
                or self.tail_iteration_extent != 0)
            or self.admission_ref != _ref(
                "l.route-factor-admission.", payload)
        ):
            raise HBVSchemaError("loop route factor admission is malformed")


def decide_loop_route_factor_admission_v1(
    subject: LoopRouteSubjectV1 | LoopBridgeAxisVectorSubjectV1,
    *, route_ref: str,
    route_factor: int,
) -> LoopRouteFactorAdmissionV1:
    """Decide semantic factor admission, before target capability/profit."""
    if route_ref not in ROUTE_FACTOR_MEANINGS_V1:
        raise HBVSchemaError("route is outside the factor ontology")
    if (
        not isinstance(route_factor, int)
        or isinstance(route_factor, bool)
        or route_factor < 1
    ):
        raise HBVSchemaError("route factor is outside the factor ontology")

    meaning = ROUTE_FACTOR_MEANINGS_V1[route_ref]
    reason = ""
    proofs = {
        "proof:Bridge-and-route-factors-have-distinct-owners",
        "proof:route-factor-does-not-own-source-trip-count",
    }
    main = None
    tail = None
    groups = None
    if subject.origin == "absent":
        reason = "route_subject_absent_after_unchanged_bridge"
    elif route_ref == SOFTWARE_PIPELINE_ROUTE:
        proofs.add("proof:pipeline-stage-independent-of-subject-trip")
        if subject.exact_trip_count is not None:
            main = subject.exact_trip_count
            tail = 0
    elif route_factor < 2 or not _power_of_two(route_factor):
        reason = "full_unroll_grouping_factor_requires_power_of_two_at_least_two"
        if subject.exact_trip_count is not None:
            main = subject.exact_trip_count
            tail = 0
    elif subject.exact_trip_count is not None:
        if route_factor > subject.exact_trip_count:
            reason = "route_grouping_factor_exceeds_exact_subject_trip_count"
            main = subject.exact_trip_count
            tail = 0
        else:
            main = (
                subject.exact_trip_count // route_factor * route_factor)
            tail = subject.exact_trip_count - main
            groups = main // route_factor
            proofs.update((
                "proof:route-factor-within-exact-subject-trip",
                "proof:static-main-tail-partition-conserves-source-trip",
            ))
    elif not subject.runtime_main_tail_certificate_ref:
        reason = "runtime_route_subject_requires_main_tail_certificate"
    else:
        proofs.update((
            subject.runtime_main_tail_certificate_ref,
            "proof:runtime-main-tail-materialization-required",
        ))

    if reason:
        disposition = "unavailable_focal_loop"
    elif route_ref == SOFTWARE_PIPELINE_ROUTE:
        disposition = "retained_live_focal_loop"
    elif subject.exact_trip_count is None:
        disposition = "runtime_factorized_main_tail_focal_loop"
    elif groups == 1 and tail == 0:
        disposition = "eliminated_focal_loop"
    elif tail:
        disposition = "factorized_main_tail_focal_loop"
    else:
        disposition = "retained_factor_group_focal_loop"
    full_elimination = (
        not reason and route_ref != SOFTWARE_PIPELINE_ROUTE
        and groups == 1 and tail == 0)
    values = {
        "schema": LOOP_ROUTE_FACTOR_ADMISSION_SCHEMA_V1,
        "route_ref": route_ref,
        "route_factor": route_factor,
        "factor_kind": meaning.factor_kind,
        "subject_ref": subject.subject_ref,
        "source_trip_count": subject.exact_trip_count,
        "main_iteration_extent": main,
        "tail_iteration_extent": tail,
        "main_group_count": groups,
        "subject_terminal_disposition": disposition,
        "full_source_iteration_elimination": full_elimination,
        "legal": not reason,
        "typed_reason": reason,
        "proof_refs": tuple(sorted(proofs)),
    }
    return LoopRouteFactorAdmissionV1(
        **values,
        admission_ref=_ref("l.route-factor-admission.", values))


@dataclass(frozen=True)
class LoopNestDimensionV1:
    """One Provider-observed dimension of a single nested route subject."""

    provider_loop_locator: str
    nesting_depth: int
    exact_trip_count: int
    parent_loop_locator: str
    provider_certificate_ref: str
    schema: str = LOOP_NEST_ROUTE_SUBJECT_SCHEMA_V1

    def __post_init__(self):
        top_level = self.nesting_depth == 0
        if (
            self.schema != LOOP_NEST_ROUTE_SUBJECT_SCHEMA_V1
            or not self.provider_loop_locator.startswith(
                "planning-cut.loop.")
            or not self.provider_loop_locator.removeprefix(
                "planning-cut.loop.").isdigit()
            or not isinstance(self.nesting_depth, int)
            or isinstance(self.nesting_depth, bool)
            or self.nesting_depth < 0
            or not isinstance(self.exact_trip_count, int)
            or isinstance(self.exact_trip_count, bool)
            or self.exact_trip_count < 2
            or top_level == bool(self.parent_loop_locator)
            or not self.provider_certificate_ref
        ):
            raise HBVSchemaError("loop-nest dimension is malformed")


@dataclass(frozen=True)
class LoopNestedRouteSubjectV1:
    """A connected loop nest governed by one route, never per-level routes."""

    dimensions: Tuple[LoopNestDimensionV1, ...]
    subject_ref: str
    schema: str = LOOP_NEST_ROUTE_SUBJECT_SCHEMA_V1

    def __post_init__(self):
        payload = {
            "schema": self.schema,
            "dimensions": tuple(dimension.__dict__
                                for dimension in self.dimensions),
        }
        if (
            self.schema != LOOP_NEST_ROUTE_SUBJECT_SCHEMA_V1
            or len(self.dimensions) < 2
            or tuple(dimension.nesting_depth
                     for dimension in self.dimensions)
            != tuple(range(len(self.dimensions)))
            or len({dimension.provider_loop_locator
                    for dimension in self.dimensions})
            != len(self.dimensions)
            or any(
                dimension.parent_loop_locator
                != self.dimensions[index - 1].provider_loop_locator
                for index, dimension in enumerate(self.dimensions)
                if index > 0)
            or self.subject_ref != _ref("l.nested-route-subject.", payload)
        ):
            raise HBVSchemaError("nested route subject is malformed")

    @classmethod
    def provider_closed(
        cls, dimensions: Tuple[LoopNestDimensionV1, ...],
    ) -> "LoopNestedRouteSubjectV1":
        payload = {
            "schema": LOOP_NEST_ROUTE_SUBJECT_SCHEMA_V1,
            "dimensions": tuple(dimension.__dict__
                                for dimension in dimensions),
        }
        return cls(
            dimensions=tuple(dimensions),
            subject_ref=_ref("l.nested-route-subject.", payload))


@dataclass(frozen=True)
class LoopNestStructuralScopeV1:
    """Provider proof of which dimensions one route may act through.

    This is artifact identity, not a second route-selection axis.  Keeping the
    dimension vector is mandatory because equal outer/inner trip counts can
    produce the same scalar factor while naming different transformations.
    """

    subject_ref: str
    dimension_mask: Tuple[bool, ...]
    dimension_trip_vector: Tuple[int, ...]
    scope_cardinality: int
    scope_ref: str
    schema: str = LOOP_NEST_STRUCTURAL_SCOPE_SCHEMA_V1

    def __post_init__(self):
        payload = {
            key: value for key, value in self.__dict__.items()
            if key != "scope_ref"
        }
        product = 1
        for selected, trip in zip(
                self.dimension_mask, self.dimension_trip_vector):
            if selected:
                product *= trip
        if (
            self.schema != LOOP_NEST_STRUCTURAL_SCOPE_SCHEMA_V1
            or not self.subject_ref
            or len(self.dimension_mask) < 2
            or len(self.dimension_mask) != len(self.dimension_trip_vector)
            or any(type(selected) is not bool
                   for selected in self.dimension_mask)
            or any(not isinstance(trip, int) or isinstance(trip, bool)
                   or trip < 2 for trip in self.dimension_trip_vector)
            or self.scope_cardinality != product
            or self.scope_ref != _ref("l.nested-route-scope.", payload)
        ):
            raise HBVSchemaError("nested route structural scope is malformed")

    @classmethod
    def provider_closed(
        cls, subject: LoopNestedRouteSubjectV1,
        *, dimension_mask: Tuple[bool, ...],
    ) -> "LoopNestStructuralScopeV1":
        trips = tuple(dimension.exact_trip_count
                      for dimension in subject.dimensions)
        product = 1
        for selected, trip in zip(dimension_mask, trips):
            if selected:
                product *= trip
        values = {
            "schema": LOOP_NEST_STRUCTURAL_SCOPE_SCHEMA_V1,
            "subject_ref": subject.subject_ref,
            "dimension_mask": tuple(dimension_mask),
            "dimension_trip_vector": trips,
            "scope_cardinality": product,
        }
        return cls(
            **values, scope_ref=_ref("l.nested-route-scope.", values))


def enumerate_loop_nest_structural_scopes_v1(
    subject: LoopNestedRouteSubjectV1,
) -> Tuple[LoopNestStructuralScopeV1, ...]:
    """Enumerate no-op and every structurally distinct expansion scope."""
    count = len(subject.dimensions)
    return tuple(
        LoopNestStructuralScopeV1.provider_closed(
            subject,
            dimension_mask=tuple(bool(bits & (1 << index))
                                 for index in range(count)))
        for bits in range(1 << count)
    )


@dataclass(frozen=True)
class LoopNestedRouteFactorAdmissionV1:
    subject_ref: str
    scope_ref: str
    route_ref: str
    route_factor: int
    factor_kind: str
    full_expansion_upper_bound: int
    legal: bool
    typed_reason: str
    proof_refs: Tuple[str, ...]
    admission_ref: str
    schema: str = LOOP_NEST_ROUTE_ADMISSION_SCHEMA_V1

    def __post_init__(self):
        payload = {
            key: value for key, value in self.__dict__.items()
            if key != "admission_ref"
        }
        if (
            self.schema != LOOP_NEST_ROUTE_ADMISSION_SCHEMA_V1
            or not self.subject_ref or not self.scope_ref
            or self.route_ref not in ROUTE_FACTOR_MEANINGS_V1
            or not isinstance(self.route_factor, int)
            or isinstance(self.route_factor, bool)
            or self.route_factor < 1
            or self.factor_kind
            != ROUTE_FACTOR_MEANINGS_V1[self.route_ref].factor_kind
            or not isinstance(self.full_expansion_upper_bound, int)
            or isinstance(self.full_expansion_upper_bound, bool)
            or self.full_expansion_upper_bound < 1
            or self.legal == bool(self.typed_reason)
            or not self.proof_refs
            or self.proof_refs != tuple(sorted(set(self.proof_refs)))
            or self.admission_ref != _ref(
                "l.nested-route-factor-admission.", payload)
        ):
            raise HBVSchemaError(
                "nested route factor admission is malformed")


def decide_loop_nested_route_factor_admission_v1(
    subject: LoopNestedRouteSubjectV1,
    scope: LoopNestStructuralScopeV1,
    *, route_ref: str,
    route_factor: int,
) -> LoopNestedRouteFactorAdmissionV1:
    """Bind one route/factor to one Provider-derived nested scope."""
    if route_ref not in ROUTE_FACTOR_MEANINGS_V1:
        raise HBVSchemaError("route is outside the nested factor ontology")
    if (not isinstance(route_factor, int) or isinstance(route_factor, bool)
            or route_factor < 1):
        raise HBVSchemaError("nested route factor is malformed")
    if scope.subject_ref != subject.subject_ref:
        raise HBVSchemaError("nested route scope belongs to another subject")

    reason = ""
    selected = any(scope.dimension_mask)
    if not selected:
        reason = "nested_route_requires_nonempty_structural_scope"
    elif route_ref == FULL_UNROLL_REORDER_ROUTE:
        if route_factor != scope.scope_cardinality:
            reason = (
                "phase_reorder_factor_must_equal_full_expansion_upper_bound")
    elif route_ref == FULL_UNROLL_VECTOR_ROUTE:
        if route_factor < 2 or not _power_of_two(route_factor):
            reason = (
                "nested_logical_factor_requires_power_of_two_at_least_two")
        elif route_factor > scope.scope_cardinality:
            reason = "nested_logical_factor_exceeds_structural_scope"
    # Pipeline stage count is independent of expansion cardinality.  The
    # Provider/materializer still owns whether this exact scope is pipeline-
    # capable; this ontology must not infer native capability from trip count.

    proofs = tuple(sorted(set((
        "proof:one-route-governs-one-nested-subject",
        "proof:nested-scope-vector-disambiguates-equal-scalar-factors",
        "proof:route-factor-kind-is-route-local",
        *tuple(dimension.provider_certificate_ref
               for dimension in subject.dimensions),
    ))))
    values = {
        "schema": LOOP_NEST_ROUTE_ADMISSION_SCHEMA_V1,
        "subject_ref": subject.subject_ref,
        "scope_ref": scope.scope_ref,
        "route_ref": route_ref,
        "route_factor": route_factor,
        "factor_kind": ROUTE_FACTOR_MEANINGS_V1[route_ref].factor_kind,
        "full_expansion_upper_bound": scope.scope_cardinality,
        "legal": not reason,
        "typed_reason": reason,
        "proof_refs": proofs,
    }
    return LoopNestedRouteFactorAdmissionV1(
        **values,
        admission_ref=_ref(
            "l.nested-route-factor-admission.", values))
