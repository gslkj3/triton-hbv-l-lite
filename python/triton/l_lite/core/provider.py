"""Ordinary loop facts; no model imports or source-origin discriminator.

Migrated from the current compiler Provider contract. Full-kernel facts remain
separate: this record alone is not sufficient input for a time predictor.
"""
from dataclasses import dataclass
from .state import Route

PIPELINE = Route.PIPELINE.value
PHASE = Route.REORDER.value
LOGICAL = Route.VECTORIZE.value

@dataclass(frozen=True, kw_only=True)
class LoopFacts:
    """One identity-free existing-loop Provider subject at the planning cut."""

    locator: str
    lower_kind: str
    upper_kind: str
    step_kind: str
    exact_static_trip_count: int | None
    carried_value_count: int
    pipeline_capable: bool
    pipeline_certificate: str
    pipeline_reason: str
    phase_capable: bool
    phase_certificate: str
    phase_reason: str
    logical_capable: bool
    logical_certificate: str
    logical_reason: str
    entry_owned: bool = True
    nesting_depth: int = 0
    parent_locator: str = ""
    nested_inner_dimension_capable: bool = False
    nested_inner_dimension_certificate: str = ""
    nested_inner_dimension_reason: str = ""
    runtime_main_tail_certificate: str = ""
    preexisting_pipeline_stage_count: int = 0
    source_operation_group_count: int
    phase_barrier_operation_count: int = 0
    native_visible_async_load_count: int
    logical_elementwise_packable_operation_count: int
    logical_load_adapter_opportunity_count: int
    logical_store_adapter_opportunity_count: int

    # Optional source observations. Old snapshots remain unknown, never zero.
    # These constrain packing but do not certify its realization.
    iteration_invariant_store_address_count: int | None = None
    explicit_synchronization_barrier_count: int | None = None
    direct_load_container_bytes: int | None = None
    direct_store_container_bytes: int | None = None
    opaque_body_region_count: int | None = None
    unconditional_entry_loop_ancestry: bool | None = None
    # Launch-bound observation for prediction, never static unroll authority.
    launch_trip_count: int | None = None
    memory_opaque_body_region_count: int | None = None
    source_load_service_counts: tuple[int, int, int, int] | None = None
    body_container_work_observed: bool = False
    body_load_container_bytes: int | None = None
    body_store_container_bytes: int | None = None

    def __post_init__(self):
        if type(self.body_container_work_observed) is not bool or any(
                value is not None and (type(value) is not int or value < 0)
                for value in (self.body_load_container_bytes, self.body_store_container_bytes)):
            raise ValueError('invalid inclusive body container work')
        if self.source_load_service_counts is not None and (
                len(self.source_load_service_counts)!=4 or
                any(type(n)is not int or n<0 for n in self.source_load_service_counts)):
            raise ValueError('invalid source load service counts')
        if (not self.locator or type(self.nesting_depth) is not int
                or self.nesting_depth < 0
                or (self.nesting_depth == 0) != (not self.parent_locator)):
            raise ValueError('inconsistent ordinary loop structure')
        if self.exact_static_trip_count is not None and (
                type(self.exact_static_trip_count) is not int or self.exact_static_trip_count < 0):
            raise ValueError('invalid exact loop trip count')
        if self.exact_static_trip_count is not None and self.runtime_main_tail_certificate:
            raise ValueError('static and runtime trip evidence conflict')
        if self.launch_trip_count is not None:
            if type(self.launch_trip_count) is not int or self.launch_trip_count < 0:
                raise ValueError('invalid launch loop trip count')
            if self.exact_static_trip_count is not None and self.launch_trip_count != self.exact_static_trip_count:
                raise ValueError('launch and static trip observations disagree')
        for capable, certificate in (
                (self.pipeline_capable, self.pipeline_certificate),
                (self.phase_capable, self.phase_certificate),
                (self.logical_capable, self.logical_certificate),
                (self.nested_inner_dimension_capable, self.nested_inner_dimension_certificate)):
            if type(capable) is not bool or capable != bool(certificate):
                raise ValueError('capability and certificate disagree')
        if self.nested_inner_dimension_capable and not self.parent_locator:
            raise ValueError('nested capability has no parent')
        counts = (self.carried_value_count, self.preexisting_pipeline_stage_count,
                  self.source_operation_group_count, self.phase_barrier_operation_count,
                  self.native_visible_async_load_count,
                  self.logical_elementwise_packable_operation_count,
                  self.logical_load_adapter_opportunity_count,
                  self.logical_store_adapter_opportunity_count)
        if any(type(n) is not int or n < 0 for n in counts):
            raise ValueError('invalid loop opportunity count')
        if self.phase_barrier_operation_count > self.source_operation_group_count:
            raise ValueError('barrier count exceeds operation groups')
        for n in (self.iteration_invariant_store_address_count,
                  self.explicit_synchronization_barrier_count,
                  self.direct_load_container_bytes, self.direct_store_container_bytes,
                  self.opaque_body_region_count, self.memory_opaque_body_region_count):
            if n is not None and (type(n) is not int or n < 0):
                raise ValueError('invalid optional memory ordering observation')
        if (self.unconditional_entry_loop_ancestry is not None and
                type(self.unconditional_entry_loop_ancestry) is not bool):
            raise ValueError('invalid loop invocation observation')

    def semantic_route_certificate(self, route_ref: str) -> str:
        """Return the Strong legality certificate, independent of coverage."""
        mapping = {
            PIPELINE: (self.pipeline_capable, self.pipeline_certificate),
            PHASE: (self.phase_capable, self.phase_certificate),
            LOGICAL: (self.logical_capable, self.logical_certificate),
        }
        try:
            capable, certificate = mapping[route_ref]
        except KeyError as error:
            raise ValueError("unknown existing-loop route") from error
        return certificate if capable else ""

    def route_service_opportunity_count(self, route_ref: str) -> int:
        """Return an operation-neutral materializer opportunity count.

        The count answers whether this pass can change anything, not which
        source/operator case produced it.  A route-local zero must not erase
        unrelated routes or the Bridge-constructed subject.
        """
        mapping = {
            PIPELINE: self.native_visible_async_load_count,
            PHASE: self.source_operation_group_count,
            LOGICAL: sum((
                self.logical_elementwise_packable_operation_count,
                self.logical_load_adapter_opportunity_count,
                self.logical_store_adapter_opportunity_count,
            )),
        }
        try:
            return int(mapping[route_ref])
        except KeyError as error:
            raise ValueError("unknown existing-loop route") from error

    def route_certificate(self, route_ref: str) -> str:
        """Return the executable route certificate at the planning cut.

        Strong legality alone is insufficient:
        the current pass must also own at least one concrete transformation
        opportunity.  This is route-local materializer coverage, not an
        operator-pattern admission rule.
        """
        certificate = self.semantic_route_certificate(route_ref)
        if certificate and self.route_service_opportunity_count(route_ref) < 1:
            return ""
        return certificate
