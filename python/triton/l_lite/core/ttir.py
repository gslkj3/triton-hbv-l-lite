"""L-owned preparation cut using native TTIR passes and independent Bridge.

This is an in-process compiler service. It does not perform backend compilation,
route selection, time prediction, or recover a failed transformation silently.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from time import perf_counter_ns
from .facts import CompilerFacts, decode_facts


@dataclass(frozen=True)
class PreparationConfig:
    # Target is explicit; this implementation has been verified on CUDA SM89.
    capability: int
    num_warps: int
    num_stages: int
    bridge_divisors: tuple[int, int, int] = (1, 1, 1)
    runtime_scalars: str | None = None

    def __post_init__(self):
        if self.capability != 89:
            raise ValueError('preparation target binding not yet validated')
        if any(type(v) is not int or v < 1 for v in (self.num_warps, self.num_stages)):
            raise ValueError('invalid preparation options')
        if len(self.bridge_divisors) != 3 or any(type(v) is not int or v < 1 or v & (v-1)
                                               for v in self.bridge_divisors):
            raise ValueError('Bridge requires three positive power-of-two divisors')

    @property
    def factor(self):
        return math.prod(self.bridge_divisors)


@dataclass(frozen=True)
class PreparedIR:
    provider: CompilerFacts
    source_ir_sha256: str
    route_input_ir_sha256: str
    route_input_ir: str
    upstream_factor: int
    grid_divisors: tuple[int, int, int]
    preparation_ns: int
    config: PreparationConfig
    source_specialization_ref: str = ''
    native_options_ref: str = ''
    native_options_json: str = ''


def prepare_module(module, config):
    """Standalone preparation; the native pipeline owns this prefix itself."""
    from triton._C.libtriton import ir, passes
    pm = ir.pass_manager(module.context)
    passes.common.add_inliner(pm)
    passes.ttir.add_rewrite_tensor_pointer(pm)
    if config.capability // 10 < 9:
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_combine(pm)
    passes.ttir.add_reorder_broadcast(pm)
    passes.common.add_cse(pm)
    passes.common.add_symbol_dce(pm)
    pm.run(module, 'l_core_native_prefix')
    return prepare_at_analysis_point(module, config)


def prepare_at_analysis_point(module, config):
    """Project passes only, after native cleanup and before native unroll."""
    from triton._C.libtriton import ir, passes
    for key in ('tt.hbv.plan_bundle', 'tt.hbv.l.static_facts'):
        if module.get_operation().get_str_attr(key) is not None:
            raise ValueError('preparation requires fresh pre-pass IR, not an existing plan or snapshot')
    started = perf_counter_ns()
    source_hash = sha256(str(module).encode()).hexdigest()
    builder = ir.builder(module.context)
    module.set_attr('tt.loop_bridge.factor', builder.get_int32_attr(config.factor))
    module.set_attr('tt.loop_bridge.requested_divisors',
                    builder.get_string_attr(json.dumps(config.bridge_divisors, separators=(',', ':'))))
    module.set_attr('tt.hbv.l.native_default_num_stages', builder.get_int32_attr(config.num_stages))
    if config.runtime_scalars is not None:
        # Bridge's compiler parser verifies the versioned runtime binding.
        module.set_attr('tt.loop_bridge.runtime_scalars', builder.get_string_attr(config.runtime_scalars))
    pm = ir.pass_manager(module.context)
    passes.ttir.add_loop_bridge_discover(pm)
    if config.factor != 1:
        passes.ttir.add_loop_bridge_program_coarsening(pm)
        # Bridge can introduce private helpers after the original inliner ran.
        # Expose their ordinary body before any route census/selection; use
        # native inlining, not a Bridge-name-specific downstream adapter.
        passes.common.add_inliner(pm)
        passes.common.add_symbol_dce(pm)
    passes.ttir.add_hbv_loop_facts(pm)
    pm.run(module, 'l_core_prepare')
    facts = decode_facts(module.get_operation().get_str_attr('tt.hbv.l.static_facts'),
                         num_warps=config.num_warps, num_stages=config.num_stages)
    divisors = tuple(module.get_int_attr('tt.loop_bridge.grid_divisor_'+axis) or 1 for axis in 'xyz')
    if divisors != config.bridge_divisors:
        raise ValueError('Bridge did not realize requested launch divisors')
    text = str(module)
    return PreparedIR(facts, source_hash, sha256(text.encode()).hexdigest(), text,
                      config.factor, divisors, perf_counter_ns()-started, config)
