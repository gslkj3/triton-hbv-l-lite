"""Shared nonpredictive AST candidate preparation."""
from dataclasses import dataclass
from time import perf_counter_ns
from .frontend import prepare_from_source, prepare_from_module
from .upstream import PreparedBatch, PreparationFailure
from .candidates import enumerate_candidates


@dataclass(frozen=True)
class PythonCandidates:
    preparation: object
    candidates: object
    original_input: object


def prepare_python_candidates(source, *, target, options, bridge_factors,
                              route_factors, binding, runtime_scalars=None):
    def prepare(factor):
        return prepare_from_source(source, target=target, metadata=options,
            factor=factor, source_ref=binding.source_ref,
            compiler_commit=binding.compiler_commit, runtime_scalars=runtime_scalars)
    return _prepare_candidates(prepare, bridge_factors, route_factors, binding)


def prepare_module_candidates(module, *, target, options, bridge_factors,
                              route_factors, binding, specialization_ref,
                              runtime_scalars=None):
    """Enumerate from the actual native frontend output, before make_ttir passes."""
    def prepare(factor):
        return prepare_from_module(module, target=target, metadata=options,
            factor=factor, source_ref=binding.source_ref,
            compiler_commit=binding.compiler_commit,
            specialization_ref=specialization_ref, runtime_scalars=runtime_scalars)
    return _prepare_candidates(prepare, bridge_factors, route_factors, binding)


def _prepare_candidates(prepare, bridge_factors, route_factors, binding):
    factors = tuple(bridge_factors)
    if any(type(f) is not int or f < 1 for f in factors):
        raise ValueError('positive integer Bridge factors required')
    started = perf_counter_ns()
    inputs, failures = [], []
    original = None
    # Original preparation is mandatory for the comparison, even when the
    # requested candidate space starts with a constructed-loop alternative.
    for factor in sorted(set(factors) | {1}):
        try:
            item = prepare(factor)
        except Exception as error:
            if factor == 1:
                raise
            failures.append(PreparationFailure(factor, type(error).__name__+':'+str(error)))
            continue
        if factor == 1:
            original = item
        if factor in factors:
            inputs.append((factor, item))
    prepared = PreparedBatch(tuple(inputs), tuple(failures), perf_counter_ns()-started)
    return PythonCandidates(prepared, enumerate_candidates(prepared, route_factors,
                            binding=binding), original)
