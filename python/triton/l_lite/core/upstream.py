"""Independent upstream preparation, before ordinary peer-route planning."""
from dataclasses import dataclass
from time import perf_counter_ns


@dataclass(frozen=True)
class PreparationFailure:
    factor: int
    reason: str


@dataclass(frozen=True)
class PreparedBatch:
    inputs: tuple[tuple[int, object], ...]
    failures: tuple[PreparationFailure, ...]
    elapsed_ns: int


def prepare_ir(module_factory, configurations):
    """Prepare fresh frontend modules using the independent compiler service.

    Configurations include their axis divisors; do not collapse distinct
    same-product axis choices under an assumed x-axis-only Bridge.
    """
    from .ttir import prepare_module
    started = perf_counter_ns()
    inputs, failures = [], []
    seen = set()
    for config in configurations:
        if config in seen:
            continue
        seen.add(config)
        try:
            item = prepare_module(module_factory(config), config)
            inputs.append((config.factor, item))
        except Exception as error:
            # Keep the exact requested construction coordinate in the reason;
            # never substitute Original or a successfully prepared sibling.
            failures.append(PreparationFailure(config.factor,
                f'divisors={config.bridge_divisors};{type(error).__name__}:{error}'))
    return PreparedBatch(tuple(inputs), tuple(failures), perf_counter_ns()-started)


def prepare(original, factors, *, binding, runtime_scalars, compiler_prepare=None):
    """Prepare each requested upstream choice once; never substitute another IR.

    The compiler owns upstream legality. It also currently owns target support;
    this adapter does not silently widen that support or import a legacy runner.
    """
    if compiler_prepare is None:
        from .frontend import prepare_from_artifact
        compiler_prepare = prepare_from_artifact
    factors = tuple(factors)
    if any(type(f) is not int or f < 1 for f in factors):
        raise ValueError('invalid upstream factor')
    started = perf_counter_ns()
    inputs, failures = [], []
    for factor in sorted(set(factors)):
        try:
            item = compiler_prepare(original, factor=factor, source_ref=binding.source_ref,
                compiler_commit=binding.compiler_commit, runtime_scalars=runtime_scalars)
            if item.upstream_factor != factor:
                raise ValueError('compiler preparation returned a different upstream choice')
            inputs.append((factor, item))
        except Exception as error:
            failures.append(PreparationFailure(factor, type(error).__name__ + ':' + str(error)))
    return PreparedBatch(tuple(inputs), tuple(failures), perf_counter_ns() - started)
