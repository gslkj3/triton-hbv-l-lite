"""Native autotuner over the same AST candidate compiler as main L.

This explicit AST API owns one invocation's candidate population. It never
consults the predictor, prunes on predicted performance, or precompiles away
resource failures before handing configurations to native autotune.
"""
from dataclasses import dataclass
from time import perf_counter_ns
from .frontend import native_options
from .candidate_frontend import prepare_python_candidates
from .selected_materialization import compile_selected


@dataclass(frozen=True)
class AutotuneResult:
    population: object
    selected: str
    timings: dict
    compilation_records: tuple
    elapsed_ns: int
    publication_permitted: bool = False


def autotune_python(source, runtime_args, *, target, options, bridge_factors,
                    route_factors, binding, grid, runtime_scalars=None,
                    reset_to_zero=None, restore_value=None, do_bench=None):
    import triton
    from triton.runtime.autotuner import Autotuner
    if (not isinstance(grid, tuple) or not 1<=len(grid)<=3 or
            any(type(g) is not int or g<1 for g in grid)):
        raise ValueError('positive concrete grid required')
    runtime_names=[name for i,name in enumerate(source.fn.arg_names)
                   if (i,) not in source.constants]
    if len(runtime_args)!=len(runtime_names):
        raise ValueError('runtime arguments must exclude AST-bound constexpr values')
    started=perf_counter_ns()
    population=prepare_python_candidates(source,target=target,options=options,
        bridge_factors=bridge_factors,route_factors=route_factors,binding=binding,
        runtime_scalars=runtime_scalars)
    candidates={'Original':None}
    candidates.update({c.planned.candidate.identity:c for c in population.candidates.candidates})
    native=native_options(options,target)
    physical=grid+(1,)*(3-len(grid))
    compiled={}
    failures={}
    records=[]

    class CandidateLauncher:
        fn=source.fn

        def run(self,*args,_l_candidate,**meta):
            # Config carries native options because native autotune always adds
            # them; require agreement rather than silently ignoring a mismatch.
            for name,value in meta.items():
                if name not in ('num_warps','num_stages','num_ctas','maxnreg') or getattr(native,name)!=value:
                    raise ValueError('autotune changed bound native option: '+name)
            if _l_candidate in failures:
                raise failures[_l_candidate]
            if _l_candidate not in compiled:
                begin=perf_counter_ns()
                bound=candidates[_l_candidate]
                try:
                    if bound is None:
                        kernel=triton.compile(source,target=target,options=native.__dict__)
                        launch_grid=physical
                    else:
                        divisors=bound.upstream_input.grid_divisors
                        if any(g%d for g,d in zip(physical,divisors)):
                            raise ValueError('Bridge grid requires an exact legal partition')
                        kernel,_=compile_selected(bound,target=target)
                        launch_grid=tuple(g//d for g,d in zip(physical,divisors))
                except Exception as error:
                    failures[_l_candidate]=error
                    records.append(dict(candidate=_l_candidate,elapsed_ns=perf_counter_ns()-begin,
                        error=type(error).__name__+':'+str(error)))
                    raise
                compiled[_l_candidate]=(kernel,launch_grid)
                records.append(dict(candidate=_l_candidate,elapsed_ns=perf_counter_ns()-begin,error=''))
            kernel,launch_grid=compiled[_l_candidate]
            if _l_candidate=='Original':
                remaining=iter(args)
                full_args=tuple(source.constants[(i,)] if (i,) in source.constants else next(remaining)
                                for i in range(len(source.fn.arg_names)))
                return kernel[launch_grid](*full_args)
            return kernel[launch_grid](*args)

    configs=[triton.Config({'_l_candidate':identity},num_warps=native.num_warps,
        num_stages=native.num_stages,num_ctas=native.num_ctas,maxnreg=native.maxnreg)
        for identity in candidates]
    tuner=Autotuner(CandidateLauncher(),runtime_names,configs,key=[],
        reset_to_zero=reset_to_zero,restore_value=restore_value,do_bench=do_bench,
        cache_results=False)
    # Never let a process-global disk-cache switch import an old candidate
    # decision: this result is bound to the freshly prepared population.
    tuner.cache_results=False
    tuner.run(*runtime_args)
    timings={config.kwargs['_l_candidate']:value
             for config,value in getattr(tuner,'configs_timings',{}).items()}
    return AutotuneResult(population,tuner.best_config.kwargs['_l_candidate'],
                          timings,tuple(records),perf_counter_ns()-started)
