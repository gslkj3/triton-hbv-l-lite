"""Predictor-free, autotuner-free default compilation of a Python kernel."""
from dataclasses import dataclass, replace
import json
from pathlib import Path
import tempfile
from .lite_entry import bind_python_call
from .candidate_frontend import prepare_python_candidates
from .frontend import native_options
from .selected_materialization import compile_selected
from .state import Route


@dataclass(frozen=True)
class DefaultCompilation:
    kernel: object
    grid: tuple
    stage_count: int
    bridge_factor: int
    route: str
    population: object
    publication_permitted: bool = False


def default_kernel(kernel, *args, grid, binding, kernel_kwargs=None, target=None):
    """Run native-default Bridge=1 and only a default-stage pipeline candidate.

No score, benefit threshold or search. Lack of a structurally eligible loop
keeps ordinary native compilation, not an error and not a pipeline success.
Explicit user options still bind the native stage count; absent options use
CUDAOptions defaults, not a separately hardcoded stage value.
"""
    import triton
    from triton._C import libtriton
    call=bind_python_call(kernel,args,kernel_kwargs or {},grid=grid,target=target)
    binding=replace(binding,source_ref='ASTSource:'+call.source.hash())
    stages=call.options['num_stages']
    population=prepare_python_candidates(call.source,target=call.target,options=call.options,
        bridge_factors=(1,),route_factors={Route.PIPELINE:(stages,)} if stages>=2 else {},
        binding=binding)
    if len(population.candidates.candidates)>1:
        raise ValueError('default mode must not construct a search population')
    if population.candidates.candidates:
        compiled,_=compile_selected(population.candidates.candidates[0],target=call.target)
        route=Route.PIPELINE.value
    else:
        # Finish the prepared prefix exactly once even when no L route applies.
        # Empty decision/materialization passes are native-path no-ops.
        prepared=population.original_input
        ir=libtriton.ir
        context=ir.context()
        ir.load_dialects(context)
        with tempfile.TemporaryDirectory(prefix='l-default-') as directory:
            path=Path(directory)/'prepared.mlir'
            path.write_text(prepared.route_input_ir)
            module=ir.parse_mlir_module(str(path),context)
            pm=ir.pass_manager(context)
            passes=libtriton.passes.ttir
            for name in ('add_hbv_loop_decision','add_loop_unroll',
                         'add_hbv_loop_materialize','add_hbv_validate_loop_plan'):
                getattr(passes,name)(pm)
            pm.run(module,'l_default_suffix')
            ttir=Path(directory)/'default.ttir'
            ttir.write_text(str(module))
            compiled=triton.compile(str(ttir),target=call.target,
                options=native_options(json.loads(prepared.native_options_json),call.target).__dict__)
        route='native_no_pipeline_candidate'
    physical=call.grid+(1,)*(3-len(call.grid))
    compiled[physical](*call.runtime_args)
    return DefaultCompilation(compiled,physical,stages,1,route,population)
