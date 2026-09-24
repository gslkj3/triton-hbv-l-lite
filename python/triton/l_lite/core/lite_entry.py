"""Bind real JIT arguments with the pinned compiler's own specialization rules."""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class BoundPythonCall:
    source: object
    target: object
    options: dict
    runtime_args: tuple
    grid: tuple


def bind_python_call(kernel, args, kwargs, *, grid, target=None):
    from triton import knobs
    from triton.compiler import ASTSource, make_backend
    from triton.runtime.jit import JITFunction, create_function_from_signature
    from triton.runtime.driver import driver
    if not isinstance(kernel,JITFunction):
        raise TypeError('expected a native @triton.jit kernel')
    target=target or driver.active.get_current_target()
    backend=make_backend(target)
    values=dict(kwargs)
    values['debug']=values.get('debug',kernel.debug) or knobs.runtime.debug
    values['instrumentation_mode']=knobs.compilation.instrumentation_mode
    for hook in kernel.pre_run_hooks:
        hook(*args,**values)
    binder=create_function_from_signature(kernel.signature,kernel.params,backend)
    bound,specialization,options=binder(*args,**values)
    options,signature,constants,attrs=kernel._pack_args(backend,values,bound,specialization,options)
    source=ASTSource(kernel,signature,constexprs=constants,attrs=attrs)
    resolved=grid(bound) if callable(grid) else grid
    if (not isinstance(resolved,(tuple,list)) or not 1<=len(resolved)<=3 or
            any(type(n) is not int or n<1 for n in resolved)):
        raise ValueError('positive launch grid required')
    runtime=tuple(v for i,v in enumerate(bound.values()) if (i,) not in constants)
    return BoundPythonCall(source,target,options.__dict__,runtime,tuple(resolved))


def autotune_kernel(kernel, *args, grid, bridge_factors, route_factors, binding,
                    kernel_kwargs=None, target=None, **tuning):
    from .native_autotune_entry import autotune_python
    call=bind_python_call(kernel,args,kernel_kwargs or {},grid=grid,target=target)
    binding=replace(binding,source_ref='ASTSource:'+call.source.hash())
    return autotune_python(call.source,call.runtime_args,target=call.target,
        options=call.options,grid=call.grid,bridge_factors=bridge_factors,
        route_factors=route_factors,binding=binding,**tuning)
