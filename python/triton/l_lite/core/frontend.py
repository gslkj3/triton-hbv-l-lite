"""Recreate a retained AST specialization without loading historical HBV options.

Native option resolution follows the pinned Triton CUDA backend. L preparation
is separate from those options and never invokes the old backend's HBV parser.
"""
from dataclasses import replace
import json
from time import perf_counter_ns
from .ttir import PreparationConfig, prepare_module


def native_options(metadata, target):
    from triton import knobs
    from triton.backends.nvidia.compiler import CUDAOptions
    if target.backend != 'cuda' or target.arch != 89:
        raise ValueError('frontend target binding not yet validated')
    names = {name for name in CUDAOptions.__dataclass_fields__
             if not name.startswith(('hbv_', 'loop_bridge_'))}
    args = {k:v for k,v in metadata.items() if k in names and v is not None}
    # JSON snapshots turn tuple-valued native options into lists. Restore the
    # native schema so frozen options remain hashable in backend compilation.
    for name, field in CUDAOptions.__dataclass_fields__.items():
        if name in args and isinstance(field.default, tuple):
            args[name] = tuple(args[name])
    # Metadata represents resolved options of the source artifact. A changed
    # environment override must not silently rebind its target.
    arch = args.setdefault('arch', f'sm{target.arch}')
    if arch != f'sm{target.arch}':
        raise ValueError('retained options and target architecture disagree')
    if args.get('ir_override'):
        raise ValueError('IR override cannot be reconstructed from AST alone')
    if args.get('instrumentation_mode') == 'consan':
        args['debug'] = True
    if args.get('num_ctas', 1) > 1:
        raise ValueError('multiple CTAs require a supported clustered target')
    if 'supported_fp8_dtypes' not in args:
        args['supported_fp8_dtypes'] = tuple(sorted(set(CUDAOptions.supported_fp8_dtypes) | {'fp8e4nv'}))
    args.setdefault('enable_fp_fusion', knobs.language.default_fp_fusion)
    args.setdefault('max_num_imprecise_acc_default', 0)
    return CUDAOptions(**args)


def prepare_from_artifact(original, *, factor, source_ref, compiler_commit, runtime_scalars=None):
    return prepare_from_source(getattr(original, 'src', None),
        target=original.metadata.target, metadata=original.metadata._asdict(),
        factor=factor, source_ref=source_ref, compiler_commit=compiler_commit,
        runtime_scalars=runtime_scalars)


def prepare_from_source(source, *, target, metadata, factor, source_ref,
                        compiler_commit, runtime_scalars=None):
    """Enter from an AST specialization; no previously compiled kernel needed."""
    from triton.compiler import ASTSource, make_backend
    from triton._C.libtriton import ir
    if not source_ref or compiler_commit != '7c56a5e40f7fd928dfd5c72902d5def0097db73a':
        raise ValueError('preparation source/compiler binding is not current')
    if not isinstance(source, ASTSource):
        raise ValueError('preparation requires retained AST signature, constants and attributes')
    started = perf_counter_ns()
    options = native_options(metadata, target)
    backend = make_backend(target)
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = source.make_ir(target, options, backend.get_codegen_implementation(options),
                            backend.get_module_map(), context)
    module.context = context
    prepared = prepare_from_module(module, target=target, metadata=metadata, factor=factor,
        source_ref=source_ref, compiler_commit=compiler_commit,
        specialization_ref='ASTSource:'+source.hash(), runtime_scalars=runtime_scalars)
    return replace(prepared, preparation_ns=perf_counter_ns()-started)


def prepare_from_module(module, *, target, metadata, factor, source_ref,
                        compiler_commit, specialization_ref, runtime_scalars=None):
    """Prepare one private copy of raw frontend TTIR; never mutate its caller.

    The native make_ttir entry can use this without regenerating Python AST IR.
    This is not an entry for already optimized/selected TTIR snapshots.
    """
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from triton._C.libtriton import ir
    if not source_ref or not specialization_ref or compiler_commit != '7c56a5e40f7fd928dfd5c72902d5def0097db73a':
        raise ValueError('preparation source/compiler binding is not current')
    started = perf_counter_ns()
    options = native_options(metadata, target)
    if runtime_scalars is not None and not isinstance(runtime_scalars, str):
        runtime_scalars = json.dumps(runtime_scalars, sort_keys=True, separators=(',', ':'))
    config = PreparationConfig(target.arch, options.num_warps, options.num_stages,
                               (factor, 1, 1), runtime_scalars=runtime_scalars)
    # The binding exposes a parser but no ModuleOp.clone. Keep the same context
    # and preserve the source module for Original and the other Bridge choices.
    context = module.context
    with TemporaryDirectory(prefix='l-core-raw-ttir-') as directory:
        path = Path(directory)/'source.ttir'
        path.write_text(str(module))
        module = ir.parse_mlir_module(str(path), context)
    module.context = context
    prepared = prepare_module(module, config)
    native_payload = {k:v for k,v in options.__dict__.items()
                      if not k.startswith(('hbv_', 'loop_bridge_'))}
    return replace(prepared, preparation_ns=perf_counter_ns()-started,
                   source_specialization_ref=specialization_ref,
                   native_options_ref=options.hash(),
                   native_options_json=json.dumps(native_payload, sort_keys=True,
                                                  separators=(',', ':'), allow_nan=False))
