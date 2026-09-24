"""Materialize only the chosen ordinary candidate after prediction.

This is the TTIR suffix, not the whole backend or a numerical validation. The
prepared input is content-checked and the compiler verifies the plan again.
"""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import tempfile
from .projection import project_direct_state


@dataclass(frozen=True)
class MaterializedSelection:
    candidate_ref: str
    input_sha256: str
    output_sha256: str
    ttir: str
    pass_order: tuple[str,...]
    publication_permitted: bool = False


def materialize_selected(bound, *, extension):
    prepared=bound.upstream_input
    text=prepared.route_input_ir
    digest=sha256(text.encode()).hexdigest()
    if digest!=prepared.route_input_ir_sha256:
        raise ValueError('selected preparation content changed')
    project_direct_state(prepared.provider,bound.planned)
    ir=extension.ir
    context=ir.context()
    ir.load_dialects(context)
    with tempfile.TemporaryDirectory(prefix='l-core-selected-') as directory:
        path=Path(directory)/'selected-input.mlir'
        path.write_text(text)
        module=ir.parse_mlir_module(str(path),context)
    if extension.passes.ttir.has_l_decision(module):
        raise ValueError('prepared input already contains a decision')
    module.set_attr('tt.hbv.plan_bundle',extension.passes.ttir.make_l_decision_attribute(
        module,bound.planned.plan.to_dict()))
    manager=ir.pass_manager(context)
    passes=extension.passes.ttir
    order=('add_hbv_loop_decision','add_loop_unroll',
           'add_hbv_loop_materialize','add_hbv_validate_loop_plan')
    for name in order:
        getattr(passes,name)(manager)
    manager.run(module,'l_core_selected_materialization')
    entry=module.get_function(module.get_entry_func_name())
    if entry.get_operation().get_str_attr('tt.hbv.l.postcondition')!='pass':
        raise ValueError('selected route lacks materialization postcondition')
    result=str(module)
    return MaterializedSelection(bound.planned.candidate.identity,digest,
        sha256(result.encode()).hexdigest(),result,order)


def compile_selected(bound, *, target):
    """Continue the selected, validated TTIR through all native backend stages.

TTIR file input deliberately skips native make_ttir: its prefix was executed
by prepare_from_source and its decision/unroll suffix by materialize_selected.
This function does not autotune, launch, or silently fall back to Original.
"""
    import json
    import triton
    from triton._C import libtriton
    from .frontend import native_options
    prepared=bound.upstream_input
    if target.backend!='cuda' or target.arch!=prepared.config.capability:
        raise ValueError('selected backend target differs from preparation')
    if not prepared.native_options_json:
        raise ValueError('selected compilation requires bound native options')
    materialized=materialize_selected(bound,extension=libtriton)
    with tempfile.TemporaryDirectory(prefix='l-core-selected-backend-') as directory:
        path=Path(directory)/'selected.ttir'
        path.write_text(materialized.ttir)
        options=native_options(json.loads(prepared.native_options_json),target)
        kernel=triton.compile(str(path),target=target, options=options.__dict__)
    return kernel,materialized
