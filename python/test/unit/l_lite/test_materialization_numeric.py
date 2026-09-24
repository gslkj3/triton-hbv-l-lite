"""Run the public MLIR regressions through this tree's own CUDA backend.

Explicit opt-in: these are output-correctness checks, not performance tests.
No Main-L prediction or contract-construction package is imported.
"""
import os
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest
import triton
from triton._C import libtriton
from triton.backends.compiler import GPUTarget

ROOT = Path(__file__).resolve().parents[4]
pytestmark = pytest.mark.skipif(
    os.environ.get('L_LITE_RUN_NUMERIC') != '1', reason='Explicit CUDA test role required')


@pytest.mark.parametrize('fixture,kind', [
    ('hbv-loop-integer-carry-materialize.mlir', 'integer'),
    ('hbv-loop-packed-axis-hints.mlir', 'float'),
    ('hbv-loop-bridge-nested-capture.mlir', 'nested'),
])
def test_public_materialization_numeric(tmp_path, monkeypatch, fixture, kind):
    import torch
    assert Path(triton.__file__).resolve().is_relative_to(ROOT)
    assert Path(libtriton.__file__).resolve().is_relative_to(ROOT)
    assert torch.cuda.get_device_capability() == (8, 9)
    # Direct CMake builds need not install the CUDA language package symlink.
    from triton.language import extra
    cuda_language = str(ROOT / 'third_party/nvidia/language')
    if cuda_language not in extra.__path__:
        extra.__path__.append(cuda_language)
    monkeypatch.setenv('TRITON_CACHE_DIR', str(tmp_path / 'cache'))
    # Fixtures contain only source IR. Generate decisions from current facts.
    # This remains a handwritten-IR test, not the Python source entry test.
    from triton.l_lite.core.ttir import PreparationConfig, prepare_module
    from triton.l_lite.core.compiler import CompilerBinding, build_candidate
    from triton.l_lite.core.candidates import BoundCandidate
    from triton.l_lite.core.state import Route
    from triton.l_lite.core.selected_materialization import compile_selected
    from triton.l_lite.core.ir_export import export_ir
    text = (ROOT / 'test/Triton' / fixture).read_text()
    assert 'tt.hbv.plan_bundle' not in text
    source = tmp_path / 'source.ttir'
    source.write_text(text)
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    module = libtriton.ir.parse_mlir_module(str(source), context)
    module.context = context
    bridge_factor = 2 if kind == 'nested' else 1
    prepared = prepare_module(module, PreparationConfig(89,4,3,(bridge_factor,1,1)))
    prepared = replace(prepared,native_options_json=json.dumps(dict(num_warps=4,num_stages=3)))
    planned = build_candidate(provider=prepared.provider,
        route=Route.REORDER if kind == 'nested' else Route.VECTORIZE,
        factor=2 if kind == 'nested' else 4,identity='numeric:'+fixture,
        binding=CompilerBinding(prepared.source_ir_sha256,
            '7c56a5e40f7fd928dfd5c72902d5def0097db73a','public-numeric'))
    kernel, materialized = compile_selected(BoundCandidate(bridge_factor,prepared,planned),
                                            target=GPUTarget('cuda',89,32))
    assert 'tt.hbv.l.static_facts' not in prepared.route_input_ir
    assert prepared.provider.whole_kernel_payload()['extractable'] is True
    for artifact in (materialized.ttir,kernel.asm['ttgir']):
        assert 'tt.hbv.l.static_facts' not in artifact
        assert 'tt.hbv.l.backend_copy_width_facts' not in artifact
        assert 'tt.hbv.l.core_plan' not in artifact
        module_header = next(line for line in artifact.splitlines()
                             if line.startswith('module '))
        assert 'tt.loop_bridge.' not in module_header
        assert 'tt.hbv.l.' not in module_header
    # Match executable operation lines, never strings in analysis attributes.
    body='\n'.join(line for line in materialized.ttir.splitlines()
                   if re.match(r'\s*(?:%\S+\s*=|tt\.store|scf\.yield)',line))
    if kind == 'nested':
        assert body.count(' = scf.for ') == 2
        loads=[i for i,line in enumerate(body.splitlines()) if ' = tt.load ' in line]
        assert len(loads)==2 and loads[1]==loads[0]+1
    else:
        assert ' = scf.for ' not in body
        assert ' = tt.join ' in body and 'tt.store' in body
        if kind == 'integer':
            assert body.count(' = tt.load ') == 4
            reduced=re.search(r'(%\w+) = "tt.reduce"',body)
            assert reduced and re.search(r'arith.addi %\w+, '+re.escape(reduced[1]),body)
        else:
            assert re.search(r'arith.mulf .*tensor<\d+x\d+x\d+xf32>',body)
            assert not any(key in body for key in ('tt.contiguity','tt.divisibility','tt.constancy'))
    export_ir(materialized.ttir,tmp_path,name='materialized.ttir',stage='post-route verified TTIR')
    (tmp_path/'kernel.backend-observations.json').write_text(
        json.dumps(kernel.metadata.l_backend_observations,indent=2,sort_keys=True))
    for stage,value in kernel.asm.items():
        path=tmp_path/('kernel.'+stage)
        if stage in ('ttir','ttgir'):
            export_ir(value,tmp_path,name=path.name,stage=stage)
        else:
            path.write_bytes(value) if isinstance(value,bytes) else path.write_text(value)
    dtype = torch.int32 if kind == 'integer' else torch.float32
    programs = 8 if kind == 'nested' else 1
    x = torch.arange(1, programs * 4 * 16 + 1, dtype=dtype, device='cuda')
    saved = x.clone()
    storage = torch.full((programs * 16 + 32,), -123, dtype=dtype, device='cuda')
    y = storage[16:-16]
    expected = x.reshape(programs, 4, 16).sum(dim=1, dtype=dtype)
    if kind != 'nested':
        expected = 7 + 3 * expected
    kernel[(programs // 2 if kind == 'nested' else 1, 1, 1)](x, y)
    torch.cuda.synchronize()
    assert torch.equal(y, expected.reshape(-1))
    assert torch.equal(x, saved)
    assert torch.all(storage[:16] == -123) and torch.all(storage[-16:] == -123)
