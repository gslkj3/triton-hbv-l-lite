"""Run the public MLIR regressions through this tree's own CUDA backend.

Explicit opt-in: these are output-correctness checks, not performance tests.
No Main-L prediction or contract-construction package is imported.
"""
import os
from pathlib import Path
import subprocess

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
    command = [str(ROOT / 'build/l-lite-public/bin/triton-opt'),
               str(ROOT / 'test/Triton' / fixture)]
    if kind == 'nested':
        command += ['-triton-loop-bridge-discover', '-triton-loop-bridge-program-coarsening']
    command += ['-triton-hbv-loop-facts', '-triton-hbv-loop-decision',
                '-triton-loop-unroll', '-triton-hbv-loop-materialize',
                '-triton-hbv-validate-loop-plan']
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=900)
    ir_path = tmp_path / 'materialized.ttir'
    ir_path.write_text(result.stdout)
    kernel = triton.compile(str(ir_path), target=GPUTarget('cuda', 89, 32),
                            options={'num_warps': 4, 'num_stages': 3})
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
