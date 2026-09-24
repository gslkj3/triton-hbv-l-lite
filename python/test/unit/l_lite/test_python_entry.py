"""Python source to selected native backend; requires explicit compiler build."""
import os
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get('L_CORE_TEST_EXTENSION'),
                               reason='explicit compiler build required')


@pytest.mark.parametrize('bridge_factor', [2, 4])
@pytest.mark.parametrize('programs', [16, 28])
def test_python_bridge_candidate_reaches_native_backend(bridge_factor, programs):
    import triton
    import triton.language as tl
    from triton._C import libtriton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from pathlib import Path
    from triton.l_lite.core.compiler import CompilerBinding
    from triton.l_lite.core.candidate_frontend import prepare_python_candidates
    from triton.l_lite.core.state import Route
    from triton.l_lite.core.selected_materialization import compile_selected

    assert Path(libtriton.__file__).resolve() == Path(os.environ['L_CORE_TEST_EXTENSION']).resolve()

    @triton.jit
    def kernel(output, SCALE: tl.constexpr):
        p = tl.program_id(0)
        tl.store(output + p, p * SCALE + 1)

    target = GPUTarget('cuda', 89, 32)
    source = ASTSource(kernel, {'output': '*i32', 'SCALE': 'constexpr'},
                       constexprs={'SCALE': 3})
    commit = '7c56a5e40f7fd928dfd5c72902d5def0097db73a'
    population = prepare_python_candidates(source, target=target, options={},
        bridge_factors=(bridge_factor,), route_factors={Route.REORDER:(bridge_factor,)},
        binding=CompilerBinding(source.hash(), commit, 'explicit-build'))
    assert not population.preparation.failures
    assert not population.candidates.rejections
    assert len(population.candidates.candidates) == 1
    bound = population.candidates.candidates[0]
    prepared = bound.upstream_input
    assert population.original_input.upstream_factor == 1
    compiled, materialized = compile_selected(bound, target=target)
    assert materialized.input_sha256 == prepared.route_input_ir_sha256
    assert compiled.asm['cubin']
    assert all(stage in compiled.asm for stage in ('ttgir', 'llir', 'ptx'))
    if os.environ.get('L_CORE_TEST_GPU') == '1':
        import torch
        output = torch.full((programs,), -999, dtype=torch.int32, device='cuda')
        compiled[(programs // bridge_factor, 1, 1)](output)
        torch.cuda.synchronize()
        expected = torch.arange(programs, dtype=torch.int32, device='cuda') * 3 + 1
        assert torch.equal(output, expected)


@pytest.mark.parametrize('route_name', ['PIPELINE', 'REORDER', 'VECTORIZE'])
@pytest.mark.parametrize('block', [32, 64])
def test_hinted_python_loop_compiles_and_runs(route_name, block):
    import triton
    import triton.language as tl
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton.l_lite.core.compiler import CompilerBinding
    from triton.l_lite.core.candidate_frontend import prepare_python_candidates
    from triton.l_lite.core.state import Route
    from triton.l_lite.core.selected_materialization import compile_selected

    @triton.jit
    def kernel(x, y, BLOCK: tl.constexpr):
        lane = tl.arange(0, BLOCK)
        acc = tl.full((BLOCK,), 0, tl.int32)
        for i in tl.range(4, num_stages=3, loop_unroll_factor=2):
            v = tl.load(x + i * BLOCK + lane)
            acc = acc + v
        tl.store(y + lane, acc)

    source = ASTSource(kernel, {'x':'*i32', 'y':'*i32', 'BLOCK':'constexpr'},
                       constexprs={'BLOCK':block})
    target = GPUTarget('cuda',89,32)
    population = prepare_python_candidates(source, target=target, options={},
        bridge_factors=(1,), route_factors={getattr(Route,route_name):(2,)},
        binding=CompilerBinding(source.hash(),
            '7c56a5e40f7fd928dfd5c72902d5def0097db73a','explicit-build'))
    assert not population.candidates.rejections
    bound = population.candidates.candidates[0]
    assert 'tt.num_stages = 3' in bound.upstream_input.route_input_ir
    kernel_compiled, materialized = compile_selected(bound,target=target)
    assert 'tt.hbv.l.replaced_loop_guidance' in materialized.ttir
    assert kernel_compiled.asm['cubin']
    if os.environ.get('L_CORE_TEST_GPU') == '1':
        import torch
        x = torch.arange(block*4,dtype=torch.int32,device='cuda')
        y = torch.empty((block,),dtype=torch.int32,device='cuda')
        kernel_compiled[(1,1,1)](x,y)
        torch.cuda.synchronize()
        assert torch.equal(y,x.reshape(4,block).sum(dim=0))




@pytest.mark.skipif(os.environ.get('L_CORE_TEST_GPU')!='1',reason='GPU autotune test')
@pytest.mark.parametrize('programs,scale',[(16,2),(28,5)])
def test_native_autotune_uses_common_python_candidates(programs,scale):
    import triton
    import triton.language as tl
    import triton.testing
    import torch
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton.l_lite.core.compiler import CompilerBinding
    from triton.l_lite.core.lite_entry import autotune_kernel
    from triton.l_lite.core.state import Route

    @triton.jit
    def kernel(output, SCALE: tl.constexpr=2):
        p=tl.program_id(0)
        tl.store(output+p,p*SCALE+1)

    output=torch.empty((programs,),dtype=torch.int32,device='cuda')
    result=autotune_kernel(kernel,output,target=GPUTarget('cuda',89,32),
        bridge_factors=(2,4),route_factors={Route.REORDER:(2,4)},
        binding=CompilerBinding('bound-by-jit-entry','7c56a5e40f7fd928dfd5c72902d5def0097db73a',
                                'explicit-build'),grid=lambda args:(args['output'].numel(),),
        kernel_kwargs={} if scale==2 else {'SCALE':scale},
        do_bench=lambda call,quantiles: triton.testing.do_bench(call,warmup=1,rep=1,
                                                               quantiles=quantiles))
    torch.cuda.synchronize()
    assert torch.equal(output,torch.arange(programs,dtype=torch.int32,device='cuda')*scale+1)
    expected={'Original'}|{c.planned.candidate.identity for c in result.population.candidates.candidates}
    assert result.timings.keys()==expected
    assert {r['candidate'] for r in result.compilation_records}==expected
    assert all(not r['error'] for r in result.compilation_records)
    assert result.selected in expected
    assert not result.publication_permitted
