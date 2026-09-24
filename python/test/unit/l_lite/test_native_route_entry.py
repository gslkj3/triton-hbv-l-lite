"""Full native JIT pipeline with externally supplied main-L analysis."""
import os
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('L_CORE_TEST_GPU') != '1', reason='GPU integration')


@pytest.mark.parametrize('route_name', ['PIPELINE','REORDER','VECTORIZE'])
@pytest.mark.parametrize('block', [32,64])
@pytest.mark.parametrize('loop_kind', ['static','dynamic','nested'])
def test_native_three_routes_with_explicit_adviser(route_name, block, loop_kind, monkeypatch):
    """Known legal decision tests compiler wiring, NOT model selection quality."""
    from hashlib import sha256
    import torch
    import triton
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.l_lite.core.analysis_hook import register_adviser
    from triton.l_lite.core.candidate_frontend import prepare_module_candidates
    from triton.l_lite.core.decision_binding import bind_selected_decision
    from triton.l_lite.core.compiler import CompilerBinding
    from triton.l_lite.core.state import Route

    @triton.jit
    def kernel(x,y,N,BLOCK:tl.constexpr,DYNAMIC:tl.constexpr,NESTED:tl.constexpr):
        lane=tl.arange(0,BLOCK)
        acc=tl.full((BLOCK,),0,tl.int32)
        for outer in range(2 if NESTED else 1):
            for i in tl.range(N if DYNAMIC else 4,num_stages=3,loop_unroll_factor=2):
                acc += tl.load(x+(outer*(N if DYNAMIC else 4)+i)*BLOCK+lane)
        tl.store(y+lane,acc)

    identity=f'test-explicit-native-route-{route_name}-{block}-{loop_kind}-v2'
    def adviser(module,metadata,options,capability):
        ref=sha256(str(module).encode()).hexdigest()
        population=prepare_module_candidates(module,target=GPUTarget('cuda',capability,32),
            options=options.__dict__,bridge_factors=(1,),
            route_factors={Route[route_name]:(2,)},specialization_ref=ref,
            binding=CompilerBinding(ref,'7c56a5e40f7fd928dfd5c72902d5def0097db73a',identity),
            at_analysis_point=True)
        assert len(population.candidates.candidates)==1, population.candidates.rejections
        bound=population.candidates.candidates[0]
        prepared=bound.upstream_input
        metadata['l_grid_divisors']=bind_selected_decision(module,config=prepared.config,
            plan_json=bound.planned.plan.canonical_json(),source_sha256=prepared.source_ir_sha256,
            route_input_sha256=prepared.route_input_ir_sha256)
        metadata['l_test_route']=route_name
    register_adviser(identity,adviser)
    monkeypatch.setenv('TRITON_L_LITE_MODE','predict')
    monkeypatch.setenv('TRITON_L_ANALYSIS_REF',identity)
    trips=5 if loop_kind=='dynamic' else 4
    total=trips*(2 if loop_kind=='nested' else 1)
    x=torch.arange(total*block,device='cuda',dtype=torch.int32)
    y=torch.empty(block,device='cuda',dtype=torch.int32)
    compiled=kernel[(1,)](x,y,trips,BLOCK=block,
        DYNAMIC=loop_kind=='dynamic',NESTED=loop_kind=='nested')
    torch.cuda.synchronize()
    assert torch.equal(y,x.reshape(total,block).sum(0))
    assert compiled.metadata.l_test_route==route_name
    assert 'tt.hbv.l.replaced_loop_guidance' in compiled.asm['ttir']
    assert 'tt.hbv.l.postcondition = "pass"' in compiled.asm['ttir']
    if loop_kind=='nested' or route_name=='PIPELINE':
        assert 'scf.for' in compiled.asm['ttir'], 'outer/pipeline loop must survive'
    assert all(stage in compiled.asm for stage in ('ttgir','llir','ptx','cubin'))

