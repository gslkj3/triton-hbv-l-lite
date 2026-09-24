import pytest


def fixture(grid=4,offset=32,stride=8,binding=True,schema=2,y=1):
    attrs=(f' attributes {{tt.loop_bridge.runtime_scalars = '
           f'{{schema = "triton.loop-bridge.runtime-scalars.v{schema}", '
           f'grid = dense<[{grid}, {y}, 1]> : tensor<3xi32>, values_by_name = {{}}}}}}'
           if binding else '')
    return 'module'+attrs+''' {
tt.func public @partition(%out: !tt.ptr<f32>) {
%pid = tt.get_program_id x : i32
'''+f'%stride = arith.constant {stride} : i32\n%offset = arith.constant dense<{offset}> : tensor<8xi32>\n'+'''
%base = arith.muli %pid, %stride : i32
%bases = tt.splat %base : i32 -> tensor<8xi32>
%lane = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
%index = arith.addi %bases, %lane : tensor<8xi32>
%second = arith.addi %index, %offset : tensor<8xi32>
%ptrs = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
%a = tt.addptr %ptrs, %index : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
%b = tt.addptr %ptrs, %second : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
%value = arith.constant dense<1.0> : tensor<8xf32>
tt.store %a, %value : tensor<8x!tt.ptr<f32>>
tt.store %b, %value : tensor<8x!tt.ptr<f32>>
tt.return
}}
'''


@pytest.mark.parametrize('kwargs,expected',[
    ({},True),({'offset':40},True),({'offset':24},False),
    ({'grid':5},False),({'binding':False},False),({'schema':1},False),
    ({'y':2},False),({'grid':2147483647},False),
    ({'grid':2,'stride':1073741824,'offset':1073741824},False),
    ({'offset':0,'binding':False},True),
])
def test_native_domain(tmp_path,kwargs,expected):
    from triton._C.libtriton import ir, passes
    p=tmp_path/'test.mlir';p.write_text(fixture(**kwargs))
    context = ir.context()
    ir.load_dialects(context)
    module = ir.parse_mlir_module(str(p), context)
    manager = ir.pass_manager(context)
    passes.ttir.add_loop_bridge_discover(manager)
    if kwargs.get('schema', 2) != 2:
        with pytest.raises(RuntimeError):
            manager.run(module, 'invalid-runtime-binding')
        return
    manager.run(module, 'runtime-binding')
    before = str(module)
    facts = passes.ttir.query_l_bridge_discovery(module)
    assert str(module) == before
    assert 'tt.loop_bridge.discovery' not in before
    assert facts['construction_legal']==expected,facts
