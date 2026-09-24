import pytest
from triton.l_lite.core.runtime_binding_attr import runtime_binding_attr


def test_typed_binding_is_consumed_without_json(tmp_path):
    from triton._C.libtriton import ir,passes
    context=ir.context();ir.load_dialects(context)
    path=tmp_path/'source.mlir'
    path.write_text('''module {
      tt.func public @kernel(%out: !tt.ptr<i32> loc("out"), %n: i32 loc("n")) {
        %p = tt.get_program_id x : i32
        %dst = tt.addptr %out, %p : !tt.ptr<i32>, i32
        tt.store %dst, %n : !tt.ptr<i32>
        tt.return
      }
    }''')
    module=ir.parse_mlir_module(str(path),context)
    attr=runtime_binding_attr(dict(schema='triton.loop-bridge.runtime-scalars.v2',
                                  grid=[8],values_by_name={'n':-7}),context)
    module.set_attr('tt.loop_bridge.runtime_scalars',attr)
    before=str(module)
    assert 'runtime_scalars = {' in before and 'n = -7 : i64' in before
    pm=ir.pass_manager(context);passes.ttir.add_loop_bridge_discover(pm)
    pm.run(module,'typed-runtime-binding')
    result=str(module)
    assert 'tt.loop_bridge.bound_scalar = -7 : i64' in result
    assert 'tt.loop_bridge.axis_extent = 8 : i64' in result
    module.set_attr('tt.loop_bridge.runtime_scalars',ir.builder(context).get_string_attr('{}'))
    pm=ir.pass_manager(context);passes.ttir.add_loop_bridge_discover(pm)
    with pytest.raises(RuntimeError):
        pm.run(module,'reject-legacy-runtime-binding')
    assert 'tt.loop_bridge.bound_scalar' not in str(module)


@pytest.mark.parametrize('grid,values', [([],{}),([0],{}),([1<<31],{}),
                                      ([1],{'n':1<<63}),([1],{'n':True})])
def test_bad_python_binding_never_enters_ir(grid,values):
    from triton._C.libtriton import ir
    context=ir.context();ir.load_dialects(context)
    with pytest.raises(ValueError):
        runtime_binding_attr(dict(schema='triton.loop-bridge.runtime-scalars.v2',
                                  grid=grid,values_by_name=values),context)
