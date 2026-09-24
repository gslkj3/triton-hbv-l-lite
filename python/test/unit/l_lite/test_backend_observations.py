"""Scoped backend diagnostics must not become IR state or cross compilations."""
from concurrent.futures import ThreadPoolExecutor
import pytest

SOURCE = """#layout = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "cuda:89"} {
 tt.func public @kernel(%ptr: tensor<128x!tt.ptr<f32>, #layout>, %lb: index, %ub: index, %step: index) {
  scf.for %i = %lb to %ub step %step {
   %value = tt.load %ptr : tensor<128x!tt.ptr<f32>, #layout>
   tt.store %ptr, %value : tensor<128x!tt.ptr<f32>, #layout>
  }
  tt.return
 }
}"""


def setup(tmp_path, name):
    from triton._C.libtriton import ir, passes
    context = ir.context()
    ir.load_dialects(context)
    path = tmp_path/(name+'.mlir')
    path.write_text(SOURCE)
    module = ir.parse_mlir_module(str(path), context)
    manager = ir.pass_manager(context)
    passes.ttgpuir.add_pipeline(manager, 3, False)
    return context, module, manager, passes.ttir.run_with_l_backend_observations


def test_backend_observation_has_no_ir_attribute(tmp_path):
    context, module, manager, run = setup(tmp_path,'plain')
    report = run(manager,module,'pipeline')
    assert report['records'] and report['records'][0]['loads']
    assert report['stage'] == 'pipeline'
    assert 'tt.hbv.l.backend_copy_width_facts' not in str(module)


def test_exception_and_nested_capture_restore_outer_scope(tmp_path):
    outer_context, outer, outer_pm, run = setup(tmp_path,'outer')
    inner_context, inner, inner_pm, _ = setup(tmp_path,'inner')
    class InnerFailure:
        def run(self, module, stage):
            inner_pm.run(module,stage)
            raise RuntimeError('deliberate test interruption')
    class Outer:
        def run(self, module, stage):
            with pytest.raises(RuntimeError,match='deliberate test interruption'):
                run(InnerFailure(),inner,'inner')
            outer_pm.run(module,stage)
    report = run(Outer(),outer,'outer')
    assert len(report['records']) == 1
    assert report['stage'] == 'outer'
    # Another compilation after both scopes have exited gets its own report.
    next_context, next_module, next_pm, _ = setup(tmp_path,'next')
    assert len(run(next_pm,next_module,'next')['records']) == 1


def test_threaded_compilations_have_independent_records(tmp_path):
    def task(index):
        context, module, manager, run = setup(tmp_path,'thread'+str(index))
        return run(manager,module,'thread'+str(index))
    with ThreadPoolExecutor(max_workers=2) as pool:
        reports = list(pool.map(task,range(2)))
    assert [r['stage'] for r in reports] == ['thread0','thread1']
    assert all(len(r['records']) == 1 for r in reports)
