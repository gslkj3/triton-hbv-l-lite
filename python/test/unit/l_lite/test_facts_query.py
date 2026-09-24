"""Analysis results are returned to the caller without modifying compiler IR."""
import json
import os
from pathlib import Path
import pytest


def test_query_is_read_only_and_matches_analysis(tmp_path):
    from triton._C.libtriton import ir, passes
    context = ir.context()
    ir.load_dialects(context)
    path = tmp_path / 'source.mlir'
    path.write_text("""module {
      tt.func public @kernel(%out: !tt.ptr<i32>) {
        %p = tt.get_program_id x : i32
        %dst = tt.addptr %out, %p : !tt.ptr<i32>, i32
        tt.store %dst, %p : !tt.ptr<i32>
        tt.return
      }
    }""")
    module = ir.parse_mlir_module(str(path), context)
    before = str(module)
    facts = passes.ttir.query_l_planning_facts(module)
    assert isinstance(facts, dict)
    assert facts['extractable'] is True
    assert facts['loop_census'] == []
    assert str(module) == before
    # Obsolete pass cannot silently recreate the removed IR transport.
    pm = ir.pass_manager(context)
    passes.ttir.add_hbv_loop_facts(pm)
    with pytest.raises(RuntimeError):
        pm.run(module, 'reject-obsolete-report-pass')
    assert 'tt.hbv.l.static_facts' not in str(module)


def test_query_reports_missing_entry_without_mutating_ir(tmp_path):
    from triton._C.libtriton import ir, passes
    context = ir.context()
    ir.load_dialects(context)
    path = tmp_path / 'empty.mlir'
    path.write_text('module {}')
    module = ir.parse_mlir_module(str(path), context)
    before = str(module)
    assert passes.ttir.query_l_planning_facts(module) == {
        'schema': 'l.core.planning-facts.v1',
        'extractable': False,
        'reason': 'entry_not_unique',
    }
    assert str(module) == before


@pytest.mark.parametrize('query_name', ['query_l_planning_facts', 'query_l_bridge_discovery'])
def test_helper_axis_analysis_does_not_annotate_input(tmp_path, query_name):
    from triton._C.libtriton import ir, passes
    context = ir.context()
    ir.load_dialects(context)
    path = tmp_path / 'helper.mlir'
    path.write_text('''module {
      tt.func private @helper(%ptr: !tt.ptr<i32>) {
        %value = tt.load %ptr : !tt.ptr<i32>
        tt.store %ptr, %value : !tt.ptr<i32>
        tt.return
      }
      tt.func public @kernel(%ptr: !tt.ptr<i32>) {
        tt.call @helper(%ptr) : (!tt.ptr<i32>) -> ()
        tt.return
      }
    }''')
    module = ir.parse_mlir_module(str(path), context)
    before = str(module)
    query = getattr(passes.ttir, query_name)
    first = query(module)
    assert str(module) == before
    assert query(module) == first
    assert str(module) == before


@pytest.mark.parametrize('fixture', (
    'hbv-loop-integer-carry-materialize.mlir',
    'hbv-loop-packed-axis-hints.mlir',
    'hbv-loop-bridge-nested-capture.mlir',
))
def test_materialization_sources_have_identical_analysis_without_ir_report(fixture):
    from triton._C.libtriton import ir, passes
    root = os.environ.get('L_CORE_QUERY_FIXTURE_ROOT')
    if not root:
        pytest.skip('requires declared current source-only fixture directory')
    context = ir.context()
    ir.load_dialects(context)
    module = ir.parse_mlir_module(str(Path(root) / fixture), context)
    before = str(module)
    facts = passes.ttir.query_l_planning_facts(module)
    assert str(module) == before
    assert facts['extractable']
    assert facts == passes.ttir.query_l_planning_facts(module)
    assert str(module) == before
