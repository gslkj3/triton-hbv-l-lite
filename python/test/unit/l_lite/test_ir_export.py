import json
import pytest
from triton.l_lite.core.ir_export import export_ir


def test_original_preserved_and_sidecar_decoded(tmp_path):
    text='module attributes {tt.hbv.l.static_facts = "{\\22loop_census\\22:[]}"} {\n}\n'
    original,reading,report=export_ir(text,tmp_path,name='materialized.ttir',stage='post-route TTIR')
    assert original.read_text()==text
    data=json.loads(report.read_text())
    assert data['attributes'][0]['value']=={'loop_census':[]}
    assert 'READING COPY ONLY' in reading.read_text()
    assert not data['reading_copy_is_compiler_input']
    with pytest.raises(FileExistsError):
        export_ir(text,tmp_path,name='materialized.ttir',stage='post-route TTIR')


def test_native_attributes_and_operations_are_not_changed(tmp_path):
    text='module attributes {"ttg.num-warps" = 4 : i32} {\n %0 = arith.constant 1 : i32\n}\n'
    _,reading,report=export_ir(text,tmp_path,name='kernel.ttgir',stage='TTGIR')
    assert reading.read_text().endswith(text)
    assert json.loads(report.read_text())['attributes']==[]
