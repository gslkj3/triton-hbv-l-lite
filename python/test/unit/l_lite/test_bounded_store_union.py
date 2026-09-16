import json
import re
import subprocess
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[4]
OPT=ROOT/'build/l-lite-public/bin/triton-opt'


def fixture(grid=4,offset=32,stride=8,binding=True,schema=2,y=1):
    payload=json.dumps(dict(schema=f'triton.loop-bridge.runtime-scalars.v{schema}',grid=[grid,y],
        **({'values_by_name':{}} if schema==2 else {'values':{}})),separators=(',',':'))
    escaped=payload.replace('"','\\22')
    attrs=f' attributes {{tt.loop_bridge.runtime_scalars = "{escaped}"}}' if binding else ''
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
    p=tmp_path/'test.mlir';p.write_text(fixture(**kwargs))
    run=subprocess.run([str(OPT),str(p),'-triton-loop-bridge-discover'],capture_output=True,text=True,timeout=30)
    assert run.returncode==0,run.stderr
    raw,=re.findall(r'tt\.loop_bridge\.discovery = "((?:[^"\\]|\\.)*)"',run.stdout)
    decoded=re.sub(r'\\([0-9A-Fa-f]{2}|.)',lambda m:chr(int(m[1],16)) if len(m[1])==2 else m[1],raw)
    facts=json.loads(decoded)
    assert facts['construction_legal']==expected,facts
