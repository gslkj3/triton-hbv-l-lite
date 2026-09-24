"""Lossless original export plus explicitly non-authoritative reading copies."""
import json
from hashlib import sha256
from pathlib import Path
import re


_INTERNAL_STRING = re.compile(r'(?P<key>tt\.hbv\.[\w.]+)\s*=\s*"(?P<value>(?:\\.|[^"\\])*)"')


def export_ir(text, directory, *, name, stage):
    """Never change compiler input: abbreviations exist only in *.reading.mlir.

    Keep exact escaped values in the sidecar, including repeated per-operation
    attributes. This is a presentation export, not an MLIR rewrite pass.
    """
    if not stage or not name or Path(name).name != name:
        raise ValueError('explicit stage and simple artifact name required')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    original = directory / name
    reading = directory / (name + '.reading.mlir')
    report = directory / (name + '.attributes.json')
    if any(p.exists() for p in (original,reading,report)):
        raise FileExistsError('refusing to overwrite exported evidence')
    attributes=[]
    def abbreviate(match):
        raw=match['value']
        decoded=re.sub(r'\\(?:([0-9a-fA-F]{2})|(.))',
                       lambda m:chr(int(m[1],16)) if m[1] else m[2],raw)
        try:
            value=json.loads(decoded)
        except (ValueError,TypeError):
            value=decoded
        index=len(attributes)
        attributes.append(dict(key=match['key'],escaped_value=raw,value=value))
        return match['key']+' = "READING_COPY_ATTRIBUTE_'+str(index)+'"'
    readable=_INTERNAL_STRING.sub(abbreviate,text)
    original.write_text(text)
    reading.write_text('// READING COPY ONLY: internal string attributes abbreviated.\n'
                       '// NOT a compiler input or original artifact; see '+report.name+'\n'+readable)
    report.write_text(json.dumps(dict(stage=stage,original=original.name,
        original_sha256=sha256(text.encode()).hexdigest(),
        reading_copy_is_compiler_input=False,attributes=attributes),indent=2,ensure_ascii=False))
    return original,reading,report
