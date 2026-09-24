"""Stable textual IR identity across loss of transient SSA printing names.

Native parsing retains operations, types, attributes and locations, but source
name hints for loop block arguments are not serialized. Normalize through the
same parser before hashing; never strip instructions or semantic attributes.
"""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory


def canonical_ir_text(module):
    from triton._C.libtriton import ir
    with TemporaryDirectory(prefix='l-core-ir-identity-') as directory:
        path = Path(directory)/'identity.ttir'
        path.write_text(str(module))
        copy = ir.parse_mlir_module(str(path), module.context)
    return str(copy)


def ir_digest(module):
    return sha256(canonical_ir_text(module).encode()).hexdigest()
