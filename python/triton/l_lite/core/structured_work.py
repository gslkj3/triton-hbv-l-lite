"""Compiler-owned container work bounds, not DRAM traffic or latency."""
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass(frozen=True)
class ByteBounds:
    lower: int
    upper: Optional[int]

    def __post_init__(self):
        if (type(self.lower) is not int or self.lower < 0 or
                self.upper is not None and
                (type(self.upper) is not int or self.upper < self.lower)):
            raise ValueError('Malformed container byte bounds')


@dataclass(frozen=True)
class StructuredWorkNode:
    kind: str
    children: Tuple['StructuredWorkNode', ...] = ()
    load_bytes: int = 0
    store_bytes: int = 0
    trip_count: Optional[int] = None

    def __post_init__(self):
        if self.kind not in {'leaf', 'sequence', 'loop', 'if_else', 'unknown'}:
            raise ValueError('Unsupported structured work node')
        if not isinstance(self.children, tuple) or any(not isinstance(c, StructuredWorkNode) for c in self.children):
            raise ValueError('Malformed structured children')
        if any(type(v) is not int or v < 0 for v in (self.load_bytes, self.store_bytes)):
            raise ValueError('Malformed container bytes')
        if self.trip_count is not None and (type(self.trip_count) is not int or self.trip_count < 0):
            raise ValueError('Malformed loop trip count')
        if self.kind != 'leaf' and (self.load_bytes or self.store_bytes):
            raise ValueError('Only leaves own direct work')
        if self.kind != 'loop' and self.trip_count is not None:
            raise ValueError('Only loops own trip counts')
        if ((self.kind in {'leaf', 'unknown'} and self.children) or
                (self.kind == 'loop' and len(self.children) != 1) or
                (self.kind == 'if_else' and len(self.children) != 2)):
            raise ValueError('Malformed structured arity')


def structured_work_from_fact(fact):
    """Decode the compiler-owned tree with no topology inference or defaults
    for required branch children. Missing optional leaf counts mean zero.
    """
    budget = [4096]
    def decode(value, depth):
        budget[0] -= 1
        if budget[0] < 0 or depth > 128:
            raise ValueError('Structured work decoding budget exceeded')
        if not isinstance(value, dict) or 'kind' not in value or set(value)-{
                'kind', 'children', 'load_bytes', 'store_bytes', 'trip_count'}:
            raise ValueError('Malformed structured work fact')
        children=value.get('children', [])
        if not isinstance(children, list):
            raise ValueError('Malformed structured fact children')
        return StructuredWorkNode(value['kind'], tuple(decode(c, depth+1) for c in children),
            value.get('load_bytes', 0), value.get('store_bytes', 0), value.get('trip_count'))
    return decode(fact, 0)


def structured_container_bounds(root: StructuredWorkNode):
    """Return independent load/store bounds; no joint latency implication.

    An if without else must explicitly carry an empty sequence as its else.
    Unknown regions never silently contribute zero. Zero-trip loops annihilate
    work even when their unexecuted body cannot be evaluated.
    """
    budget = [4096]

    def walk(node, depth):
        budget[0] -= 1
        if budget[0] < 0 or depth > 128:
            raise ValueError('Structured work evaluation budget exceeded')
        if node.kind == 'unknown':
            return ByteBounds(0, None), ByteBounds(0, None)
        if node.kind == 'leaf':
            return ByteBounds(node.load_bytes, node.load_bytes), ByteBounds(node.store_bytes, node.store_bytes)
        if node.kind == 'loop' and node.trip_count == 0:
            return ByteBounds(0, 0), ByteBounds(0, 0)
        children = [walk(child, depth+1) for child in node.children]
        def combine(axis):
            values = [v[axis] for v in children]
            if node.kind == 'sequence':
                return ByteBounds(sum(v.lower for v in values),
                    None if any(v.upper is None for v in values) else sum(v.upper for v in values))
            if node.kind == 'if_else':
                return ByteBounds(min(v.lower for v in values),
                    None if any(v.upper is None for v in values) else max(v.upper for v in values))
            value = values[0]
            if node.trip_count is None:
                return ByteBounds(0, 0 if value.upper == 0 else None)
            return ByteBounds(value.lower*node.trip_count,
                None if value.upper is None else value.upper*node.trip_count)
        return combine(0), combine(1)

    return walk(root, 0)
