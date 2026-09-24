"""Typed launch binding at the Python/compiler boundary, never JSON in IR."""
import json


def runtime_binding_attr(payload, context):
    from triton._C.libtriton import ir
    # Existing caller snapshots may be JSON on disk; decode before IR transport.
    if isinstance(payload,str):
        payload=json.loads(payload)
    if not isinstance(payload,dict) or set(payload)!={'schema','grid','values_by_name'}:
        raise ValueError('runtime binding requires schema, grid and named values')
    grid=payload['grid']; values=payload['values_by_name']
    if (payload['schema']!='triton.loop-bridge.runtime-scalars.v2' or
        not isinstance(grid,(list,tuple)) or not 1<=len(grid)<=3 or
        any(type(v) is not int or not 1<=v<=2147483647 for v in grid) or
        not isinstance(values,dict) or
        any(not isinstance(k,str) or not k or type(v) is not int or
            not -(1<<63)<=v<(1<<63) for k,v in values.items())):
        raise ValueError('runtime binding fields outside typed domain')
    builder=ir.builder(context)
    return builder.get_dictionary_attr({
        'schema':builder.get_string_attr(payload['schema']),
        'grid':ir.make_attr(list(grid),context),
        'values_by_name':builder.get_dictionary_attr({
            k:builder.get_int64_attr(v) for k,v in values.items()})})
