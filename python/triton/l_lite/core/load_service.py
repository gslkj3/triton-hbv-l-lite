"""Ordinary-loop source service descriptions, not final async eligibility.

The width reported here is what source AxisInfo proves along each axis. Native
layout and later rewrites may change the scheduled width; no candidate is rejected.
"""
import math

SERVICE_KINDS=('scalar','wide_tensor','narrow_tensor','unknown')


def conditional_innermost_buffer_bytes(facts, stages):
    """Source-only buffer hypothesis with sequential reuse, not kernel total.

    Excludes reduction/layout scratch and unselected outer services. None means
    this projection is unavailable, not that the candidate is infeasible.
    """
    from .state import Loop, innermost
    if type(stages)is not int or stages<1:
        raise ValueError('positive integer stage count required')
    if stages==1:return 0
    census=facts.get('loop_census');accesses=facts.get('native_pointer_axis_facts')
    if not isinstance(census,list) or not isinstance(accesses,list):return None
    leaves=innermost(tuple(Loop(l['locator'],l.get('parent_locator') or None,None)for l in census))
    if not leaves:return None
    owned={l['locator']:l.get('entry_owned') for l in census}
    if any(owned[l.locator] is not True for l in leaves):return None
    parts=[conditional_loop_buffer_payload(accesses,l.locator)['payload_bytes']for l in leaves]
    if any(p is None for p in parts):return None
    return max(parts)*(stages-1)


def conditional_loop_buffer_payload(accesses, locator):
    """Payload if source width survives and wide loads are pipelined.

    Reuses planning-cut AxisInfo, not backend load ordinals or kernel identity.
    This is neither final eligibility nor a capacity rejection certificate.
    Unknown width, payload, or nested control is not silently zero-filled.
    """
    description=describe_loop_loads(accesses,locator)
    if not description['all_direct']:
        return dict(payload_bytes=None, reason='nested_control_requires_schedule', loads=[])
    by_ordinal={a['memory_ordinal']:a for a in accesses}
    rows=[]
    for load in description['loads']:
        access=by_ordinal[load['memory_ordinal']]
        widths=load['source_proven_width_bits_by_axis']
        bits=access.get('element_bits')
        if widths is None or type(bits)is not int or bits%8:
            return dict(payload_bytes=None,reason='unknown_source_width_or_byte_size',loads=rows)
        included=max(widths)>=32
        shape=[] if access.get('scalar_pointer') is True else access.get('pointer_shape')
        if not isinstance(shape,list) or any(type(n)is not int or n<=0 for n in shape):
            return dict(payload_bytes=None,reason='unknown_source_shape',loads=rows)
        payload=math.prod(shape)*bits//8 if included else 0
        rows.append(dict(memory_ordinal=load['memory_ordinal'],payload_bytes=payload,
                         source_width_bits=max(widths),conditionally_buffered=included))
    return dict(payload_bytes=sum(r['payload_bytes'] for r in rows),reason=None,loads=rows,
                final_async_eligibility='unresolved')


def source_service_coordinates(members):
    """Potential source load invocations, not issued transactions or traffic."""
    result={}
    for kind in SERVICE_KINDS:
        total=0
        for member in members:
            values=[member.get(n) for n in ('invocations_per_program','source_iterations','source_'+kind+'_loads')]
            if any(v==0 for v in values):continue
            if any(type(v)is not int or v<0 for v in values):break
            total+=math.prod(values)
        else:
            result['log1p_source_'+kind+'_load_invocations']=math.log1p(total)
    return result


def describe_loop_loads(accesses, locator):
    if not isinstance(locator, str) or not locator:
        raise ValueError('ordinary loop locator required')
    selected=[]
    for access in accesses:
        if access.get('operation')!='tt.load':
            continue
        ancestry=access.get('loop_ancestry_innermost_first')
        if ancestry is None:
            raise ValueError('memory operation missing loop ownership')
        if not ancestry or ancestry[0]!=locator:
            continue
        bits=access.get('element_bits')
        valid_bits=type(bits)is int and bits>0
        kind='unknown'
        widths=None
        if access.get('scalar_pointer') is True and valid_bits:
            kind='scalar'
            widths=[bits]
        elif access.get('scalar_pointer') is False and access.get('available') is True and valid_bits:
            cont=access.get('contiguity_elements');div=access.get('divisibility_bytes')
            shape=access.get('pointer_shape');masked=access.get('mask_present')
            mask=access.get('mask_constancy_elements') if masked else None
            arrays=[cont,div,shape]+([mask] if masked else [])
            complete=(type(masked)is bool and isinstance(shape,list) and bool(shape)
                      and all(isinstance(a,list) and len(a)==len(shape) for a in arrays)
                      and all(type(n)is int and n>0 for a in arrays for n in a))
            if complete:
                widths=[min(c,max(d//max(bits//8,1),1),s,
                            mask[i] if masked else s)*bits
                        for i,(c,d,s)in enumerate(zip(cont,div,shape))]
                kind='tensor_source_wide' if max(widths)>=32 else 'tensor_source_narrow'
        selected.append(dict(memory_ordinal=access['memory_ordinal'],kind=kind,
                             source_proven_width_bits_by_axis=widths,
                             direct_for_body=access.get('direct_for_body')))
    counts={k:sum(r['kind']==k for r in selected)
            for k in ('scalar','tensor_source_wide','tensor_source_narrow','unknown')}
    return dict(loads=selected,counts=counts,final_async_eligibility='unresolved',
                all_direct=all(r['direct_for_body'] is True for r in selected))
