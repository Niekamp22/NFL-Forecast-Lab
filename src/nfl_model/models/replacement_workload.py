"""Experimental transfer of identified injury vacancies from the team reserve."""
import numpy as np
import json
from pathlib import Path


def replacement_policy():
    return json.loads(Path(__file__).with_suffix('.json').read_text(encoding='utf-8'))


def replacement_allocation(values, reserve, weights, positions, unavailable, eligible, total, strengths):
    """Same-position usage evidence only; never invent more team opportunities.

    `weights` includes the unavailable players' pre-injury workload estimates.
    `values` is the ordinary allocation after excluding unavailable players.
    Strengths are fixed using earlier-year evaluation, separately by position/stat.
    Unknown/excluded-player reserve is not automatically treated as an injury vacancy.
    """
    values=np.asarray(values,dtype=float).copy()
    weights=np.asarray(weights,dtype=float)
    if np.isinf(weights).any():
        raise ValueError('Invalid workload evidence')
    weights=np.maximum(0,np.nan_to_num(weights,nan=0))
    positions=np.asarray(positions)
    unavailable=np.asarray(unavailable,dtype=bool)
    eligible=np.asarray(eligible,dtype=bool)&~unavailable
    if len({len(values),len(weights),len(positions),len(unavailable),len(eligible)})!=1:
        raise ValueError('Replacement inputs must have matching lengths')
    if not np.isfinite([total,reserve]).all() or total<0 or reserve<0 or not np.isfinite(values).all() or (values<0).any():
        raise ValueError('Invalid opportunity budget')
    if not np.isclose(values.sum()+reserve,total):
        raise ValueError('Initial allocation does not reconcile with team budget')
    if any(not 0<=s<=1 for s in strengths.values()):
        raise ValueError('Replacement strength must be between zero and one')
    additions=np.zeros(len(values))
    denominator=max(float(weights.sum()),float(total),1e-12)
    proposals=[]
    for pos,strength in strengths.items():
        receivers=eligible&(positions==pos)&(weights>0)
        vacancy=total*float(weights[unavailable&(positions==pos)].sum())/denominator
        amount=strength*vacancy if receivers.any() else 0.
        proposals.append((receivers,amount))
    requested=sum(amount for _,amount in proposals)
    scale=min(1.,reserve/requested) if requested>0 else 0.
    for mask,amount in proposals:
        if amount>0:
            additions[mask]+=scale*amount*weights[mask]/weights[mask].sum()
    return values+additions,max(0.,reserve-float(additions.sum())),additions
