"""Research-only coherent allocation of a proposed team passing-TD expectation."""
import numpy as np
from .player_roles import audit_allocations


def reconcile_passing_tds(players,budgets,targets):
    out=players.copy();updated=budgets.copy()
    for team,target in targets.items():
        if not np.isfinite(target) or target<0:raise ValueError('Invalid TD target')
        g=out[out.team.eq(team)];reserve=g.index[g.is_unallocated]
        if len(reserve)!=1:raise ValueError('Require team reserve')
        eligible=g.receiving_tds.notna()&g.receptions.notna()
        ids=g.index[eligible];capacity=g.loc[ids,'receptions'].clip(lower=0).to_numpy()
        target=min(target,float(capacity.sum()))
        weights=g.loc[ids,'receiving_tds'].clip(lower=0).to_numpy()
        if weights.sum()==0:weights=capacity.copy()
        allocation=np.zeros(len(ids));left=target
        for _ in range(len(ids)+1):
            room=np.maximum(0,capacity-allocation);active=room>1e-12
            if left<=1e-10 or not active.any():break
            w=np.where(active,weights,0.)
            if w.sum()==0:w=room
            added=np.minimum(room,left*w/w.sum());allocation+=added;left-=added.sum()
        if left>1e-7:raise ValueError('TD allocation did not converge')
        out.loc[ids,'receiving_tds']=allocation
        passer=g.index[~g.is_unallocated&g.passing_tds.notna()]
        if len(passer)>1:raise ValueError('Multiple passing roles')
        out.loc[reserve[0],'passing_tds']=0.
        out.loc[passer[0] if len(passer) else reserve[0],'passing_tds']=allocation.sum()
        updated.loc[updated.team.eq(team),['passing_tds','receiving_tds']]=allocation.sum()
    audit_allocations(out,updated)
    return out,updated
