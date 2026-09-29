"""The narrowly validated WR rate policy for newly frozen forecasts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .player_refinement import evidence_rows,refine_allocations


def refinement_policy():
    return json.loads(Path(__file__).with_name('receiving_stability.json').read_text())


def apply_current_refinement(players,budgets,history,cutoff):
    policy=refinement_policy()
    if policy['rate_strengths']!={'WR':{'receiving_yards':20}} or policy['workload_blend']!=0:
        raise ValueError('Operational refinement requires a reviewed WR-only policy')
    evidence=evidence_rows(players,history,cutoff)
    out,updated=refine_allocations(players,budgets,evidence,policy['rate_strengths'])
    keys=['game_id','team','player_id']
    facts=evidence[evidence.stat.eq('receiving_yards')].set_index(keys)
    aligned=facts.reindex(pd.MultiIndex.from_frame(out[keys]))
    personal=(players.receiving_yards/players.targets.where(players.targets.gt(0))).to_numpy()
    count=aligned.opportunities.to_numpy();prior=aligned.prior_rate.to_numpy()
    applied=(~out.is_unallocated & out.position.eq('WR')).to_numpy()&np.isfinite(personal)&(count>0)&np.isfinite(prior)
    out['receiving_rate_stabilized']=applied
    for name,values in [('receiving_personal_rate',personal),('receiving_prior_rate',prior),('receiving_rate_opportunities',count),('receiving_rate_prior_strength',np.full(len(out),20.))]:
        out[name]=np.where(applied,values,np.nan)
    return out,updated,policy
