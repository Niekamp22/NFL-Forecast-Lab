"""Conserved, prior-only workload/rate refinements; disabled unless explicitly selected."""
import numpy as np
import pandas as pd
from .player_roles import FIELDS,audit_allocations

PAIRS={'receptions':'targets','receiving_yards':'targets','receiving_tds':'targets',
       'rushing_yards':'carries','rushing_tds':'carries','fg_made':'fg_att','pat_made':'pat_att'}


def evidence_rows(players,history,cutoff):
    if history.kickoff.ge(pd.Timestamp(cutoff)).any():raise ValueError('Refinement history must precede cutoff')
    rows=[]
    team=history.groupby(['team','game_id'])[['targets','carries']].sum(min_count=1)
    by_player={pid:g.sort_values('kickoff') for pid,g in history.groupby('player_id')}
    league=history.groupby('position')[list(PAIRS)+list(set(PAIRS.values()))].sum(min_count=1)
    for r in players[~players.is_unallocated].itertuples():
        h=by_player.get(r.player_id,history.iloc[:0]);eff=h.tail(10)
        other=h[~h.team.eq(r.team)];stint=h[h.team.eq(r.team)]
        if len(other):stint=stint[stint.kickoff.gt(other.kickoff.max())]
        for stat,opp in PAIRS.items():
            v=eff.dropna(subset=[stat,opp]);n=v[opp].sum()
            prior=league.loc[r.position,stat]/league.loc[r.position,opp] if r.position in league.index and league.loc[r.position,opp]>0 else np.nan
            rows.append(dict(game_id=r.game_id,team=r.team,player_id=r.player_id,stat=stat,opportunities=n,prior_rate=prior))
        for opp in ['targets','carries']:
            latest=stint.tail(2).dropna(subset=[opp]);shares=[]
            for v in latest.itertuples():
                total=team.loc[(r.team,v.game_id),opp]
                if total>0:shares.append(getattr(v,opp)/total)
            rows.append(dict(game_id=r.game_id,team=r.team,player_id=r.player_id,stat=opp,opportunities=len(shares),prior_rate=np.mean(shares) if shares else np.nan))
    return pd.DataFrame(rows)


def refine_allocations(players,budgets,evidence,rate_strengths=None,workload_blend=0.):
    """Apply a fixed experiment setting; all evidence must have been built pregame.

    Rates retain missing estimates. Workload changes are bounded by team budgets;
    unknown production stays in reserve. QB passing totals follow receiving totals.
    """
    if not 0<=workload_blend<=1:raise ValueError('Invalid workload blend')
    out=players.copy();e=evidence.set_index(['game_id','team','player_id','stat'])
    if not e.index.is_unique:raise ValueError('Duplicate refinement evidence')
    original_rates={s:out[s]/out[o].where(out[o].gt(0)) for s,o in PAIRS.items()}
    for team,g in out.groupby('team'):
        reserve=g.index[g.is_unallocated]
        if len(reserve)!=1:raise ValueError('Require one explicit team reserve')
        ri=reserve[0];named=g.index[~g.is_unallocated]
        if workload_blend:
            for opp,stats in [('targets',['receptions','receiving_yards','receiving_tds']),('carries',['rushing_yards','rushing_tds'])]:
                total=float(g[opp].sum());old=out.loc[named,opp].copy();new=old.copy()
                for i in named:
                    r=out.loc[i];key=(r.game_id,team,r.player_id,opp)
                    if pd.notna(old.loc[i]) and key in e.index and pd.notna(e.loc[key,'prior_rate']):
                        new.loc[i]=(1-workload_blend)*old.loc[i]+workload_blend*e.loc[key,'prior_rate']*total
                if new.sum()>total:new*=total/new.sum()
                out.loc[named,opp]=new;out.loc[ri,opp]=max(0,total-new.sum())
                for stat in stats:
                    default=float(g[stat].sum())/total if total else 0.
                    rates=original_rates[stat].loc[named]
                    out.loc[named,stat]=new*rates
                    # Preserve supported zero-volume predictions.
                    zeros=new.eq(0)&players.loc[named,stat].notna()
                    out.loc[named[zeros],stat]=0.
                    out.loc[ri,stat]=(out.loc[ri,opp]+new[rates.isna()].sum())*default
        for stat,opp in PAIRS.items():
            r=out.loc[named]
            strength=r.position.map(lambda pos:(rate_strengths or {}).get(pos,{}).get(stat,0)).to_numpy(float)
            if (strength<0).any() or not np.isfinite(strength).all():raise ValueError('Invalid rate strength')
            if not strength.any():continue
            keys=pd.MultiIndex.from_arrays([r.game_id,r.team,r.player_id,[stat]*len(r)])
            facts=e.reindex(keys);n=facts.opportunities.to_numpy();prior=facts.prior_rate.to_numpy()
            volume=r[opp].to_numpy();production=r[stat].to_numpy(copy=True)
            valid=(strength>0)&(n>0)&np.isfinite(prior)&(volume>0)&np.isfinite(production)
            production[valid]=(production[valid]*n[valid]+prior[valid]*strength[valid]*volume[valid])/(n[valid]+strength[valid])
            out.loc[named,stat]=production
        out.loc[g.index,'receptions']=np.minimum(out.loc[g.index,'receptions'],out.loc[g.index,'targets'])
        out.loc[g.index,'receiving_tds']=np.minimum(out.loc[g.index,'receiving_tds'],out.loc[g.index,'receptions'])
        for stat,target in [('completions','receptions'),('passing_yards','receiving_yards'),('passing_tds','receiving_tds')]:
            resolved=named[out.loc[named,stat].notna()]
            if len(resolved)>1:raise ValueError('Multiple resolved passing roles')
            destination=resolved[0] if len(resolved) else ri
            out.loc[ri,stat]=0.
            out.loc[destination,stat]=out.loc[g.index,target].sum()
    updated=budgets.copy()
    for i,r in updated.iterrows():
        g=out[out.team.eq(r.team)]
        for stat in FIELDS:updated.loc[i,stat]=g[stat].sum()
        for stat in ['targets','carries']:updated.loc[i,'unallocated_'+stat]=g.loc[g.is_unallocated,stat].iloc[0]
    audit_allocations(out,updated)
    return out,updated
