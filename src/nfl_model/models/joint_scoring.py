"""Reconcile expected offensive events to an explicit team points budget.

This transform requires a separately estimated allowance for conversions and
non-offensive scoring. It never treats the discrepancy itself as that allowance.
"""
import numpy as np
import pandas as pd
from .player_roles import FIELDS,audit_allocations


def scoring_policy():
    return dict(version='shared_points_v1',history_days=365,
                other_points='Prior completed league games: scoreboard minus offensive TD, FG and PAT points; nonnegative mean',
                target='Frozen independent team score; reconcile offensive TDs and PATs within opportunity caps',
                selection='Accounting consistency; reused 2024-25 Weeks 2-9 comparison improves QB/WR TD and PAT MAE, with small TE TD regression. No team-score accuracy claim.')


def league_other_points(history,home_teams):
    """Estimate an allowance from actual prior events, never model disagreement."""
    if history.empty:raise ValueError('Need prior scoring observations')
    cols=['receiving_tds','rushing_tds','fg_made','pat_made']
    t=history.groupby(['game_id','team'],as_index=False)[cols].sum(min_count=1)
    scores=history.groupby('game_id')[['home_score','away_score']].first()
    if not set(t.game_id).issubset(set(home_teams.index)):raise ValueError('Missing home-team evidence')
    actual=np.array([scores.loc[r.game_id,'home_score' if home_teams.loc[r.game_id]==r.team else 'away_score'] for r in t.itertuples()])
    residual=actual-6*(t.receiving_tds+t.rushing_tds)-3*t.fg_made-t.pat_made
    if not np.isfinite(residual).all():raise ValueError('Incomplete historical scoring components')
    return max(0.,float(residual.mean()))


def audit_joint_scoring(players):
    for _,g in players.groupby(['game_id','team']):
        if g.scoring_team_points.nunique()!=1 or g.scoring_other_points.nunique()!=1:
            raise ValueError('Inconsistent shared score metadata')
        points=6*(g.receiving_tds.sum()+g.rushing_tds.sum())+3*g.fg_made.sum()+g.pat_made.sum()+g.scoring_other_points.iloc[0]
        if not np.isclose(points,g.scoring_team_points.iloc[0],atol=1e-7):raise ValueError('Shared team score does not reconcile')


def reconcile_scoring(players,budgets,score_targets):
    out=players.copy();updated=budgets.copy();details=[]
    if score_targets.duplicated(['game_id','team']).any():raise ValueError('Duplicate score targets')
    keys=['game_id','team']
    if set(map(tuple,score_targets[keys].to_numpy()))!=set(map(tuple,players[keys].drop_duplicates().to_numpy())):
        raise ValueError('Score targets must cover the exact player cohort')
    for row in score_targets.itertuples():
        if not np.isfinite([row.score,row.other_points]).all() or min(row.score,row.other_points)<0:
            raise ValueError('Score targets must be finite and nonnegative')
        ix=out.index[out.game_id.eq(row.game_id)&out.team.eq(row.team)]
        if not len(ix):raise ValueError('Missing team allocation')
        reserve=ix[out.loc[ix,'is_unallocated']]
        if len(reserve)!=1:raise ValueError('Require one reserve per team')
        before=out.loc[ix]
        touchdowns=float(before.receiving_tds.sum()+before.rushing_tds.sum())
        if touchdowns<=0:raise ValueError('No supported touchdown allocation')
        pat_attempt_per_td=min(1.,max(0.,float(before.pat_att.sum())/touchdowns))
        pat_made_per_td=min(pat_attempt_per_td,max(0.,float(before.pat_made.sum())/touchdowns))
        other=min(float(row.other_points),float(row.score))
        fg_points=3*float(before.fg_made.sum())
        fg_scale=min(1.,max(0.,row.score-other)/fg_points) if fg_points>0 else 1.
        target_td=max(0.,row.score-other-fg_points*fg_scale)/(6+pat_made_per_td)
        factor=target_td/touchdowns
        # A common multiplier preserves relative event rates until an event hits
        # its opportunity cap; remaining TDs are redistributed among supported
        # nonzero rates. Missing player estimates never become numerical.
        event_columns=['receiving_tds','rushing_tds']
        values=before[event_columns].to_numpy(float)
        caps=before[['receptions','carries']].to_numpy(float)
        if (np.isfinite(values)&(values>0)&~np.isfinite(caps)).any():
            raise ValueError('Touchdown estimate has no supported opportunity capacity')
        supported=np.isfinite(values)&np.isfinite(caps)&(values>0)
        capacity=np.where(supported,np.maximum(caps,0),0)
        if target_td>capacity.sum()+1e-7:
            raise ValueError('Point budget exceeds supported touchdown capacity')
        weights=np.where(supported,values,0)
        low=0.;high=max(1.,factor)
        while np.minimum(weights*high,capacity).sum()<target_td-1e-10:high*=2
        for _ in range(60):
            middle=(low+high)/2
            if np.minimum(weights*middle,capacity).sum()<target_td:low=middle
            else:high=middle
        allocated=np.minimum(weights*((low+high)/2),capacity)
        # Supported zeros stay zero; missing stats retain NaN.
        allocated=np.where(np.isfinite(values),allocated,np.nan)
        out.loc[ix,event_columns]=allocated
        for stat in ['fg_att','fg_made']:out.loc[ix,stat]*=fg_scale
        for stat,total in [('pat_att',target_td*pat_attempt_per_td),('pat_made',target_td*pat_made_per_td)]:
            old=float(before[stat].sum())
            if old>0:out.loc[ix,stat]=before[stat]*(total/old)
            else:
                # Unsupported kicker roles remain in the explicit team reserve.
                out.loc[ix,stat]=before[stat]*0
                out.loc[reserve[0],stat]=total
        resolved=ix[~out.loc[ix,'is_unallocated']&out.loc[ix,'passing_tds'].notna()]
        if len(resolved)>1:raise ValueError('Multiple passing roles')
        out.loc[reserve[0],'passing_tds']=0.
        destination=resolved[0] if len(resolved) else reserve[0]
        out.loc[destination,'passing_tds']=out.loc[ix,'receiving_tds'].sum()
        points=6*(out.loc[ix,'receiving_tds'].sum()+out.loc[ix,'rushing_tds'].sum())+3*out.loc[ix,'fg_made'].sum()+out.loc[ix,'pat_made'].sum()+other
        if not np.isclose(points,row.score,atol=1e-7):raise ValueError('Scoring budget does not reconcile')
        out.loc[ix,'scoring_td_multiplier']=factor
        out.loc[ix,'scoring_other_points']=other
        out.loc[ix,'scoring_team_points']=row.score
        details.append(dict(game_id=row.game_id,team=row.team,score=row.score,other_points=other,td_multiplier=factor,fg_multiplier=fg_scale))
    for i,row in updated.iterrows():
        g=out[out.game_id.eq(row.game_id)&out.team.eq(row.team)]
        for stat in FIELDS:updated.loc[i,stat]=g[stat].sum()
    audit_allocations(out,updated)
    audit_joint_scoring(out)
    return out,updated,pd.DataFrame(details)
