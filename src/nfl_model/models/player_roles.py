"""Conserved team opportunities with explicit unresolved role shares."""
import hashlib
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from .weekly_forecast import earliest_snapshot
from ..season_weighting import weights as season_weights,policy as season_policy

FIELDS = ['attempts','completions','passing_yards','passing_tds','passing_interceptions',
          'targets','receptions','receiving_yards','receiving_tds','carries','rushing_yards','rushing_tds','fg_att','fg_made','pat_att','pat_made']


ROLE_POLICY = 'current_season_qb_v4'


def checked_player_snapshot(root, current=False):
    folder, meta = earliest_snapshot(root)
    if current:
        candidates=[]
        for path in Path(root).glob('*/manifest.json'):
            item=json.loads(path.read_text())
            if item.get('role_policy')==ROLE_POLICY and pd.Timestamp(item['created_at'])<pd.Timestamp(item['earliest_kickoff']):
                candidates.append((pd.Timestamp(item['created_at']),str(path.parent),item))
        if candidates:
            _,selected,meta=min(candidates,key=lambda x:x[:2]);folder=Path(selected)
    path = folder/'predictions.parquet'
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta['predictions_sha256']:
        raise ValueError('Player prediction checksum mismatch')
    return folder, meta, pd.read_parquet(path)


def allocate(total, weights, historical_total):
    """Reserve excluded/unknown role shares; never amplify supported weights."""
    weights = np.maximum(0,np.nan_to_num(np.asarray(weights,dtype=float),nan=0))
    denominator = max(float(weights.sum()), float(historical_total), 1e-12)
    values = total*weights/denominator
    return values, max(0,float(total-values.sum()))


def resolve_current_qb(depth_ids,latest_qbs,eligible_ids):
    """Depth QB1 plus a substantive majority in the latest game; not confirmation."""
    if len(depth_ids)!=1 or latest_qbs.empty:return None
    usage=latest_qbs.groupby('player_id').attempts.sum(min_count=1).dropna()
    if usage.empty or usage.sum()<=0:return None
    leader=usage.idxmax()
    if leader==depth_ids[0] and leader in set(eligible_ids) and usage[leader]>=10 and usage[leader]/usage.sum()>.5:
        return leader
    return None


def audit_allocations(players, teams):
    for row in teams.itertuples():
        group = players[players.team.eq(row.team)]
        for stat in FIELDS:
            if not np.isclose(group[stat].sum(),getattr(row,stat),atol=1e-7):
                raise ValueError(f'{row.team}: {stat} does not reconcile')
        if not np.isclose(group.passing_yards.sum(),group.receiving_yards.sum()):
            raise ValueError('Passing and receiving yards differ')
        if not np.isclose(group.completions.sum(),group.receptions.sum()):
            raise ValueError('Completions and catches differ')
        if not np.isclose(group.passing_tds.sum(),group.receiving_tds.sum()):
            raise ValueError('Passing and receiving TDs differ')
        if group.targets.sum() > group.attempts.sum()+1e-7:
            raise ValueError('Targets exceed pass attempts')


def current_stint(history, player_id, team):
    observed=history[history.player_id.eq(player_id)].sort_values('kickoff')
    other=observed[~observed.team.eq(team)]
    if not other.empty:observed=observed[observed.kickoff.gt(other.kickoff.max())]
    return observed[observed.team.eq(team)].tail(5)


def observed_workload(prior, team_games, stat, budget, season=None):
    """Recency-weighted shares over observed games only; missing rows aren't zeros."""
    pairs=prior[['game_id','kickoff',stat]+(['season'] if season is not None else [])].merge(team_games[['game_id',stat]],on='game_id',suffixes=('_player','_team'),validate='one_to_one').sort_values('kickoff')
    pairs=pairs.dropna(subset=[stat+'_player',stat+'_team'])
    pairs=pairs[pairs[stat+'_team'].gt(0)]
    if pairs.empty:return 0.
    shares=(pairs[stat+'_player']/pairs[stat+'_team']).clip(0,1)
    w=season_weights(pairs,season,'player_share') if season is not None else np.arange(1,len(pairs)+1)
    return float(np.average(shares,weights=w)*budget)


def freeze_role_forecast(root, season, week):
    root=Path(root)
    target=root/'player_role_forecasts'/str(season)/f'week_{week:02d}'
    if list(target.glob('*/manifest.json')):
        saved,saved_meta,_=checked_player_snapshot(target,current=True)
        if saved_meta.get('role_policy')==ROLE_POLICY:return saved
    original,meta,long=checked_player_snapshot(root/'player_opportunity_forecasts'/str(season)/f'week_{week:02d}')
    now=pd.Timestamp.now(tz='UTC')
    if now>=pd.Timestamp(meta['earliest_kickoff']): raise ValueError('Cannot freeze after kickoff')
    sources=[Path(p) for p in meta['input_hashes']]
    for p in sources:
        if hashlib.sha256(p.read_bytes()).hexdigest()!=meta['input_hashes'][str(p)]: raise ValueError(f'Pinned source changed: {p}')
    roster_path=next(p for p in sources if p.name=='player_week.parquet')
    depth_path=roster_path.parent/'player_week_depth.parquet'
    sources.append(depth_path)
    roster=pd.read_parquet(roster_path)
    roster=roster[roster.season.eq(season)&roster.week.eq(week)]
    depth=pd.read_parquet(depth_path)
    depth=depth[depth.season.eq(season)&depth.week.eq(week)&depth.depth_timestamp.le(now)]
    schedule=pd.read_parquet(next(p for p in sources if p.name=='games.parquet'))
    schedule=schedule[schedule.game_type.eq('REG')].copy()
    schedule['kickoff']=pd.to_datetime(schedule.gameday.astype(str)+' '+schedule.gametime.astype(str)).dt.tz_localize('America/New_York').dt.tz_convert('UTC')
    history=pd.concat([pd.read_parquet(p) for p in sources if p.name.startswith('stats_player_week_')],ignore_index=True)
    history=history[history.season_type.eq('REG')]
    history=history.merge(schedule[['game_id','kickoff','home_score','away_score']],on='game_id',validate='many_to_one')
    cutoff=pd.Timestamp(meta['created_at'])
    history=history[history.kickoff.lt(cutoff)&history.kickoff.ge(cutoff-pd.Timedelta(days=365))&history.home_score.notna()&history.away_score.notna()]
    if history.duplicated(['game_id','player_id']).any(): raise ValueError('Duplicate player stats')
    team_games=history.groupby(['team','game_id','kickoff','season'],as_index=False)[FIELDS].sum(min_count=1)
    current=schedule[schedule.season.eq(season)&schedule.week.eq(week)]
    if current.empty or current.kickoff.isna().any() or current.kickoff.le(now).any(): raise ValueError('Invalid target schedule')
    if current[['home_score','away_score']].notna().any().any(): raise ValueError('Target results already present')
    outputs=[];budgets=[]
    for team, observations in long.groupby('team'):
        games=team_games[team_games.team.eq(team)].sort_values('kickoff').tail(5)
        if len(games)<3: raise ValueError(f'{team}: fewer than three historical team games')
        if games[FIELDS].isna().any().any(): raise ValueError('Incomplete team opportunity history')
        weights=season_weights(games,season,'team_volume'); weights/=weights.sum()
        totals={s:float(games[s].to_numpy()@weights) for s in FIELDS}
        totals['targets']=min(totals['targets'],totals['attempts'])
        candidates=roster[roster.team.eq(team)&roster.position.isin(['QB','RB','WR','TE','K'])].copy()
        # Current eligibility does not certify game-day participation.
        candidates=candidates[candidates.roster_status.eq('ACT')&~candidates.injury_status.eq('Out').fillna(False)]
        game=current[current.home_team.eq(team)|current.away_team.eq(team)].iloc[0]
        teamhist=history[history.team.eq(team)&history.game_id.isin(games.game_id)]
        old_weights={gid:w for gid,w in zip(games.game_id,weights)}
        qb_depth=depth[depth.team.eq(team)&depth.pos_abb.eq('QB')&pd.to_numeric(depth.pos_rank).eq(1)].player_id.dropna().unique()
        qb_stats=teamhist[teamhist.position.eq('QB')].copy()
        qb_stats['weighted_attempts']=qb_stats.attempts*qb_stats.game_id.map(old_weights)
        leader=qb_stats.groupby('player_id').weighted_attempts.sum().idxmax() if not qb_stats.empty else None
        # Depth and last game's actual usage must agree, too: old starters may have changed.
        latest_qb=qb_stats[qb_stats.game_id.eq(games.iloc[-1].game_id)]
        last_leader=latest_qb.loc[latest_qb.attempts.idxmax(),'player_id'] if not latest_qb.empty else None
        qb=resolve_current_qb(qb_depth,latest_qb,candidates.player_id)
        player_rows=[]
        for r in candidates.itertuples():
            prior=current_stint(history,r.player_id,team)
            baseline=observations[observations.player_id.eq(r.player_id)]
            row=dict(season=season,week=week,game_id=game.game_id,kickoff_utc=game.kickoff,team=team,
                     opponent=game.away_team if team==game.home_team else game.home_team,
                     player_id=r.player_id,player_name=r.player_name,position=r.position,is_unallocated=False,
                     role='Likely starting QB (unconfirmed)' if r.player_id==qb else 'QB role unresolved' if r.position=='QB' else 'Recent team usage',
                     history_games=int(prior.game_id.nunique()),
                     expected_snap_share=getattr(r,'prior_offensive_snap_pct_3',np.nan),
                     review_flags='Conditional role estimate; game-day active status unknown'+('; No recent usage on this team' if prior.empty else '')+('; Injury report unknown' if pd.isna(r.injury_status) else f'; Injury: {r.injury_status}'))
            if r.position=='QB' and qb is None: row['review_flags']+='; Depth and usage do not establish a starter'
            for s in FIELDS: row[s]=np.nan
            for opportunity in ['targets','carries']:
                row['_'+opportunity]=observed_workload(prior,team_games[team_games.team.eq(team)],opportunity,totals[opportunity],season)
            for s in ['receptions','receiving_yards','receiving_tds','rushing_yards','rushing_tds','fg_att','fg_made','pat_att','pat_made']:
                found=baseline[baseline.stat.eq(s)]
                row['_rate_'+s]=float(found.efficiency.iloc[0]) if len(found) and s not in ['fg_att','pat_att'] else np.nan
                row['_estimate_'+s]=float(found.projection.iloc[0]) if len(found) else np.nan
            # Reweight efficiency evidence separately from current-team workload.
            efficiency=history[history.player_id.eq(r.player_id)].sort_values('kickoff').tail(10)
            for stat,opportunity in [('receptions','targets'),('receiving_yards','targets'),('receiving_tds','targets'),('rushing_yards','carries'),('rushing_tds','carries'),('fg_made','fg_att'),('pat_made','pat_att')]:
                valid=efficiency.dropna(subset=[stat,opportunity])
                w=season_weights(valid,season,'player_rate',recency=False)
                denominator=float(valid[opportunity]@w)
                row['_rate_'+stat]=float(valid[stat]@w)/denominator if denominator>0 else np.nan
            player_rows.append(row)
        p=pd.DataFrame(player_rows)
        if p.empty: raise ValueError(f'{team}: no eligible candidates')
        reserve={**{c:np.nan for c in p.columns},'season':season,'week':week,'game_id':game.game_id,'kickoff_utc':game.kickoff,
                 'team':team,'opponent':p.opponent.iloc[0],'player_id':f'UNALLOCATED_{team}', 'player_name':'Unallocated / unresolved roles',
                 'position':'TEAM','is_unallocated':True,'role':'Unassigned workload','history_games':len(games),'review_flags':'Excluded players, missing rates, or unresolved starter; not a named-player prediction',**{s:0.0 for s in FIELDS}}
        for opportunity,derived in [('targets',['receptions','receiving_yards','receiving_tds']),('carries',['rushing_yards','rushing_tds'])]:
            values,remaining=allocate(totals[opportunity],p['_'+opportunity],float(games[opportunity].to_numpy()@weights))
            known=p.history_games.gt(0)
            p.loc[known,opportunity]=values[known.to_numpy()]
            reserve[opportunity]=remaining
            for stat in derived:
                rate=p['_rate_'+stat]
                default=totals[stat]/totals[opportunity] if totals[opportunity]>0 else 0
                valid=known&rate.notna()
                p.loc[valid,stat]=values[valid.to_numpy()]*rate[valid]
                # Unknown efficiency retains its opportunities but production stays in reserve.
                reserve[stat]=(remaining+values[(~valid).to_numpy()].sum())*default
        # Coherent completions and receiving TDs with bounded per-target rates.
        p['receptions']=np.minimum(p.receptions,p.targets)
        p['receiving_tds']=np.minimum(p.receiving_tds,p.receptions)
        reserve['receptions']=min(reserve['receptions'],reserve['targets']+float(p.targets[p.receptions.isna()].sum()))
        reserve['receiving_tds']=min(reserve['receiving_tds'],reserve['receptions'])
        for stat,target_stat in [('completions','receptions'),('passing_yards','receiving_yards'),('passing_tds','receiving_tds')]:
            totals[stat]=float(p[target_stat].sum()+reserve[target_stat])
        for stat in ['attempts','completions','passing_yards','passing_tds','passing_interceptions']:
            if qb is not None: p.loc[p.player_id.eq(qb),stat]=totals[stat]
            else: reserve[stat]=totals[stat]
        # Kicker budgets use the same opportunity allocation rule.
        for attempted,made in [('fg_att','fg_made'),('pat_att','pat_made')]:
            k=p.position.eq('K')
            kw=p.loc[k,'_estimate_'+attempted].fillna(0).to_numpy()
            assigned,left=allocate(totals[attempted],kw,totals[attempted])
            p.loc[k,attempted]=assigned
            rate=p.loc[k,'_rate_'+made].clip(0,1)
            p.loc[k,made]=assigned*rate
            reserve[attempted]=left
            reserve[made]=(left+assigned[rate.isna().to_numpy()].sum())*(totals[made]/totals[attempted] if totals[attempted] else 0)
        p=pd.concat([p,pd.DataFrame([reserve])],ignore_index=True)
        p=p.drop(columns=[c for c in p if c.startswith('_')])
        outputs.append(p)
        budgets.append(dict(team=team,game_id=game.game_id,team_history_games=len(games),qb_resolved=qb is not None,
                            **{s:float(p[s].sum()) for s in FIELDS},unallocated_targets=float(reserve['targets']),unallocated_carries=float(reserve['carries'])))
    players=pd.concat(outputs,ignore_index=True);teams=pd.DataFrame(budgets)
    audit_allocations(players,teams)
    folder=target/(now.strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8]);folder.mkdir(parents=True,exist_ok=False)
    players.to_parquet(folder/'predictions.parquet',index=False);players.to_csv(folder/'predictions.csv',index=False)
    teams.to_parquet(folder/'team_budgets.parquet',index=False)
    info=dict(created_at=str(now),earliest_kickoff=meta['earliest_kickoff'],season=season,week=week,players=int((~players.is_unallocated).sum()),role_policy=ROLE_POLICY,season_weighting=season_policy(),
              source_forecast=str(original.resolve()),method='Five-game team budget; current-season-weighted observed-game current-stint workload shares; depth/usage QB agreement; ten-game player efficiency',
              selection_reason='Default operational view for coherent team/player totals, not a demonstrated accuracy winner. Defense experiments remain research options.',
              uncertainty='No calibrated intervals or active probabilities. Snap share is a prior-usage proxy, not a participation forecast. Named-player gaps stay missing; team reserve is explicit.',
              input_hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources+[original/'predictions.parquet']},
              predictions_sha256=hashlib.sha256((folder/'predictions.parquet').read_bytes()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (folder/'manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    return folder
