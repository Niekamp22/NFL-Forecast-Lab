"""Team defense factors applied consistently to frozen role allocations."""
import hashlib
import json
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
from .player_roles import FIELDS,checked_player_snapshot,audit_allocations
from .player_matchup import split_context,fit_split,apply_split


def adjust_allocations(players,budgets,factors):
    out=players.copy();totals=budgets.copy()
    for row in factors.itertuples():
        mask=out.team.eq(row.team)
        for stat in FIELDS:out.loc[mask,'baseline_'+stat]=out.loc[mask,stat]
        for stat in ['attempts','targets','completions','receptions','passing_tds','receiving_tds','passing_interceptions']:
            out.loc[mask,stat]*=row.pass_volume_multiplier
        for stat in ['passing_yards','receiving_yards']:
            out.loc[mask,stat]*=row.pass_volume_multiplier*row.pass_rate_multiplier
        for stat in ['carries','rushing_tds']:out.loc[mask,stat]*=row.rush_volume_multiplier
        out.loc[mask,'rushing_yards']*=row.rush_volume_multiplier*row.rush_rate_multiplier
        for field in ['pass_volume_multiplier','pass_rate_multiplier','rush_volume_multiplier','rush_rate_multiplier','defense_status']:
            out.loc[mask,field]=getattr(row,field)
        for stat in FIELDS:totals.loc[totals.team.eq(row.team),stat]=out.loc[mask,stat].sum()
        reserve=out[mask&out.is_unallocated]
        totals.loc[totals.team.eq(row.team),'unallocated_targets']=reserve.targets.sum()
        totals.loc[totals.team.eq(row.team),'unallocated_carries']=reserve.carries.sum()
    audit_allocations(out,totals)
    return out,totals


def freeze_defense_roles(root,season,week):
    root=Path(root);target=root/'player_role_defense_forecasts'/str(season)/f'week_{week:02d}'
    if list(target.glob('*/manifest.json')):return checked_player_snapshot(target)[0]
    original,meta,players=checked_player_snapshot(root/'player_role_forecasts'/str(season)/f'week_{week:02d}')
    now=pd.Timestamp.now(tz='UTC')
    if now>=pd.Timestamp(meta['earliest_kickoff']):raise ValueError('Cannot freeze after kickoff')
    if season<=2025:raise ValueError('2022–2023 training / 2024–2025 evaluation require a later prospective season')
    sources=[Path(p) for p in meta['input_hashes']]
    for path in sources:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['input_hashes'][str(path)]:raise ValueError(f'Pinned source changed: {path}')
    schedule=pd.read_parquet(next(p for p in sources if p.name=='games.parquet'))
    schedule=schedule[schedule.game_type.eq('REG')].copy()
    schedule['kickoff']=pd.to_datetime(schedule.gameday.astype(str)+' '+schedule.gametime.astype(str)).dt.tz_localize('America/New_York').dt.tz_convert('UTC')
    stats=pd.concat([pd.read_parquet(p) for p in sources if p.name.startswith('stats_player_week_')],ignore_index=True)
    stats=stats[stats.season_type.eq('REG')]
    if stats.duplicated(['game_id','player_id']).any():raise ValueError('Duplicate player stats')
    stats=stats.merge(schedule[['game_id','kickoff','home_team','away_team','home_score','away_score']],on='game_id',validate='many_to_one')
    cutoff=pd.Timestamp(meta['created_at'])
    stats=stats[stats.kickoff.lt(cutoff)&stats.home_score.notna()&stats.away_score.notna()].copy()
    if not (stats.team.eq(stats.home_team)|stats.team.eq(stats.away_team)).all():raise ValueError('Team mismatch')
    stats['opponent']=np.where(stats.team.eq(stats.home_team),stats.away_team,stats.home_team)
    team_games=stats.groupby(['season','week','game_id','team','opponent','kickoff'],as_index=False)[['targets','receiving_yards','carries','rushing_yards']].sum(min_count=1)
    models={};metrics=[];histories=[];contexts_now={}
    for kind,opportunity,yards in [('pass','targets','receiving_yards'),('rush','carries','rushing_yards')]:
        allowed=team_games.rename(columns={'opponent':'defense',opportunity:'opportunities',yards:'yards'})
        rows=[]
        for team,group in team_games.groupby('team'):
            group=group.sort_values('kickoff')
            for index,r in enumerate(group.itertuples()):
                if r.season not in [2022,2023,2024,2025]:continue
                past=group.iloc[max(0,index-10):index]
                past=past[past.kickoff.ge(r.kickoff-pd.Timedelta(days=365))&past.kickoff.lt(r.kickoff)]
                if len(past)<3:continue
                recent=past.tail(5)
                if recent[opportunity].isna().any():continue
                volume=float(np.average(recent[opportunity],weights=np.arange(1,len(recent)+1)))
                complete=past.dropna(subset=[opportunity,yards])
                rate=complete[yards].sum()/complete[opportunity].sum() if complete[opportunity].sum()>0 else np.nan
                context=split_context(allowed,r.opponent,r.kickoff)
                rows.append(dict(season=r.season,week=r.week,game_id=r.game_id,team=team,opponent=r.opponent,
                                 base_volume=volume,base_rate=rate,actual_opportunities=getattr(r,opportunity),actual=getattr(r,yards),**context))
        frame=pd.DataFrame(rows);model=fit_split(frame);models[kind]=model
        v,rate,prediction=apply_split(frame.base_volume,frame.base_rate,frame.volume_factor,frame.rate_factor,model)
        frame['baseline']=frame.base_volume*frame.base_rate;frame['adjusted']=prediction;frame['kind']=kind
        evaluation=frame[frame.season.isin([2024,2025])].dropna(subset=['baseline','adjusted','actual'])
        metrics.append(dict(kind=kind,games=len(evaluation),baseline_mae=(evaluation.baseline-evaluation.actual).abs().mean(),
                            adjusted_mae=(evaluation.adjusted-evaluation.actual).abs().mean(),**model))
        histories.append(frame)
        for team,group in players.groupby('team'):
            opponent=group.opponent.iloc[0]
            context=split_context(allowed,opponent,cutoff)
            valid=np.isfinite(context['volume_factor']) and np.isfinite(context['rate_factor'])
            contexts_now[(team,kind)]=dict(context,volume_multiplier=max(0,1+model['volume']*context['volume_factor']) if valid else 1.,
                                         rate_multiplier=max(0,1+model['rate']*context['rate_factor']) if valid else 1.,valid=valid)
    factors=[]
    for team,group in players.groupby('team'):
        row=dict(team=team,opponent=group.opponent.iloc[0],defense_status='Adjusted')
        for kind in ['pass','rush']:
            c=contexts_now[(team,kind)]
            for key,value in c.items():row[kind+'_'+key]=value
            if not c['valid']:row['defense_status']='Partial or missing defense history; unavailable factors unchanged'
        factors.append(row)
    factors=pd.DataFrame(factors)
    budgets_path=original/'team_budgets.parquet'
    adjusted,budgets=adjust_allocations(players,pd.read_parquet(budgets_path),factors)
    folder=target/(now.strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8]);folder.mkdir(parents=True,exist_ok=False)
    adjusted.to_parquet(folder/'predictions.parquet',index=False);budgets.to_parquet(folder/'team_budgets.parquet',index=False)
    factors.to_csv(folder/'defense_factors.csv',index=False);pd.DataFrame(metrics).to_csv(folder/'evaluation.csv',index=False)
    pd.concat(histories,ignore_index=True).to_parquet(folder/'historical_predictions.parquet',index=False)
    info=dict(meta,created_at=str(now),baseline_snapshot=str(original.resolve()),feature_cutoff=str(cutoff),models=models,
              method='Experimental defense-adjusted role allocation: team-level opportunity and yardage efficiency multipliers; fixed player shares, including reserves',
              selection_reason='Opt-in comparison. Train 2022–2023, reused evaluation 2024–2025 at team level only; not a validation of historical player role allocations.',
              limits='Passing multiplier learned on targets and also applied to attempts, catches, TDs and interceptions at unchanged per-opportunity rates. Rushing TDs scale with carries. No direct defense TD, interception, catch-rate or kicker adjustment. Team score forecast unchanged.',
              input_hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources+[original/'predictions.parquet',original/'manifest.json',budgets_path]},
              predictions_sha256=hashlib.sha256((folder/'predictions.parquet').read_bytes()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (folder/'manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    return folder
