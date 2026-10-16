"""Separately versioned, genuinely pre-kickoff tests for remaining weekly games."""
import hashlib
import json
from pathlib import Path
import uuid
import pandas as pd

from .models.player_roles import checked_player_snapshot,audit_allocations
from .models.joint_scoring import reconcile_scoring,league_other_points,scoring_policy,audit_joint_scoring
from .models.role_defense import adjust_allocations
from .player_views import load_pinned_history
from .dashboard import load_forecast


def testing_root(root,season,week,defense=False):
    return Path(root)/'player_testing_forecasts'/('defense' if defense else 'baseline')/str(season)/f'week_{week:02d}'


def future_game_ids(players,now):
    games=players[['game_id','kickoff_utc']].drop_duplicates()
    if games.game_id.duplicated().any():raise ValueError('Conflicting game kickoff times')
    return set(games.loc[pd.to_datetime(games.kickoff_utc,utc=True)>pd.Timestamp(now),'game_id'])


def load_testing_snapshot(root,season,week,defense=False):
    candidates=[]
    for path in testing_root(root,season,week,defense).glob('*/manifest.json'):
        m=json.loads(path.read_text())
        created=pd.Timestamp(m['created_at'])
        if m.get('forecast_scope')!='remaining_games_test' or created>=pd.Timestamp(m['earliest_kickoff']):
            raise ValueError('Invalid testing forecast timing or scope')
        if pd.Timestamp(m['history_cutoff'])>created:raise ValueError('Testing inputs are after creation')
        candidates.append((created,str(path.parent),m))
    if not candidates:return None
    _,folder,meta=min(candidates,key=lambda x:x[:2]);folder=Path(folder)
    for name,key in [('predictions.parquet','predictions_sha256'),('team_budgets.parquet','budgets_sha256')]:
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=meta[key]:raise ValueError('Testing forecast checksum mismatch')
    p=pd.read_parquet(folder/'predictions.parquet')
    if set(p.game_id)!=set(meta['game_ids']):raise ValueError('Testing game coverage mismatch')
    if (pd.to_datetime(p.kickoff_utc,utc=True)<=pd.Timestamp(meta['created_at'])).any():raise ValueError('Testing forecast contains a started game')
    original_family='player_role_defense_forecasts' if defense else 'player_role_forecasts'
    _,parent,_=checked_player_snapshot(Path(root)/original_family/str(season)/f'week_{week:02d}',current=True)
    if meta['parent_predictions_sha256']!=parent['predictions_sha256']:raise ValueError('Testing parent forecast changed')
    audit_allocations(p,pd.read_parquet(folder/'team_budgets.parquet'));audit_joint_scoring(p)
    return folder,meta,p


def create_testing_week(root,season,week):
    root=Path(root)
    existing=[load_testing_snapshot(root,season,week,d) for d in [False,True]]
    if any(existing):
        if not all(existing):raise ValueError('Incomplete testing pair; inspect before resuming')
        return [item[0] for item in existing]
    baseline,bm,p=checked_player_snapshot(root/'player_role_forecasts'/str(season)/f'week_{week:02d}',current=True)
    defense,dm,_=checked_player_snapshot(root/'player_role_defense_forecasts'/str(season)/f'week_{week:02d}',current=True)
    now=pd.Timestamp.now(tz='UTC');game_ids=future_game_ids(p,now)
    if not game_ids:raise ValueError('No future games remain for a prospective test')
    p=p[p.game_id.isin(game_ids)].copy()
    budgets=pd.read_parquet(baseline/'team_budgets.parquet');budgets=budgets[budgets.game_id.isin(game_ids)]
    for meta in [bm,dm]:
        for value,expected in meta['input_hashes'].items():
            if hashlib.sha256(Path(value).read_bytes()).hexdigest()!=expected:raise ValueError('Pinned testing source changed')
    history=load_pinned_history(bm)
    cutoff=pd.Timestamp(bm.get('history_cutoff',bm.get('feature_cutoff',bm['created_at'])))
    history=history[(history.kickoff<cutoff)&(history.kickoff>=cutoff-pd.Timedelta(days=365))]
    schedule_path=next(Path(value) for value in bm['input_hashes'] if Path(value).name=='games.parquet')
    schedule=pd.read_parquet(schedule_path)
    other=league_other_points(history,schedule.set_index('game_id').home_team)
    team_folder,_,games=load_forecast(root,season,week)
    targets=[]
    for game in games[games.game_id.isin(game_ids)].itertuples():
        for side in ['home','away']:
            targets.append(dict(game_id=game.game_id,team=getattr(game,side+'_team'),score=getattr(game,'projected_'+side+'_score'),other_points=other))
    targets=pd.DataFrame(targets)
    joint,joint_budget,_=reconcile_scoring(p,budgets,targets)
    factors=pd.read_csv(defense/'defense_factors.csv');factors=factors[factors.team.isin(joint.team)]
    adjusted,adjusted_budget=adjust_allocations(joint,joint_budget,factors)
    adjusted,adjusted_budget,_=reconcile_scoring(adjusted,adjusted_budget,targets)
    created=pd.Timestamp.now(tz='UTC');earliest=pd.to_datetime(p.kickoff_utc,utc=True).min()
    if created>=earliest:raise ValueError('Kickoff reached before the testing version was saved')
    folders=[]
    for d,frame,budget,parent,meta in [(False,joint,joint_budget,baseline,bm),(True,adjusted,adjusted_budget,defense,dm)]:
        folder=testing_root(root,season,week,d)/(created.strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8])
        folder.mkdir(parents=True,exist_ok=False)
        frame.to_parquet(folder/'predictions.parquet',index=False);budget.to_parquet(folder/'team_budgets.parquet',index=False)
        if d:
            (folder/'evaluation.csv').write_bytes((defense/'evaluation.csv').read_bytes())
            factors.to_csv(folder/'defense_factors.csv',index=False)
        sources={**meta['input_hashes']}
        for source in [baseline/'predictions.parquet',baseline/'manifest.json',defense/'predictions.parquet',defense/'manifest.json',defense/'defense_factors.csv',team_folder/'predictions.parquet',team_folder/'manifest.json']:
            sources[str(source.resolve())]=hashlib.sha256(source.read_bytes()).hexdigest()
        info=dict(meta,created_at=str(created),earliest_kickoff=str(earliest),history_cutoff=str(cutoff),
                  forecast_scope='remaining_games_test',role_policy='joint_scoring_v9',joint_scoring_policy=scoring_policy(),
                  parent_snapshot=str(parent.resolve()),parent_predictions_sha256=meta['predictions_sha256'],
                  game_ids=sorted(game_ids),excluded_started_games=sorted(set(games.game_id)-game_ids),
                  players=int((~frame.is_unallocated).sum()),games=len(game_ids),input_hashes=sources,
                  selection_reason='User-requested Week 6 prospective test; original forecasts retained. Only games starting after this version was created are eligible.',
                  predictions_sha256=hashlib.sha256((folder/'predictions.parquet').read_bytes()).hexdigest(),
                  budgets_sha256=hashlib.sha256((folder/'team_budgets.parquet').read_bytes()).hexdigest(),
                  code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).parent/'models/joint_scoring.py']})
        (folder/'manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8');folders.append(folder)
    return folders
