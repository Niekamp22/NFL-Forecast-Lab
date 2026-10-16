"""After-kickoff comparisons from pinned pregame data; never prospective tests."""
import hashlib,json,uuid
from pathlib import Path
import pandas as pd
from .models.player_roles import checked_player_snapshot,audit_allocations
from .models.joint_scoring import reconcile_scoring,league_other_points,scoring_policy,audit_joint_scoring
from .models.role_defense import adjust_allocations
from .player_views import load_pinned_history
from .dashboard import load_forecast


def comparison_root(root,season,week,defense=False):
    return Path(root)/'player_comparison_forecasts'/('defense' if defense else 'baseline')/str(season)/f'week_{week:02d}'


def load_retrospective_snapshot(root,season,week,defense=False):
    candidates=[]
    for path in comparison_root(root,season,week,defense).glob('*/manifest.json'):
        m=json.loads(path.read_text())
        if m.get('forecast_scope')!='retrospective_comparison' or m.get('eligible_for_accuracy') is not False:
            raise ValueError('Retrospective comparison must be excluded from prospective accuracy')
        candidates.append((pd.Timestamp(m['created_at']),str(path.parent),m))
    if not candidates:return None
    _,folder,m=min(candidates,key=lambda x:x[:2]);folder=Path(folder)
    for name,key in [('predictions.parquet','predictions_sha256'),('team_budgets.parquet','budgets_sha256')]:
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=m[key]:raise ValueError('Comparison checksum mismatch')
    p=pd.read_parquet(folder/'predictions.parquet')
    kickoffs=pd.to_datetime(p.kickoff_utc,utc=True)
    if set(p.game_id)!=set(m['game_ids']):raise ValueError('Comparison game coverage mismatch')
    if (kickoffs>pd.Timestamp(m['created_at'])).any():raise ValueError('Retrospective comparison contains a future game')
    if pd.Timestamp(m['history_cutoff'])>=kickoffs.min():raise ValueError('Comparison inputs must precede kickoff')
    family='player_role_defense_forecasts' if defense else 'player_role_forecasts'
    _,parent,_=checked_player_snapshot(Path(root)/family/str(season)/f'week_{week:02d}',current=True)
    if parent['predictions_sha256']!=m['parent_predictions_sha256']:raise ValueError('Comparison parent changed')
    audit_allocations(p,pd.read_parquet(folder/'team_budgets.parquet'));audit_joint_scoring(p)
    return folder,m,p


def create_retrospective_week(root,season,week,game_ids):
    root=Path(root);game_ids=set(game_ids)
    if not game_ids:raise ValueError('Explicit comparison game IDs required')
    existing=[load_retrospective_snapshot(root,season,week,d) for d in [False,True]]
    if any(existing):
        if not all(existing) or any(set(item[1]['game_ids'])!=game_ids for item in existing):raise ValueError('Inspect existing comparison before changing scope')
        return [item[0] for item in existing]
    baseline,bm,p=checked_player_snapshot(root/'player_role_forecasts'/str(season)/f'week_{week:02d}',current=True)
    defense,dm,_=checked_player_snapshot(root/'player_role_defense_forecasts'/str(season)/f'week_{week:02d}',current=True)
    if not game_ids.issubset(set(p.game_id)):raise ValueError('Unknown comparison game')
    p=p[p.game_id.isin(game_ids)].copy()
    if (pd.to_datetime(p.kickoff_utc,utc=True)>pd.Timestamp.now(tz='UTC')).any():raise ValueError('Use prospective testing for games that have not started')
    budgets=pd.read_parquet(baseline/'team_budgets.parquet');budgets=budgets[budgets.game_id.isin(game_ids)]
    for meta in [bm,dm]:
        for value,expected in meta['input_hashes'].items():
            if hashlib.sha256(Path(value).read_bytes()).hexdigest()!=expected:raise ValueError('Pinned comparison source changed')
    cutoff=pd.Timestamp(bm.get('history_cutoff',bm.get('feature_cutoff',bm['created_at'])))
    if cutoff>=pd.to_datetime(p.kickoff_utc,utc=True).min():raise ValueError('No pregame source cutoff')
    history=load_pinned_history(bm)
    history=history[(history.kickoff<cutoff)&(history.kickoff>=cutoff-pd.Timedelta(days=365))]
    schedule=pd.read_parquet(next(Path(v) for v in bm['input_hashes'] if Path(v).name=='games.parquet'))
    other=league_other_points(history,schedule.set_index('game_id').home_team)
    team_folder,_,games=load_forecast(root,season,week)
    targets=pd.DataFrame([dict(game_id=g.game_id,team=getattr(g,side+'_team'),score=getattr(g,'projected_'+side+'_score'),other_points=other) for g in games[games.game_id.isin(game_ids)].itertuples() for side in ['home','away']])
    joint,jb,_=reconcile_scoring(p,budgets,targets)
    factors=pd.read_csv(defense/'defense_factors.csv');factors=factors[factors.team.isin(joint.team)]
    adjusted,ab=adjust_allocations(joint,jb,factors);adjusted,ab,_=reconcile_scoring(adjusted,ab,targets)
    created=pd.Timestamp.now(tz='UTC');folders=[]
    for d,frame,budget,parent,meta in [(False,joint,jb,baseline,bm),(True,adjusted,ab,defense,dm)]:
        folder=comparison_root(root,season,week,d)/(created.strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8]);folder.mkdir(parents=True,exist_ok=False)
        frame.to_parquet(folder/'predictions.parquet',index=False);budget.to_parquet(folder/'team_budgets.parquet',index=False)
        if d:(folder/'evaluation.csv').write_bytes((defense/'evaluation.csv').read_bytes())
        sources=dict(meta['input_hashes'])
        for source in [baseline/'predictions.parquet',baseline/'manifest.json',defense/'predictions.parquet',defense/'manifest.json',defense/'defense_factors.csv',team_folder/'predictions.parquet',team_folder/'manifest.json']:
            sources[str(source.resolve())]=hashlib.sha256(source.read_bytes()).hexdigest()
        info=dict(meta,created_at=str(created),history_cutoff=str(cutoff),forecast_scope='retrospective_comparison',eligible_for_accuracy=False,
                  role_policy='joint_scoring_v9',joint_scoring_policy=scoring_policy(),parent_snapshot=str(parent.resolve()),parent_predictions_sha256=meta['predictions_sha256'],
                  game_ids=sorted(game_ids),games=len(game_ids),players=int((~frame.is_unallocated).sum()),input_hashes=sources,
                  selection_reason='Explicit user-requested after-kickoff comparison using only original pinned pregame inputs. Separate from prospective accuracy.',
                  predictions_sha256=hashlib.sha256((folder/'predictions.parquet').read_bytes()).hexdigest(),budgets_sha256=hashlib.sha256((folder/'team_budgets.parquet').read_bytes()).hexdigest(),
                  code_sha256={v.name:hashlib.sha256(v.read_bytes()).hexdigest() for v in [Path(__file__),Path(__file__).parent/'models/joint_scoring.py']})
        (folder/'manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8');folders.append(folder)
    return folders
