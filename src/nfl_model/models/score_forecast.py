"""Add pregame totals to an existing frozen margin snapshot without changing it."""
import hashlib
import json
import uuid
from datetime import datetime,timezone
from pathlib import Path

import pandas as pd

from .ratings import RatingConfig
from .totals import replay_totals,projected_scores


def freeze_scores(root:Path,season:int,week:int)->Path:
    now=pd.Timestamp(datetime.now(timezone.utc))
    total_path=Path(json.loads((root/'models/totals/latest.json').read_text())['path'])
    total_run=json.loads((total_path/'manifest.json').read_text())
    candidates=[]
    for path in (root/'snapshots'/str(season)/f'week_{week:02d}').glob('*/manifest.json'):
        m=json.loads(path.read_text());candidates.append((pd.Timestamp(m['created_at']),path.parent,m))
    if not candidates:raise ValueError('Freeze margin forecasts first')
    _,margin_path,margin_run=min(candidates,key=lambda item:(item[0],str(item[1])))
    if now>=pd.Timestamp(margin_run['earliest_kickoff']):raise ValueError('Week has already begun')
    live=json.loads((Path(margin_run['live_build'])/'manifest.json').read_text())
    source=next(s for s in live['sources'] if s['dataset']=='schedules')
    path=Path(source['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=source['sha256']:raise ValueError('Schedule checksum mismatch')
    if pd.Timestamp(source['retrieved_at'])>=pd.Timestamp(margin_run['earliest_kickoff']):raise ValueError('Source after cutoff')
    schedule=pd.read_parquet(path,columns=['season','week','game_id','game_type','gameday','gametime','home_team','away_team','home_score','away_score','location'])
    games=schedule[schedule.season.between(2021,season)&schedule.game_type.eq('REG')]
    target=games[games.season.eq(season)&games.week.eq(week)]
    kickoff=pd.to_datetime(target.gameday.astype(str)+' '+target.gametime.astype(str)).dt.tz_localize('America/New_York').dt.tz_convert('UTC').min()
    if pd.isna(kickoff) or now>=kickoff or target[['home_score','away_score']].notna().any().any():raise ValueError('Target week must be unplayed with future kickoff')
    games=games[(games.season<season)|games.week.le(week)]
    config=total_run['configs'][total_run['selected_on_validation']]
    totals,states=replay_totals(games,RatingConfig(config['half_life_days'],config['ridge']),[season])
    totals=totals[totals.week.eq(week)].copy()
    totals['selected_total']=totals.predicted_total if total_run['selected_on_validation']=='team_scoring' else totals.league_total
    margin_file=margin_path/'predictions.parquet';margin_hash=hashlib.sha256(margin_file.read_bytes()).hexdigest()
    if margin_run.get('predictions_sha256') and margin_hash!=margin_run['predictions_sha256']:raise ValueError('Margin checksum mismatch')
    margins=pd.read_parquet(margin_file)
    if set(totals.game_id)!=set(margins.game_id):raise ValueError('Forecast cohorts differ')
    joined=totals.merge(margins[['game_id','home_team','away_team','success_model_margin']],on=['game_id','home_team','away_team'],validate='one_to_one')
    if len(joined)!=len(margins):raise ValueError('Forecast team mismatch')
    scores=projected_scores(joined.selected_total,joined.success_model_margin)
    output=pd.concat([joined[['game_id','home_team','away_team','selected_total','success_model_margin']],scores],axis=1)
    folder=root/'snapshots_scores'/str(season)/f'week_{week:02d}'/(now.strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8]);folder.mkdir(parents=True,exist_ok=False)
    output.to_parquet(folder/'predictions.parquet',index=False);output.to_csv(folder/'predictions.csv',index=False)
    manifest={'created_at':str(now),'earliest_kickoff':str(kickoff),'season':season,'week':week,'totals_model':str(total_path),'margin_snapshot':str(margin_path),
              'margin_sha256':margin_hash,'legacy_margin_creation_hash_available':bool(margin_run.get('predictions_sha256')),'source':source,'games':len(output),
              'prediction_sha256':hashlib.sha256((folder/'predictions.parquet').read_bytes()).hexdigest(),'weekly_models':[s for s in states if s['week']==week],
              'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    lines=[f'# Score projections: {season} Week {week}','',f'Frozen at {now}. Totals model: {total_run["selected_on_validation"]}. Margin model: existing success-rate snapshot.','',
           '| Away | Home | Away projected score | Home projected score | Projected total |','|---|---|---:|---:|---:|']
    for r in output.itertuples():lines.append(f'| {r.away_team} | {r.home_team} | {r.projected_away_score:.1f} | {r.projected_home_score:.1f} | {r.score_total:.1f} |')
    lines+=['','Continuous model estimates; not calibrated win probabilities. Unresolved QB situations remain unadjusted. Original margin forecasts are unchanged. The grading command currently grades margins only; totals grading is a subsequent extension.']
    report='\n'.join(lines);(folder/'report.md').write_text(report,encoding='utf-8');Path(f'docs/score_forecast_{season}_week_{week:02d}.md').write_text(report,encoding='utf-8')
    return folder
