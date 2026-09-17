"""Grade margins, totals, scores and three-way probabilities on one frozen cohort."""
import hashlib
import json
import uuid
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.client import NFLVerseClient
from .grading import grade_predictions
from .weekly_forecast import earliest_snapshot,checked_predictions
from .probability import probability_score,reliability


def grade_full_predictions(predictions:pd.DataFrame,schedule:pd.DataFrame,pbp:pd.DataFrame)->tuple[pd.DataFrame,dict]:
    required=['selected_total','score_total','projected_home_score','projected_away_score','p_home','p_tie','p_away']
    if not np.isfinite(predictions[required].to_numpy(dtype=float)).all():raise ValueError('Non-finite score/probability forecast')
    if (predictions[['selected_total','score_total','projected_home_score','projected_away_score']]<0).any().any():raise ValueError('Negative score/total')
    if not np.allclose(predictions.projected_home_score+predictions.projected_away_score,predictions.score_total) or not np.allclose(predictions.projected_home_score-predictions.projected_away_score,predictions.success_model_margin):raise ValueError('Inconsistent projected scores')
    # Validate probabilities even when all results are pending.
    probs=predictions[['p_away','p_tie','p_home']].to_numpy()
    if (probs<0).any() or (probs>1).any() or not np.allclose(probs.sum(axis=1),1):raise ValueError('Invalid probabilities')
    graded,summary=grade_predictions(predictions,schedule,pbp)
    final=graded.status.eq('graded')
    graded['actual_total']=(graded.home_score+graded.away_score).where(final)
    summary['totals_and_scores']={}
    for model,target in [('selected_total','actual_total'),('score_total','actual_total'),('projected_home_score','home_score'),('projected_away_score','away_score')]:
        error=(graded[model]-graded[target]).where(final)
        graded[f'{model}_error']=error
        summary['totals_and_scores'][model]={'games':int(final.sum()),'mae':float(error.abs().mean()) if final.any() else None,'rmse':float(np.sqrt((error**2).mean())) if final.any() else None,'mean_error':float(error.mean()) if final.any() else None}
    summary['probabilities']=probability_score(graded)
    return graded,summary


def grade_full_week(client:NFLVerseClient,season:int,week:int,refresh:bool=False)->Path:
    snapshot,meta=earliest_snapshot(client.root/'forecasts'/str(season)/f'week_{week:02d}')
    predictions=checked_predictions(snapshot,meta)
    if not predictions.season.eq(season).all() or not predictions.week.eq(week).all():raise ValueError('Forecast season/week mismatch')
    frames,refs={},[]
    for name,year in [('schedules',None),('play_by_play',season)]:
        path=client.download(name,year,force=refresh);frames[name]=pd.read_parquet(path)
        refs.append({**json.loads((path.parent/'manifest.json').read_text()),'path':str(path.resolve())})
    graded,summary=grade_full_predictions(predictions,frames['schedules'],frames['play_by_play'])
    now=datetime.now(timezone.utc);stamp=now.strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8]
    folder=client.root/'evaluations_full'/str(season)/f'week_{week:02d}'/stamp;folder.mkdir(parents=True,exist_ok=False)
    graded.to_parquet(folder/'grades.parquet',index=False);graded.to_csv(folder/'grades.csv',index=False)
    reliability(graded).to_csv(folder/'reliability.csv',index=False)
    info={'created_at':now.isoformat(),'snapshot':str(snapshot),'snapshot_hash':meta['predictions_sha256'],'selection':'earliest pre-kickoff unified forecast',
          'sources':refs,'summary':summary,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (folder/'manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    report=f'# Complete forecast grading — {season} Week {week}\n\nGraded **{summary["graded"]} / {summary["total_forecasts"]}** games.\n\nPending or conflicting games contribute no errors or probability scores. END GAME evidence must agree with the schedule.\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n\nEvery grading version pins its result sources. Later corrections produce a new evaluation; original forecasts are never rewritten. Reliability bins from a single week are descriptive only.\n'
    (folder/'report.md').write_text(report,encoding='utf-8');Path(f'docs/full_grading_{season}_week_{week:02d}.md').write_text(report,encoding='utf-8')
    return folder
