"""Grade the frozen operational player forecast; missing stats never become zero."""
import hashlib
import json
import uuid
from pathlib import Path

import pandas as pd

from .grading import grade_predictions
from .player_roles import FIELDS, checked_player_snapshot


def grade_player_rows(predictions,schedule,pbp,stats):
    games=predictions[['game_id']].drop_duplicates().merge(schedule[['game_id','home_team','away_team']],on='game_id',how='left',validate='one_to_one')
    games['score_rating_margin']=0.0;games['success_model_margin']=0.0
    finals,_=grade_predictions(games,schedule,pbp)
    if stats.duplicated(['game_id','player_id']).any(): raise ValueError('Duplicate player result rows')
    long=predictions[~predictions.is_unallocated].melt(id_vars=['season','week','game_id','team','player_id','player_name','position'],value_vars=FIELDS,var_name='stat',value_name='projection')
    actual=stats.reindex(columns=['game_id','team','player_id']+FIELDS).melt(id_vars=['game_id','team','player_id'],value_vars=FIELDS,var_name='stat',value_name='actual')
    out=long.merge(actual,on=['game_id','team','player_id','stat'],how='left',validate='one_to_one')
    out=out.merge(finals[['game_id','status']],on='game_id',how='left',validate='many_to_one')
    out.loc[out.status.eq('graded')&out.actual.isna(),'status']='missing_player_stat'
    out.loc[out.projection.isna(),'status']='no_forecast'
    valid=out.status.eq('graded')
    out['actual']=out.actual.where(valid)
    out['error']=(out.projection-out.actual).where(valid)
    return out


def grade_role_week(client,season,week,refresh=False,defense=False):
    root=client.root
    folder,meta,predictions=checked_player_snapshot(root/('player_role_defense_forecasts' if defense else 'player_role_forecasts')/str(season)/f'week_{week:02d}')
    refs=[];frames={}
    for dataset,year in [('schedules',None),('play_by_play',season),('player_stats',season)]:
        path=client.download(dataset,year,force=refresh)
        frames[dataset]=pd.read_parquet(path)
        refs.append({**json.loads((path.parent/'manifest.json').read_text()),'path':str(path.resolve())})
    grades=grade_player_rows(predictions,frames['schedules'],frames['play_by_play'],frames['player_stats'])
    metrics=grades[grades.status.eq('graded')].groupby(['position','stat']).error.agg(observations='size',mae=lambda x:x.abs().mean(),bias='mean').reset_index()
    now=pd.Timestamp.now(tz='UTC')
    output=root/('player_defense_evaluations' if defense else 'player_evaluations')/str(season)/f'week_{week:02d}'/(now.strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8])
    output.mkdir(parents=True,exist_ok=False)
    grades.to_parquet(output/'grades.parquet',index=False);metrics.to_csv(output/'metrics.csv',index=False)
    run=dict(created_at=str(now),snapshot=str(folder.resolve()),snapshot_hash=meta['predictions_sha256'],sources=refs,
             statuses=grades.status.value_counts().to_dict(),graded=int(grades.status.eq('graded').sum()),
             grades_sha256=hashlib.sha256((output/'grades.parquet').read_bytes()).hexdigest())
    (output/'manifest.json').write_text(json.dumps(run,indent=2),encoding='utf-8')
    return output
