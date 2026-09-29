"""Accuracy cohorts use saved forecast information, never target-game outcomes."""
import numpy as np
import pandas as pd

KEY=['game_id','team','player_id']


def accuracy_cohorts(rows, predictions):
    p=predictions.loc[~predictions.is_unallocated].copy()
    if p.duplicated(KEY).any(): raise ValueError('Duplicate forecast player keys')
    high=(p.position.eq('QB') & p.attempts.ge(15)
          | p.position.eq('RB') & p.carries.ge(10)
          | p.position.eq('WR') & p.targets.ge(5)
          | p.position.eq('TE') & p.targets.ge(3)
          | p.position.eq('K') & p.fg_att.ge(1))
    p['workload_group']=np.where(high,'Substantial projected workload','Lower or unresolved projected workload')
    p['experience_group']=np.select([p.history_games.eq(0),p.history_games.between(1,2),p.history_games.ge(3)],
        ['No recent team games','1–2 recent team games','3+ recent team games'],default='Unknown history')
    out=rows.merge(p[KEY+['workload_group','experience_group']],on=KEY,how='left',validate='many_to_one')
    out[['workload_group','experience_group']]=out[['workload_group','experience_group']].fillna('Unknown history')
    return out


def accuracy_summary(rows, groups=('position','stat')):
    result=[]
    for key,g in rows.groupby(list(groups),dropna=False,sort=True):
        valid=g[g.status.eq('graded') & g.projection.notna() & g.actual.notna()]
        error=valid.projection-valid.actual
        supported=g.projection.notna()
        final=g.game_result_status.eq('graded') if 'game_result_status' in g else g.status.isin(['graded','missing_player_stat'])
        eligible=int((supported & final).sum())
        result.append(dict(zip(groups,key if isinstance(key,tuple) else (key,))) | dict(
            observations=len(valid),players=valid.player_id.nunique(),
            final_estimates=eligible,coverage=len(valid)/eligible if eligible else np.nan,
            average_miss=error.abs().mean(),median_miss=error.abs().median(),
            rmse=np.sqrt((error**2).mean()),average_bias=error.mean()))
    return pd.DataFrame(result)
