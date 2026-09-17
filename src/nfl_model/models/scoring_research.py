"""Pregame football-feature corrections to team scores and combined totals."""
import hashlib
import json
import uuid
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES={
    'success':['success_rate'],
    'epa':['epa_per_play'],
    'production':['epa_per_play','success_rate','explosive_pass_rate','explosive_rush_rate','sack_rate','turnover_rate'],
}


def score_metrics(frame,home='home_prediction',away='away_prediction'):
    h=frame[home]-frame.home_score;a=frame[away]-frame.away_score
    return dict(games=len(frame),team_mae=float((h.abs().mean()+a.abs().mean())/2),
                team_rmse=float(np.sqrt(np.mean(np.r_[h*h,a*a]))),
                total_mae=float((h+a).abs().mean()),margin_mae=float((h-a).abs().mean()),
                bias=float((h.mean()+a.mean())/2))


def ridge_predict(train,target,columns,ridge):
    x=train[columns].to_numpy(float);z=target[columns].to_numpy(float)
    valid=np.isfinite(x);counts=valid.sum(axis=0)
    means=np.divide(np.where(valid,x,0).sum(axis=0),counts,out=np.zeros(len(columns)),where=counts>0)
    mx=~valid;mz=~np.isfinite(z)
    x=np.where(mx,means,x);z=np.where(mz,means,z)
    scale=x.std(axis=0);scale=np.where(scale>1e-8,scale,1)
    x=np.column_stack([np.ones(len(x)),(x-means)/scale,mx]);z=np.column_stack([np.ones(len(z)),(z-means)/scale,mz])
    penalty=np.eye(x.shape[1])*ridge;penalty[0,0]=1e-8
    beta=np.linalg.solve(x.T@x+penalty,x.T@train.residual.to_numpy(float))
    return z@beta


def feature_rows(base,teams,feature_set,mode):
    fields=[f'{side}_{metric}_5' for side in ['offense','defense'] for metric in FEATURES[feature_set]]
    if teams.duplicated(['game_id','team']).any():raise ValueError('Duplicate team feature keys')
    for side in ['offense','defense']:
        if (teams[f'{side}_max_source_week_5']>=teams.week).fillna(False).any():raise ValueError('Non-pregame feature')
    frame=base.copy()
    for venue in ['home','away']:
        subset=teams[['game_id','team']+fields].rename(columns={'team':venue+'_team',**{f:venue+'_'+f for f in fields}})
        frame=frame.merge(subset,on=['game_id',venue+'_team'],validate='one_to_one',how='left',indicator=True)
        if not frame._merge.eq('both').all():raise ValueError('Missing team feature row')
        frame=frame.drop(columns='_merge')
    if mode=='total':
        names=[]
        for f in fields:
            name='sum_'+f;frame[name]=frame['home_'+f]+frame['away_'+f];names.append(name)
        frame['residual']=frame.home_score+frame.away_score-frame.score_total
        return frame,names
    pieces=[];names=['is_home']
    for own,opp in [('home','away'),('away','home')]:
        row=frame[['season','week','game_id','gameday']].copy();row['venue']=own;row['is_home']=float(own=='home')
        row['residual']=frame[own+'_score']-frame['projected_'+own+'_score']
        for f in fields:
            name='own_'+f;row[name]=frame[own+'_'+f]
            if name not in names:names.append(name)
            name='opponent_'+f;row[name]=frame[opp+'_'+f]
            if name not in names:names.append(name)
        pieces.append(row)
    return pd.concat(pieces,ignore_index=True),names


def chronological_corrections(frame,columns,ridge):
    result=[]
    for (season,week),target in frame.groupby(['season','week'],sort=True):
        cutoff=pd.to_datetime(target.gameday).min()
        earlier=(frame.season<season)|((frame.season==season)&(frame.week<week))
        train=frame[earlier&frame.residual.notna()&(pd.to_datetime(frame.gameday)<cutoff)]
        copy=target.copy()
        copy['correction']=ridge_predict(train,target,columns,ridge) if train.game_id.nunique()>=32 else 0.
        result.append(copy)
    return pd.concat(result,ignore_index=True)


def apply_corrections(base,corrections,mode,strength):
    out=base.copy()
    if mode=='total':
        c=out.game_id.map(corrections.set_index('game_id').correction)
        total=np.maximum(out.score_total+strength*c,(out.projected_home_score-out.projected_away_score).abs())
        change=(total-out.score_total)/2
        out['home_prediction']=out.projected_home_score+change
        out['away_prediction']=out.projected_away_score+change
    else:
        indexed=corrections.set_index(['game_id','venue']).correction
        for venue in ['home','away']:
            c=[indexed.loc[(g,venue)] for g in out.game_id]
            out[venue+'_prediction']=np.maximum(0,out['projected_'+venue+'_score']+strength*np.asarray(c))
    return out


def run(root):
    root=Path(root)
    totals=Path(json.loads((root/'models/totals/latest.json').read_text())['path'])
    tm=json.loads((totals/'manifest.json').read_text())
    efficiency=Path(tm['efficiency_build']);em=json.loads((efficiency/'manifest.json').read_text())
    source=Path(tm['source']['path']);features=Path(em['team_feature_path'])
    for p,expected in [(source,tm['source']['sha256']),(features,em['team_feature_sha256'])]:
        if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError('Source checksum mismatch')
    schedule=pd.read_parquet(source)
    base=pd.read_parquet(totals/'projected_scores.parquet').merge(schedule[['game_id','gameday','home_score','away_score']],on='game_id',validate='one_to_one')
    if base[['home_score','away_score']].isna().any().any():raise ValueError('Incomplete historical outcomes')
    teams=pd.read_parquet(features)
    candidates={};trials=[]
    baseline=base.assign(home_prediction=base.projected_home_score,away_prediction=base.projected_away_score)
    for mode in ['total','team_scores']:
        for feature_set in FEATURES:
            frame,names=feature_rows(base,teams,feature_set,mode)
            # Selection only sees 2022-23 outcomes. Evaluation is run for the winner.
            for ridge in [10,100,1000]:
                correction=chronological_corrections(frame[frame.season<=2023],names,ridge)
                for strength in [.25,.5,1.]:
                    p=apply_corrections(base[base.season<=2023],correction,mode,strength)
                    key=f'{mode}_{feature_set}_{ridge}_{strength}'
                    trials.append(dict(key=key,mode=mode,features=feature_set,ridge=ridge,strength=strength,**score_metrics(p)))
    chosen=min(trials,key=lambda x:(x['team_mae'],x['key']))
    frame,names=feature_rows(base,teams,chosen['features'],chosen['mode'])
    correction=chronological_corrections(frame,names,chosen['ridge'])
    proposed=apply_corrections(base,correction,chosen['mode'],chosen['strength'])
    metrics=[]
    for label,p in [('baseline',baseline),('selected',proposed)]:
        for split,mask in [('selection',p.season<=2023),('evaluation',p.season>=2024),('2024',p.season==2024),('2025',p.season==2025),('early_evaluation',(p.season>=2024)&p.week.le(6))]:
            metrics.append(dict(model=label,split=split,**score_metrics(p[mask])))
    evalmask=base.season.ge(2024)
    a=((baseline.home_prediction-baseline.home_score).abs()+(baseline.away_prediction-baseline.away_score).abs())/2
    b=((proposed.home_prediction-proposed.home_score).abs()+(proposed.away_prediction-proposed.away_score).abs())/2
    # Resample entire season/week clusters, keeping correlated games and team scores together.
    delta=pd.DataFrame({'season':base.season,'week':base.week,'delta':b-a})[evalmask].groupby(['season','week']).delta.agg(['sum','count'])
    rng=np.random.default_rng(2026);sample=rng.integers(0,len(delta),size=(3000,len(delta)))
    boot=delta['sum'].to_numpy()[sample].sum(axis=1)/delta['count'].to_numpy()[sample].sum(axis=1)
    interval=np.quantile(boot,[.025,.975]).tolist()
    stamp=pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8]
    folder=root/'models/scoring_research'/stamp;folder.mkdir(parents=True)
    pd.DataFrame(trials).to_csv(folder/'selection.csv',index=False);pd.DataFrame(metrics).to_csv(folder/'metrics.csv',index=False)
    proposed.to_parquet(folder/'predictions.parquet',index=False)
    info=dict(created_at=str(pd.Timestamp.now(tz='UTC')),selected=chosen,features=names,team_mae_difference_ci=interval,
              inputs={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,features,totals/'projected_scores.parquet',efficiency/'manifest.json']},
              limits='Reused 2024-25 evaluation; observed historical revisions, not certified publication times. Cold-start correction zero before 32 earlier games. No player availability, weather, or market inputs. No live model promotion.',
              code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (folder/'manifest.json').write_text(json.dumps(info,indent=2))
    (folder.parent/'latest.json').write_text(json.dumps({'path':str(folder.resolve())}))
    print(json.dumps(chosen));print(pd.DataFrame(metrics).to_string(index=False));print('Paired week-cluster interval (selected minus baseline)',interval);print(folder)
    return folder
