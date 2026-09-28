"""Research-only weekly CatBoost replay; no live forecast integration."""
import numpy as np
import pandas as pd


def replay_boosted(frame,features,mode,depth=2,loss='RMSE',iterations=200):
    from catboost import CatBoostRegressor
    if mode not in ['direct','residual']:
        raise ValueError('Unsupported score target')
    if frame.duplicated(['game_id','venue']).any() or not frame.groupby('game_id').size().eq(2).all():
        raise ValueError('Require paired unique game rows')
    if not (pd.to_datetime(frame.latest_feature_date)<pd.to_datetime(frame.gameday)).all():
        raise ValueError('Non-pregame features')
    if any(c not in features for c in ['is_home','base_prediction']):
        raise ValueError('Missing required features')
    if any(c in features for c in ['actual','residual','game_id','season','week']):
        raise ValueError('Outcome or identity leakage')
    output=[]
    for (season,week),target in frame.groupby(['season','week'],sort=True):
        cutoff=pd.to_datetime(target.gameday).min()
        earlier=(frame.season<season)|((frame.season==season)&(frame.week<week))
        train=frame[earlier&(pd.to_datetime(frame.gameday)<cutoff)]
        row=target.copy();row['training_games']=train.game_id.nunique()
        if train.game_id.nunique()<32:
            row['raw_prediction']=target.base_prediction if mode=='direct' else 0.
        else:
            x=train[features].to_numpy(float);z=target[features].to_numpy(float)
            if not np.isfinite(x).all() or not np.isfinite(z).all():
                raise ValueError('Incomplete feature matrix')
            model=CatBoostRegressor(iterations=iterations,depth=depth,learning_rate=.03,l2_leaf_reg=10,
                loss_function=loss,random_seed=20260923,thread_count=2,verbose=False,allow_writing_files=False)
            model.fit(x,train.actual.to_numpy() if mode=='direct' else (train.actual-train.base_prediction).to_numpy())
            row['raw_prediction']=model.predict(z)
        output.append(row)
    return pd.concat(output,ignore_index=True)


def combine_predictions(frame,mode,strength=1.):
    out=frame.copy()
    if mode=='direct':out['prediction']=np.maximum(0.,out.raw_prediction)
    elif mode=='residual':out['prediction']=np.maximum(0.,out.base_prediction+strength*out.raw_prediction)
    else:raise ValueError('Unsupported score target')
    return out
