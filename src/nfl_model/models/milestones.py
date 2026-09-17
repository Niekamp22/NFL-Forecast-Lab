"""Experimental, chronologically evaluated comparable-game milestone estimates.

No normal/Poisson assumption: preserve discrete zeros and the observed tails.
Current role predictions are a different policy from historical rolling baselines;
transfer to them remains unvalidated and is disclosed in the UI and manifests.
"""
import hashlib
import json
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
from ..player_views import POSITION_STATS,load_pinned_history,with_total_yards
from .player_roles import checked_player_snapshot

NEIGHBORS=300
MIN_TRAIN=200


def thresholds(stat):
    if stat=='passing_yards':return list(range(100,426,25))
    if stat=='total_yards':return list(range(25,226,25))
    if stat in ['receiving_yards','rushing_yards']:return [10,25,40,50,60,75,90,100,125,150,175,200]
    if stat=='attempts':return [15,20,25,30,35,40,45,50]
    if stat=='completions':return [10,15,20,25,30,35,40]
    if stat=='carries':return [1,3,5,10,15,20,25,30]
    if stat in ['targets','receptions']:return [1,2,3,4,5,6,8,10,12,15]
    return [1,2,3,4,5]


def historical_records(history):
    """Strictly earlier five recorded games, one-year lookback, >=3 valid stats."""
    history=with_total_yards(history)
    output=[]
    for pid,group in history.groupby('player_id'):
        group=group.sort_values(['kickoff','game_id'])
        columns=sorted({s for values in POSITION_STATS.values() for s in values}|{'total_yards'})
        values=group[columns].to_numpy(dtype=float)
        dates=group.kickoff.to_numpy()
        column_index={s:i for i,s in enumerate(columns)}
        for i,r in enumerate(group.itertuples()):
            if r.season not in [2022,2023,2024,2025] or r.position not in POSITION_STATS:continue
            start=max(0,i-5)
            keep=(dates[start:i]<r.kickoff)&(dates[start:i]>=r.kickoff-pd.Timedelta(days=365))
            past=values[start:i][keep]
            weights=np.arange(1,len(past)+1)
            for stat in POSITION_STATS[r.position]+(['total_yards'] if r.position=='RB' else []):
                column=column_index[stat];actual=values[i,column]
                valid=np.isfinite(past[:,column])
                if valid.sum()<3 or not np.isfinite(actual):continue
                expected=float(np.average(past[valid,column],weights=weights[valid]))
                output.append(dict(player_id=pid,game_id=r.game_id,season=r.season,week=r.week,
                                   position=r.position,stat=stat,expected=expected,actual=actual))
    return pd.DataFrame(output)


def comparable_probabilities(train,expected,levels):
    """Monotone P(actual >= threshold), with fixed Jeffreys half-count smoothing."""
    pool=train.dropna(subset=['expected','actual'])
    if len(pool)<MIN_TRAIN or not np.isfinite(expected):return None
    if expected<pool.expected.min() or expected>pool.expected.max():return None
    distances=np.abs(pool.expected.to_numpy()-expected)
    # Stable tie handling keeps zero-heavy stats reproducible.
    selected=np.argsort(distances,kind='stable')[:NEIGHBORS]
    actual=pool.actual.to_numpy()[selected]
    hits=(actual[:,None]>=np.asarray(levels)[None,:]).sum(axis=0)
    return dict(probability=(hits+.5)/(len(actual)+1),hits=hits,samples=len(actual),
                expected_low=float(pool.expected.iloc[selected].min()),expected_high=float(pool.expected.iloc[selected].max()))


def evaluate_records(records):
    training=records[records.season.isin([2022,2023])].copy()
    evaluations=[];metrics=[];reliability=[]
    for (position,stat),pool in training.groupby(['position','stat'],sort=True):
        levels=thresholds(stat)
        test=records[records.season.isin([2024,2025])&records.position.eq(position)&records.stat.eq(stat)]
        local=[]
        frequency=((pool.actual.to_numpy()[:,None]>=np.asarray(levels)).sum(axis=0)+.5)/(len(pool)+1)
        for row in test.itertuples():
            result=comparable_probabilities(pool,row.expected,levels)
            if result is None:continue
            for j,level in enumerate(levels):
                local.append(dict(position=position,stat=stat,game_id=row.game_id,player_id=row.player_id,season=row.season,
                                  threshold=level,probability=float(result['probability'][j]),outcome=int(row.actual>=level),frequency=float(frequency[j])))
        if not local:continue
        frame=pd.DataFrame(local);evaluations.append(frame)
        for level,g in frame.groupby('threshold'):
            metrics.append(dict(position=position,stat=stat,threshold=int(level),games=len(g),
                                brier=float(((g.probability-g.outcome)**2).mean()),
                                frequency_brier=float(((g.frequency-g.outcome)**2).mean()),
                                mean_probability=float(g.probability.mean()),observed_rate=float(g.outcome.mean()),
                                eligible_test_games=len(test),evaluated_games=int(len(g))))
        frame['bin']=np.minimum((frame.probability*10).astype(int),9)
        for bin_number,g in frame.groupby('bin'):
            reliability.append(dict(position=position,stat=stat,bin_low=bin_number/10,bin_high=(bin_number+1)/10,
                                    threshold_observations=len(g),mean_probability=g.probability.mean(),observed_rate=g.outcome.mean()))
    return training,pd.concat(evaluations,ignore_index=True),pd.DataFrame(metrics),pd.DataFrame(reliability)


def build_milestones(root,season,week,defense=False):
    root=Path(root)
    family='defense' if defense else 'baseline'
    original,meta,players=checked_player_snapshot(root/('player_role_defense_forecasts' if defense else 'player_role_forecasts')/str(season)/f'week_{week:02d}',current=True)
    now=pd.Timestamp.now(tz='UTC')
    if season<=2025:raise ValueError('Prospective milestones require a season after the evaluation years')
    if now>=pd.Timestamp(meta['earliest_kickoff']):raise ValueError('Cannot freeze milestones after kickoff')
    model_root=root/'models/player_milestones';pointer=model_root/'latest.json'
    raw_hashes={p:h for p,h in meta['input_hashes'].items() if Path(p).name.startswith('stats_player_week_') or Path(p).name=='games.parquet'}
    model=None
    if pointer.exists():
        candidate=Path(json.loads(pointer.read_text())['path'])
        model_meta=json.loads((candidate/'manifest.json').read_text())
        if model_meta['raw_hashes']==raw_hashes:model=candidate
    if model is None:
        history=load_pinned_history(meta)
        records=historical_records(history)
        training,evaluations,metrics,reliability=evaluate_records(records)
        model=model_root/(now.strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8]);model.mkdir(parents=True,exist_ok=False)
        training.to_parquet(model/'training.parquet',index=False)
        evaluations.to_parquet(model/'evaluation_predictions.parquet',index=False)
        metrics.to_csv(model/'metrics.csv',index=False);reliability.to_csv(model/'reliability.csv',index=False)
        model_meta=dict(created_at=str(now),training_years=[2022,2023],evaluation_years=[2024,2025],raw_hashes=raw_hashes,
                        neighbors=NEIGHBORS,min_training=MIN_TRAIN,history='Prior five observed games in 365 days, at least three valid observations per stat',
                        method='Nearest pregame stat expectations within position, smoothed empirical exceedance frequencies; fixed parameters, no evaluation tuning',
                        limits='Observed-player cohort only. Historical rolling expectations differ from live role/defense forecasts. Reused retrospective evaluation is not validation of live probabilities; no active probability or joint-stat distribution.',
                        artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in model.iterdir() if p.is_file()},
                        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (model/'manifest.json').write_text(json.dumps(model_meta,indent=2),encoding='utf-8')
        pointer.write_text(json.dumps({'path':str(model.resolve())}),encoding='utf-8')
    else:
        for p,h in model_meta['artifacts'].items():
            if hashlib.sha256((model/p).read_bytes()).hexdigest()!=h:raise ValueError('Probability artifact integrity failure')
        training=pd.read_parquet(model/'training.parquet')
    outputs=[]
    players=with_total_yards(players[~players.is_unallocated])
    pools={(position,stat):g for (position,stat),g in training.groupby(['position','stat'])}
    for player in players.itertuples():
        for stat in POSITION_STATS[player.position]+(['total_yards'] if player.position=='RB' else []):
            expected=getattr(player,stat);levels=thresholds(stat)
            pool=pools.get((player.position,stat),training.iloc[:0])
            reason='experimental'
            if pd.isna(expected):reason='projection_unavailable'
            elif player.history_games<3:reason='limited_current_team_history'
            elif len(pool)<MIN_TRAIN:reason='insufficient_training'
            result=comparable_probabilities(pool,expected,levels) if reason=='experimental' else None
            if reason=='experimental' and result is None:reason='outside_training_support'
            for j,level in enumerate(levels):
                outputs.append(dict(season=season,week=week,game_id=player.game_id,team=player.team,player_id=player.player_id,position=player.position,
                                    stat=stat,projection=expected,threshold=level,status=reason,
                                    probability=float(result['probability'][j]) if result else np.nan,
                                    comparable_games=result['samples'] if result else 0,
                                    comparable_hits=int(result['hits'][j]) if result else 0,
                                    comparable_expected_low=result['expected_low'] if result else np.nan,
                                    comparable_expected_high=result['expected_high'] if result else np.nan))
    frame=pd.DataFrame(outputs)
    target=root/'player_milestones'/family/str(season)/f'week_{week:02d}'
    for existing in target.glob('*/manifest.json'):
        old=json.loads(existing.read_text())
        if old['forecast_sha256']==meta['predictions_sha256'] and old['model_path']==str(model.resolve()):
            if hashlib.sha256((existing.parent/'probabilities.parquet').read_bytes()).hexdigest()!=old['probabilities_sha256']:raise ValueError('Frozen probability checksum mismatch')
            return existing.parent
    folder=target/(now.strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8]);folder.mkdir(parents=True,exist_ok=False)
    frame.to_parquet(folder/'probabilities.parquet',index=False)
    info=dict(created_at=str(now),earliest_kickoff=meta['earliest_kickoff'],forecast_path=str(original.resolve()),forecast_sha256=meta['predictions_sha256'],
              model_path=str(model.resolve()),model_manifest_sha256=hashlib.sha256((model/'manifest.json').read_bytes()).hexdigest(),
              probabilities_sha256=hashlib.sha256((folder/'probabilities.parquet').read_bytes()).hexdigest(),rows=len(frame),status_counts=frame.status.value_counts().to_dict(),
              semantics='P(recorded game stat >= milestone); integer thresholds include equality. Conditional on existing role and observed participation. Not sportsbook settlement rules.')
    (folder/'manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    return folder
