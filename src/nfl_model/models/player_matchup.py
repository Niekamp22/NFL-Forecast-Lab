"""Separate opponent opportunity volume from production per opportunity."""
import hashlib
import json
import uuid

import numpy as np
import pandas as pd

from .weekly_forecast import earliest_snapshot

OPPS = {'passing_yards':'attempts', 'receiving_yards':'targets', 'rushing_yards':'carries'}


def split_context(games, opponent, cutoff):
    past = games[(games.kickoff < cutoff) & (games.kickoff >= cutoff-pd.Timedelta(days=365))]
    own = past[past.defense.eq(opponent)].sort_values('kickoff').tail(5)
    # Paired complete observations: no numerator/denominator from different games.
    past = past.dropna(subset=['yards','opportunities'])
    own = own.dropna(subset=['yards','opportunities'])
    if len(own) < 3 or past.empty or own.opportunities.sum() <= 0 or past.opportunities.sum() <= 0:
        return dict(volume_factor=np.nan, rate_factor=np.nan, defense_games=len(own))
    volume = own.opportunities.mean()
    league_volume = past.opportunities.mean()
    rate = own.yards.sum()/own.opportunities.sum()
    league_rate = past.yards.sum()/past.opportunities.sum()
    return dict(volume_factor=volume/league_volume-1, rate_factor=rate/league_rate-1 if league_rate > 0 else np.nan,
                defense_games=len(own), defense_opportunities=volume, league_opportunities=league_volume,
                defense_rate=rate, league_rate=league_rate)


def fit_split(frame):
    train = frame[frame.season.isin([2022,2023])].dropna(subset=['base_volume','base_rate','actual_opportunities','actual','volume_factor','rate_factor'])
    if len(train) < 30:
        raise ValueError('Not enough split-model training rows')
    def slope(x, residual):
        # Fixed 10% ridge shrinkage, signed slopes: direction is learned, not assumed.
        ss = float(x@x)
        return float(np.clip(x@residual/(1.1*ss), -1, 1)) if ss else 0.0
    volume = slope((train.base_volume*train.volume_factor).to_numpy(), (train.actual_opportunities-train.base_volume).to_numpy())
    active = train[train.actual_opportunities.gt(0)]
    expected = active.actual_opportunities*active.base_rate
    rate = slope((expected*active.rate_factor).to_numpy(), (active.actual-expected).to_numpy())
    return dict(volume=volume, rate=rate)


def apply_split(volume, rate, volume_factor, rate_factor, model):
    adjusted_volume = np.maximum(0, volume*(1+model['volume']*volume_factor))
    adjusted_rate = np.maximum(0, rate*(1+model['rate']*rate_factor))
    return adjusted_volume, adjusted_rate, adjusted_volume*adjusted_rate


def grouped_effects(evaluation):
    rows = []
    for axis in ['volume_factor','rate_factor']:
        data = evaluation.copy()
        data['bucket'] = pd.cut(data[axis], [-np.inf,-.10,.10,np.inf], labels=['Below league (>10%)','Near league (±10%)','Above league (>10%)'])
        for bucket, group in data.groupby('bucket', observed=True):
            difference = group.actual-group.baseline
            clusters = pd.DataFrame({'game_id':group.game_id,'difference':difference}).groupby('game_id').difference.agg(['sum','count'])
            rng = np.random.default_rng(2026)
            samples = rng.integers(0,len(clusters),size=(1000,len(clusters)))
            means = clusters['sum'].to_numpy()[samples].sum(axis=1)/clusters['count'].to_numpy()[samples].sum(axis=1)
            lower,upper = np.quantile(means,[.025,.975])
            opp_total = group.actual_opportunities.sum()
            rows.append(dict(axis=axis,bucket=str(bucket),player_games=len(group),unique_games=len(clusters),
                             expected_yards=group.baseline.mean(),actual_yards=group.actual.mean(),
                             yards_above_expectation=difference.mean(),ci_low=lower,ci_high=upper,
                             opportunities_above_expectation=(group.actual_opportunities-group.base_volume).mean(),
                             efficiency_above_expectation=((group.actual-group.actual_opportunities*group.base_rate).sum()/opp_total if opp_total else np.nan)))
    return pd.DataFrame(rows)


def run_player_matchup(client, season, week):
    root = client.root
    now = pd.Timestamp.now(tz='UTC')
    previous, previous_meta = earliest_snapshot(root/'player_defense_forecasts'/str(season)/f'week_{week:02d}')
    if now >= pd.Timestamp(previous_meta['earliest_kickoff']):
        raise ValueError('Cannot freeze after kickoff')
    # Reuse exactly the source versions behind the first defense experiment.
    from pathlib import Path
    paths = [Path(p) for p in previous_meta['input_hashes']]
    for path in paths:
        if hashlib.sha256(path.read_bytes()).hexdigest() != previous_meta['input_hashes'][str(path)]:
            raise ValueError(f'Pinned input changed: {path}')
    player_paths = [p for p in paths if p.name.startswith('stats_player_week_')]
    stats = pd.concat([pd.read_parquet(p) for p in player_paths],ignore_index=True)
    schedule = pd.read_parquet(next(p for p in paths if p.name=='games.parquet'))
    schedule = schedule[schedule.game_type.eq('REG')].copy()
    schedule['kickoff'] = pd.to_datetime(schedule.gameday.astype(str)+' '+schedule.gametime.astype(str)).dt.tz_localize('America/New_York').dt.tz_convert('UTC')
    cutoff = pd.Timestamp(previous_meta['feature_cutoff'])
    stats = stats[stats.season_type.eq('REG') & stats.player_id.notna()]
    stats = stats.merge(schedule[['game_id','kickoff','home_team','away_team','home_score','away_score']],on='game_id',validate='many_to_one')
    stats = stats[stats.kickoff.lt(cutoff) & stats.home_score.notna() & stats.away_score.notna()].copy()
    stats['defense'] = np.where(stats.team.eq(stats.home_team),stats.away_team,stats.home_team)
    history_path = previous/'historical_matchups.parquet'
    history = pd.read_parquet(history_path)
    groups = {pid:g.sort_values('kickoff') for pid,g in stats.groupby('player_id')}
    base_rows = []
    # Small NumPy windows avoid repeated whole-frame filtering in the backtest.
    for pid, group in groups.items():
        for stat, opportunity in OPPS.items():
            dates = group.kickoff.to_numpy()
            counts = group[opportunity].to_numpy(dtype=float)
            yards = group[stat].to_numpy(dtype=float)
            for index, actual in enumerate(group.itertuples()):
                if actual.season not in [2022,2023,2024,2025]:
                    continue
                start = max(0,index-10)
                keep = (dates[start:index] >= actual.kickoff-pd.Timedelta(days=365)) & (dates[start:index] < actual.kickoff)
                count = counts[start:index][keep]; value = yards[start:index][keep]
                recent = count[-5:]; valid = np.isfinite(recent)
                volume = np.average(recent[valid],weights=np.arange(1,len(recent)+1)[valid]) if valid.any() else np.nan
                pairs = np.isfinite(count) & np.isfinite(value) & (count>=0)
                rate = value[pairs].sum()/count[pairs].sum() if count[pairs].sum()>0 else np.nan
                base_rows.append(dict(player_id=pid,game_id=actual.game_id,stat=stat,base_volume=volume,base_rate=rate,actual_opportunities=counts[index]))
    history = history.merge(pd.DataFrame(base_rows),on=['player_id','game_id','stat'],validate='one_to_one')
    calendar = schedule.set_index('game_id').kickoff
    models, metrics, buckets = {}, [], []
    source_prediction = next(p for p in paths if p.name=='predictions.parquet')
    predictions = pd.read_parquet(source_prediction)
    predictions['opportunity_projection'] = predictions.projection
    predictions['matchup_adjustment_applied'] = False
    for (position,stat), frame in history.groupby(['position','stat']):
        opportunity = OPPS[stat]
        # Position-specific opportunity allowance and yards per opportunity.
        relevant = stats[stats.position.eq(position)]
        games = relevant.groupby(['game_id','kickoff','defense'],as_index=False)[[stat,opportunity]].sum(min_count=1).rename(columns={stat:'yards',opportunity:'opportunities'})
        contexts = {}
        for row in frame.itertuples():
            key = (row.game_id,row.opponent)
            if key not in contexts:
                contexts[key] = split_context(games,row.opponent,calendar[row.game_id])
        context = pd.DataFrame([contexts[(r.game_id,r.opponent)] for r in frame.itertuples()],index=frame.index)
        frame = frame.drop(columns=['defense_games'],errors='ignore').join(context)
        model = fit_split(frame)
        models[f'{position}:{stat}'] = model
        volume,rate,adjusted = apply_split(frame.base_volume,frame.base_rate,frame.volume_factor,frame.rate_factor,model)
        frame['split_projection'] = adjusted
        frame['volume_projection'] = volume
        frame['rate_projection'] = rate
        evaluation = frame[frame.season.isin([2024,2025])].dropna(subset=['baseline','adjusted','split_projection','actual','actual_opportunities'])
        metrics.append(dict(position=position,stat=stat,games=len(evaluation),volume_coefficient=model['volume'],rate_coefficient=model['rate'],
                            baseline_mae=(evaluation.baseline-evaluation.actual).abs().mean(),
                            defense_mae=(evaluation.adjusted-evaluation.actual).abs().mean(),
                            split_mae=(evaluation.split_projection-evaluation.actual).abs().mean()))
        table = grouped_effects(evaluation)
        table['position']=position;table['stat']=stat;buckets.append(table)
        for column in frame:
            history.loc[frame.index,column]=frame[column]
        context_now = {}
        for index,row in predictions[predictions.position.eq(position)&predictions.stat.eq(stat)].iterrows():
            if row.opponent not in context_now:
                context_now[row.opponent]=split_context(games,row.opponent,cutoff)
            c=context_now[row.opponent]
            for field,value in c.items(): predictions.loc[index,field]=value
            if pd.notna(row.projection) and np.isfinite(c['volume_factor']) and np.isfinite(c['rate_factor']):
                v,r,p=apply_split(row.projected_opportunities,row.efficiency,c['volume_factor'],c['rate_factor'],model)
                predictions.loc[index,'projection']=p
                predictions.loc[index,'adjusted_opportunities']=v
                predictions.loc[index,'adjusted_efficiency']=r
                predictions.loc[index,'matchup_adjustment_applied']=True
            else:
                predictions.loc[index,'review_flags']+='; Split matchup unavailable: original estimate retained if available'
    folder=root/'player_matchup_forecasts'/str(season)/f'week_{week:02d}'/(now.strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True,exist_ok=False)
    predictions.to_parquet(folder/'predictions.parquet',index=False)
    history.to_parquet(folder/'historical_matchups.parquet',index=False)
    pd.DataFrame(metrics).to_csv(folder/'historical_errors.csv',index=False)
    pd.concat(buckets,ignore_index=True).to_csv(folder/'grouped_effects.csv',index=False)
    meta=dict(previous_meta,created_at=str(now),models=models,
              method='Split matchup experiment: separately adjusted opportunities × adjusted yards per opportunity. Position-specific opponent history, prior five games, 365-day limit.',
              evaluation='2022–2023 train-only signed coefficients; paired reused 2024–2025 evaluation on observed player-stat rows. Descriptive groups and game-cluster bootstrap intervals are not causal controls for injuries, role, schedule or game script.',
              input_hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[history_path,previous/'manifest.json']},
              predictions_sha256=hashlib.sha256((folder/'predictions.parquet').read_bytes()).hexdigest(),
              code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (folder/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return folder
