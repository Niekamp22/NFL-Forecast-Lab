"""Historical matchup examples and a train-only opponent yardage adjustment."""
import hashlib
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from .player_projection import opportunity_projection
from .weekly_forecast import earliest_snapshot

YARDS = {'QB':['passing_yards'], 'RB':['rushing_yards', 'receiving_yards'],
         'WR':['receiving_yards'], 'TE':['receiving_yards']}


def defense_context(games, opponent, cutoff, metric):
    past = games[(games.kickoff < cutoff) & (games.kickoff >= cutoff-pd.Timedelta(days=365))]
    prior = past[past.defense.eq(opponent)].sort_values('kickoff').tail(5)
    allowed = prior[metric].mean()
    league = past[metric].mean()
    n = int(prior[metric].notna().sum())
    ratio = allowed/league-1 if n >= 3 and pd.notna(league) and league > 0 else np.nan
    return dict(defense_allowed=allowed, league_allowed=league, defense_games=n, defense_factor=ratio)


def fit_adjustment(frame):
    """Fixed ridge penalty and 0..1 slope; fit only 2022–23, never evaluation rows."""
    train = frame[frame.season.isin([2022, 2023])].dropna(subset=['baseline', 'actual', 'defense_factor'])
    if len(train) < 30:
        raise ValueError('Insufficient training matchups')
    x = (train.baseline * train.defense_factor).to_numpy()
    residual = (train.actual-train.baseline).to_numpy()
    return float(np.clip(x@residual/(x@x+10000.0), 0, 1))


def run_player_defense(client, season, week):
    root = client.root
    now = pd.Timestamp.now(tz='UTC')
    original, original_meta = earliest_snapshot(root/'player_opportunity_forecasts'/str(season)/f'week_{week:02d}')
    prediction_file = original/'predictions.parquet'
    if hashlib.sha256(prediction_file.read_bytes()).hexdigest() != original_meta['predictions_sha256']:
        raise ValueError('Opportunity snapshot integrity failure')
    if now >= pd.Timestamp(original_meta['earliest_kickoff']):
        raise ValueError('Cannot create defense forecasts after target kickoff')
    sources = [prediction_file]
    schedule_path = client.download('schedules')
    sources.append(schedule_path)
    schedule = pd.read_parquet(schedule_path)
    schedule = schedule[schedule.game_type.eq('REG')].copy()
    schedule['kickoff'] = pd.to_datetime(schedule.gameday.astype(str)+' '+schedule.gametime.astype(str), errors='coerce').dt.tz_localize('America/New_York').dt.tz_convert('UTC')
    frames = []
    for year in range(2021, season+1):
        path = client.download('player_stats', year)
        sources.append(path)
        frames.append(pd.read_parquet(path))
    stats = pd.concat(frames, ignore_index=True)
    stats = stats[stats.season_type.eq('REG') & stats.player_id.notna()]
    if stats.duplicated(['game_id', 'player_id']).any():
        raise ValueError('Duplicate player-game rows')
    stats = stats.merge(schedule[['game_id','kickoff','home_team','away_team','home_score','away_score']], on='game_id', validate='many_to_one')
    cutoff = pd.Timestamp(original_meta['created_at'])
    stats = stats[stats.kickoff.lt(cutoff) & stats.home_score.notna() & stats.away_score.notna()].copy()
    if not (stats.team.eq(stats.home_team) | stats.team.eq(stats.away_team)).all():
        raise ValueError('Player team does not match schedule')
    stats['defense'] = np.where(stats.team.eq(stats.home_team), stats.away_team, stats.home_team)
    games = stats.groupby(['game_id','kickoff','defense'], as_index=False)[['receiving_yards','rushing_yards']].sum(min_count=1)
    # Receiving yards allowed includes all positions and excludes sack losses.
    contexts = {}
    rows = []
    for pid, group in stats[stats.position.isin(YARDS)].groupby('player_id'):
        group = group.sort_values('kickoff')
        for index, actual in enumerate(group.itertuples()):
            if actual.season not in [2022, 2023, 2024, 2025]:
                continue
            past = group.iloc[max(0,index-10):index]
            for stat in YARDS[actual.position]:
                metric = 'rushing_yards' if stat == 'rushing_yards' else 'receiving_yards'
                key = (actual.game_id, actual.defense, metric)
                if key not in contexts:
                    contexts[key] = defense_context(games, actual.defense, actual.kickoff, metric)
                baseline = opportunity_projection(past, actual.kickoff, stat)['projection']
                if pd.isna(baseline) or pd.isna(getattr(actual,stat)):
                    continue
                recent = past[past.kickoff.ge(actual.kickoff-pd.Timedelta(days=365))].tail(5)
                rows.append(dict(season=actual.season, week=actual.week, game_id=actual.game_id,
                                 player_id=pid, player_name=actual.player_display_name, team=actual.team,
                                 opponent=actual.defense, position=actual.position, stat=stat,
                                 prior_player_average=recent[stat].mean(), prior_player_games=int(recent[stat].notna().sum()),
                                 baseline=baseline, actual=getattr(actual,stat), **contexts[key]))
    history = pd.DataFrame(rows)
    models, summaries = {}, []
    history['adjusted'] = np.nan
    for (position,stat), frame in history.groupby(['position','stat']):
        beta = fit_adjustment(frame)
        models[f'{position}:{stat}'] = beta
        prediction = np.maximum(0, frame.baseline*(1+beta*frame.defense_factor))
        history.loc[frame.index,'adjusted'] = prediction
        evaluation = frame[frame.season.isin([2024,2025]) & frame.defense_factor.notna()].copy()
        adjusted = np.maximum(0, evaluation.baseline*(1+beta*evaluation.defense_factor))
        summaries.append(dict(position=position, stat=stat, coefficient=beta, games=len(evaluation),
                              baseline_mae=float((evaluation.baseline-evaluation.actual).abs().mean()),
                              defense_mae=float((adjusted-evaluation.actual).abs().mean())))
    predictions = pd.read_parquet(prediction_file)
    predictions['opportunity_projection'] = predictions.projection
    predictions['defense_adjustment_applied'] = False
    for index, row in predictions.iterrows():
        model_key = f'{row.position}:{row.stat}'
        if model_key not in models:
            continue
        metric = 'rushing_yards' if row.stat == 'rushing_yards' else 'receiving_yards'
        context = defense_context(games, row.opponent, cutoff, metric)
        for field, value in context.items():
            predictions.loc[index,field] = value
        beta = models[model_key]
        predictions.loc[index,'defense_coefficient'] = beta
        if pd.notna(row.projection) and pd.notna(context['defense_factor']):
            predictions.loc[index,'projection'] = max(0,row.projection*(1+beta*context['defense_factor']))
            predictions.loc[index,'defense_adjustment_applied'] = True
        else:
            predictions.loc[index,'review_flags'] += '; Defense adjustment unavailable: using original estimate if available'
    version = now.strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8]
    folder = root/'player_defense_forecasts'/str(season)/f'week_{week:02d}'/version
    folder.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(folder/'predictions.parquet', index=False)
    predictions.to_csv(folder/'predictions.csv', index=False)
    history.to_parquet(folder/'historical_matchups.parquet', index=False)
    metrics = pd.DataFrame(summaries)
    metrics['mae_improvement'] = metrics.baseline_mae-metrics.defense_mae
    metrics.to_csv(folder/'historical_errors.csv',index=False)
    # Explicit selection for illustrations, not evidence of a general effect.
    examples = history[history.position.eq('WR') & history.season.isin([2024,2025]) & history.prior_player_average.between(65,90) & history.prior_player_games.eq(5) & history.defense_factor.gt(0)]
    examples = examples.assign(change_from_average=examples.actual-examples.prior_player_average).sort_values('change_from_average',ascending=False)
    examples.to_csv(folder/'wr_examples.csv',index=False)
    meta = dict(original_meta, created_at=str(now), opportunity_snapshot=str(original.resolve()),
                method='Experimental defense yardage adjustment: opportunity projection × (1 + fitted coefficient × prior defense deviation from league average). QB passing, RB rushing/receiving, WR/TE receiving only.',
                evaluation='Fixed coefficients trained on 2022–2023; paired 2024–2025 evaluation reused during research. Observed player-stat rows only; no active-status backtest. Opponent averages are descriptive, not schedule-adjusted or causal.',
                models=models, feature_cutoff=str(cutoff),
                input_hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                predictions_sha256=hashlib.sha256((folder/'predictions.parquet').read_bytes()).hexdigest(),
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (folder/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return folder
