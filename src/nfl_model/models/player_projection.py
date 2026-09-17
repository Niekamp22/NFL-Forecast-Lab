"""Conditional recent-role baseline; no active probability or matchup adjustment."""
import hashlib
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

STATS = {
    'QB': ['attempts', 'completions', 'passing_yards', 'passing_tds', 'passing_interceptions', 'carries', 'rushing_yards', 'rushing_tds'],
    'RB': ['carries', 'rushing_yards', 'rushing_tds', 'targets', 'receptions', 'receiving_yards', 'receiving_tds'],
    'WR': ['targets', 'receptions', 'receiving_yards', 'receiving_tds', 'carries', 'rushing_yards'],
    'TE': ['targets', 'receptions', 'receiving_yards', 'receiving_tds'],
    'K': ['fg_att', 'fg_made', 'pat_att', 'pat_made'],
}


OPPORTUNITIES = {
    'completions': 'attempts', 'passing_yards': 'attempts',
    'passing_tds': 'attempts', 'passing_interceptions': 'attempts',
    'rushing_yards': 'carries', 'rushing_tds': 'carries',
    'receptions': 'targets', 'receiving_yards': 'targets', 'receiving_tds': 'targets',
    'fg_made': 'fg_att', 'pat_made': 'pat_att',
}


def opportunity_projection(history, cutoff, stat):
    """Five-game workload times pooled ten-game production per opportunity.

    Separate windows let workload respond faster than efficiency. No fitted
    hyperparameters. Zero historical opportunities leave the rate unknown.
    """
    opportunity = OPPORTUNITIES.get(stat, stat)
    volume, n = recent_projection(history, cutoff, opportunity)
    if opportunity == stat:
        return dict(projection=volume, opportunity=opportunity, projected_opportunities=volume,
                    efficiency=1.0, efficiency_opportunities=np.nan, history_games=n)
    past = history[(history.kickoff < cutoff) & (history.kickoff >= cutoff-pd.Timedelta(days=365))].sort_values('kickoff').tail(10)
    valid = past[stat].notna() & past[opportunity].notna() & past[opportunity].ge(0)
    denominator = float(past.loc[valid, opportunity].sum())
    rate = float(past.loc[valid, stat].sum()) / denominator if denominator > 0 else np.nan
    return dict(projection=volume*rate, opportunity=opportunity, projected_opportunities=volume,
                efficiency=rate, efficiency_opportunities=denominator, history_games=n)


def recent_projection(history, cutoff, stat):
    """Up to five observed games in 365 days, weights 1..n, strictly before cutoff.

    Missing observations stay missing; absent game rows are not inserted as zeros.
    """
    past = history[(history.kickoff < cutoff) & (history.kickoff >= cutoff - pd.Timedelta(days=365))].sort_values('kickoff').tail(5)
    valid = past[stat].notna()
    n = int(valid.sum())
    if not n:
        return np.nan, 0
    weights = np.arange(1, len(past) + 1)[valid.to_numpy()]
    return float(np.average(past.loc[valid, stat], weights=weights)), n


def build_player_projections(client, season, week, opportunity_model=False):
    root = client.root
    now = pd.Timestamp.now(tz='UTC')
    build = Path(json.loads((root / 'processed/latest.json').read_text())['path'])
    roster_path = build / 'player_week.parquet'
    roster = pd.read_parquet(roster_path)
    roster = roster[roster.season.eq(season) & roster.week.eq(week) & roster.position.isin(STATS)].copy()
    if roster.empty:
        raise ValueError('No player roster for requested week; build the current database first')
    schedule_path = client.download('schedules')
    schedule = pd.read_parquet(schedule_path)
    schedule = schedule[schedule.game_type.eq('REG')].copy()
    schedule['kickoff'] = pd.to_datetime(schedule.gameday.astype(str) + ' ' + schedule.gametime.astype(str), errors='coerce').dt.tz_localize('America/New_York').dt.tz_convert('UTC')
    target = schedule[schedule.season.eq(season) & schedule.week.eq(week)]
    if target.empty or target.kickoff.isna().any() or (target.kickoff <= now).any():
        raise ValueError('Player forecasts must be created before every target kickoff')
    if target[['home_score', 'away_score']].notna().any().any():
        raise ValueError('Target results already present')
    sources = [roster_path, schedule_path]
    frames = []
    for year in range(2021, season + 1):
        path = client.download('player_stats', year)
        sources.append(path)
        frames.append(pd.read_parquet(path))
    history = pd.concat(frames, ignore_index=True)
    history = history[history.season_type.eq('REG') & history.position.isin(STATS) & history.player_id.notna()]
    if history.duplicated(['game_id', 'player_id']).any():
        raise ValueError('Duplicate player-game statistics')
    history = history.merge(schedule[['game_id', 'kickoff', 'home_score', 'away_score']], on='game_id', validate='many_to_one')
    history = history[history.kickoff.lt(now) & history.home_score.notna() & history.away_score.notna()]
    groups = {pid: g.sort_values('kickoff') for pid, g in history.groupby('player_id')}
    rows = []
    for player in roster.itertuples():
        games = target[target.home_team.eq(player.team) | target.away_team.eq(player.team)]
        if len(games) != 1:
            continue
        game = games.iloc[0]
        past = groups.get(player.player_id, history.iloc[:0])
        flags = ['Conditional on a similar role; active status unconfirmed']
        blocked = pd.isna(player.roster_status) or player.roster_status != 'ACT' or (pd.notna(player.injury_status) and player.injury_status == 'Out')
        if blocked:
            flags.append('Projection withheld: roster/injury status')
        if pd.isna(player.injury_status):
            flags.append('Injury report unknown')
        else:
            flags.append(f'Injury designation: {player.injury_status}')
        recent = past[past.kickoff.ge(now - pd.Timedelta(days=365))]
        if not recent.empty and recent.iloc[-1].team != player.team:
            flags.append('Team changed since last observed game')
        if not recent.empty and now - recent.iloc[-1].kickoff > pd.Timedelta(days=45):
            flags.append('No recent game in 45 days')
        if player.position == 'QB':
            flags.append('Starter not confirmed; backup/partial games affect average')
        for stat in STATS[player.position]:
            estimate, n = recent_projection(past, now, stat)
            details = {}
            if opportunity_model:
                details = opportunity_projection(past, now, stat)
                estimate = details.pop('projection')
                n = details.pop('history_games')
                details['baseline_projection'] = recent_projection(past, now, stat)[0]
                if blocked:
                    details['baseline_projection'] = np.nan
                    details['projected_opportunities'] = np.nan
            rows.append(dict(season=season, week=week, game_id=game.game_id, kickoff_utc=game.kickoff,
                             team=player.team, opponent=game.away_team if player.team == game.home_team else game.home_team,
                             player_id=player.player_id, player_name=player.player_name, position=player.position,
                             stat=stat, projection=np.nan if blocked else estimate, history_games=n, **details,
                             roster_status=player.roster_status, review_flags='; '.join(flags + (['Limited history'] if n < 3 else []))))
    predictions = pd.DataFrame(rows)
    # Sequential retrospective benchmark: target rows never enter their own inputs.
    errors = []
    for pid, group in groups.items():
        for index, actual in enumerate(group.itertuples()):
            if actual.season not in [2024, 2025]:
                continue
            available = group.iloc[max(0, index-10):index]
            past = available.tail(5)
            past = past[past.kickoff.ge(actual.kickoff - pd.Timedelta(days=365)) & past.kickoff.lt(actual.kickoff)]
            for stat in STATS[actual.position]:
                valid = past[stat].notna()
                n = int(valid.sum())
                estimate = float(np.average(past.loc[valid, stat], weights=np.arange(1, len(past)+1)[valid.to_numpy()])) if n else np.nan
                value = getattr(actual, stat)
                candidate = opportunity_projection(available, actual.kickoff, stat)['projection'] if opportunity_model else estimate
                if n and pd.notna(value) and pd.notna(candidate):
                    errors.append(dict(position=actual.position, stat=stat, error=candidate-value, baseline_error=estimate-value))
    errors = pd.DataFrame(errors)
    metrics = errors.groupby(['position', 'stat']).error.agg(games='size', mae=lambda x: x.abs().mean(), bias='mean').reset_index()
    if opportunity_model:
        comparison = errors.groupby(['position', 'stat']).baseline_error.agg(baseline_mae=lambda x:x.abs().mean()).reset_index()
        metrics = metrics.merge(comparison, on=['position', 'stat'])
        metrics['mae_improvement'] = metrics.baseline_mae - metrics.mae
    stamp = now.strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid.uuid4().hex[:8]
    folder = root / ('player_opportunity_forecasts' if opportunity_model else 'player_forecasts') / str(season) / f'week_{week:02d}' / stamp
    folder.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(folder / 'predictions.parquet', index=False)
    predictions.to_csv(folder / 'predictions.csv', index=False)
    metrics.to_csv(folder / 'historical_errors.csv', index=False)
    meta = dict(created_at=str(now), season=season, week=week, earliest_kickoff=str(target.kickoff.min()),
                rows=len(predictions), players=int(predictions.player_id.nunique()),
                method='Last five observed player games within 365 days; linear recency weights 1..n; no opponent or availability adjustment',
                evaluation='2024–2025 observed stat-row cohort only; excludes missing rows and no-history players, not an unconditional active/inactive backtest. Sources retrieved retrospectively.',
                input_hashes={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                predictions_sha256=hashlib.sha256((folder / 'predictions.parquet').read_bytes()).hexdigest(),
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if opportunity_model:
        meta['method'] = 'Five-game weighted opportunities times ten-game pooled efficiency; 365-day lookback; experimental, no opponent/active adjustment'
        meta['evaluation'] += ' Paired comparison uses identical rows with both predictions available; positive MAE improvement favors the opportunity model. No model promotion based on this reused evaluation.'
        meta['evaluation_candidate_rows'] = len(errors)
    (folder / 'manifest.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return folder
