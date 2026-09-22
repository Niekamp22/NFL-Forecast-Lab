"""Consumer display helpers; do not alter forecast values."""
import hashlib
import json
import pandas as pd


def matching_player_results(root, season, week, defense, meta):
    family = 'player_defense_evaluations' if defense else 'player_evaluations'
    candidates = []
    if not meta.get('predictions_sha256'):
        return None
    for path in (root / family / str(season) / f'week_{week:02d}').glob('*/manifest.json'):
        run = json.loads(path.read_text())
        if run.get('snapshot_hash') == meta.get('predictions_sha256'):
            candidates.append((pd.Timestamp(run['created_at']), path, run))
    if not candidates:
        return None
    _, path, run = max(candidates, key=lambda x: x[0])
    grades = path.parent / 'grades.parquet'
    if hashlib.sha256(grades.read_bytes()).hexdigest() != run.get('grades_sha256'):
        raise ValueError('Results checksum mismatch')
    return path.parent, run, pd.read_parquet(grades)


def game_status(kickoff, verified_final=False, now=None):
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    if verified_final:
        return 'Final · verified results available'
    if pd.Timestamp(kickoff) <= now:
        return 'Started · final result not verified here'
    return 'Upcoming'


def date_label(value):
    if value is None or pd.isna(value):
        return 'Unavailable'
    return pd.Timestamp(value).tz_convert('America/New_York').strftime('%b %d, %Y · %I:%M %p ET')


def verified_game_scores(result):
    """Require explicit final-game evidence, including when all player stats are missing."""
    if result is None:
        return {}
    rows=result[2]
    if not {'game_result_status','home_score','away_score'}.issubset(rows.columns):
        return {}
    finals=rows.loc[rows.game_result_status.eq('graded'),['game_id','home_score','away_score']].drop_duplicates()
    if finals.game_id.duplicated().any():
        raise ValueError('Conflicting verified final scores')
    return {r.game_id:(r.home_score,r.away_score) for r in finals.itertuples()}
