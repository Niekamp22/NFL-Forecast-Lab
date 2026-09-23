"""Reviewed, slate-scoped news evidence. Never infer a numerical injury penalty."""
import json
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


def load_availability(root, season, week, as_of):
    path = Path(root)/'availability_updates'/str(season)/f'week_{week:02d}.json'
    if not path.exists():
        return None, None
    evidence = json.loads(path.read_text(encoding='utf-8'))
    if (evidence['season'], evidence['week']) != (season, week):
        raise ValueError('Availability evidence belongs to another slate')
    cutoff = pd.Timestamp(as_of)
    reviewed = pd.Timestamp(evidence['reviewed_at'])
    if reviewed.tzinfo is None or reviewed > cutoff:
        raise ValueError('Availability review is after the forecast cutoff')
    seen = set()
    starters = set()
    for item in evidence['players']:
        key = (item['team'], item['player_id'])
        if key in seen:
            raise ValueError('Duplicate availability evidence')
        seen.add(key)
        if item['action'] not in {'starter', 'unavailable', 'uncertain', 'note'}:
            raise ValueError('Unsupported availability action')
        published = pd.Timestamp(item['published_at'])
        if published.tzinfo is None or published > reviewed:
            raise ValueError('Source published after availability review')
        if urlparse(item['source_url']).scheme != 'https':
            raise ValueError('Availability evidence requires a source URL')
        if item['action'] == 'starter':
            if item['team'] in starters:
                raise ValueError('Conflicting confirmed starters')
            starters.add(item['team'])
    return path, evidence


def validate_availability(evidence, roster, history, current):
    if not evidence:
        return
    if pd.Timestamp(evidence['reviewed_at']) >= current.kickoff.min():
        raise ValueError('Availability review must precede the target slate')
    for item in evidence['players']:
        match = roster[roster.team.eq(item['team']) & roster.player_id.eq(item['player_id'])]
        if len(match) != 1 or match.iloc[0].player_name != item['player_name']:
            raise ValueError('Availability player identity does not match the pinned roster')
        if item['action'] == 'starter' and match.iloc[0].position != 'QB':
            raise ValueError('Starter evidence must identify a quarterback')
        for game_id in item.get('shortened_games', []):
            observed = history[history.team.eq(item['team']) & history.player_id.eq(item['player_id']) & history.game_id.eq(game_id)]
            if len(observed) != 1:
                raise ValueError('Injury-shortened game not in pregame player history')


def workload_history(prior, item):
    """Remove verified partial appearances from workload shares, never actual results."""
    return prior[~prior.game_id.isin(item.get('shortened_games', []))]
