"""Timestamp-based depth-chart selection; no inference from actual game usage."""
import pandas as pd

from .identity import resolve_ids
from .player_week import regular


def select_depth(depth: pd.DataFrame, schedule: pd.DataFrame, crosswalk: pd.DataFrame,
                 as_of: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one team chart snapshot before kickoff and no later than build cutoff.

    These source timestamps do not certify historical retrieval availability.
    Keep multiple positions per player in this separate long-form table.
    """
    depth = resolve_ids(depth, crosswalk)
    depth["depth_timestamp"] = pd.to_datetime(depth.dt, utc=True, errors="raise")
    unresolved = depth[depth.player_id.isna()].copy()
    unresolved["source_dataset"] = "depth_charts"
    schedule = regular(schedule)
    local = pd.to_datetime(schedule.gameday.astype(str) + " " + schedule.gametime.astype(str), errors="coerce")
    kickoffs = local.dt.tz_localize("America/New_York", ambiguous="raise", nonexistent="raise").dt.tz_convert("UTC")
    pieces = []
    for (_, game), kickoff in zip(schedule.iterrows(), kickoffs):
        if pd.isna(kickoff):
            continue
        for side in ["home_team", "away_team"]:
            eligible = depth[(depth.team == game[side]) & (depth.depth_timestamp < kickoff) & (depth.depth_timestamp <= as_of)]
            if eligible.empty:
                continue
            latest = eligible.depth_timestamp.max()
            chosen = eligible[eligible.depth_timestamp == latest].copy()
            chosen["season"] = game.season
            chosen["week"] = game.week
            chosen["game_id"] = game.game_id
            chosen["kickoff_utc"] = kickoff
            chosen["availability_certified_pregame"] = False
            pieces.append(chosen)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(), unresolved
