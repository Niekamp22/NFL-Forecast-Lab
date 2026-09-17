"""Football-only efficiency from scrimmage plays, aggregated before target weeks."""
from __future__ import annotations

import pandas as pd

METRICS = ["epa_per_play", "epa_per_dropback", "epa_per_rush", "success_rate",
           "explosive_pass_rate", "explosive_rush_rate", "sack_rate", "turnover_rate"]


def contributions(pbp: pd.DataFrame) -> pd.DataFrame:
    """Create sums/counts: sacks and scrambles are dropbacks, not designed rushes.

    Exclude no-play penalties, special teams, kneels and spikes. Explosive passes
    gain 20+ yards on pass attempts excluding sacks; designed rushes gain 10+.
    Unknown values do not enter either the numerator or denominator.
    """
    required = ["season", "week", "season_type", "game_id", "posteam", "defteam", "play_id",
                "play_type", "epa", "success", "qb_dropback", "rush_attempt", "pass_attempt",
                "sack", "interception", "fumble_lost", "yards_gained"]
    missing = set(required) - set(pbp)
    if missing:
        raise ValueError(f"Missing play-by-play fields: {sorted(missing)}")
    if pbp.duplicated(["game_id", "play_id"]).any():
        raise ValueError("Duplicate game/play IDs")
    plays = pbp[pbp.season_type.eq("REG") & pbp.play_type.isin(["pass", "run"]) & pbp.posteam.notna() & pbp.defteam.notna()].copy()
    dropback = plays.qb_dropback.eq(1)
    rush = plays.rush_attempt.eq(1) & plays.qb_dropback.eq(0)
    passes = plays.pass_attempt.eq(1) & plays.sack.eq(0)
    values = {
        "epa_per_play": plays.epa,
        "epa_per_dropback": plays.epa.where(dropback),
        "epa_per_rush": plays.epa.where(rush),
        "success_rate": plays.success,
        "explosive_pass_rate": plays.yards_gained.ge(20).astype(float).where(passes & plays.yards_gained.notna()),
        "explosive_rush_rate": plays.yards_gained.ge(10).astype(float).where(rush & plays.yards_gained.notna()),
        "sack_rate": plays.sack.where(dropback),
        "turnover_rate": (plays.interception.eq(1) | plays.fumble_lost.eq(1)).astype(float).where(plays.interception.notna() & plays.fumble_lost.notna()),
    }
    for name, series in values.items():
        plays[f"{name}_sum"] = series.fillna(0)
        plays[f"{name}_count"] = series.notna().astype(int)
    return plays


def build_team_weeks(pbp: pd.DataFrame, schedule: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build REG scheduled team-week rows with play-weighted prior-game windows.

    Window membership follows scheduled games (so missing games are not silently
    replaced by older ones). Include only completed games in strictly earlier weeks.
    Defense values are opponent production allowed; negative defensive EPA is good.
    """
    games = schedule[schedule.game_type.eq("REG")].copy()
    if games.game_id.duplicated().any():
        raise ValueError("Duplicate schedule game IDs")
    plays = contributions(pbp)
    unknown = set(plays.game_id) - set(games.game_id)
    if unknown:
        raise ValueError(f"PBP games absent from schedule: {sorted(unknown)}")
    for field in ["season", "week"]:
        mapped = plays.game_id.map(games.set_index("game_id")[field])
        if not plays[field].eq(mapped).all():
            raise ValueError(f"PBP {field} disagrees with schedule")
    counts = [f"{m}_{suffix}" for m in METRICS for suffix in ["sum", "count"]]
    observations = []
    for side, teamcol in [("offense", "posteam"), ("defense", "defteam")]:
        group = plays.groupby(["season", "week", "game_id", teamcol])[counts].sum().reset_index().rename(columns={teamcol: "team"})
        group["side"] = side
        observations.append(group)
    observations = pd.concat(observations, ignore_index=True)
    records = []
    for _, game in games.iterrows():
        for venue, opponent in [("home", "away"), ("away", "home")]:
            team = game[f"{venue}_team"]
            row = {"season": game.season, "week": game.week, "team": team, "game_id": game.game_id,
                   "opponent": game[f"{opponent}_team"], "is_home": venue == "home", "game_date": game.gameday,
                   "rest_days": game.get(f"{venue}_rest"), "pregame_certified": False}
            prior = games[(games.season == game.season) & (games.week < game.week) &
                          ((games.home_team == team) | (games.away_team == team))].sort_values("week")
            for label, window in [("1", 1), ("3", 3), ("5", 5), ("std", None)]:
                scheduled = prior.tail(window) if window else prior
                complete = scheduled[scheduled.home_score.notna() & scheduled.away_score.notna()]
                row[f"prior_games_expected_{label}"] = len(scheduled)
                for side in ["offense", "defense"]:
                    obs = observations[(observations.team == team) & (observations.side == side) & observations.game_id.isin(complete.game_id)]
                    row[f"{side}_games_observed_{label}"] = len(obs)
                    row[f"{side}_max_source_week_{label}"] = obs.week.max() if len(obs) else None
                    for metric in METRICS:
                        denominator = int(obs[f"{metric}_count"].sum())
                        row[f"{side}_{metric}_{label}"] = obs[f"{metric}_sum"].sum() / denominator if denominator else float("nan")
                        row[f"{side}_{metric}_{label}_n"] = denominator
            records.append(row)
    result = pd.DataFrame(records)
    if result.duplicated(["season", "week", "team"]).any():
        raise ValueError("Duplicate team-week keys")
    return result, observations
