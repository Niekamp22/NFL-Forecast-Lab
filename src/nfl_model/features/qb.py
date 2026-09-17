"""QB observations and prior-game scenarios without hindsight starter labels."""
import pandas as pd


def qb_games(pbp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attribute dropbacks to passer IDs, or rusher IDs for scrambles only."""
    plays = pbp[pbp.season_type.eq("REG") & pbp.qb_dropback.eq(1) & pbp.play_type.isin(["pass", "run"])].copy()
    plays["player_id"] = plays.passer_player_id
    scramble = plays.qb_scramble.eq(1)
    plays.loc[scramble, "player_id"] = plays.loc[scramble, "rusher_player_id"]
    missing = plays[plays.player_id.isna()].copy()
    valid = plays[plays.player_id.notna()].copy()
    valid["team"] = valid.posteam
    metrics = {"epa": "epa", "success": "success", "cpoe": "cpoe", "sack": "sack", "scramble": "qb_scramble", "interception": "interception"}
    keys = ["season", "week", "game_id", "team", "player_id"]
    group = valid.groupby(keys)
    result = group.size().rename("dropbacks").to_frame()
    for name, source in metrics.items():
        result[f"{name}_sum"] = group[source].sum(min_count=1)
        result[f"{name}_n"] = group[source].count()
        result[f"{name}_rate"] = result[f"{name}_sum"] / result[f"{name}_n"].replace(0, float("nan"))
    return result.reset_index(), missing


def qb_scenarios(observations: pd.DataFrame, schedule: pd.DataFrame, prior_weight: float = 100) -> pd.DataFrame:
    """For each game/team use its most recent game's highest-dropback QB as a proxy.

    This is a mechanical prior-usage scenario, not an expected/confirmed starter.
    QB performance follows IDs across teams. Last-five observed games are shrunk
    toward the league's earlier-game mean with an explicit 100-observation prior.
    The prior is an initial smoothing choice, not an assigned point value.
    """
    rows = []
    for (season, week), games in schedule.groupby(["season", "week"], sort=True):
        history = observations[(observations.season < season) | ((observations.season == season) & (observations.week < week))]
        for game in games.itertuples():
            for team in [game.home_team, game.away_team]:
                row = dict(season=season, week=week, game_id=game.game_id, team=team, player_id=None,
                           starter_evidence="prior_usage_proxy", confirmed_starter=False, prior_games=0)
                previous = history[history.team == team].sort_values(["season", "week"])
                if len(previous):
                    last = previous.iloc[-1]
                    candidates = previous[(previous.season == last.season) & (previous.week == last.week)].sort_values(["dropbacks", "player_id"], ascending=[False, True])
                    player = candidates.iloc[0].player_id
                    career = history[history.player_id == player].sort_values(["season", "week"])
                    recent = career.tail(5)
                    row.update(player_id=player, prior_games=len(career), recent_games=len(recent),
                               last_source_season=int(recent.season.iloc[-1]), last_source_week=int(recent.week.iloc[-1]))
                    for metric in ["epa", "success", "cpoe", "sack", "scramble", "interception"]:
                        n = recent[f"{metric}_n"].sum()
                        league_n = history[f"{metric}_n"].sum()
                        league_mean = history[f"{metric}_sum"].sum()/league_n if league_n else float("nan")
                        row[f"prior_{metric}"] = (recent[f"{metric}_sum"].sum() + prior_weight * league_mean)/(n+prior_weight) if n else float("nan")
                        row[f"recent_{metric}_n"] = n
                rows.append(row)
    return pd.DataFrame(rows)
