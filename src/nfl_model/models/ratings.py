"""Regularized, opponent-adjusted team strength in scoreboard points.

The model fits margin = home rating - away rating + home-field effect.
Only earlier weeks enter each fit. No market data or player availability inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RatingConfig:
    half_life_days: float = 120
    ridge: float = 12


def fit_ratings(history: pd.DataFrame, teams: list[str], cutoff: pd.Timestamp,
                config: RatingConfig) -> tuple[pd.Series, float, float]:
    """Weighted ridge least squares; team ratings shrink toward league average.

    Offseason time naturally decays older games. Home field is estimated with
    the same weights. A neutral-site game gets no home-field adjustment.
    Returns team ratings, home-field estimate, and a home-field-only baseline.
    """
    if config.half_life_days <= 0 or config.ridge <= 0:
        raise ValueError("half life and ridge must be positive")
    if history.empty:
        return pd.Series(0.0, index=teams), 0.0, 0.0
    dates = pd.to_datetime(history.gameday)
    if (dates >= cutoff).any():
        raise ValueError("History contains a game at or after prediction cutoff")
    index = {team: i for i, team in enumerate(teams)}
    x = np.zeros((len(history), len(teams) + 1))
    for row_number, row in enumerate(history.itertuples()):
        x[row_number, index[row.home_team]] = 1
        x[row_number, index[row.away_team]] = -1
    venue = history.location.eq("Home").to_numpy(dtype=float)
    x[:, -1] = venue
    margin = (history.home_score - history.away_score).to_numpy(dtype=float)
    age = (cutoff - dates).dt.total_seconds().to_numpy() / 86400
    weights = np.exp2(-age / config.half_life_days)
    penalty = np.eye(x.shape[1]) * config.ridge
    penalty[-1, -1] = 1e-8  # Numerically stable when historical games are all neutral.
    coefficient = np.linalg.solve(x.T @ (weights[:, None] * x) + penalty, x.T @ (weights * margin))
    baseline = float(np.sum(weights * venue * margin) / max(np.sum(weights * venue), 1e-8))
    return pd.Series(coefficient[:-1], index=teams), float(coefficient[-1]), baseline


def replay(games: pd.DataFrame, config: RatingConfig, predict_seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit once before each REG week, then freeze all predictions for that week."""
    if games.game_id.duplicated().any():
        raise ValueError("Duplicate schedule games")
    if not games.location.isin(["Home", "Neutral"]).all():
        raise ValueError("Unknown home/neutral location")
    teams = sorted(set(games.home_team) | set(games.away_team))
    predictions, states = [], []
    for (season, week), target in games[games.season.isin(predict_seasons)].groupby(["season", "week"], sort=True):
        cutoff = pd.to_datetime(target.gameday).min()
        earlier_week = (games.season < season) | ((games.season == season) & (games.week < week))
        history = games[earlier_week & games.home_score.notna() & games.away_score.notna() & (pd.to_datetime(games.gameday) < cutoff)]
        ratings, home_field, baseline = fit_ratings(history, teams, cutoff, config)
        for team, rating in ratings.items():
            states.append(dict(season=season, week=week, team=team, rating_points=rating, home_field=home_field,
                               training_games=len(history), cutoff_date=cutoff, last_training_date=history.gameday.max() if len(history) else None))
        for game in target.itertuples():
            venue = float(game.location == "Home")
            predictions.append(dict(season=season, week=week, game_id=game.game_id, home_team=game.home_team,
                                    away_team=game.away_team, gameday=game.gameday, location=game.location,
                                    predicted_margin=float(ratings[game.home_team] - ratings[game.away_team] + home_field * venue),
                                    baseline_margin=baseline * venue, actual_margin=game.home_score - game.away_score,
                                    training_games=len(history), cutoff_date=cutoff,
                                    last_training_date=history.gameday.max() if len(history) else None))
    return pd.DataFrame(predictions), pd.DataFrame(states)


def score(predictions: pd.DataFrame) -> dict:
    """Margin errors include ties; winner accuracy excludes tied final scores."""
    completed = predictions[predictions.actual_margin.notna()]
    output = {"games": len(completed)}
    for name, column in [("ratings", "predicted_margin"), ("home_field_only", "baseline_margin")]:
        error = completed[column] - completed.actual_margin
        decisive = completed[completed.actual_margin.ne(0)]
        output[name] = {"mae": float(error.abs().mean()), "rmse": float(np.sqrt((error**2).mean())),
                        "winner_accuracy": float((np.sign(decisive[column]) == np.sign(decisive.actual_margin)).mean()),
                        "winner_games": len(decisive), "mean_error": float(error.mean())}
    return output
