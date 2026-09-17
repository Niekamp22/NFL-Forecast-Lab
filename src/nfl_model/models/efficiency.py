"""Chronological EPA/success-rate residual correction to frozen score ratings."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_SETS = {"epa": ["epa_per_play"], "success": ["success_rate"], "epa_success": ["epa_per_play", "success_rate"]}


def assemble_features(predictions: pd.DataFrame, teams: pd.DataFrame, window: str, feature_set: str) -> tuple[pd.DataFrame, list[str]]:
    """Join by game/team, with explicit football feature allowlist and timing checks."""
    if teams.duplicated(["game_id", "team"]).any():
        raise ValueError("Duplicate team feature keys")
    fields = [f"{side}_{metric}_{window}" for side in ["offense", "defense"] for metric in FEATURE_SETS[feature_set]]
    for side in ["offense", "defense"]:
        source_week = teams[f"{side}_max_source_week_{window}"]
        if (source_week >= teams.week).fillna(False).any():
            raise ValueError("Team feature uses same-week or future observations")
    result = predictions.copy()
    for venue in ["home", "away"]:
        selected = teams[["game_id", "team"] + fields].rename(columns={"team": f"{venue}_team", **{c: f"{venue}_{c}" for c in fields}})
        result = result.merge(selected, on=["game_id", f"{venue}_team"], how="left", validate="one_to_one", indicator=True)
        if not result._merge.eq("both").all():
            raise ValueError("Missing team feature row")
        result = result.drop(columns="_merge")
    names = []
    for field in fields:
        name = f"difference_{field}"
        result[name] = result[f"home_{field}"] - result[f"away_{field}"]
        names.append(name)
    return result, names


def fit_correction(train: pd.DataFrame, target: pd.DataFrame, columns: list[str], ridge: float) -> tuple[np.ndarray, dict]:
    """Train-only imputation/scaling, missing indicators, then residual ridge fit."""
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    x = train[columns].to_numpy(dtype=float)
    z = target[columns].to_numpy(dtype=float)
    count = np.isfinite(x).sum(axis=0)
    mean = np.divide(np.nansum(x, axis=0), count, out=np.zeros(len(columns)), where=count > 0)
    missing_x, missing_z = ~np.isfinite(x), ~np.isfinite(z)
    x = np.where(missing_x, mean, x)
    z = np.where(missing_z, mean, z)
    scale = np.std(x, axis=0)
    scale = np.where(scale > 1e-8, scale, 1)
    x = np.column_stack([np.ones(len(x)), (x-mean)/scale, missing_x.astype(float)])
    z = np.column_stack([np.ones(len(z)), (z-mean)/scale, missing_z.astype(float)])
    residual = (train.actual_margin - train.predicted_margin).to_numpy(dtype=float)
    penalty = np.eye(x.shape[1]) * ridge
    penalty[0, 0] = 1e-8
    beta = np.linalg.solve(x.T @ x + penalty, x.T @ residual)
    return z @ beta, {"means": mean.tolist(), "scales": scale.tolist(), "coefficients": beta.tolist(), "features": columns}


def replay_correction(frame: pd.DataFrame, columns: list[str], ridge: float, seasons: list[int]) -> tuple[pd.DataFrame, list[dict]]:
    """Residual models see only completed prior weeks; base predictions are pregame."""
    results, states = [], []
    for (season, week), target in frame[frame.season.isin(seasons)].groupby(["season", "week"], sort=True):
        prior = (frame.season < season) | ((frame.season == season) & (frame.week < week))
        cutoff = pd.to_datetime(target.cutoff_date).min()
        train = frame[prior & frame.actual_margin.notna() & (pd.to_datetime(frame.gameday) < cutoff)]
        if train.empty:
            raise ValueError("Residual model needs prior completed games")
        correction, state = fit_correction(train, target, columns, ridge)
        output = target.copy()
        output["score_rating_margin"] = output.predicted_margin
        output["predicted_margin"] = output.predicted_margin + correction
        output["residual_training_games"] = len(train)
        results.append(output)
        states.append({"season": int(season), "week": int(week), "cutoff_date": str(cutoff),
                       "training_games": len(train), "last_training_date": str(train.gameday.max()), **state})
    return pd.concat(results, ignore_index=True), states
