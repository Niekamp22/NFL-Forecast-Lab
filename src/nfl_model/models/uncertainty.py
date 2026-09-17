"""Paired season-stratified week-cluster bootstrap of margin-error improvements."""
import numpy as np
import pandas as pd


def paired_interval(baseline: pd.DataFrame, candidate: pd.DataFrame, repetitions: int = 10000, seed: int = 2026) -> dict:
    """Resample whole weeks within each season, preserving paired game errors.

    Positive values mean lower candidate MAE. This approximation does not remove
    serial dependence across weeks or bias from repeatedly inspecting evaluation years.
    """
    if baseline.game_id.duplicated().any() or candidate.game_id.duplicated().any():
        raise ValueError("Duplicate predictions")
    if set(baseline.game_id) != set(candidate.game_id):
        raise ValueError("Prediction cohorts differ")
    pair = baseline.merge(candidate[["game_id", "predicted_margin", "actual_margin"]], on="game_id", suffixes=("_base", "_candidate"), validate="one_to_one")
    if not np.allclose(pair.actual_margin_base, pair.actual_margin_candidate, equal_nan=True):
        raise ValueError("Outcome mismatch")
    pair = pair[pair.actual_margin_base.notna()].copy()
    pair["gain"] = (pair.predicted_margin_base-pair.actual_margin_base).abs() - (pair.predicted_margin_candidate-pair.actual_margin_base).abs()
    rng = np.random.default_rng(seed)
    numerator, denominator = np.zeros(repetitions), np.zeros(repetitions)
    for _, season in pair.groupby("season"):
        clusters = season.groupby("week").gain.agg(["sum", "count"])
        sample = rng.integers(0, len(clusters), size=(repetitions, len(clusters)))
        numerator += clusters["sum"].to_numpy()[sample].sum(axis=1)
        denominator += clusters["count"].to_numpy()[sample].sum(axis=1)
    draws = numerator / denominator
    return {"games": len(pair), "mae_improvement": float(pair.gain.mean()), "ci95": np.quantile(draws, [.025, .975]).tolist(),
            "repetitions": repetitions, "seed": seed, "method": "paired week clusters stratified by season"}
