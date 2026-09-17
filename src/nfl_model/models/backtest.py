"""Chronological tuning and held-out evaluation, with immutable result artifacts."""
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .ratings import RatingConfig, replay, score


def run_backtest(root: Path) -> Path:
    """2021 warm-up, 2022–23 tuning, 2024–25 untouched evaluation seasons."""
    history_folder = Path(json.loads((root / "processed/history/latest.json").read_text())["path"])
    manifest = json.loads((history_folder / "manifest.json").read_text())
    source = next(s for s in manifest["sources"] if s["dataset"] == "schedules")
    source_path = Path(source["path"])
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError("Schedule source checksum mismatch")
    # Explicit football-only allowlist. Raw market columns never enter this model.
    columns = ["season", "week", "game_id", "game_type", "gameday", "home_team", "away_team", "home_score", "away_score", "location"]
    games = pd.read_parquet(source_path, columns=columns)
    games = games[games.season.between(2021, 2025) & games.game_type.eq("REG")].copy()
    if set(games.season) != set(range(2021, 2026)):
        raise ValueError("Backtest requires all seasons 2021–2025")
    candidates = []
    for half_life in [60, 120, 240]:
        for ridge in [4, 12, 32]:
            config = RatingConfig(half_life, ridge)
            predictions, _ = replay(games[games.season <= 2023], config, [2022, 2023])
            metrics = score(predictions)
            candidates.append({"half_life_days": half_life, "ridge": ridge, "validation_mae": metrics["ratings"]["mae"]})
    tuning = pd.DataFrame(candidates).sort_values(["validation_mae", "half_life_days", "ridge"])
    selected = tuning.iloc[0]
    config = RatingConfig(float(selected.half_life_days), float(selected.ridge))
    predictions, states = replay(games, config, [2022, 2023, 2024, 2025])
    predictions["split"] = predictions.season.map({2022: "tuning", 2023: "tuning", 2024: "holdout", 2025: "holdout"})
    metrics = {"by_season": {str(y): score(g) for y, g in predictions.groupby("season")},
               "holdout": score(predictions[predictions.split.eq("holdout")])}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    folder = root / "models" / "baseline" / stamp
    folder.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(folder / "predictions.parquet", index=False)
    states.to_parquet(folder / "rating_history.parquet", index=False)
    tuning.to_csv(folder / "tuning.csv", index=False)
    run = {"version": stamp, "config": config.__dict__, "metrics": metrics, "schedule_source": source,
           "history_build": str(history_folder), "warmup": [2021], "tuning_seasons": [2022, 2023], "holdout_seasons": [2024, 2025],
           "publication_time_certified": False,
           "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}}
    (folder / "manifest.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    lines = ["# Team-rating baseline backtest", "", "Ratings are scoreboard points above/below league average. Predicted margin = home rating minus away rating plus estimated home field (zero venue adjustment at neutral sites).", "",
             f"Selected half-life: {config.half_life_days:g} days; ridge penalty: {config.ridge:g}.", "",
             "2021 initializes history. Nine configurations were compared using 2022–2023 MAE only. Hyperparameters were then frozen for 2024–2025. Ratings refit before each week using earlier completed games, including earlier holdout games as they become available.", "",
             "| Season | Split | Games | Rating MAE | Baseline MAE | Rating RMSE | Baseline RMSE | Winner accuracy |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for year, result in metrics["by_season"].items():
        a, b = result["ratings"], result["home_field_only"]
        lines.append(f"| {year} | {'Holdout' if int(year)>2023 else 'Tuning'} | {result['games']} | {a['mae']:.3f} | {b['mae']:.3f} | {a['rmse']:.3f} | {b['rmse']:.3f} | {a['winner_accuracy']:.1%} |")
    holdout = metrics["holdout"]
    lines += ["", "## Combined holdout", "", "```json", json.dumps(holdout, indent=2), "```", "",
              "MAE is average absolute margin error in points; RMSE penalizes large misses more. Winner accuracy excludes actual ties; a zero predicted margin counts as no correct winner. The home-field-only baseline estimates a decayed average home margin using identical training cutoffs and the selected half-life.", "",
              "## Limits", "", "This is a score-only benchmark, not the final model. No sportsbook inputs, win probabilities, totals, player/QB adjustments or betting recommendations. Historical results use currently retrieved corrected schedules; original publication timing is not certified. No uncertainty interval is estimated. Future model choices must not repeatedly tune against this same holdout and call it untouched.", "",
              f"Artifacts: {folder.resolve()}"]
    report = "\n".join(lines)
    (folder / "report.md").write_text(report, encoding="utf-8")
    Path("docs/baseline_backtest.md").write_text(report, encoding="utf-8")
    pointer = folder.parent / "latest.json"
    temp = pointer.with_name(f"latest_{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps({"path": str(folder.resolve()), "version": stamp}, indent=2), encoding="utf-8")
    temp.replace(pointer)
    return folder
