"""Predefined feature ablations; selection only on 2022–23 chronological errors."""
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .ratings import RatingConfig, replay, score
from .efficiency import FEATURE_SETS, assemble_features, replay_correction


def run_efficiency_experiment(root: Path) -> Path:
    """Compare EPA, success and their combination against the frozen baseline."""
    base_folder = Path(json.loads((root / "models/baseline/latest.json").read_text())["path"])
    baseline = json.loads((base_folder / "manifest.json").read_text())
    history_folder = Path(baseline["history_build"])
    source = Path(baseline["schedule_source"]["path"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != baseline["schedule_source"]["sha256"]:
        raise ValueError("Baseline schedule checksum mismatch")
    columns = ["season", "week", "game_id", "game_type", "gameday", "home_team", "away_team", "home_score", "away_score", "location"]
    games = pd.read_parquet(source, columns=columns)
    games = games[games.season.between(2021, 2025) & games.game_type.eq("REG")]
    base, _ = replay(games, RatingConfig(**baseline["config"]), [2021, 2022, 2023, 2024, 2025])
    team_path = history_folder / "team_week.parquet"
    teams = pd.read_parquet(team_path)
    comparisons = {"score_ratings": base[base.season >= 2022].copy()}
    selection, tuning, fitted = {}, [], {}
    for feature_set in FEATURE_SETS:
        trials = []
        for window in ["3", "5", "std"]:
            frame, names = assemble_features(base, teams, window, feature_set)
            for ridge in [10, 100, 1000]:
                validation, _ = replay_correction(frame[frame.season <= 2023], names, ridge, [2022, 2023])
                metric = score(validation)["ratings"]["mae"]
                trial = {"feature_set": feature_set, "window": window, "ridge": ridge, "validation_mae": metric}
                trials.append(trial)
                tuning.append(trial)
        selected = sorted(trials, key=lambda x: (x["validation_mae"], x["ridge"], x["window"]))[0]
        selection[feature_set] = selected
        frame, names = assemble_features(base, teams, selected["window"], feature_set)
        comparisons[feature_set], fitted[feature_set] = replay_correction(frame, names, selected["ridge"], [2022, 2023, 2024, 2025])
    validation_mae = {name: score(frame[frame.season.isin([2022, 2023])])["ratings"]["mae"] for name, frame in comparisons.items()}
    chosen = min(validation_mae, key=validation_mae.get)
    metrics = {name: {"validation": score(frame[frame.season <= 2023])["ratings"],
                      "evaluation": score(frame[frame.season >= 2024])["ratings"],
                      "by_season": {str(y): score(g)["ratings"] for y, g in frame.groupby("season")}} for name, frame in comparisons.items()}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    folder = root / "models" / "efficiency" / stamp
    folder.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(tuning).to_csv(folder / "tuning.csv", index=False)
    for name, frame in comparisons.items():
        frame.to_parquet(folder / f"predictions_{name}.parquet", index=False)
    (folder / "weekly_models.json").write_text(json.dumps(fitted, indent=2), encoding="utf-8")
    run = {"version": stamp, "selected_on_validation": chosen, "selected_configs": selection, "metrics": metrics,
           "baseline_build": str(base_folder), "team_feature_path": str(team_path),
           "team_feature_sha256": hashlib.sha256(team_path.read_bytes()).hexdigest(),
           "evaluation_status": "2024–2025 previously examined holdout; not fresh", "publication_time_certified": False,
           "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}}
    (folder / "manifest.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    lines = ["# EPA and success-rate experiment", "", f"Selected using 2022–2023 only: **{chosen}**.", "",
             "The score-rating baseline is frozen. Each candidate learns a ridge correction to its pregame margin using home-minus-away offensive and defensive EPA/play and/or success rate. Windows: last 3, last 5 or season-to-date; penalties: 10, 100, 1000. All 27 trials are reported. Missing inputs use training-only means plus missingness indicators; scaling also uses training data only. Week 1 history remains missing in the original data.", "",
             "The correction trains weekly on prior-game residuals starting in 2021. Its baseline settings were previously selected on 2022–2023; those years are tuning results, not independent evaluation. All candidate settings are selected before evaluating 2024–2025. Those evaluation years have already been viewed in earlier work, so this is not a fresh holdout.", "",
             "| Model | Tuning MAE | 2024 MAE | 2025 MAE | Evaluation MAE | Evaluation RMSE | Winner accuracy |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, result in metrics.items():
        a = result["evaluation"]
        lines.append(f"| {name} | {result['validation']['mae']:.3f} | {result['by_season']['2024']['mae']:.3f} | {result['by_season']['2025']['mae']:.3f} | {a['mae']:.3f} | {a['rmse']:.3f} | {a['winner_accuracy']:.1%} |")
    lines += ["", "## Configurations selected on tuning years", "", "```json", json.dumps(selection, indent=2), "```", "",
              "## Interpretation limits", "", "MAE is margin error in points; evaluation covers the same 544 games for each model. Winner accuracy excludes ties. Changes are descriptive, not proof of statistical significance. No sportsbook data, calibrated probabilities or betting conclusions. Retrospective corrections/publication timing remain a limitation. This experiment does not replace the baseline or live team report automatically.", "", f"Artifacts: {folder.resolve()}"]
    report = "\n".join(lines)
    (folder / "report.md").write_text(report, encoding="utf-8")
    Path("docs/efficiency_experiment.md").write_text(report, encoding="utf-8")
    pointer = folder.parent / "latest.json"
    temp = pointer.with_name(f"latest_{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps({"path": str(folder.resolve()), "version": stamp}, indent=2), encoding="utf-8")
    temp.replace(pointer)
    return folder
