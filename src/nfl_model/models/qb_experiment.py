"""Uncertainty audit and conservative prior-usage QB feature experiment."""
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..features.qb import qb_games, qb_scenarios
from .uncertainty import paired_interval
from .ratings import RatingConfig, replay, score
from .efficiency import assemble_features, replay_correction


def run_qb_experiment(root: Path) -> Path:
    """Use prior usage only for QB identity; tune on 2022–23, reuse 2024–25 evaluation."""
    eff_path = Path(json.loads((root / "models/efficiency/latest.json").read_text())["path"])
    eff = json.loads((eff_path / "manifest.json").read_text())
    baseline_path = Path(eff["baseline_build"])
    baseline = json.loads((baseline_path / "manifest.json").read_text())
    hist = json.loads((Path(baseline["history_build"]) / "manifest.json").read_text())
    saved_base = pd.read_parquet(eff_path / "predictions_score_ratings.parquet")
    saved_success = pd.read_parquet(eff_path / "predictions_success.parquet")
    uncertainty = paired_interval(saved_base[saved_base.season >= 2024], saved_success[saved_success.season >= 2024])
    schedule = pd.read_parquet(baseline["schedule_source"]["path"], columns=["season", "week", "game_id", "game_type", "gameday", "home_team", "away_team", "home_score", "away_score", "location"])
    schedule = schedule[schedule.season.between(2021, 2025) & schedule.game_type.eq("REG")]
    observations, unmatched, source_refs = [], [], []
    for source in hist["sources"]:
        if source["dataset"] != "play_by_play":
            continue
        path = Path(source["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("PBP checksum mismatch")
        observed, missing = qb_games(pd.read_parquet(path))
        observations.append(observed)
        unmatched.append(missing)
        source_refs.append(source)
    observed = pd.concat(observations, ignore_index=True)
    scenarios = qb_scenarios(observed, schedule)
    base, _ = replay(schedule, RatingConfig(**baseline["config"]), [2021, 2022, 2023, 2024, 2025])
    selected_success = eff["selected_configs"]["success"]
    frame, names = assemble_features(base, pd.read_parquet(eff["team_feature_path"]), selected_success["window"], "success")
    for venue in ["home", "away"]:
        selected = scenarios[["game_id", "team", "prior_epa", "prior_cpoe", "player_id"]].rename(columns={"team": f"{venue}_team", "player_id": f"{venue}_qb_proxy", "prior_epa": f"{venue}_qb_epa", "prior_cpoe": f"{venue}_qb_cpoe"})
        frame = frame.merge(selected, on=["game_id", f"{venue}_team"], validate="one_to_one")
    for metric in ["epa", "cpoe"]:
        column = f"qb_{metric}_difference"
        frame[column] = frame[f"home_qb_{metric}"] - frame[f"away_qb_{metric}"]
        names.append(column)
    trials = []
    for ridge in [10, 100, 1000]:
        validation, _ = replay_correction(frame[frame.season <= 2023], names, ridge, [2022, 2023])
        trials.append({"ridge": ridge, "validation_mae": score(validation)["ratings"]["mae"]})
    chosen = min(trials, key=lambda r: r["validation_mae"])
    predictions, states = replay_correction(frame, names, chosen["ridge"], [2022, 2023, 2024, 2025])
    metrics = {str(y): score(g)["ratings"] for y, g in predictions.groupby("season")}
    qb_uncertainty = paired_interval(saved_success[saved_success.season >= 2024], predictions[predictions.season >= 2024])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    folder = root / "models/qb" / stamp
    folder.mkdir(parents=True, exist_ok=False)
    observed.to_parquet(folder / "qb_game.parquet", index=False)
    scenarios.to_parquet(folder / "qb_week_scenarios.parquet", index=False)
    pd.concat(unmatched, ignore_index=True).to_parquet(folder / "unattributed_dropbacks.parquet", index=False)
    predictions.to_parquet(folder / "predictions.parquet", index=False)
    (folder / "weekly_models.json").write_text(json.dumps(states, indent=2), encoding="utf-8")
    run = {"version": stamp, "efficiency_build": str(eff_path), "success_uncertainty": uncertainty, "qb_vs_success_uncertainty": qb_uncertainty,
           "trials": trials, "selected": chosen, "metrics": metrics, "sources": source_refs, "starter_method": "previous team game's highest-dropback player, no current-game starter labels",
           "prior_observations": 100, "publication_time_certified": False,
           "code_sha256": {str(p.relative_to(Path(__file__).parent.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.parent.rglob("*.py")}}
    (folder / "manifest.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    lines = ["# Uncertainty and QB experiment", "", "## Success-rate improvement uncertainty", "", "```json", json.dumps(uncertainty, indent=2), "```", "",
             "Positive improvement means lower MAE. The 95% percentile interval resamples paired whole weeks within each evaluation season (10,000 draws). It does not capture all cross-week dependence or repeated-holdout selection bias. If the interval includes zero, these results do not clearly separate improvement from sampling variation.", "",
             "## QB layer", "", f"Built {len(observed):,} QB-game observations, {len(scenarios):,} team-week scenarios and preserved {sum(len(x) for x in unmatched)} unattributed dropbacks.", "",
             "Metrics: EPA/dropback, success, CPOE on eligible passes, sacks, scrambles and interceptions. Player IDs attribute plays; scrambles use rusher ID. Histories begin in 2021, so prior-games counts are observed history, not complete careers. Prior last-five QB-game metrics shrink toward the earlier league mean with an explicit 100-observation prior. This smoothing choice is not a QB point value.", "",
             "The QB identity is a prior-usage proxy, not a confirmed starter. It cannot anticipate a debut, injury replacement, signing or trade. The experiment deliberately does not use retrospectively populated schedule starter IDs. No claimed injury-aware QB adjustment is made.", "",
             "The correction adds prior QB EPA and CPOE differences to the success-rate features. Penalty is selected on 2022–2023 only. Evaluation years 2024–2025 have already been examined and are not fresh.", "",
             "| Season | QB + success MAE | Winner accuracy |", "|---|---:|---:|"]
    for year, metric in metrics.items():
        lines.append(f"| {year} | {metric['mae']:.3f} | {metric['winner_accuracy']:.1%} |")
    lines += ["", "## QB addition versus success-rate model", "", "```json", json.dumps(qb_uncertainty, indent=2), "```", "",
              "All experiments remain separate from production reports. Prospective forecasts should freeze source and model versions before kickoff and label any QB identity assumptions."]
    report = "\n".join(lines)
    (folder / "report.md").write_text(report, encoding="utf-8")
    Path("docs/qb_experiment.md").write_text(report, encoding="utf-8")
    temp = folder.parent / f"latest_{uuid.uuid4().hex}.tmp"
    temp.write_text(json.dumps({"path": str(folder.resolve()), "version": stamp}), encoding="utf-8")
    temp.replace(folder.parent / "latest.json")
    return folder
