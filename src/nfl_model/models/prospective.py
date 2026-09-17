"""Freeze baseline and success-model forecasts before the earliest week kickoff."""
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .ratings import RatingConfig, replay
from .efficiency import assemble_features, replay_correction


def freeze_week(root: Path, season: int, week: int) -> Path:
    """Save immutable pre-kickoff forecasts using pinned input/model versions."""
    now = pd.Timestamp(datetime.now(timezone.utc))
    eff_path = Path(json.loads((root / "models/efficiency/latest.json").read_text())["path"])
    eff = json.loads((eff_path / "manifest.json").read_text())
    base_path = Path(eff["baseline_build"])
    baseline = json.loads((base_path / "manifest.json").read_text())
    live_path = Path(json.loads((root / "processed/latest.json").read_text())["path"])
    live = json.loads((live_path / "manifest.json").read_text())
    src = next(s for s in live["sources"] if s["dataset"] == "schedules")
    schedule = pd.read_parquet(src["path"], columns=["season", "week", "game_id", "game_type", "gameday", "gametime", "home_team", "away_team", "home_score", "away_score", "location"])
    games = schedule[schedule.game_type.eq("REG") & schedule.season.between(2021, season)]
    target = games[games.season.eq(season) & games.week.eq(week)]
    if target.empty or target[["home_score", "away_score"]].notna().any().any():
        raise ValueError("Target week missing or already contains results")
    kickoff = pd.to_datetime(target.gameday.astype(str)+" "+target.gametime.astype(str), errors="raise").dt.tz_localize("America/New_York").dt.tz_convert("UTC").min()
    if pd.isna(kickoff) or now >= kickoff:
        raise ValueError("Can only freeze before the earliest target kickoff")
    hist = json.loads((Path(baseline["history_build"]) / "manifest.json").read_text())
    refs = hist["sources"] + live["sources"]
    for source in refs:
        if pd.Timestamp(source["retrieved_at"]) > now or pd.Timestamp(source["retrieved_at"]) >= kickoff:
            raise ValueError("Source retrieval is after cutoff")
        if hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("Source checksum mismatch")
    games = games[(games.season < season) | (games.week <= week)]
    base, _ = replay(games, RatingConfig(**baseline["config"]), sorted(games.season.unique()))
    historical = pd.read_parquet(eff["team_feature_path"])
    current = pd.read_parquet(live_path / "team_week.parquet")
    teams = pd.concat([historical[historical.season < season], current[current.season == season]], ignore_index=True)
    config = eff["selected_configs"]["success"]
    frame, names = assemble_features(base, teams, config["window"], "success")
    corrected, models = replay_correction(frame, names, config["ridge"], [season])
    prediction = corrected[corrected.week == week][["season", "week", "game_id", "home_team", "away_team", "score_rating_margin", "predicted_margin"]].rename(columns={"predicted_margin": "success_model_margin"})
    folder = root / "snapshots" / str(season) / f"week_{week:02d}" / (now.strftime("%Y%m%dT%H%M%S%fZ")+"_"+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    prediction.to_parquet(folder / "predictions.parquet", index=False)
    prediction.to_csv(folder / "predictions.csv", index=False)
    manifest = {"created_at": str(now), "earliest_kickoff": str(kickoff), "season": season, "week": week,
                "predictions_sha256": hashlib.sha256((folder / "predictions.parquet").read_bytes()).hexdigest(),
                "efficiency_build": str(eff_path), "live_build": str(live_path), "sources": refs,
                "models": [s for s in models if s["week"] == week], "games": len(prediction),
                "qb_adjustment": "none; experimental QB scenarios excluded", "margin_convention": "positive means home team by that many points",
                "model_code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return folder
