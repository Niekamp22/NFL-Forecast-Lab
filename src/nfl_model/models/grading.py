"""Grade frozen forecasts using matching schedule scores and PBP end-game evidence."""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.client import NFLVerseClient

MODELS = ["score_rating_margin", "success_model_margin"]


def grade_predictions(predictions: pd.DataFrame, schedule: pd.DataFrame, pbp: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Require END GAME evidence and agreement on scores before treating a game as final."""
    if predictions.game_id.duplicated().any() or schedule.game_id.duplicated().any():
        raise ValueError("Duplicate prediction or schedule game IDs")
    if not np.isfinite(predictions[MODELS].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite forecast")
    columns = ["game_id", "home_team", "away_team", "home_score", "away_score"]
    merged = predictions.merge(schedule[columns], on="game_id", how="left", suffixes=("", "_result"), validate="one_to_one", indicator=True)
    present = merged._merge.eq("both")
    if (present & ((merged.home_team != merged.home_team_result) | (merged.away_team != merged.away_team_result))).any():
        raise ValueError("Schedule teams disagree with frozen forecast")
    end = pbp[pbp.desc.astype("string").str.strip().str.upper().eq("END GAME")][["game_id", "total_home_score", "total_away_score"]].drop_duplicates()
    if end.game_id.duplicated().any():
        raise ValueError("Conflicting END GAME scores")
    merged = merged.merge(end, on="game_id", how="left", validate="one_to_one")
    merged["status"] = "pending"
    merged.loc[~present, "status"] = "missing_schedule"
    scored = merged.home_score.notna() & merged.away_score.notna()
    ended = merged.total_home_score.notna() & merged.total_away_score.notna()
    merged.loc[present & scored & ~ended, "status"] = "awaiting_end_game_evidence"
    agrees = merged.home_score.eq(merged.total_home_score) & merged.away_score.eq(merged.total_away_score)
    merged.loc[present & scored & ended & ~agrees, "status"] = "score_conflict"
    merged.loc[present & scored & ended & agrees, "status"] = "graded"
    final = merged.status.eq("graded")
    merged["actual_margin"] = (merged.home_score - merged.away_score).where(final)
    summary = {"total_forecasts": len(merged), "graded": int(final.sum()), "statuses": merged.status.value_counts().to_dict(), "models": {}}
    for model in MODELS:
        error = merged[model] - merged.actual_margin
        merged[f"{model}_error"] = error
        merged[f"{model}_absolute_error"] = error.abs()
        decisive = final & merged.actual_margin.ne(0)
        merged[f"{model}_winner_correct"] = pd.Series(pd.NA, index=merged.index, dtype="boolean")
        merged.loc[decisive, f"{model}_winner_correct"] = np.sign(merged.loc[decisive, model]).eq(np.sign(merged.loc[decisive, "actual_margin"]))
        summary["models"][model] = {"mae": float(error.abs().mean()) if final.any() else None,
            "rmse": float(np.sqrt((error**2).mean())) if final.any() else None,
            "mean_error": float(error.mean()) if final.any() else None,
            "winner_accuracy": float(merged.loc[decisive, f"{model}_winner_correct"].mean()) if decisive.any() else None,
            "winner_games": int(decisive.sum())}
    return merged.drop(columns="_merge"), summary


def grade_week(client: NFLVerseClient, season: int, week: int, refresh: bool = False) -> Path:
    """Grade the earliest valid frozen run; publish a separate immutable evaluation."""
    snapshot_root = client.root / "snapshots" / str(season) / f"week_{week:02d}"
    candidates = []
    for path in snapshot_root.glob("*/manifest.json"):
        meta = json.loads(path.read_text())
        if pd.Timestamp(meta["created_at"]) < pd.Timestamp(meta["earliest_kickoff"]):
            candidates.append((pd.Timestamp(meta["created_at"]), path.parent, meta))
    if not candidates:
        raise ValueError("No valid pre-kickoff snapshot for requested week")
    _, snapshot, meta = sorted(candidates, key=lambda x: (x[0], str(x[1])))[0]
    pred_path = snapshot / "predictions.parquet"
    pred_hash = hashlib.sha256(pred_path.read_bytes()).hexdigest()
    if meta.get("predictions_sha256") and meta["predictions_sha256"] != pred_hash:
        raise ValueError("Frozen prediction checksum mismatch")
    predictions = pd.read_parquet(pred_path)
    if not predictions.season.eq(season).all() or not predictions.week.eq(week).all() or len(predictions) != meta["games"]:
        raise ValueError("Frozen forecast metadata mismatch")
    refs, frames = [], {}
    for name, year in [("schedules", None), ("play_by_play", season)]:
        path = client.download(name, year, force=refresh)
        refs.append({**json.loads((path.parent / "manifest.json").read_text()), "path": str(path.resolve())})
        frames[name] = pd.read_parquet(path)
    graded, summary = grade_predictions(predictions, frames["schedules"], frames["play_by_play"])
    now = datetime.now(timezone.utc)
    folder = client.root / "evaluations" / str(season) / f"week_{week:02d}" / (now.strftime("%Y%m%dT%H%M%S%fZ")+"_"+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    graded.to_parquet(folder / "grades.parquet", index=False)
    graded.to_csv(folder / "grades.csv", index=False)
    run = {"created_at": now.isoformat(), "snapshot": str(snapshot.resolve()), "selection": "earliest pre-kickoff snapshot",
           "snapshot_manifest_sha256": hashlib.sha256((snapshot / "manifest.json").read_bytes()).hexdigest(),
           "predictions_sha256": pred_hash, "creation_hash_verified": bool(meta.get("predictions_sha256")),
           "sources": refs, "summary": summary, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (folder / "manifest.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    lines = [f"# Prospective evaluation: {season} Week {week}", "", f"Graded: **{summary['graded']} / {summary['total_forecasts']}**.", "",
             f"Frozen predictions: `{snapshot.name}`. Selection uses the earliest pre-kickoff run to avoid choosing a favorable revision.", "",
             "Final scores must match schedule and END GAME play-by-play evidence. Other games remain pending or explicitly flagged. Missing results never become zero error.", "",
             "```json", json.dumps(summary, indent=2), "```", "", "## Game status", "", "```text",
             graded[["game_id", "home_team", "away_team", "status", "actual_margin"]].to_string(index=False), "```", "",
             "## Result-source freshness", ""]
    for source in refs:
        lines.append(f"- {source['dataset']}: source updated {source.get('source_updated_at')}; retrieved {source['retrieved_at']}.")
    lines += ["", "Use --force-refresh to fetch current results. Every grading run is preserved; later source corrections create a new evaluation, not revised predictions."]
    if not run["creation_hash_verified"]:
        lines += ["", "Legacy snapshot: no creation-time prediction hash was recorded. This evaluation pins its current hash but cannot prove that the file has never changed since creation."]
    report = "\n".join(lines)
    (folder / "report.md").write_text(report, encoding="utf-8")
    Path(f"docs/prospective_{season}_week_{week:02d}.md").write_text(report, encoding="utf-8")
    return folder
