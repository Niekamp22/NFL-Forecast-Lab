"""Read-only helpers for displaying canonical forecasts and matching results."""
import hashlib
import json
from pathlib import Path

import pandas as pd

from .models.weekly_forecast import earliest_snapshot, checked_predictions


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def artifact_path(value, project):
    path = Path(value)
    return path if path.is_absolute() else project / path


def available_weeks(root):
    return sorted({(int(p.parent.parent.parent.name), int(p.parent.parent.name.removeprefix("week_")))
                   for p in (root / "forecasts").glob("*/week_*/*/manifest.json")}, reverse=True)


def load_forecast(root, season, week):
    folder, meta = earliest_snapshot(root / "forecasts" / str(season) / f"week_{week:02d}")
    if not meta.get("predictions_sha256"):
        raise ValueError("Unified forecast has no creation checksum")
    return folder, meta, checked_predictions(folder, meta).sort_values(["kickoff_utc", "game_id"])


def matching_evaluation(root, project, folder, meta):
    candidates = []
    for path in (root / "evaluations_full" / str(meta["season"]) / f"week_{meta['week']:02d}").glob("*/manifest.json"):
        run = read_json(path)
        if (artifact_path(run["snapshot"], project).resolve() == folder.resolve()
                and run.get("snapshot_hash") == meta["predictions_sha256"]):
            candidates.append((pd.Timestamp(run["created_at"]), str(path), run))
    if not candidates:
        return None
    _, path, run = max(candidates, key=lambda item: item[:2])
    return Path(path).parent, run


def verify_context(meta, project):
    """Validate pinned context before displaying it beside a frozen forecast."""
    for value, expected in meta.get("input_hashes", {}).items():
        path = artifact_path(value, project)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Pinned input checksum mismatch: {path.name}")


def team_filter(frame, team):
    return frame if team == "All teams" else frame[frame.home_team.eq(team) | frame.away_team.eq(team)]
