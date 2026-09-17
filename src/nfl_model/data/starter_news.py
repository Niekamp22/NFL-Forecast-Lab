"""Append-only, human-reviewed source observations with strict week and time scope."""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CLAIMS = {"announced_starter", "conditional_starter", "starter_unresolved", "in_game_exit"}


def import_review(root: Path, review_path: Path) -> int:
    """Store reviewed summaries, not scraped article copies. Identical imports are idempotent."""
    records = json.loads(review_path.read_text(encoding="utf-8"))
    folder = root / "raw/starter_news"
    folder.mkdir(parents=True, exist_ok=True)
    count = 0
    for record in records:
        identity = record["evidence_id"]
        if not identity or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in identity):
            raise ValueError("Invalid evidence ID")
        if record["claim_type"] not in CLAIMS or not record["source_url"].startswith("https://"):
            raise ValueError("Invalid evidence claim or URL")
        observed = pd.Timestamp(record["observed_at"])
        if observed.tzinfo is None or observed > pd.Timestamp.now(tz="UTC"):
            raise ValueError("Observation timestamp must be timezone-aware and not future")
        if pd.Timestamp(record["published_date"]).date() > observed.date():
            raise ValueError("Publication date is later than observation")
        if record["claim_type"] in {"announced_starter", "conditional_starter", "in_game_exit"} and not record.get("player_id"):
            raise ValueError("Player-specific claims require a source-matched ID")
        path = folder / f"{identity}.json"
        if path.exists():
            previous = json.loads(path.read_text())
            if previous["observation"] != record:
                raise ValueError("Cannot overwrite evidence; add a new ID with supersedes")
            continue
        envelope = {"recorded_at": datetime.now(timezone.utc).isoformat(), "observation": record}
        with path.open("x", encoding="utf-8") as file:
            json.dump(envelope, file, indent=2)
        count += 1
    return count


def observations_as_of(records: list[dict], season: int, week: int, cutoff: pd.Timestamp) -> list[dict]:
    """Use both observed and local recorded time. Old publication dates cannot backdate knowledge."""
    if cutoff.tzinfo is None:
        raise ValueError("Cutoff must be timezone-aware")
    eligible = [r for r in records if r["observation"]["season"] == season and r["observation"]["week"] == week
                and pd.Timestamp(r["recorded_at"]) < cutoff and pd.Timestamp(r["observation"]["observed_at"]) < cutoff]
    superseded = {r["observation"].get("supersedes") for r in eligible}
    return [r for r in eligible if r["observation"]["evidence_id"] not in superseded]
