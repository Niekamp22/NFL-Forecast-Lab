"""Publish reproducible, versioned normalized tables from explicit raw versions."""
from __future__ import annotations

import json
import logging
import uuid
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .catalog import DATASETS
from .client import NFLVerseClient
from .depth import select_depth
from .identity import build_identity
from .player_week import build_player_weeks
from ..features.team_week import build_team_weeks

LOG = logging.getLogger(__name__)


def build_database(client: NFLVerseClient, seasons: list[int], force: bool = False) -> dict:
    """Build normalized REG observations. Latest pointer is committed only on success."""
    started = datetime.now(timezone.utc)
    sources, manifests = {}, []
    for name in ["players", "weekly_rosters", "player_stats", "snap_counts", "injuries", "depth_charts", "schedules", "play_by_play"]:
        frames = []
        for season in sorted(set(seasons)) if "{season}" in DATASETS[name].filename else [None]:
            path = client.download(name, season, force)
            frame = pd.read_parquet(path)
            manifest = json.loads((path.parent / "manifest.json").read_text())
            manifests.append({**manifest, "path": str(path.resolve())})
            frames.append(frame)
        sources[name] = pd.concat(frames, ignore_index=True)
        if name == "schedules":
            sources[name] = sources[name][sources[name].season.isin(seasons)]
    master, crosswalk, conflicts = build_identity(sources["players"], sources["weekly_rosters"])
    tables, report = build_player_weeks(sources, crosswalk)
    teams, game_observations = build_team_weeks(sources["play_by_play"], sources["schedules"])
    tables["team_week"] = teams
    tables["team_game_observations"] = game_observations
    report["team_week_rows"] = len(teams)
    aliases = pd.concat([
        sources["players"][["gsis_id", "display_name"]].rename(columns={"gsis_id": "player_id", "display_name": "player_name"}).assign(source="players"),
        sources["weekly_rosters"][["gsis_id", "full_name"]].rename(columns={"gsis_id": "player_id", "full_name": "player_name"}).assign(source="weekly_rosters"),
    ], ignore_index=True).drop_duplicates()
    tables["player_name_aliases"] = aliases
    # Only select charts for weeks with observed player-week records, not every future scheduled game.
    weeks = tables["player_week"][["season", "week"]].drop_duplicates()
    schedule = sources["schedules"].merge(weeks, on=["season", "week"], how="inner")
    depth, unresolved_depth = select_depth(sources["depth_charts"], schedule, crosswalk, pd.Timestamp(started))
    tables.update({"master_players": master, "player_id_crosswalk": crosswalk,
                   "identity_conflicts": conflicts, "player_week_depth": depth,
                   "unresolved_depth_players": unresolved_depth})
    report["master_rows"] = len(master)
    report["master_rows_without_gsis"] = int(master.player_id.isna().sum())
    report["master_nonstandard_source_ids"] = int(master.canonical_id_format.eq("source_nonstandard").sum())
    report["conflicting_crosswalk_rows"] = len(conflicts)
    report["unresolved_depth_rows"] = len(unresolved_depth)
    report["depth_rows"] = len(depth)
    report["multi_team_player_weeks"] = int((tables["player_week"].groupby(["season", "week", "player_id"]).team.nunique() > 1).sum())
    version = started.strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    folder = client.root / "processed" / version
    folder.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        # Unresolved source rows retain every field; mixed source representations
        # (for example jersey number string vs float) use textual audit columns.
        if name.startswith("unresolved"):
            for col in table.select_dtypes(include="object"):
                table[col] = table[col].astype("string")
        table.to_parquet(folder / f"{name}.parquet", index=False)
    manifest = {"build_version": version, "built_at": started.isoformat(), "seasons": seasons,
                "schema_version": 2, "pregame_certified": False, "sources": manifests, "quality": report,
                "code_sha256": {str(p.relative_to(Path(__file__).parent.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.parent.rglob("*.py")}}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (folder / "quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pointer = client.root / "processed" / "latest.json"
    temp = pointer.with_name(f"latest_{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps({"build_version": version, "path": str(folder.resolve())}, indent=2), encoding="utf-8")
    temp.replace(pointer)
    LOG.warning("Unresolved weekly players: %s; identity conflicts: %s. See %s", report["unresolved_rows_by_source"], len(conflicts), folder)
    return manifest
