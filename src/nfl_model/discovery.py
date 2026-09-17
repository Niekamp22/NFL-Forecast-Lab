"""Generate a data map from real Parquet contents, with explicit missing-data evidence."""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from .data.catalog import DATASETS
from .data.client import NFLVerseClient, DataUnavailableError

LOG = logging.getLogger(__name__)


def inspect_sources(client: NFLVerseClient, season: int, week: int, force: bool = False) -> None:
    """Inspect current files or clearly labeled historical samples when absent."""
    records = []
    lines = ["# nflverse data map", "", f"Inspection time (UTC): {datetime.now(timezone.utc).isoformat()}",
             f"Requested season/week: {season}/{week}", "",
             "## Architecture and evidence", "",
             "Use public nflverse GitHub release assets directly from Python with requests and pandas/pyarrow. "
             "nflreadpy is the official Python alternative; direct assets give us explicit immutable retrieval versions. "
             "The stats_player and stats_team releases are the current weekly statistics sources; avoid legacy player_stats filenames.", "",
             "Raw bytes are preserved under data/raw/<dataset>/<season-or-all>/<retrieval-version>/. "
             "Each manifest records URL, asset ID, source update timestamp, retrieval timestamp, bytes and SHA-256. "
             "latest.json is only a cache pointer. Old versions remain intact. Processed data and snapshots are separate.", "",
             "## Point-in-time limits", "",
             "These are retrospective source files, NOT certified pre-kickoff snapshots. Downloading today cannot reconstruct "
             "what was published before a historical kickoff. Source update times are file-level, not row-level availability. "
             "Never use same-week results, snaps, participation, season aggregates, current player status or corrected future "
             "records as historical pregame inputs. Depth-chart timestamps need a strict before-kickoff selection. "
             "Preserve source versions now; a full snapshot selection engine is a later phase. No joins or predictors are built here.", "",
             "Names are display-only. Inspect the ID fields below and build explicit crosswalks later. "
             "Unmatched IDs must be retained and reported. Trades do not cover all signings, releases, practice-squad and IR transactions; "
             "weekly roster differences are observations, not proof of a specific transaction type.", "",
             "## Official references", "",
             "- [Release assets](https://github.com/nflverse/nflverse-data/releases)",
             "- [Update schedule](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html)",
             "- [Dataset documentation](https://nflreadr.nflverse.com/reference/)",
             "- [Official Python reader](https://nflreadpy.nflverse.com/)", "",
             "Documentation says injuries stopped after 2024; live asset contents below take precedence for observed availability, "
             "but a listed file does not establish completeness or restoration of a production feed. "
             "Participation from 2023 is documented as postseason-only. Depth charts since 2025 use timestamps, not weeks.", ""]
    for name, spec in DATASETS.items():
        record = {"dataset": name, "priority": spec.priority}
        lines += [f"## {name} — {spec.priority}", "", f"Source: https://github.com/nflverse/nflverse-data/releases/tag/{spec.tag}",
                  f"Asset pattern: `{spec.filename}`. Update cadence: {spec.frequency}.", ""]
        try:
            available = client.available_seasons(name)
            sample_season = season if season in available else (max(available) if available else None)
            path = client.download(name, sample_season, force)
            frame = pd.read_parquet(path)
            columns = list(frame.columns)
            year_col = next((c for c in ("season", "year") if c in columns), None)
            if name == "draft" and "draft_year" in columns:
                year_col = "draft_year"
            if name in ("players", "contracts", "teams"):
                year_col = None
            years = sorted(pd.to_numeric(frame[year_col], errors="coerce").dropna().astype(int).unique().tolist()) if year_col else []
            current = frame[frame[year_col] == season] if year_col else (frame if sample_season == season else frame.iloc[:0])
            week_col = next((c for c in ("week", "game_week", "week_num") if c in columns), None)
            weekly = current[current[week_col] == week] if week_col else current.iloc[:0]
            type_col = next((c for c in ("season_type", "game_type") if c in weekly), None)
            if type_col:
                weekly = weekly[weekly[type_col].isin(["REG", "Regular"])]
            status = (f"Week {week}: {len(weekly):,} regular-season rows" if week_col and len(current)
                      else f"{len(current):,} season rows; no week field—Week {week} unverified" if len(current)
                      else "Not season-indexed; current master/reference file inspected" if name in ("players", "contracts", "teams")
                      else "No requested-season rows found")
            record.update({"status": status, "rows": len(frame), "current_rows": len(current),
                           "week_rows": len(weekly), "available_seasons": available or years,
                           "columns": columns, "sample_path": str(path),
                           "manifest": json.loads((path.parent / "manifest.json").read_text())})
            ids = [c for c in columns if "id" in c.lower().split("_") or c.lower().endswith("_id")]
            teams = [c for c in columns if "team" in c.lower() or c in ("posteam", "defteam")]
            games = [c for c in columns if "game" in c.lower() and ("id" in c.lower() or "date" in c.lower())]
            lines += [f"Available seasons (asset inventory, or rows for combined files): {available or years or 'not season-indexed'}.",
                      f"Inspected `{path.name}`: {len(frame):,} rows, {len(columns)} columns.",
                      f"**{season} availability: {status}.**", "",
                      f"Identifier candidates (actual columns; semantics require dataset dictionary): `{', '.join(ids) or 'none'}`.",
                      f"Team fields: `{', '.join(teams) or 'none'}`. Game fields: `{', '.join(games) or 'none'}`.", "",
                      "Leakage: " + ("Current mutable roster, injury, depth or player state requires publication-time evidence; never backfill current status into past games."
                          if name in ("players", "rosters", "weekly_rosters", "injuries", "depth_charts", "contracts")
                          else "Game outcomes and usage are postgame observations; lag before use. Season totals include later games. Draft/combine/trade data requires event and publication dates."), "",
                      "<details><summary>All inspected columns</summary>", "", "`" + "`, `".join(columns) + "`", "", "</details>", ""]
            evidence = weekly if not weekly.empty else current
            preferred = ["game_id", "season", "week", "home_team", "away_team", "home_score", "away_score", "player_id", "gsis_id", "pfr_player_id", "player_display_name", "full_name", "team", "passing_yards", "offense_snaps", "report_status", "dt", "timestamp"]
            selected = [c for c in preferred if c in evidence]
            if not evidence.empty and selected:
                lines += ["Observed examples (not predictions):", "", "```text", evidence[selected].head(3).to_string(index=False), "```", ""]
            print(f"{name}: {status}", flush=True)
        except (DataUnavailableError, requests.RequestException, ValueError, OSError) as exc:
            record["error"] = str(exc)
            LOG.warning("Inspection incomplete for %s: %s", name, exc)
            lines += [f"**Unavailable or inspection failed:** {exc}", ""]
        records.append(record)
    output = Path("docs")
    output.mkdir(exist_ok=True)
    (output / "availability.json").write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    Path("nflverse_data_map.md").write_text("\n".join(lines), encoding="utf-8")
