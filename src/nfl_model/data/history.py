"""Season-at-a-time historical team backfill and cross-source reconciliation."""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .client import NFLVerseClient
from ..features.team_week import build_team_weeks

LOG = logging.getLogger(__name__)


def reconcile(pbp: pd.DataFrame, stats: pd.DataFrame, schedule: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Compare source totals independently of the efficiency play filter.

    These are internal nflverse consistency checks, not independent-provider proof.
    Retain every disagreement, including attribution/definition differences.
    """
    issues = []
    for side in ["home", "away"]:
        scores = pbp.groupby("game_id")[f"{side}_score"].last()
        for _, game in schedule.iterrows():
            actual = scores.get(game.game_id, float("nan"))
            expected = game[f"{side}_score"]
            if pd.notna(expected) and (pd.isna(actual) or actual != expected):
                issues.append(dict(game_id=game.game_id, team=game[f"{side}_team"], metric="final_score", pbp=actual, team_stats=expected))
    # Use raw statistical yardage fields (not efficiency yards_gained); retain all
    # recorded stat-bearing plays including kneels. Exclude two-point attempts.
    plays = pbp[pbp.two_point_attempt.ne(1)].copy()
    values = pd.DataFrame({"game_id": plays.game_id, "team": plays.posteam,
                           "passing_yards": plays.passing_yards,
                           "rushing_yards": plays.rushing_yards.add(plays.lateral_rushing_yards, fill_value=0),
                           "sacks_suffered": plays.sack,
                           "passing_interceptions": plays.interception})
    totals = values.groupby(["game_id", "team"]).sum(min_count=1)
    # No passing/rushing yards can legitimately mean zero for a whole team-game;
    # comparison remains unknown if the entire team-game is absent.
    totals = totals.fillna(0)
    metrics = ["passing_yards", "rushing_yards", "sacks_suffered", "passing_interceptions"]
    if stats.duplicated(["game_id", "team"]).any():
        raise ValueError("Duplicate team-stat game/team keys")
    expected_keys = {(g.game_id, team) for g in schedule.itertuples() if pd.notna(g.home_score) and pd.notna(g.away_score) for team in (g.home_team, g.away_team)}
    actual_keys = set(zip(stats.game_id, stats.team))
    for game_id, team in sorted(expected_keys - actual_keys):
        issues.append(dict(game_id=game_id, team=team, metric="missing_team_stats", pbp=None, team_stats=None))
    for _, row in stats.iterrows():
        key = (row.game_id, row.team)
        for metric in metrics:
            observed = totals.loc[key, metric] if key in totals.index else float("nan")
            expected = row[metric]
            if pd.isna(observed) or pd.isna(expected) or abs(observed - expected) > 1e-6:
                issues.append(dict(game_id=row.game_id, team=row.team, metric=metric, pbp=observed, team_stats=expected))
    completed = schedule[schedule.home_score.notna() & schedule.away_score.notna()]
    summary = {"scheduled_games": len(schedule), "completed_games": len(completed), "pbp_games": pbp.game_id.nunique(),
               "missing_completed_games": sorted(set(completed.game_id) - set(pbp.game_id)),
               "extra_pbp_games": sorted(set(pbp.game_id) - set(schedule.game_id)),
               "team_stat_rows": len(stats), "teams": sorted(set(schedule.home_team) | set(schedule.away_team)),
               "pbp_rows": len(pbp), "discrepancies": len(issues)}
    return summary, pd.DataFrame(issues, columns=["game_id", "team", "metric", "pbp", "team_stats"])


def backfill_history(client: NFLVerseClient, seasons: list[int], force: bool = False) -> Path:
    """Publish historical team tables separately from the live player/report build."""
    started = datetime.now(timezone.utc)
    version = started.strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    folder = client.root / "processed" / "history" / version
    folder.mkdir(parents=True, exist_ok=False)
    sources, summaries, schemas, mismatches, features, observations = [], [], {}, [], [], []
    def fetch(name, season=None):
        path = client.download(name, season, force)
        sources.append({**json.loads((path.parent / "manifest.json").read_text()), "path": str(path.resolve())})
        return pd.read_parquet(path)
    schedules = fetch("schedules")
    for season in sorted(set(seasons)):
        LOG.info("Backfilling %s", season)
        pbp = fetch("play_by_play", season)
        stats = fetch("team_stats", season)
        if set(pbp.season.dropna().unique()) != {season} or set(stats.season.dropna().unique()) != {season}:
            raise ValueError(f"Unexpected season values in {season} assets")
        schemas[str(season)] = {"pbp": {c: str(t) for c, t in pbp.dtypes.items()}, "team_stats": {c: str(t) for c, t in stats.dtypes.items()}}
        # Raw assets preserve playoffs; normalized efficiency currently has REG semantics.
        games = schedules[(schedules.season == season) & schedules.game_type.eq("REG")]
        if games.empty:
            raise ValueError(f"No schedule for {season}")
        reg = pbp[pbp.season_type.eq("REG")]
        regular_stats = stats[stats.season_type.eq("REG")]
        summary, differences = reconcile(reg, regular_stats, games)
        summary["season"] = season
        summaries.append(summary)
        differences["season"] = season
        mismatches.append(differences)
        table, obs = build_team_weeks(pbp, games)
        for column in table:
            if "_max_source_week_" in column and (table[column] >= table.week).fillna(False).any():
                raise ValueError("Historical future-week leakage")
        table.to_parquet(folder / f"team_week_{season}.parquet", index=False)
        features.append(table)
        observations.append(obs)
        print(f"{season}: {summary['completed_games']} completed REG games; {len(reg):,} PBP rows; {len(differences)} reconciliation differences", flush=True)
    pd.concat(features, ignore_index=True).to_parquet(folder / "team_week.parquet", index=False)
    pd.concat(observations, ignore_index=True).to_parquet(folder / "team_game_observations.parquet", index=False)
    pd.concat(mismatches, ignore_index=True).to_parquet(folder / "reconciliation.parquet", index=False)
    pd.concat(mismatches, ignore_index=True).to_csv(folder / "reconciliation.csv", index=False)
    (folder / "schemas.json").write_text(json.dumps(schemas, indent=2), encoding="utf-8")
    manifest = {"version": version, "built_at": started.isoformat(), "seasons": seasons, "sources": sources,
                "summary": summaries, "pregame_certified": False,
                "code_sha256": {str(p.relative_to(Path(__file__).parent.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.parent.rglob("*.py")}}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = ["# Historical data: 2021 onward", "", f"Build: {version}", "",
             "REG metrics only; raw files also retain postseason data. Sources are retrospective nflverse releases, not certified historical pregame snapshots.", "",
             "| Season | Completed games | PBP games | PBP rows | Stat differences |", "|---|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s['season']} | {s['completed_games']} | {s['pbp_games']} | {s['pbp_rows']:,} | {s['discrepancies']} |")
    lines += ["", "Discrepancies are preserved in reconciliation.csv/parquet. Comparisons use raw stat-bearing plays, excluding two-point attempts; efficiency metrics use their separate documented play filter.",
              "", "Exact input schemas and source manifests are stored alongside these tables. Same-week and later-week results are excluded from rolling metrics. Ratings and training have not begun."]
    (folder / "report.md").write_text("\n".join(lines), encoding="utf-8")
    pointer = folder.parent / "latest.json"
    temp = folder.parent / f"latest_{uuid.uuid4().hex}.tmp"
    temp.write_text(json.dumps({"path": str(folder.resolve()), "version": version}, indent=2), encoding="utf-8")
    temp.replace(pointer)
    return folder
