"""Reviewable QB evidence; no inferred confirmations or arbitrary point values."""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

KEY = ["season", "week", "team", "player_id"]


def assemble_qb_evidence(state: pd.DataFrame, usage: pd.DataFrame, depth: pd.DataFrame,
                         season: int, week: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Union roster QBs and listed QBs; compare only earlier-week usage.

    Missing IDs stay in a separate table. Neither missing injuries nor agreement
    between depth and usage establishes a confirmed starting quarterback.
    """
    roster = state[state.season.eq(season) & state.week.eq(week) & state.position.eq("QB")].copy()
    charts = depth[depth.season.eq(season) & depth.week.eq(week) & depth.pos_abb.eq("QB")].copy()
    unresolved = charts[charts.player_id.isna()].copy()
    charts = charts[charts.player_id.notna()].copy()
    if roster.duplicated(KEY).any():
        raise ValueError("Duplicate QB roster keys")
    chart_cols = KEY + ["player_name", "pos_rank", "depth_timestamp", "kickoff_utc"]
    charts = charts[chart_cols].drop_duplicates()
    if charts.duplicated(KEY).any():
        raise ValueError("Conflicting QB depth ranks or timestamps")
    charts = charts.rename(columns={"player_name": "depth_name", "pos_rank": "listed_qb_rank"})
    selected = KEY + ["player_name", "roster_status", "injury_status", "practice_status", "injury_record_present", "roster_record_present"]
    evidence = roster[selected].merge(charts, on=KEY, how="outer", validate="one_to_one")
    evidence["player_name"] = evidence.player_name.fillna(evidence.depth_name)
    # Historical QB position is taken from that historical week's roster record.
    historical_qbs = state[state.position.eq("QB")][KEY].drop_duplicates()
    prior = usage[usage.season.eq(season) & usage.week.lt(week)].merge(historical_qbs, on=KEY, how="inner", validate="one_to_one")
    prior = prior[prior.snap_record_present.eq(True) & prior.offensive_snaps.notna()]
    evidence["prior_usage_week"] = pd.Series(pd.NA, index=evidence.index, dtype="Int64")
    evidence["prior_offensive_snaps"] = float("nan")
    evidence["prior_offensive_snap_pct"] = float("nan")
    evidence["team_prior_usage_leader_id"] = pd.Series(pd.NA, index=evidence.index, dtype="string")
    for team, rows in evidence.groupby("team"):
        observed = prior[prior.team.eq(team)]
        if observed.empty:
            continue
        last_week = observed.week.max()
        last = observed[observed.week.eq(last_week)]
        leaders = last[last.offensive_snaps.eq(last.offensive_snaps.max()) & last.offensive_snaps.gt(0)]
        leader = leaders.player_id.iloc[0] if len(leaders) == 1 else pd.NA
        evidence.loc[rows.index, "team_prior_usage_leader_id"] = leader
        for index, row in rows.iterrows():
            player = last[last.player_id.eq(row.player_id)]
            if len(player):
                evidence.at[index, "prior_usage_week"] = int(last_week)
                evidence.at[index, "prior_offensive_snaps"] = player.offensive_snaps.iloc[0]
                evidence.at[index, "prior_offensive_snap_pct"] = player.offensive_snap_pct.iloc[0]
    evidence["confirmed_starter"] = False
    flags = []
    for row in evidence.itertuples():
        issues = []
        if pd.isna(row.roster_record_present) or not row.roster_record_present:
            issues.append("no_roster_record")
        if pd.isna(row.listed_qb_rank):
            issues.append("not_on_depth_chart")
        if pd.isna(row.injury_record_present) or not row.injury_record_present:
            issues.append("injury_report_unavailable")
        elif pd.isna(row.injury_status):
            issues.append("no_game_designation")
        if pd.notna(row.injury_status) and row.injury_status in ["Out", "Doubtful", "Questionable"]:
            issues.append("injury_designation")
        if pd.notna(row.practice_status) and row.practice_status != "Full Participation in Practice":
            issues.append("practice_restriction")
        if pd.notna(row.roster_status) and row.roster_status != "ACT":
            issues.append("non_active_roster_status")
        if pd.notna(row.listed_qb_rank) and row.listed_qb_rank == 1 and pd.notna(row.team_prior_usage_leader_id) and row.player_id != row.team_prior_usage_leader_id:
            issues.append("depth_usage_disagree")
        flags.append("; ".join(issues))
    evidence["review_flags"] = flags
    if (evidence.prior_usage_week >= week).fillna(False).any():
        raise ValueError("Same-week usage leak")
    return evidence.sort_values(["team", "listed_qb_rank", "player_id"], na_position="last"), unresolved


def build_qb_evidence(root: Path, season: int, week: int) -> Path:
    """Save an immutable evidence report from one completed build and its sources."""
    build = Path(json.loads((root / "processed/latest.json").read_text())["path"])
    manifest = json.loads((build / "manifest.json").read_text())
    frames, hashes = {}, {}
    for name in ["player_week", "player_week_usage", "player_week_depth"]:
        path = build / f"{name}.parquet"
        frames[name] = pd.read_parquet(path)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    evidence, unresolved = assemble_qb_evidence(frames["player_week"], frames["player_week_usage"], frames["player_week_depth"], season, week)
    if evidence.empty:
        raise ValueError("No QB evidence for requested season/week; build that week first")
    now = datetime.now(timezone.utc)
    relevant = [s for s in manifest["sources"] if s["dataset"] in ["weekly_rosters", "injuries", "depth_charts", "snap_counts", "schedules"]]
    latest_retrieval = max(pd.Timestamp(s["retrieved_at"]) for s in relevant)
    evidence["all_sources_retrieved_before_kickoff"] = evidence.kickoff_utc.notna() & (pd.to_datetime(evidence.kickoff_utc, utc=True) > latest_retrieval)
    # This flag describes file availability only, not confirmation of roster/medical facts.
    stamp = now.strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    folder = root / "processed/qb_evidence" / stamp
    folder.mkdir(parents=True, exist_ok=False)
    evidence.to_parquet(folder / "qb_evidence.parquet", index=False)
    unresolved.to_parquet(folder / "unresolved_qb_depth.parquet", index=False)
    evidence.to_csv(folder / "qb_evidence.csv", index=False)
    info = {"created_at": now.isoformat(), "season": season, "week": week, "input_build": str(build), "input_hashes": hashes,
            "sources": relevant, "rows": len(evidence), "teams": evidence.team.nunique(), "unresolved_depth_rows": len(unresolved),
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (folder / "manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    lines = [f"# QB evidence: {season} Week {week}", "", f"{len(evidence)} player records across {evidence.team.nunique()} teams. {len(unresolved)} unresolved depth-chart records retained separately.", "",
             "Listed QB1/QB2 are depth-chart ranks, not confirmed starters. Injury reports may be absent. Previous usage is from earlier weeks only; it does not predict an injury replacement or roster change.", "",
             "## Review priority", ""]
    conflicts = evidence[evidence.review_flags.str.contains("depth_usage_disagree")]
    if conflicts.empty:
        lines.append("No listed QB1 disagrees with a uniquely identified previous-game usage leader in the available data.")
    else:
        lines += ["```text", conflicts[["team", "player_name", "team_prior_usage_leader_id", "review_flags"]].to_string(index=False), "```"]
    for team, group in evidence.groupby("team"):
        lines += ["", f"## {team}", "", "```text", group[["listed_qb_rank", "player_name", "player_id", "roster_status", "injury_status", "practice_status", "prior_usage_week", "prior_offensive_snaps", "review_flags"]].to_string(index=False), "```"]
    lines += ["", "## Source freshness", ""]
    for source in relevant:
        lines.append(f"- {source['dataset']}: updated {source.get('source_updated_at')}; retrieved {source['retrieved_at']}.")
    lines += ["", "The all_sources_retrieved_before_kickoff field is a file-timing check only. No starting-QB probabilities or point adjustments are assigned. Use build --force-refresh to update inputs, then rerun qb-evidence. Existing forecasts remain frozen."]
    from .starter_news import observations_as_of
    news = [json.loads(p.read_text()) for p in (root / "raw/starter_news").glob("*.json")]
    reviewed = observations_as_of(news, season, week, pd.Timestamp(now))
    lines += ["", "## Reviewed starter reporting", "", "These are separately reviewed news observations as of this report's creation, not overrides of nflverse fields. Confirm the claim scope and timestamps before using any historical evidence."]
    for record in reviewed:
        observation = record["observation"]
        lines += ["", f"- {observation['team']} — {observation['claim_type']}: {observation['summary']} [Source]({observation['source_url']}) Observed {observation['observed_at']}; recorded {record['recorded_at']}."]
    if not reviewed:
        lines.append("No reviewed news observations for this specific week were recorded before this report.")
    (folder / "reviewed_news.json").write_text(json.dumps(reviewed, indent=2), encoding="utf-8")
    report = "\n".join(lines)
    (folder / "report.md").write_text(report, encoding="utf-8")
    Path(f"docs/qb_evidence_{season}_week_{week:02d}.md").write_text(report, encoding="utf-8")
    return folder
