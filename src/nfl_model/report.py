"""Readable team observations with provenance and visible coverage limits."""
import json
from pathlib import Path

import pandas as pd

from .features.team_week import METRICS


def team_report(root: Path, season: int, week: int, team: str) -> str:
    """Read one completed build without refreshing or mixing source versions."""
    pointer = json.loads((root / "processed" / "latest.json").read_text())
    folder = Path(pointer["path"])
    manifest = json.loads((folder / "manifest.json").read_text())
    def read(name):
        return pd.read_parquet(folder / f"{name}.parquet")
    teams = read("team_week")
    target = teams[(teams.season == season) & (teams.week == week) & (teams.team == team)]
    if len(target) != 1:
        raise ValueError(f"No unique scheduled REG game for {team}, {season} Week {week}; check team code, season or bye week.")
    target = target.iloc[0]
    lines = [f"# {team} — {season} Week {week}", "", f"Opponent: {target.opponent} | {'Home' if target.is_home else 'Away'} | Date: {target.game_date} | Rest: {target.rest_days} days", "",
             f"Build: {manifest['build_version']} ({manifest['built_at']})", "",
             "This is an observation report. Historical pregame availability is not certified. Missing data is unknown, not zero.", "",
             "## Recent team efficiency — play-by-play", "",
             "Defense metrics show opponent production allowed; lower EPA/success/explosive rates are better. Defensive sack/turnover rates measure events generated.", "",
             "| Metric | Off last 1 | Def last 1 | Off last 3 | Def last 3 | Off last 5 | Def last 5 | Off STD | Def STD |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for metric in METRICS:
        values = []
        for window in ["1", "3", "5", "std"]:
            for side in ["offense", "defense"]:
                value = target[f"{side}_{metric}_{window}"]
                values.append("Unknown" if pd.isna(value) else f"{value:.3f}" if metric.startswith("epa") else f"{value:.1%}")
        lines.append(f"| {metric.replace('_', ' ')} | " + " | ".join(values) + " |")
    lines += ["", "Coverage (observed offense/defense games versus scheduled prior games):"]
    for w in ["1", "3", "5", "std"]:
        lines.append(f"- {w}: {target[f'offense_games_observed_{w}']}/{target[f'defense_games_observed_{w}']} of {target[f'prior_games_expected_{w}']} games. Offensive EPA denominator: {target[f'offense_epa_per_play_{w}_n']} plays.")
    state = read("player_week")
    state = state[(state.season == season) & (state.week == week) & (state.team == team)]
    lines += ["", "## Roster and injury observations — weekly rosters / injuries", "",
              f"Player-week records: {len(state)}. Roster records: {int(state.roster_record_present.eq(True).sum())}."]
    injury = state[state.injury_record_present.eq(True)]
    if injury.empty:
        lines += ["No injury reports in this build for this team/week. Health and game availability are unknown."]
    else:
        lines += ["", "```text", injury[["player_name", "position", "injury_status", "practice_status", "report_primary_injury"]].fillna("Unknown").to_string(index=False), "```"]
    lines += ["", "Roster statuses: " + str(state.roster_status.value_counts(dropna=False).to_dict()), "",
              "## Depth-chart listings — timestamped charts", "", "These are source depth ranks, not confirmed starters or predicted active players."]
    depth = read("player_week_depth")
    depth = depth[(depth.season == season) & (depth.week == week) & (depth.team == team)]
    if depth.empty:
        lines += ["No eligible depth chart in this build."]
    else:
        lines += [f"Chart timestamp: {depth.depth_timestamp.max()}."]
        for label, selected in [("Quarterbacks", depth[depth.pos_abb.eq("QB") & depth.pos_rank.le(3)]),
                                ("Rank-one positions (including offensive line and defense)", depth[depth.pos_rank.eq(1)])]:
            lines += ["", f"### {label}", "", "```text", selected[["pos_abb", "pos_rank", "player_name", "player_id"]].drop_duplicates().sort_values(["pos_abb", "pos_rank"]).to_string(index=False), "```"]
    usage = read("player_week_usage")
    history = usage[(usage.season == season) & (usage.week < week) & (usage.team == team) & usage.snap_record_present.eq(True)]
    lines += ["", "## Recent snap leaders — actual prior-game usage", ""]
    if history.empty:
        lines += ["No earlier snap observations in this season."]
    else:
        latest_week = history.week.max()
        history = history[history.week == latest_week]
        master = read("master_players")[["player_id", "player_name"]].drop_duplicates("player_id")
        history = history.merge(master, on="player_id", how="left", validate="many_to_one")
        lines += [f"Observed Week {latest_week}; this is not a Week {week} lineup prediction."]
        for side in ["offensive", "defensive", "special_teams"]:
            selected = history.sort_values(f"{side}_snaps", ascending=False).head(6)
            lines += ["", f"### {side.replace('_', ' ').title()}", "", "```text", selected[["player_name", "player_id", f"{side}_snaps", f"{side}_snap_pct"]].to_string(index=False), "```"]
    lines += ["", "## Data gaps", "", "- PFR advanced stats are excluded; their partial coverage does not block this report.",
              "- Snapshot certification, replacement rankings and inferred starters remain unimplemented.",
              "- A game ID appearing in play-by-play does not prove every play is complete; counts reflect observed eligible plays."]
    unresolved = read("unresolved_players")
    team_unresolved = unresolved[(unresolved.season == season) & (unresolved.week <= week) & (unresolved.team == team)]
    lines.append(f"- Unresolved weekly source rows for this team through Week {week}: {len(team_unresolved)}. Full source records remain in unresolved_players.parquet.")
    lines += ["", "## Source freshness (UTC)", "", "Files are the exact versions used by this build. Refresh is explicit; these dates may be older than the report date.", "",
              "| Source | Season | Source updated | Retrieved |", "|---|---|---|---|"]
    for source in manifest["sources"]:
        lines.append(f"| {source['dataset']} | {source.get('season') or 'all'} | {source.get('source_updated_at') or 'Unknown'} | {source['retrieved_at']} |")
    return "\n".join(lines) + "\n"
