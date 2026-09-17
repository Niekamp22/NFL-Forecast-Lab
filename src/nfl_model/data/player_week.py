"""Player-week observation tables and strictly prior-week usage summaries.

Source availability reports are retrospective observations, not certified pregame
features. Actual usage is written separately and never used to label expected active.
"""
from __future__ import annotations

import pandas as pd

from .identity import resolve_ids

KEY = ["season", "week", "team", "player_id"]
SNAPS = {"offense_snaps": "offensive_snaps", "offense_pct": "offensive_snap_pct",
         "defense_snaps": "defensive_snaps", "defense_pct": "defensive_snap_pct",
         "st_snaps": "special_teams_snaps", "st_pct": "special_teams_snap_pct"}
STATS = ["attempts", "passing_yards", "passing_tds", "passing_interceptions", "passing_epa", "passing_cpoe",
         "carries", "rushing_yards", "rushing_tds", "rushing_epa", "targets", "receptions", "receiving_yards",
         "receiving_tds", "receiving_epa", "target_share", "air_yards_share", "sacks_suffered", "def_tackles_solo", "def_sacks"]


def regular(frame: pd.DataFrame) -> pd.DataFrame:
    """This first normalized table explicitly supports regular-season weeks only."""
    field = next((c for c in ("season_type", "game_type") if c in frame), None)
    return frame[frame[field].eq("REG")].copy() if field else frame.copy()


def assert_unique(frame: pd.DataFrame, name: str) -> None:
    if frame.duplicated(KEY).any():
        raise ValueError(f"{name}: duplicate player-week keys; refusing a many-to-many join")


def build_player_weeks(sources: dict[str, pd.DataFrame], crosswalk: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    """Outer-union all resolved weekly observations; quarantine every unresolved row."""
    normalized, unresolved = {}, []
    for name in ["weekly_rosters", "player_stats", "snap_counts", "injuries"]:
        frame = regular(sources[name])
        if name == "player_stats":
            frame = frame.rename(columns={"player_id": "gsis_id"})
        if name == "snap_counts":
            frame = frame.rename(columns={"pfr_player_id": "pfr_id"})
        frame = resolve_ids(frame, crosswalk)
        bad = frame[frame.player_id.isna()].copy()
        bad["source_dataset"] = name
        unresolved.append(bad)
        frame = frame[frame.player_id.notna()].copy()
        assert_unique(frame, name)
        normalized[name] = frame
    keys = pd.concat([f[KEY] for f in normalized.values()], ignore_index=True).drop_duplicates()
    state = keys.copy()
    roster = normalized["weekly_rosters"]
    roster = roster[KEY + ["full_name", "position", "status", "depth_chart_position"]].rename(
        columns={"full_name": "player_name", "status": "roster_status", "depth_chart_position": "roster_depth_position"})
    roster["roster_record_present"] = True
    state = state.merge(roster, on=KEY, how="left", validate="one_to_one")
    # CUT/RET are roster source records, but do not imply membership on the active roster.
    state["rostered_flag"] = pd.Series(pd.NA, index=state.index, dtype="boolean")
    state.loc[state.roster_status.isin(["ACT", "DEV", "RES", "INA", "EXE"]), "rostered_flag"] = True
    state.loc[state.roster_status.isin(["CUT", "RET"]), "rostered_flag"] = False
    injury = normalized["injuries"]
    injury_columns = ["report_status", "report_primary_injury", "report_secondary_injury", "practice_status", "practice_primary_injury", "practice_secondary_injury"]
    injury = injury[KEY + injury_columns].rename(columns={"report_status": "injury_status"})
    injury["injury_record_present"] = True
    state = state.merge(injury, on=KEY, how="left", validate="one_to_one")
    # A missing report is unknown, never a healthy classification or an active prediction.
    state["expected_active_flag"] = pd.Series(pd.NA, index=state.index, dtype="boolean")
    state.loc[state.injury_status.eq("Out") | state.roster_status.isin(["CUT", "RET", "DEV", "RES"]), "expected_active_flag"] = False
    state["availability_certified_pregame"] = False

    usage = keys.copy()
    snaps = normalized["snap_counts"][KEY + list(SNAPS)].rename(columns=SNAPS)
    snaps["snap_record_present"] = True
    usage = usage.merge(snaps, on=KEY, how="left", validate="one_to_one")
    stats = normalized["player_stats"][KEY + STATS].copy()
    stats["stat_record_present"] = True
    usage = usage.merge(stats, on=KEY, how="left", validate="one_to_one")
    usage["actually_played_flag"] = pd.Series(pd.NA, index=usage.index, dtype="boolean")
    total = usage[["offensive_snaps", "defensive_snaps", "special_teams_snaps"]].sum(axis=1, min_count=1)
    usage.loc[total.gt(0), "actually_played_flag"] = True
    # Zero snaps do not prove a player was inactive; game-day active lists are not available here.
    usage["actually_active_flag"] = pd.Series(pd.NA, index=usage.index, dtype="boolean")
    usage.loc[total.gt(0), "actually_active_flag"] = True
    usage["usage_is_postgame"] = True
    state = add_prior_usage(state, usage)

    schedule = regular(sources["schedules"])
    game_teams = pd.concat([schedule[["season", "week", "game_id", side]].rename(columns={side: "team"}) for side in ["home_team", "away_team"]])
    if game_teams.duplicated(["season", "week", "team"]).any():
        raise ValueError("Multiple scheduled games per team/week")
    state = state.merge(game_teams, on=["season", "week", "team"], how="left", validate="many_to_one")
    state["season_type"] = "REG"
    usage["season_type"] = "REG"
    assert_unique(state, "player_week")
    unresolved_frame = pd.concat(unresolved, ignore_index=True)
    report = {"player_week_rows": len(state), "usage_rows": len(usage),
              "unresolved_rows_by_source": unresolved_frame.source_dataset.value_counts().to_dict(),
              "missing_schedule_rows": int(state.game_id.isna().sum()),
              "weeks": state.groupby(["season", "week"]).size().rename("rows").reset_index().to_dict("records"),
              "limits": ["Availability is retrospective, not certified pregame.",
                         "Missing injury reports and missing snaps remain unknown.",
                         "Prior usage resets by season and uses only observed earlier weeks.",
                         "Postseason normalization, role estimates and roster movement are not inferred in this stage."]}
    return {"player_week": state, "player_week_usage": usage, "unresolved_players": unresolved_frame}, report


def add_prior_usage(state: pd.DataFrame, usage: pd.DataFrame) -> pd.DataFrame:
    """Calculate observed-game averages using Weeks < N; missing games are not zero-filled."""
    result = state.copy()
    for col in ["offensive_snap_pct", "defensive_snap_pct", "special_teams_snap_pct"]:
        for window in [1, 3, 5]:
            result[f"prior_{col}_{window}"] = float("nan")
            result[f"prior_{col}_{window}_observations"] = 0
    result["prior_usage_max_week"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    groups = {key: group.sort_values("week") for key, group in usage.groupby(["season", "player_id"])}
    for (season, player_id), rows in result.groupby(["season", "player_id"]):
        history = groups.get((season, player_id))
        for index, row in rows.iterrows():
            prior = history[history.week < row.week]
            # Ambiguous multi-team weeks are excluded from history rather than arbitrarily ordered.
            prior = prior[~prior.week.duplicated(keep=False)]
            observed = prior[prior.snap_record_present.eq(True)]
            if not observed.empty:
                result.at[index, "prior_usage_max_week"] = int(observed.week.max())
            for col in ["offensive_snap_pct", "defensive_snap_pct", "special_teams_snap_pct"]:
                for window in [1, 3, 5]:
                    values = observed.tail(window)[col].dropna()
                    result.at[index, f"prior_{col}_{window}"] = values.mean()
                    result.at[index, f"prior_{col}_{window}_observations"] = len(values)
    if (result.prior_usage_max_week >= result.week).fillna(False).any():
        raise ValueError("Same-week usage leakage")
    return result
