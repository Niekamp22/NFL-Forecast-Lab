"""Explicit asset patterns verified against the nflverse release catalog."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Dataset:
    tag: str
    filename: str
    priority: str = "CORE"
    frequency: str = "Not documented; inspect source updated_at"


DATASETS = {
    "play_by_play": Dataset("pbp", "play_by_play_{season}.parquet", frequency="After game days; corrections midweek"),
    "schedules": Dataset("schedules", "games.parquet", frequency="Every 5 minutes in season"),
    "player_stats": Dataset("stats_player", "stats_player_week_{season}.parquet", frequency="Same as play-by-play"),
    "team_stats": Dataset("stats_team", "stats_team_week_{season}.parquet", frequency="Same as play-by-play"),
    "players": Dataset("players", "players.parquet"),
    "weekly_rosters": Dataset("weekly_rosters", "roster_weekly_{season}.parquet", frequency="Daily 07:00 UTC"),
    "rosters": Dataset("rosters", "roster_{season}.parquet", "USEFUL", "Daily 07:00 UTC"),
    "injuries": Dataset("injuries", "injuries_{season}.parquet", frequency="Documentation says discontinued after 2024; verify live rows"),
    "depth_charts": Dataset("depth_charts", "depth_charts_{season}.parquet", frequency="Daily 07:00 UTC; timestamped since 2025"),
    "snap_counts": Dataset("snap_counts", "snap_counts_{season}.parquet", frequency="00/06/12/18 UTC"),
    "participation": Dataset("pbp_participation", "pbp_participation_{season}.parquet", frequency="After postseason since 2023; not an in-season feed"),
    "trades": Dataset("trades", "trades.parquet", "USEFUL"),
    **{f"ngs_{kind}": Dataset("nextgen_stats", f"ngs_{kind}.parquet", "USEFUL", "Nightly 03–05 ET") for kind in ("passing", "rushing", "receiving")},
    **{f"advanced_{kind}": Dataset("pfr_advstats", f"advstats_week_{kind}_{{season}}.parquet", "USEFUL", "Daily 07:00 UTC") for kind in ("pass", "rush", "rec", "def")},
    "draft": Dataset("draft_picks", "draft_picks.parquet", "USEFUL"),
    "combine": Dataset("combine", "combine.parquet", "EXPERIMENTAL"),
    "qbr": Dataset("espn_data", "qbr_week_level.parquet", "USEFUL"),
    "ftn_charting": Dataset("ftn_charting", "ftn_charting_{season}.parquet", "EXPERIMENTAL", "00/06/12/18 UTC; provider dependent"),
    "contracts": Dataset("contracts", "historical_contracts.parquet", "EXPERIMENTAL"),
    "officials": Dataset("officials", "officials.parquet", "EXPERIMENTAL"),
    "teams": Dataset("teams", "teams_colors_logos.parquet", "USEFUL"),
}
