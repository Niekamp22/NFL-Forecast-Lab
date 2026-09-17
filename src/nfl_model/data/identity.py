"""Resolve identities only through explicit source IDs, never player names."""
from __future__ import annotations

import pandas as pd

ID_FIELDS = ["gsis_id", "pfr_id", "espn_id", "pff_id", "esb_id", "nfl_id", "otc_id", "smart_id", "sportradar_id", "sleeper_id"]


def clean_id(values: pd.Series) -> pd.Series:
    """Preserve string IDs and normalize numeric IDs serialized with a .0 suffix."""
    result = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return result.mask(result.isin(["", "nan", "None", "<NA>"]))


def build_identity(players: pd.DataFrame, rosters: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build master records and a conflict-aware long-form alternate ID crosswalk."""
    master = players.copy().rename(columns={"gsis_id": "player_id", "display_name": "player_name", "college_name": "college", "status": "source_current_status"})
    master["player_id"] = clean_id(master.player_id)
    valid = master[master.player_id.notna()]
    if valid.player_id.duplicated().any():
        raise ValueError("Duplicate GSIS IDs in player master")
    # Keep ID-less master rows in the master; they cannot participate in canonical joins.
    missing = rosters.loc[rosters.gsis_id.notna() & ~rosters.gsis_id.isin(master.player_id)].copy()
    if not missing.empty:
        missing = missing.sort_values(["season", "week"]).drop_duplicates("gsis_id", keep="last")
        extra = missing.rename(columns={"gsis_id": "player_id", "full_name": "player_name"})
        extra = extra[[c for c in master if c in extra]]
        master = pd.concat([master, extra], ignore_index=True)
    pieces = []
    for source, frame in [("players", players), ("weekly_rosters", rosters)]:
        for field in ID_FIELDS:
            if field not in frame:
                continue
            part = pd.DataFrame({"player_id": clean_id(frame.gsis_id), "id_type": field,
                                 "alternate_id": clean_id(frame[field]), "source": source})
            pieces.append(part.dropna(subset=["player_id", "alternate_id"]))
    crosswalk = pd.concat(pieces, ignore_index=True).drop_duplicates()
    counts = crosswalk.groupby(["id_type", "alternate_id"]).player_id.transform("nunique")
    crosswalk["ambiguous"] = counts > 1
    conflicts = crosswalk[crosswalk.ambiguous].copy()
    for field in ID_FIELDS + ["jersey_number"]:
        if field in master:
            master[field] = clean_id(master[field])
    master["canonical_id_format"] = "source_nonstandard"
    master.loc[master.player_id.str.fullmatch(r"\d{2}-\d{7}", na=False), "canonical_id_format"] = "gsis"
    master.loc[master.player_id.isna(), "canonical_id_format"] = "missing"
    return master, crosswalk, conflicts


def resolve_ids(frame: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Attach canonical IDs. Conflicting alternate IDs remain unresolved."""
    result = frame.copy()
    candidates = []
    direct = clean_id(result.get("gsis_id", pd.Series(pd.NA, index=result.index, dtype="string")))
    candidates.append(direct.rename("direct"))
    for field in ID_FIELDS:
        if field == "gsis_id" or field not in result:
            continue
        mapping = crosswalk[(crosswalk.id_type == field) & ~crosswalk.ambiguous].drop_duplicates("alternate_id").set_index("alternate_id").player_id
        candidates.append(clean_id(result[field]).map(mapping).rename(field))
    matrix = pd.concat(candidates, axis=1)
    selected = matrix.bfill(axis=1).iloc[:, 0].astype("string")
    conflicts = matrix.ne(selected, axis=0).fillna(False).any(axis=1)
    result["player_id"] = selected.mask(conflicts)
    result["identity_status"] = "resolved"
    result.loc[result.player_id.isna(), "identity_status"] = "unmatched"
    result.loc[conflicts, "identity_status"] = "conflicting_ids"
    return result
