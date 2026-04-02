"""
Match transformer — maps parsed FDCO DataFrames to database-ready records.

Responsibilities:
  - Resolve team canonical names -> team_ids via database lookup.
  - Add derived label columns (over_25, total_goals, result_ftr).
  - Validate that no post-match information enters match rows.
  - Deduplicate matches before DB insertion.

This module is intentionally thin. Heavy cleaning lives in fdco_loader.py.
The transformer only does DB ID resolution and final preparation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


def resolve_team_ids(
    matches_df: pd.DataFrame,
    name_to_id: dict[str, int],
) -> pd.DataFrame:
    """
    Replace home_team_canonical and away_team_canonical columns with integer IDs.

    Parameters
    ----------
    matches_df : DataFrame with columns 'home_team_canonical', 'away_team_canonical'
    name_to_id : dict mapping canonical team name -> team_id (from teams table)

    Returns
    -------
    DataFrame with 'home_team_id' and 'away_team_id' columns added.
    Original canonical name columns are dropped.

    Raises
    ------
    ValueError if any team name cannot be resolved to an ID.
    """
    df = matches_df.copy()

    unresolved = set()
    for col in ("home_team_canonical", "away_team_canonical"):
        missing = set(df[col].unique()) - set(name_to_id.keys())
        unresolved.update(missing)

    if unresolved:
        raise ValueError(
            f"Cannot resolve team names to IDs: {sorted(unresolved)}. "
            f"Run scripts/init_db.py first to seed the teams table."
        )

    df["home_team_id"] = df["home_team_canonical"].map(name_to_id)
    df["away_team_id"] = df["away_team_canonical"].map(name_to_id)
    df = df.drop(columns=["home_team_canonical", "away_team_canonical"])

    return df


def resolve_team_ids_for_stats(
    stats_df: pd.DataFrame,
    name_to_id: dict[str, int],
) -> pd.DataFrame:
    """
    Replace team_canonical column in stats DataFrame with team_id.
    """
    df = stats_df.copy()

    unresolved = set(df["team_canonical"].unique()) - set(name_to_id.keys())
    if unresolved:
        raise ValueError(
            f"Cannot resolve team names in stats: {sorted(unresolved)}"
        )

    df["team_id"] = df["team_canonical"].map(name_to_id)
    df = df.drop(columns=["team_canonical"])
    return df


def add_surrogate_keys(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    Add an integer surrogate primary key column to a DataFrame.
    IDs are sequential starting from 1 (unless an offset is provided for append mode).

    For production use, prefer database SEQUENCE / AUTOINCREMENT.
    This helper is used for bulk inserts into DuckDB where autoincrement
    is simulated with row_number assignments.
    """
    df = df.copy()
    df.insert(0, id_col, range(1, len(df) + 1))
    return df


def dedup_matches(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows based on match_id.
    Keeps the first occurrence. Logs a warning if duplicates are found.
    """
    n_before = len(matches_df)
    matches_df = matches_df.drop_duplicates(subset=["match_id"], keep="first")
    n_after = len(matches_df)
    if n_before != n_after:
        log.warning(
            "Removed %d duplicate match rows (kept %d).", n_before - n_after, n_after
        )
    return matches_df


def validate_no_future_leakage(matches_df: pd.DataFrame) -> None:
    """
    Assert that result columns (goals, FTR, over_25) are only non-null
    when the match kickoff is in the past relative to the computed_at time.

    In practice for batch historical loading, this always passes.
    This exists as a guard for online/live data integration later.

    Raises AssertionError if leakage is detected.
    """
    if "computed_at_utc" not in matches_df.columns:
        return

    future_rows = matches_df[
        (matches_df["kickoff_utc"] > matches_df["computed_at_utc"])
        & matches_df["goals_home"].notna()
    ]

    if not future_rows.empty:
        raise AssertionError(
            f"Leakage detected: {len(future_rows)} rows have result data "
            f"but kickoff is in the future relative to computed_at_utc.\n"
            f"{future_rows[['match_id', 'kickoff_utc', 'computed_at_utc', 'goals_home']].head()}"
        )


def prepare_matches_for_insert(
    matches_df: pd.DataFrame,
    name_to_id: dict[str, int],
) -> pd.DataFrame:
    """
    Full preparation pipeline: dedup -> resolve IDs -> validate -> select columns.

    Returns a DataFrame ready for direct insertion into the matches table.
    """
    df = dedup_matches(matches_df)
    df = resolve_team_ids(df, name_to_id)
    validate_no_future_leakage(df)

    # Select only columns that exist in the matches schema
    schema_cols = [
        "match_id", "competition_id", "season", "gameweek", "kickoff_utc",
        "home_team_id", "away_team_id", "venue",
        "goals_home", "goals_away", "goals_ht_home", "goals_ht_away",
        "result_ftr", "total_goals", "over_25",
        "source", "api_football_id", "fdco_row_key",
    ]

    # Only keep columns that are present (some may not be in FDCO data)
    available_cols = [c for c in schema_cols if c in df.columns]
    return df[available_cols]


def prepare_stats_for_insert(
    stats_df: pd.DataFrame,
    name_to_id: dict[str, int],
    id_offset: int = 0,
) -> pd.DataFrame:
    """
    Prepare team_match_stats DataFrame for DB insertion.
    """
    if stats_df.empty:
        return stats_df

    df = resolve_team_ids_for_stats(stats_df, name_to_id)

    schema_cols = [
        "match_id", "team_id", "is_home",
        "shots", "shots_on_target",
        "xg_for", "xg_against",
        "possession_pct", "corners", "fouls",
        "yellow_cards", "red_cards",
        "source",
    ]
    available_cols = [c for c in schema_cols if c in df.columns]
    df = df[available_cols].copy()
    df.insert(0, "stat_id", range(id_offset + 1, id_offset + 1 + len(df)))
    return df


def prepare_odds_for_insert(
    odds_df: pd.DataFrame,
    id_offset: int = 0,
) -> pd.DataFrame:
    """
    Prepare odds_snapshots DataFrame for DB insertion.
    Adds surrogate snapshot_id starting from id_offset + 1.
    """
    if odds_df.empty:
        return odds_df

    schema_cols = [
        "match_id", "bookmaker_id", "market_type", "snapshot_type",
        "snapshot_utc", "line",
        "odds_over", "odds_under",
        "implied_prob_over", "implied_prob_under",
        "margin", "source",
    ]
    available_cols = [c for c in schema_cols if c in odds_df.columns]
    df = odds_df[available_cols].copy()
    df.insert(0, "snapshot_id", range(id_offset + 1, id_offset + 1 + len(df)))
    return df
