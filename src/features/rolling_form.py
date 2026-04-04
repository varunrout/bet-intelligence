"""
Rolling form feature engineering for Football Market Intelligence System.

Design principles
-----------------
1. All rolling windows look BACKWARD only — no current match data is included.
2. `pandas.shift(1)` before every `.rolling()` call enforces this.
3. xG columns are computed as NULL placeholders until API-Football data is added.
4. Minimum 1 match of history required; features will be NaN for first appearances.

Usage
-----
    from src.features.rolling_form import build_rolling_features
    features_df = build_rolling_features(matches_df, stats_df)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Rolling windows to compute (in matches, not days)
WINDOWS = [3, 5]


# ---------------------------------------------------------------------------
# Step 1 — Build long-format team-match event table
# ---------------------------------------------------------------------------


def build_team_match_events(
    matches_df: pd.DataFrame,
    stats_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create one row per (team, match) from the matches and stats tables.

    Each row records what happened from that team's perspective:
        goals_for, goals_against, won, drew, clean_sheet,
        over_25, shots, shots_on_target, corners, yellow_cards.

    Parameters
    ----------
    matches_df : DataFrame from the matches table.
                 Required columns: match_id, kickoff_utc, home_team_id,
                 away_team_id, goals_home, goals_away, result_ftr, over_25
    stats_df   : DataFrame from the team_match_stats table.
                 Required columns: match_id, team_id, is_home,
                 shots, shots_on_target, corners, yellow_cards

    Returns
    -------
    DataFrame sorted by (team_id, kickoff_utc). One row per team per match.
    """
    required_match_cols = {
        "match_id", "kickoff_utc", "home_team_id", "away_team_id",
        "goals_home", "goals_away", "result_ftr", "over_25",
    }
    missing = required_match_cols - set(matches_df.columns)
    if missing:
        raise ValueError(f"matches_df missing columns: {missing}")

    # ── Home team rows ──────────────────────────────────────────────────────
    home = matches_df[list(required_match_cols)].copy()
    home["team_id"]       = home["home_team_id"]
    home["opp_team_id"]   = home["away_team_id"]
    home["is_home"]       = True
    home["goals_for"]     = home["goals_home"]
    home["goals_against"] = home["goals_away"]
    home["won"]           = (home["result_ftr"] == "H").astype(float)
    home["drew"]          = (home["result_ftr"] == "D").astype(float)
    home["clean_sheet"]   = (home["goals_away"] == 0).astype(float)

    # ── Away team rows ──────────────────────────────────────────────────────
    away = matches_df[list(required_match_cols)].copy()
    away["team_id"]       = away["away_team_id"]
    away["opp_team_id"]   = away["home_team_id"]
    away["is_home"]       = False
    away["goals_for"]     = away["goals_away"]
    away["goals_against"] = away["goals_home"]
    away["won"]           = (away["result_ftr"] == "A").astype(float)
    away["drew"]          = (away["result_ftr"] == "D").astype(float)
    away["clean_sheet"]   = (away["goals_home"] == 0).astype(float)

    base_cols = [
        "match_id", "kickoff_utc", "team_id", "opp_team_id", "is_home",
        "goals_for", "goals_against", "won", "drew", "clean_sheet", "over_25",
    ]
    events = pd.concat([home[base_cols], away[base_cols]], ignore_index=True)

    # ── Merge with stats ────────────────────────────────────────────────────
    stat_cols = ["match_id", "team_id", "shots", "shots_on_target", "corners", "yellow_cards",
                 "xg_for", "xg_against"]
    available_stat_cols = [c for c in stat_cols if c in stats_df.columns]
    events = events.merge(
        stats_df[available_stat_cols],
        on=["match_id", "team_id"],
        how="left",
    )

    events["over_25"] = pd.to_numeric(events["over_25"], errors="coerce")

    events = events.sort_values(["team_id", "kickoff_utc"]).reset_index(drop=True)
    return events


# ---------------------------------------------------------------------------
# Step 2 — Compute rolling statistics (strictly backward-looking)
# ---------------------------------------------------------------------------


def _rolling_lookback(series: pd.Series, window: int, agg: str = "mean") -> pd.Series:
    """
    Compute a rolling aggregate using only past observations (not current).

    shift(1) ensures the current match is NOT included in the window.
    min_periods=1 gives a result even with < window matches of history.

    Parameters
    ----------
    series : ordered Series for one team
    window : lookback window in matches
    agg    : 'mean' or 'sum'

    Returns
    -------
    Series of rolling values aligned to original index.
    """
    shifted = series.shift(1)
    if agg == "mean":
        return shifted.rolling(window, min_periods=1).mean()
    elif agg == "sum":
        return shifted.rolling(window, min_periods=1).sum()
    else:
        raise ValueError(f"Unknown agg: {agg}")


def compute_rolling_stats(
    events_df: pd.DataFrame,
    windows: list[int] = WINDOWS,
) -> pd.DataFrame:
    """
    Add rolling feature columns to the team-match events DataFrame.

    All rolling values at row i reflect only matches 0..i-1 for that team —
    standard shift(1) + rolling pattern.

    Parameters
    ----------
    events_df : output of build_team_match_events(), sorted by (team_id, kickoff_utc)
    windows   : list of window sizes to compute (default [3, 5])

    Returns
    -------
    events_df with new rolling columns appended.
    """
    df = events_df.copy()
    df = df.sort_values(["team_id", "kickoff_utc"])

    # Metrics to aggregate and the column mapping
    avg_metrics = {
        "goals_for":         "goals_scored",
        "goals_against":     "goals_conceded",
        "shots":             "shots",
        "shots_on_target":   "shots_on_target",
        "xg_for":            "xg_for",       # rolling xG scored (populated after Understat fetch)
        "xg_against":        "xg_against",   # rolling xG conceded
    }
    sum_metrics = {
        "won":         "wins",
        "clean_sheet": "clean_sheets",
        "over_25":     "over25",
    }

    for source_col, name in avg_metrics.items():
        if source_col not in df.columns:
            log.debug("Column %s not found — skipping rolling for %s.", source_col, name)
            continue
        for w in windows:
            col_name = f"{name}_avg{w}"
            df[col_name] = (
                df.groupby("team_id")[source_col]
                .transform(lambda x, w=w: _rolling_lookback(x, w, agg="mean"))
            )

    for source_col, name in sum_metrics.items():
        if source_col not in df.columns:
            continue
        for w in windows:
            col_name = f"{name}_last{w}"
            df[col_name] = (
                df.groupby("team_id")[source_col]
                .transform(lambda x, w=w: _rolling_lookback(x, w, agg="sum"))
            )

    return df


# ---------------------------------------------------------------------------
# Step 3 — Pivot to match-level wide format
# ---------------------------------------------------------------------------


def pivot_to_match_features(
    events_with_rolling: pd.DataFrame,
    matches_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert long-format (team × match) rolling features to wide-format:
    one row per match with home_{feature} and away_{feature} columns.

    Parameters
    ----------
    events_with_rolling : output of compute_rolling_stats()
    matches_df          : original matches DataFrame (for join base)

    Returns
    -------
    DataFrame indexed on match_id with home_ and away_ prefixed feature columns.
    """
    # Identify rolling feature columns (anything not in the base event columns)
    base_cols = {
        "match_id", "kickoff_utc", "team_id", "opp_team_id", "is_home",
        "goals_for", "goals_against", "won", "drew", "clean_sheet", "over_25",
        "shots", "shots_on_target", "corners", "yellow_cards",
        "xg_for", "xg_against",  # current-match values — excluded to prevent leakage
    }
    feature_cols = [c for c in events_with_rolling.columns if c not in base_cols]

    home_events = events_with_rolling[events_with_rolling["is_home"]].copy()
    away_events = events_with_rolling[~events_with_rolling["is_home"]].copy()

    home_feats = home_events[["match_id"] + feature_cols].copy()
    home_feats = home_feats.rename(columns={c: f"home_{c}" for c in feature_cols})

    away_feats = away_events[["match_id"] + feature_cols].copy()
    away_feats = away_feats.rename(columns={c: f"away_{c}" for c in feature_cols})

    base = matches_df[["match_id", "kickoff_utc"]].copy()
    result = base.merge(home_feats, on="match_id", how="left")
    result = result.merge(away_feats, on="match_id", how="left")

    return result.set_index("match_id")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_rolling_features(
    matches_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    windows: list[int] = WINDOWS,
) -> pd.DataFrame:
    """
    Full rolling feature pipeline. Returns one row per match with
    home_ and away_ prefixed rolling statistics.

    This is the primary entry point for Phase 3 feature engineering.

    Parameters
    ----------
    matches_df : from DB matches table
    stats_df   : from DB team_match_stats table
    windows    : list of lookback windows (default [3, 5])

    Returns
    -------
    DataFrame indexed on match_id. Columns:
        kickoff_utc,
        home_{metric}_avg{w}, home_{metric}_last{w},
        away_{metric}_avg{w}, away_{metric}_last{w}

    Leakage guarantee
    -----------------
    All rolling values at match M use only matches with kickoff < M's kickoff.
    Enforced by shift(1) before every rolling() call.
    No information from the current or future matches is ever included.
    """
    log.info("Building team match event table...")
    events = build_team_match_events(matches_df, stats_df)
    log.info("  %d team-match event rows.", len(events))

    log.info("Computing rolling statistics (windows: %s)...", windows)
    events_with_rolling = compute_rolling_stats(events, windows=windows)

    log.info("Pivoting to match-level wide format...")
    features = pivot_to_match_features(events_with_rolling, matches_df)
    log.info("  %d match feature rows, %d features each.", len(features), len(features.columns))

    return features
