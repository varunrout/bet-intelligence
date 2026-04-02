"""
Context and Elo feature engineering for Football Market Intelligence System.

Produces per-match features that capture:
  - Rest days since each team's last match
  - Match congestion (matches in the last 14 days)
  - Elo ratings at kickoff (pre-match, strictly backward)

Leakage notes
-------------
Rest days and congestion are computed from the team's PREVIOUS matches only.
Elo ratings use only results from prior completed matches.
All timestamps are UTC.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Elo configuration
ELO_INITIAL  = 1500.0
ELO_K_FACTOR = 20.0
ELO_SCALE    = 400.0


# ---------------------------------------------------------------------------
# Rest days and match congestion
# ---------------------------------------------------------------------------


def _build_team_kicks(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Long-format table: one row per (team, match) with only match_id,
    kickoff_utc, and team_id. Used for time-based lookbacks.
    """
    home = matches_df[["match_id", "kickoff_utc", "home_team_id"]].rename(
        columns={"home_team_id": "team_id"}
    )
    away = matches_df[["match_id", "kickoff_utc", "away_team_id"]].rename(
        columns={"away_team_id": "team_id"}
    )
    kicks = pd.concat([home, away], ignore_index=True)
    return kicks.sort_values(["team_id", "kickoff_utc"]).reset_index(drop=True)


def compute_rest_days(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rest days for the home and away team in each match.

    rest_days = number of whole calendar days between the team's previous
    match kickoff and this match's kickoff.

    First appearance of a team → NaN (no prior match in dataset).

    Returns
    -------
    DataFrame with columns:
        match_id, home_rest_days, away_rest_days, rest_differential
    """
    kicks = _build_team_kicks(matches_df)

    # Previous kickoff per team (shift within team group)
    kicks["prev_kickoff"] = kicks.groupby("team_id")["kickoff_utc"].shift(1)
    kicks["rest_days"] = (
        (kicks["kickoff_utc"] - kicks["prev_kickoff"])
        .dt.total_seconds()
        .div(86400)
        .apply(lambda x: int(x) if pd.notna(x) else None)
    )

    # Merge back to match level
    home_rest = kicks.merge(
        matches_df[["match_id", "home_team_id"]],
        left_on=["match_id", "team_id"],
        right_on=["match_id", "home_team_id"],
    )[["match_id", "rest_days"]].rename(columns={"rest_days": "home_rest_days"})

    away_rest = kicks.merge(
        matches_df[["match_id", "away_team_id"]],
        left_on=["match_id", "team_id"],
        right_on=["match_id", "away_team_id"],
    )[["match_id", "rest_days"]].rename(columns={"rest_days": "away_rest_days"})

    result = matches_df[["match_id"]].merge(home_rest, on="match_id", how="left")
    result = result.merge(away_rest, on="match_id", how="left")
    result["rest_differential"] = result["home_rest_days"] - result["away_rest_days"]

    return result[["match_id", "home_rest_days", "away_rest_days", "rest_differential"]]


def compute_congestion(
    matches_df: pd.DataFrame,
    window_days: int = 14,
) -> pd.DataFrame:
    """
    Count how many matches each team played in the {window_days} days
    BEFORE each kickoff (not including the current match).

    A team playing 3 EPL matches in 14 days is congested.
    A team with 1 match in 14 days is fresh.

    Parameters
    ----------
    matches_df   : matches table
    window_days  : lookback window in days (default 14)

    Returns
    -------
    DataFrame with columns:
        match_id, home_matches_in_{window_days}_days, away_matches_in_{window_days}_days
    """
    kicks = _build_team_kicks(matches_df)
    kicks = kicks.sort_values(["team_id", "kickoff_utc"])

    # Use pandas time-based rolling with a UTC index
    kicks["kickoff_utc_naive"] = kicks["kickoff_utc"].dt.tz_localize(None)

    congestion_rows = []

    for team_id, grp in kicks.groupby("team_id"):
        grp = grp.set_index("kickoff_utc_naive").sort_index()
        window_str = f"{window_days}D"
        # Count matches in prior window_days days — excluding current match
        # closed='left' means the current timestamp is NOT included
        grp["congestion"] = (
            grp["match_id"]
            .rolling(window_str, closed="left")
            .count()
            .fillna(0)
            .astype(int)
        )
        grp = grp.reset_index()
        grp["team_id"] = team_id  # restore team_id for join key
        congestion_rows.append(grp[["match_id", "team_id", "congestion"]])

    congestion_df = pd.concat(congestion_rows, ignore_index=True)
    congestion_df = congestion_df.rename(columns={"congestion": "cong"})

    # Map back to home/away — merge on (match_id, team_id) to avoid Cartesian product
    kicks = kicks.reset_index(drop=True)
    kicks_with_cong = kicks.merge(congestion_df, on=["match_id", "team_id"], how="left")

    home_c = kicks_with_cong.merge(
        matches_df[["match_id", "home_team_id"]],
        left_on=["match_id", "team_id"],
        right_on=["match_id", "home_team_id"],
        how="inner",
    )[["match_id", "cong"]].rename(columns={"cong": f"home_matches_in_{window_days}_days"})

    away_c = kicks_with_cong.merge(
        matches_df[["match_id", "away_team_id"]],
        left_on=["match_id", "team_id"],
        right_on=["match_id", "away_team_id"],
        how="inner",
    )[["match_id", "cong"]].rename(columns={"cong": f"away_matches_in_{window_days}_days"})

    result = matches_df[["match_id"]].merge(home_c, on="match_id", how="left")
    result = result.merge(away_c, on="match_id", how="left")

    return result


# ---------------------------------------------------------------------------
# Elo ratings
# ---------------------------------------------------------------------------


def compute_elo_ratings(
    matches_df: pd.DataFrame,
    k_factor: float = ELO_K_FACTOR,
    initial_rating: float = ELO_INITIAL,
) -> pd.DataFrame:
    """
    Compute Elo ratings for all teams at each match kickoff.

    The rating returned for each match is the PRE-MATCH rating —
    the state before that match's result is processed.

    Algorithm
    ---------
    Standard Elo with:
        expected_home = 1 / (1 + 10^((R_away - R_home) / 400))
        score_home    = 1.0 (win), 0.5 (draw), 0.0 (loss)
        R_home_new    = R_home + K * (score_home - expected_home)

    Ties in kickoff times within the same match are not possible
    (each match has a unique kickoff). Teams starting their first
    season get initial_rating.

    Parameters
    ----------
    matches_df     : must have match_id, kickoff_utc, home_team_id,
                     away_team_id, result_ftr (H/D/A), goals_home, goals_away
    k_factor       : Elo update magnitude (default 20)
    initial_rating : Starting rating for new teams (default 1500)

    Returns
    -------
    DataFrame with columns: match_id, home_elo, away_elo, elo_differential
    (all PRE-match ratings)
    """
    ratings: dict[int, float] = {}
    rows = []

    matches_sorted = matches_df.sort_values("kickoff_utc").copy()

    for _, match in matches_sorted.iterrows():
        h_id = int(match["home_team_id"])
        a_id = int(match["away_team_id"])

        r_h = ratings.get(h_id, initial_rating)
        r_a = ratings.get(a_id, initial_rating)

        # Store PRE-match ratings as features
        rows.append({
            "match_id":        match["match_id"],
            "home_elo":        round(r_h, 2),
            "away_elo":        round(r_a, 2),
            "elo_differential": round(r_h - r_a, 2),
        })

        # Skip rating update if result not available
        ftr = match.get("result_ftr")
        if pd.isna(ftr) or ftr not in ("H", "D", "A"):
            continue

        # Expected scores
        e_h = 1.0 / (1.0 + 10.0 ** ((r_a - r_h) / ELO_SCALE))
        e_a = 1.0 - e_h

        # Actual scores
        if ftr == "H":
            s_h, s_a = 1.0, 0.0
        elif ftr == "D":
            s_h, s_a = 0.5, 0.5
        else:  # A
            s_h, s_a = 0.0, 1.0

        # Update
        ratings[h_id] = r_h + k_factor * (s_h - e_h)
        ratings[a_id] = r_a + k_factor * (s_a - e_a)

    elo_df = pd.DataFrame(rows)
    log.info(
        "Elo computed for %d matches. Final ratings range: [%.0f, %.0f].",
        len(elo_df),
        min(ratings.values()) if ratings else initial_rating,
        max(ratings.values()) if ratings else initial_rating,
    )
    return elo_df


# ---------------------------------------------------------------------------
# Combined context feature builder
# ---------------------------------------------------------------------------


def build_context_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all context features for every match.

    Combines: rest days + congestion + Elo into one DataFrame.

    Returns
    -------
    DataFrame indexed on match_id with all context columns.
    """
    log.info("Computing rest days...")
    rest = compute_rest_days(matches_df)

    log.info("Computing match congestion (14-day window)...")
    cong = compute_congestion(matches_df, window_days=14)

    log.info("Computing Elo ratings...")
    elo = compute_elo_ratings(matches_df)

    result = rest.merge(cong, on="match_id", how="left")
    result = result.merge(elo, on="match_id", how="left")
    result = result.set_index("match_id")

    log.info("Context features complete: %d rows, %d columns.", len(result), len(result.columns))
    return result
