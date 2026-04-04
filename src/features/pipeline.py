"""
Feature engineering pipeline orchestrator for Football Market Intelligence System.

Sequence
--------
1. Load matches, stats, and odds from DuckDB.
2. Compute rolling form features (rolling_form.py).
3. Compute context features: rest days, congestion, Elo (context_features.py).
4. Extract market features from odds (market_features.py).
5. Merge all features into one row per match.
6. Compute derived cross-team features.
7. Write to engineered_features table (idempotent via UNIQUE constraint).

Leakage policy
--------------
All features in the final DataFrame use only information available
strictly before kickoff_utc. This is enforced at each sub-module level.
compute_features() performs a final audit before inserting into the DB.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.context_features import build_context_features
from src.features.market_features import extract_market_features
from src.features.rolling_form import build_rolling_features
from src.utils.config import DB_PATH
from src.utils.db import get_connection, run_query, upsert_dataframe

log = logging.getLogger(__name__)

FEATURE_VERSION = "v1"

# Columns that must NOT appear in the feature set
FORBIDDEN_FEATURE_COLS = {
    "goals_home", "goals_away", "total_goals",
    "over_25", "result_ftr", "goals_ht_home", "goals_ht_away",
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_matches(competition_key: str, db_path: Path) -> pd.DataFrame:
    sql = """
        SELECT
            m.match_id, m.kickoff_utc, m.season,
            m.home_team_id, m.away_team_id,
            m.goals_home, m.goals_away,
            m.total_goals, m.over_25,
            m.result_ftr
        FROM matches m
        JOIN competitions c ON m.competition_id = c.competition_id
        WHERE c.short_code = ?
        ORDER BY m.kickoff_utc
    """
    df = run_query(sql, params=[competition_key.upper()], db_path=db_path)
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True)
    return df


def _load_stats(competition_key: str, db_path: Path) -> pd.DataFrame:
    sql = """
        SELECT s.match_id, s.team_id, s.is_home,
               s.shots, s.shots_on_target,
               s.corners, s.fouls, s.yellow_cards, s.red_cards,
               s.xg_for, s.xg_against
        FROM team_match_stats s
        JOIN matches m ON s.match_id = m.match_id
        JOIN competitions c ON m.competition_id = c.competition_id
        WHERE c.short_code = ?
    """
    return run_query(sql, params=[competition_key.upper()], db_path=db_path)


def _load_odds(competition_key: str, db_path: Path) -> pd.DataFrame:
    sql = """
        SELECT o.match_id, bk.short_code,
               o.odds_over, o.odds_under,
               o.implied_prob_over, o.implied_prob_under,
               o.margin, o.snapshot_type
        FROM odds_snapshots o
        JOIN bookmakers bk ON o.bookmaker_id = bk.bookmaker_id
        JOIN matches m ON o.match_id = m.match_id
        JOIN competitions c ON m.competition_id = c.competition_id
        WHERE c.short_code = ?
          AND o.market_type = 'ou25'
    """
    return run_query(sql, params=[competition_key.upper()], db_path=db_path)


# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------


def _audit_for_leakage(features_df: pd.DataFrame) -> None:
    """
    Final check before DB insertion.
    Raises AssertionError if any forbidden target column is found.
    """
    leaked = set(features_df.columns) & FORBIDDEN_FEATURE_COLS
    if leaked:
        raise AssertionError(
            f"LEAKAGE DETECTED: forbidden columns found in feature set: {leaked}. "
            f"Remove them before inserting into engineered_features."
        )
    log.debug("Leakage audit passed.")


# ---------------------------------------------------------------------------
# Derived cross-team features
# ---------------------------------------------------------------------------


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features from home + away rolling stats.
    All inputs are already backward-looking at this point.
    """
    df = df.copy()

    # Combined expected goals proxy
    # = avg of (home attack + away attack) — rough measure of total expected activity
    if "home_goals_scored_avg5" in df.columns and "away_goals_scored_avg5" in df.columns:
        df["combined_goals_avg5"] = (
            df["home_goals_scored_avg5"] + df["away_goals_scored_avg5"]
        ) / 2

    # Attack proxy = (home attack + away attack) adjusted by (home defence + away defence)
    # Approximates expected total goals using both teams' form
    # attack_proxy = (H_scored_avg5 + H_conceded_avg5 * 0 - wait wrong)
    # Better: average of (home goals scored + away goals against) and (away goals scored + home goals against)
    # = (home_attack + away_attack) / 2 where attack = avg goals the team contributes to over matches
    if all(c in df.columns for c in [
        "home_goals_scored_avg5", "away_goals_conceded_avg5",
        "away_goals_scored_avg5", "home_goals_conceded_avg5",
    ]):
        df["attack_proxy"] = (
            (df["home_goals_scored_avg5"] + df["away_goals_conceded_avg5"])
            + (df["away_goals_scored_avg5"] + df["home_goals_conceded_avg5"])
        ) / 2

    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def build_features(
    competition_key: str = "EPL",
    db_path: Path = DB_PATH,
    feature_version: str = FEATURE_VERSION,
) -> pd.DataFrame:
    """
    Build the complete pre-match feature set for all matches in a competition.

    Steps:
        1. Load data from DB
        2. Rolling form features
        3. Context features (rest, congestion, Elo)
        4. Market features (odds)
        5. Merge + derived features
        6. Leakage audit

    Returns
    -------
    DataFrame indexed on match_id. Does NOT write to DB — use
    save_features() for that.
    """
    log.info("Loading data for competition: %s", competition_key)
    matches = _load_matches(competition_key, db_path)
    stats   = _load_stats(competition_key, db_path)
    odds    = _load_odds(competition_key, db_path)

    n_m, n_s, n_o = len(matches), len(stats), len(odds)
    log.info("  Loaded %d matches, %d stat rows, %d odds rows.", n_m, n_s, n_o)

    if matches.empty:
        raise RuntimeError(f"No matches found for competition '{competition_key}'.")

    # ── Rolling form ─────────────────────────────────────────────────────────
    log.info("Step 1/3 — Rolling form features...")
    rolling_feats = build_rolling_features(matches, stats)

    # ── Context features ─────────────────────────────────────────────────────
    log.info("Step 2/3 — Context features (rest, congestion, Elo)...")
    context_feats = build_context_features(matches)

    # ── Market features ───────────────────────────────────────────────────────
    log.info("Step 3/3 — Market features...")
    market_feats = extract_market_features(odds)

    # ── Merge ─────────────────────────────────────────────────────────────────
    log.info("Merging feature sets...")
    features = rolling_feats.join(context_feats, how="left")
    features = features.join(market_feats, how="left")

    # ── Derived features ──────────────────────────────────────────────────────
    features = _add_derived_features(features)

    # ── Add metadata ──────────────────────────────────────────────────────────
    features["computed_at_utc"] = matches.set_index("match_id")["kickoff_utc"]
    features["feature_version"] = feature_version
    features["is_neutral"]      = False

    # ── Leakage audit ─────────────────────────────────────────────────────────
    _audit_for_leakage(features)

    log.info(
        "Feature build complete: %d rows × %d columns.",
        len(features), len(features.columns),
    )
    return features


def save_features(
    features_df: pd.DataFrame,
    db_path: Path = DB_PATH,
    force: bool = False,
) -> int:
    """
    Write the feature DataFrame to the engineered_features table.

    Uses INSERT OR IGNORE based on UNIQUE (match_id, feature_version)
    so the operation is idempotent — safe to run multiple times.

    Parameters
    ----------
    features_df : DataFrame returned by build_features()
    db_path     : path to DuckDB file
    force       : when True, DELETE existing rows for this feature_version
                  before inserting.  Use after adding new data sources
                  (e.g. Understat xG) so existing rows are fully refreshed.

    Returns
    -------
    int : number of new rows inserted
    """
    df = features_df.reset_index()  # match_id back as column

    # ── Force rebuild: clear existing rows before re-inserting ────────────
    if force:
        with get_connection(db_path) as con:
            n_del = con.execute(
                "SELECT COUNT(*) FROM engineered_features WHERE feature_version = ?",
                [FEATURE_VERSION],
            ).fetchone()[0]
            con.execute(
                "DELETE FROM engineered_features WHERE feature_version = ?",
                [FEATURE_VERSION],
            )
        log.info(
            "Force rebuild: deleted %d existing feature rows (version='%s').",
            n_del, FEATURE_VERSION,
        )

    # Schema column ordering — only keep columns that exist in both the
    # DataFrame and the DB schema. Extra columns are silently dropped.
    schema_cols = [
        "match_id", "computed_at_utc",
        "home_goals_scored_avg5", "home_goals_conceded_avg5",
        "home_shots_avg5", "home_shots_on_target_avg5",
        "home_wins_last5", "home_clean_sheets_last5", "home_over25_last5",
        "home_goals_scored_avg3", "home_goals_conceded_avg3",
        "away_goals_scored_avg5", "away_goals_conceded_avg5",
        "away_shots_avg5", "away_shots_on_target_avg5",
        "away_wins_last5", "away_clean_sheets_last5", "away_over25_last5",
        "away_goals_scored_avg3", "away_goals_conceded_avg3",
        "home_xg_for_avg5", "home_xg_against_avg5",
        "away_xg_for_avg5", "away_xg_against_avg5",
        "combined_goals_avg5", "attack_proxy",
        "home_rest_days", "away_rest_days", "rest_differential",
        "home_matches_in_14_days", "away_matches_in_14_days",
        "is_neutral",
        "home_elo", "away_elo", "elo_differential",
        "pin_implied_prob_over", "pin_odds_over", "pin_odds_under", "pin_margin",
        "b365_implied_prob_over", "b365_odds_over", "b365_odds_under", "b365_margin",
        "avg_implied_prob_over", "max_implied_prob_over",
        "pin_b365_divergence",
        "opening_implied_prob_over", "closing_implied_prob_over",
        "opening_margin", "odds_movement_over",
        "home_key_absences", "away_key_absences",
        "feature_version",
    ]

    # Build final insert DataFrame with only matched columns
    insert_df = pd.DataFrame()
    for col in schema_cols:
        if col in df.columns:
            insert_df[col] = df[col]
        else:
            insert_df[col] = None

    # Add surrogate key
    with get_connection(db_path) as con:
        try:
            current_max = con.execute(
                "SELECT COALESCE(MAX(feature_id), 0) FROM engineered_features"
            ).fetchone()[0]
        except Exception:
            current_max = 0

    insert_df.insert(0, "feature_id", range(current_max + 1, current_max + 1 + len(insert_df)))

    n = upsert_dataframe(
        insert_df,
        "engineered_features",
        conflict_columns=["match_id", "feature_version"],
        db_path=db_path,
    )
    log.info("Saved %d feature rows to engineered_features.", n)
    return n
