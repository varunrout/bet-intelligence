"""
Feature preparation utilities for Football Market Intelligence System — Phase 4.

Defines three feature set groupings keyed by modelling purpose:
  market_only  — Pinnacle closing implied probability only (logistic baseline).
  form_only    — rolling form + xG + context + Elo, no market signal.
  all          — full feature set (market + form), intended for tree models.

The prepare() function selects, validates, and optionally cleans a feature
matrix from the master DataFrame loaded from the engineered_features table.

Design rules
------------
- Column lists reference features with ≥ 99% DB coverage.
- xG rolling features (home/away_xg_for/against_avg5) are included since
  Understat data was ingested in Phase 4b — ~99.3% coverage.
- prepare() never modifies the input df; it always returns a copy.
- When drop_nan_rows=True the returned index is a strict subset of df.index,
  making it safe to recover the target via df.loc[X.index, "over_25"].
"""

from __future__ import annotations

import warnings

import pandas as pd

# ── Individual feature group lists ────────────────────────────────────────────

ROLLING_FEATURES: list[str] = [
    # 5-match rolling averages
    "home_goals_scored_avg5",
    "home_goals_conceded_avg5",
    "home_shots_avg5",
    "home_shots_on_target_avg5",
    "home_wins_last5",
    "home_clean_sheets_last5",
    "home_over25_last5",
    "home_goals_scored_avg3",
    "home_goals_conceded_avg3",
    # xG rolling averages (populated via Understat Phase 4b ingestion — ~99.3% coverage)
    "home_xg_for_avg5",
    "home_xg_against_avg5",
    "away_xg_for_avg5",
    "away_xg_against_avg5",
    "away_goals_scored_avg5",
    "away_goals_conceded_avg5",
    "away_shots_avg5",
    "away_shots_on_target_avg5",
    "away_wins_last5",
    "away_clean_sheets_last5",
    "away_over25_last5",
    "away_goals_scored_avg3",
    "away_goals_conceded_avg3",
    # Derived cross-team rolling features
    "combined_goals_avg5",
    "attack_proxy",
]

CONTEXT_FEATURES: list[str] = [
    "home_rest_days",
    "away_rest_days",
    "rest_differential",
    "home_matches_in_14_days",
    "away_matches_in_14_days",
]

ELO_FEATURES: list[str] = [
    "home_elo",
    "away_elo",
    "elo_differential",
]

MARKET_FEATURES: list[str] = [
    "pin_implied_prob_over",
    "pin_odds_over",
    "pin_odds_under",
    "pin_margin",
    "b365_implied_prob_over",
    "b365_odds_over",
    "b365_odds_under",
    "b365_margin",
    "avg_implied_prob_over",
    "max_implied_prob_over",
    "pin_b365_divergence",
]

# ── Feature set registry ───────────────────────────────────────────────────────
#
# market_only  : single-feature logistic regression — sets the market ceiling.
# form_only    : all non-market pre-match features — measures pure form signal.
# all          : union of market and form features — intended for gradient boosting.

FEATURE_SETS: dict[str, list[str]] = {
    "market_only": [
        "pin_implied_prob_over",
    ],
    "form_only": ROLLING_FEATURES + CONTEXT_FEATURES + ELO_FEATURES,
    "all": ROLLING_FEATURES + CONTEXT_FEATURES + ELO_FEATURES + MARKET_FEATURES,
}


# ── Public API ─────────────────────────────────────────────────────────────────


def prepare(
    df: pd.DataFrame,
    feature_set: str,
    drop_nan_rows: bool = True,
) -> pd.DataFrame:
    """
    Select and optionally clean features from the modelling DataFrame.

    Parameters
    ----------
    df            : Master DataFrame loaded from the engineered_features join.
                    Must be sorted by kickoff_utc (chronological order).
                    Must contain the columns listed in FEATURE_SETS[feature_set].
    feature_set   : One of 'market_only', 'form_only', 'all'.
    drop_nan_rows : If True, rows with any NaN in the selected columns are
                    dropped.  The returned index is then a strict subset of
                    df.index — use df.loc[X.index, "over_25"] to align labels.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.  Index is preserved from df (or a subset when
        drop_nan_rows=True).  All selected columns are cast to float64.
    y : pd.Series
        Binary target (over_25) aligned to X.index.  Requires 'over_25'
        column in df.

    Raises
    ------
    ValueError
        If feature_set is not a key in FEATURE_SETS.

    Notes
    -----
    Columns in FEATURE_SETS[feature_set] that are not present in df are
    silently skipped with a UserWarning.  This allows the function to work
    gracefully with older feature versions that do not contain every column.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature_set '{feature_set}'. "
            f"Available: {sorted(FEATURE_SETS.keys())}"
        )

    requested = FEATURE_SETS[feature_set]
    available = [c for c in requested if c in df.columns]
    missing   = [c for c in requested if c not in df.columns]

    if missing:
        warnings.warn(
            f"prepare(): feature_set='{feature_set}' — {len(missing)} column(s) "
            f"not found in df and will be skipped: "
            f"{missing[:6]}{'...' if len(missing) > 6 else ''}",
            UserWarning,
            stacklevel=2,
        )

    X = df[available].copy().apply(pd.to_numeric, errors="coerce")

    if drop_nan_rows:
        X = X.dropna()

    y = df.loc[X.index, "over_25"].astype(float)
    return X, y
