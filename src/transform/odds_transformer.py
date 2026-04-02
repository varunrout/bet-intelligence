"""
Odds transformation utilities for Football Market Intelligence System.

Computes:
  - Implied probabilities from decimal odds
  - Bookmaker margin (overround)
  - Margin-free (true) probabilities
  - Odds movement between snapshots
  - CLV (closing line value) labels

All inputs and outputs use decimal odds (European format).
All probabilities are in [0, 1].

Design rules:
  - No I/O here. Pure numeric transformations only.
  - All functions are stateless and vectorisable.
  - Raises ValueError if input odds are invalid (<=1.0).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Sanity bounds on bookmaker margin (overround).
# Values outside this range are suspicious and logged as warnings.
MARGIN_MIN = -0.02  # -2% (small negatives can appear in avg/max odds; flag only large negatives)
MARGIN_MAX = 0.20   # 20%


# ---------------------------------------------------------------------------
# Implied probability
# ---------------------------------------------------------------------------


def implied_prob(decimal_odds: float) -> float:
    """
    Convert a single decimal odds value to implied probability.

    Parameters
    ----------
    decimal_odds : float
        Must be > 1.0.

    Returns
    -------
    float : implied probability in (0, 1)
    """
    if decimal_odds <= 1.0:
        raise ValueError(f"Invalid decimal odds: {decimal_odds}. Must be > 1.0.")
    return 1.0 / decimal_odds


def implied_prob_series(odds: pd.Series) -> pd.Series:
    """Vectorised implied_prob for a pandas Series."""
    invalid = odds[odds <= 1.0]
    if not invalid.empty:
        raise ValueError(
            f"Invalid odds values (<=1.0) at indices: {invalid.index.tolist()}"
        )
    return 1.0 / odds


# ---------------------------------------------------------------------------
# Bookmaker margin
# ---------------------------------------------------------------------------


def compute_margin_two_way(odds_over: float, odds_under: float) -> float:
    """
    Compute bookmaker margin for a two-outcome market (e.g. Over/Under).

    margin = (1/odds_over + 1/odds_under) - 1

    A fair book has margin = 0.
    Typical bookmaker margins: 2–6% for sharp books, 5–10% for recreational.

    Parameters
    ----------
    odds_over, odds_under : float
        Decimal odds for each side.

    Returns
    -------
    float : margin as a proportion (e.g. 0.04 = 4%)
    """
    total = implied_prob(odds_over) + implied_prob(odds_under)
    margin = total - 1.0

    if not (MARGIN_MIN <= margin <= MARGIN_MAX):
        log.warning(
            "Unusual margin %.4f for odds (%.3f, %.3f).", margin, odds_over, odds_under
        )

    return margin


def compute_margin_three_way(
    odds_home: float, odds_draw: float, odds_away: float
) -> float:
    """
    Compute bookmaker margin for a three-outcome market (1X2).

    margin = (1/H + 1/D + 1/A) - 1
    """
    total = implied_prob(odds_home) + implied_prob(odds_draw) + implied_prob(odds_away)
    margin = total - 1.0

    if not (MARGIN_MIN <= margin <= MARGIN_MAX):
        log.warning(
            "Unusual margin %.4f for odds (%.3f, %.3f, %.3f).",
            margin,
            odds_home,
            odds_draw,
            odds_away,
        )

    return margin


# ---------------------------------------------------------------------------
# Margin-free (true) probabilities
# ---------------------------------------------------------------------------


def remove_margin_two_way(
    odds_over: float, odds_under: float
) -> tuple[float, float]:
    """
    Return margin-free probabilities for a two-outcome market.

    Uses the proportional (basic) margin removal method:
        true_prob_i = implied_prob_i / sum_of_implied_probs

    This is the simplest method and appropriate for symmetric markets.
    For 1X2, Shin's method or power method is more accurate — see
    compute_shin_probs_three_way for that case.

    Returns
    -------
    (prob_over, prob_under) : both in (0, 1), sum to 1.0
    """
    p_over = implied_prob(odds_over)
    p_under = implied_prob(odds_under)
    total = p_over + p_under
    return p_over / total, p_under / total


def remove_margin_three_way(
    odds_home: float, odds_draw: float, odds_away: float
) -> tuple[float, float, float]:
    """
    Return margin-free probabilities for a three-outcome market (1X2).

    Uses proportional method. For higher accuracy, use Shin's method.

    Returns
    -------
    (prob_home, prob_draw, prob_away) : all in (0, 1), sum to 1.0
    """
    p_h = implied_prob(odds_home)
    p_d = implied_prob(odds_draw)
    p_a = implied_prob(odds_away)
    total = p_h + p_d + p_a
    return p_h / total, p_d / total, p_a / total


# ---------------------------------------------------------------------------
# Odds movement
# ---------------------------------------------------------------------------


def odds_movement(opening_odds: float, closing_odds: float) -> float:
    """
    Compute the absolute change in odds from opening to closing.

    Negative value = odds drifted (moved out, less favoured).
    Positive value = odds shortened (moved in, more favoured).

    We define movement as: opening - closing
    So a positive value means the outcome became more favoured (odds fell).
    """
    return opening_odds - closing_odds


def prob_movement(opening_odds: float, closing_odds: float) -> float:
    """
    Compute the change in implied probability from opening to closing.

    Positive = market became more confident in the outcome.
    """
    return implied_prob(closing_odds) - implied_prob(opening_odds)


# ---------------------------------------------------------------------------
# Closing line value (CLV)
# ---------------------------------------------------------------------------


def compute_clv(odds_taken: float, closing_odds: float) -> float:
    """
    Compute closing line value (CLV) for a bet.

    CLV = (odds_taken / closing_odds) - 1

    Positive CLV: you got better odds than the closing line.
    Negative CLV: you got worse odds than the closing line.

    CLV is the primary quality signal for a betting process.
    Sustained positive CLV is evidence of genuine edge.

    Parameters
    ----------
    odds_taken   : decimal odds at bet placement
    closing_odds : decimal odds at market close (pre-kickoff)
    """
    if closing_odds <= 1.0:
        raise ValueError(f"Invalid closing odds: {closing_odds}")
    return (odds_taken / closing_odds) - 1.0


def compute_clv_series(
    odds_taken: pd.Series, closing_odds: pd.Series
) -> pd.Series:
    """Vectorised CLV computation."""
    return (odds_taken / closing_odds) - 1.0


# ---------------------------------------------------------------------------
# Bulk DataFrame helpers
# ---------------------------------------------------------------------------


def enrich_ou_odds_row(
    odds_over: float,
    odds_under: float,
    snapshot_type: str = "closing",
) -> dict:
    """
    Compute all derived fields for one Over/Under odds pair.

    Returns a dict suitable for direct insertion into odds_snapshots.
    Returns None if either odds value is missing or invalid.
    """
    try:
        if pd.isna(odds_over) or pd.isna(odds_under):
            return None
        if odds_over <= 1.0 or odds_under <= 1.0:
            return None

        margin = compute_margin_two_way(odds_over, odds_under)
        prob_over, prob_under = remove_margin_two_way(odds_over, odds_under)

        return {
            "odds_over": round(odds_over, 4),
            "odds_under": round(odds_under, 4),
            "implied_prob_over": round(prob_over, 6),
            "implied_prob_under": round(prob_under, 6),
            "margin": round(margin, 6),
            "snapshot_type": snapshot_type,
        }
    except (ValueError, ZeroDivisionError) as e:
        log.warning("Failed to enrich OU odds row: %s", e)
        return None


def enrich_ou_dataframe(
    df: pd.DataFrame,
    over_col: str,
    under_col: str,
    snapshot_type: str = "closing",
) -> pd.DataFrame:
    """
    Add implied probability and margin columns to a DataFrame with OU odds.

    Parameters
    ----------
    df           : DataFrame with odds columns
    over_col     : column name of the over odds
    under_col    : column name of the under odds
    snapshot_type: label for the snapshot ('opening', 'closing', 'timestamped')

    Returns
    -------
    DataFrame with additional columns:
        implied_prob_over, implied_prob_under, margin, snapshot_type
    """
    df = df.copy()

    valid_mask = (
        df[over_col].notna()
        & df[under_col].notna()
        & (df[over_col] > 1.0)
        & (df[under_col] > 1.0)
    )

    df.loc[valid_mask, "implied_prob_over"] = (
        1.0 / df.loc[valid_mask, over_col]
    )
    df.loc[valid_mask, "implied_prob_under"] = (
        1.0 / df.loc[valid_mask, under_col]
    )

    total = df.loc[valid_mask, "implied_prob_over"] + df.loc[valid_mask, "implied_prob_under"]
    df.loc[valid_mask, "margin"] = total - 1.0

    # Proportional margin removal
    df.loc[valid_mask, "implied_prob_over"] = (
        df.loc[valid_mask, "implied_prob_over"] / total
    )
    df.loc[valid_mask, "implied_prob_under"] = (
        df.loc[valid_mask, "implied_prob_under"] / total
    )

    df["snapshot_type"] = snapshot_type

    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        log.warning("%d rows had invalid/missing OU odds and were skipped.", n_invalid)

    return df
