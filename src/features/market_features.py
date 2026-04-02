"""
Market feature extraction for Football Market Intelligence System.

Extracts pre-match odds features from the odds_snapshots table.
Treats Pinnacle as the sharp/efficient price signal and Bet365 as
the recreational book benchmark.

Feature categories
------------------
1. Pinnacle implied probability and margin
2. Bet365 implied probability and margin
3. Market average and maximum
4. PIN vs B365 divergence signal

Leakage policy
--------------
Only uses the 'closing' snapshot from FDCO data.
In Phase 7 (CLV modeling), opening vs closing distinction matters.
For now all FDCO odds are treated as pre-match closing odds.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

# Short codes as stored in bookmakers table
_BOOKMAKER_MAP = {
    "PIN":  "pin",
    "B365": "b365",
    "AVG":  "avg",
    "MAX":  "max",
}


def extract_market_features(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot odds_snapshots into one row per match with bookmaker columns.

    Parameters
    ----------
    odds_df : DataFrame from odds_snapshots table joined with bookmakers.
              Required columns:
                match_id, short_code (bookmaker),
                implied_prob_over, odds_over, odds_under, margin

    Returns
    -------
    DataFrame indexed on match_id with columns:
        pin_implied_prob_over, pin_odds_over, pin_odds_under, pin_margin,
        b365_implied_prob_over, b365_odds_over, b365_odds_under, b365_margin,
        avg_implied_prob_over, max_implied_prob_over,
        pin_b365_divergence,
        opening_implied_prob_over, closing_implied_prob_over (duplicated from PIN)
    """
    required_cols = {
        "match_id", "short_code", "implied_prob_over",
        "odds_over", "odds_under", "margin",
    }
    missing = required_cols - set(odds_df.columns)
    if missing:
        raise ValueError(f"odds_df missing columns: {missing}")

    rows = {}

    for match_id, grp in odds_df.groupby("match_id"):
        row: dict = {"match_id": match_id}

        for short_code, prefix in _BOOKMAKER_MAP.items():
            bk_row = grp[grp["short_code"] == short_code]
            if bk_row.empty:
                row[f"{prefix}_implied_prob_over"] = None
                row[f"{prefix}_odds_over"]         = None
                row[f"{prefix}_odds_under"]        = None
                row[f"{prefix}_margin"]            = None
                continue

            # If somehow multiple rows (duplicate data), take first
            bk_row = bk_row.iloc[0]

            row[f"{prefix}_implied_prob_over"] = bk_row.get("implied_prob_over")
            row[f"{prefix}_odds_over"]         = bk_row.get("odds_over")
            row[f"{prefix}_odds_under"]        = bk_row.get("odds_under")
            row[f"{prefix}_margin"]            = bk_row.get("margin")

        rows[match_id] = row

    if not rows:
        log.warning("No market features extracted — odds_df is empty.")
        return pd.DataFrame()

    market_df = pd.DataFrame(rows.values())

    # Divergence signal: |PIN − B365| implied probability
    if "pin_implied_prob_over" in market_df.columns and "b365_implied_prob_over" in market_df.columns:
        market_df["pin_b365_divergence"] = (
            market_df["pin_implied_prob_over"] - market_df["b365_implied_prob_over"]
        ).abs()
    else:
        market_df["pin_b365_divergence"] = None

    # Legacy columns for Phase 7 CLV compatibility
    # With FDCO data there is only one snapshot type (closing).
    # When The Odds API data is added, these will be populated from different timestamps.
    market_df["opening_implied_prob_over"] = market_df.get("pin_implied_prob_over")
    market_df["closing_implied_prob_over"] = market_df.get("pin_implied_prob_over")
    market_df["opening_margin"]            = market_df.get("pin_margin")
    market_df["odds_movement_over"]        = 0.0  # no movement with single snapshot

    log.info(
        "Market features extracted: %d matches, PIN coverage: %d/%d, B365 coverage: %d/%d.",
        len(market_df),
        market_df["pin_implied_prob_over"].notna().sum(),
        len(market_df),
        market_df["b365_implied_prob_over"].notna().sum(),
        len(market_df),
    )

    return market_df.set_index("match_id")
