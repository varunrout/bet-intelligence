"""
Descriptive analytics for Football Market Intelligence System.

All functions return DataFrames ready for plotting or reporting.
All database queries use pre-match information only (no future leakage possible
for descriptive stats computed on completed historical matches).

Sections:
  1. Market overview — season-level base rates
  2. Bookmaker margin analysis
  3. Implied probability calibration
  4. Team tendency analysis
  5. Odds distribution
  6. Market consensus / bookmaker agreement
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import DB_PATH
from src.utils.db import run_query


# ---------------------------------------------------------------------------
# 1. Market overview
# ---------------------------------------------------------------------------


def season_summary(db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Over-2.5 rate and goal statistics per season.

    Returns
    -------
    DataFrame with columns:
        season, matches, over25_n, over25_rate, avg_goals,
        avg_goals_home, avg_goals_away
    """
    return run_query(
        """
        SELECT
            season,
            COUNT(*)                                  AS matches,
            SUM(over_25::INTEGER)                     AS over25_n,
            ROUND(AVG(over_25::INTEGER), 4)           AS over25_rate,
            ROUND(AVG(total_goals), 3)                AS avg_goals,
            ROUND(AVG(goals_home), 3)                 AS avg_goals_home,
            ROUND(AVG(goals_away), 3)                 AS avg_goals_away
        FROM matches
        GROUP BY season
        ORDER BY season
        """,
        db_path=db_path,
    )


def monthly_over25_rate(db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Over-2.5 rate by calendar month across all seasons.
    Useful for detecting seasonality (early-season vs late-season patterns).
    """
    return run_query(
        """
        SELECT
            EXTRACT(MONTH FROM kickoff_utc)::INTEGER  AS month,
            COUNT(*)                                   AS matches,
            ROUND(AVG(over_25::INTEGER), 4)            AS over25_rate,
            ROUND(AVG(total_goals), 3)                 AS avg_goals
        FROM matches
        GROUP BY month
        ORDER BY month
        """,
        db_path=db_path,
    )


def goals_distribution(db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Distribution of total goals per match across all seasons.
    """
    return run_query(
        """
        SELECT
            total_goals,
            COUNT(*) AS n,
            ROUND(COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (), 4) AS pct
        FROM matches
        GROUP BY total_goals
        ORDER BY total_goals
        """,
        db_path=db_path,
    )


def result_distribution(db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Home win / Draw / Away win distribution per season.
    """
    return run_query(
        """
        SELECT
            season,
            result_ftr,
            COUNT(*) AS n,
            ROUND(COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (PARTITION BY season), 4) AS pct
        FROM matches
        WHERE result_ftr IS NOT NULL
        GROUP BY season, result_ftr
        ORDER BY season, result_ftr
        """,
        db_path=db_path,
    )


# ---------------------------------------------------------------------------
# 2. Bookmaker margin analysis
# ---------------------------------------------------------------------------


def bookmaker_margin_by_season(db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Average bookmaker margin per bookmaker per season.

    Returns
    -------
    DataFrame: season, bookmaker, n, avg_margin_pct, min_margin, max_margin
    """
    return run_query(
        """
        SELECT
            m.season,
            bk.name                             AS bookmaker,
            bk.is_sharp,
            COUNT(*)                            AS n,
            ROUND(AVG(o.margin) * 100, 3)       AS avg_margin_pct,
            ROUND(MIN(o.margin) * 100, 3)       AS min_margin_pct,
            ROUND(MAX(o.margin) * 100, 3)       AS max_margin_pct,
            ROUND(STDDEV(o.margin) * 100, 3)    AS std_margin_pct
        FROM odds_snapshots o
        JOIN matches    m  ON o.match_id     = m.match_id
        JOIN bookmakers bk ON o.bookmaker_id = bk.bookmaker_id
        WHERE o.margin IS NOT NULL
        GROUP BY m.season, bk.name, bk.is_sharp
        ORDER BY m.season, avg_margin_pct
        """,
        db_path=db_path,
    )


def implied_prob_summary(db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Distribution of implied over-2.5 probability by bookmaker.
    Shows how each book prices the market on average.
    """
    return run_query(
        """
        SELECT
            bk.name                                     AS bookmaker,
            COUNT(*)                                    AS n,
            ROUND(AVG(o.implied_prob_over), 4)          AS mean_prob_over,
            ROUND(STDDEV(o.implied_prob_over), 4)       AS std_prob_over,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP
                  (ORDER BY o.implied_prob_over), 4)    AS p25,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP
                  (ORDER BY o.implied_prob_over), 4)    AS median_prob_over,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP
                  (ORDER BY o.implied_prob_over), 4)    AS p75
        FROM odds_snapshots o
        JOIN bookmakers bk ON o.bookmaker_id = bk.bookmaker_id
        WHERE o.implied_prob_over IS NOT NULL
        GROUP BY bk.name
        ORDER BY mean_prob_over
        """,
        db_path=db_path,
    )


def sharp_vs_recreational_margin(db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Head-to-head margin comparison: Pinnacle vs Bet365 per match.
    Shows how the margin spread varies across the odds range.
    """
    return run_query(
        """
        WITH pin AS (
            SELECT o.match_id, o.margin AS pin_margin, o.implied_prob_over AS pin_prob_over
            FROM odds_snapshots o
            JOIN bookmakers bk ON o.bookmaker_id = bk.bookmaker_id
            WHERE bk.short_code = 'PIN'
        ),
        b365 AS (
            SELECT o.match_id, o.margin AS b365_margin, o.implied_prob_over AS b365_prob_over
            FROM odds_snapshots o
            JOIN bookmakers bk ON o.bookmaker_id = bk.bookmaker_id
            WHERE bk.short_code = 'B365'
        )
        SELECT
            p.match_id,
            p.pin_margin,
            b.b365_margin,
            ROUND(b.b365_margin - p.pin_margin, 5)          AS margin_premium,
            p.pin_prob_over,
            b.b365_prob_over,
            ROUND(p.pin_prob_over - b.b365_prob_over, 5)    AS prob_disagreement
        FROM pin p
        JOIN b365 b ON p.match_id = b.match_id
        """,
        db_path=db_path,
    )


# ---------------------------------------------------------------------------
# 3. Implied probability calibration
# ---------------------------------------------------------------------------


def calibration_data(
    bookmaker_short_code: str = "PIN",
    n_bins: int = 10,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """
    Compute calibration bins: compare implied probability vs actual over-2.5 rate.

    Each row is one probability bucket. A perfectly calibrated book gives
    mean_implied_prob ≈ actual_over25_rate in every bin.

    Parameters
    ----------
    bookmaker_short_code : 'PIN' (Pinnacle), 'B365' (Bet365), 'AVG', 'MAX'
    n_bins               : number of equal-width probability bins

    Returns
    -------
    DataFrame: bin_lower, bin_upper, n, mean_implied_prob, actual_over25_rate, calibration_error
    """
    raw = run_query(
        f"""
        SELECT
            o.implied_prob_over,
            m.over_25::INTEGER AS outcome
        FROM odds_snapshots o
        JOIN matches    m  ON o.match_id     = m.match_id
        JOIN bookmakers bk ON o.bookmaker_id = bk.bookmaker_id
        WHERE bk.short_code = '{bookmaker_short_code}'
          AND o.implied_prob_over IS NOT NULL
          AND m.over_25          IS NOT NULL
        """,
        db_path=db_path,
    )

    if raw.empty:
        return pd.DataFrame()

    raw["bin"] = pd.cut(
        raw["implied_prob_over"],
        bins=n_bins,
        labels=False,
        include_lowest=True,
    )
    binned = raw.groupby("bin").agg(
        n=("outcome", "count"),
        mean_implied_prob=("implied_prob_over", "mean"),
        actual_over25_rate=("outcome", "mean"),
    ).reset_index(drop=True)

    binned["calibration_error"] = (
        binned["mean_implied_prob"] - binned["actual_over25_rate"]
    )

    # Bin edges
    bin_edges = pd.cut(
        raw["implied_prob_over"], bins=n_bins, include_lowest=True
    ).cat.categories
    binned["bin_lower"] = [b.left  for b in bin_edges]
    binned["bin_upper"] = [b.right for b in bin_edges]

    cols = ["bin_lower", "bin_upper", "n", "mean_implied_prob",
            "actual_over25_rate", "calibration_error"]
    return binned[cols].round(4)


def overall_calibration_metrics(
    bookmaker_short_code: str = "PIN",
    db_path: Path = DB_PATH,
) -> dict:
    """
    Compute overall calibration summary for one bookmaker.

    Returns dict with:
        mean_implied_prob   - average implied prob across all matches
        actual_over25_rate  - actual frequency of over-2.5
        avg_overround_bias  - how much the book overestimates probability on average
        brier_score         - probabilistic accuracy metric (lower = better)
        log_loss            - log loss of implied probs vs outcomes
    """
    from scipy.stats import entropy

    raw = run_query(
        f"""
        SELECT o.implied_prob_over, m.over_25::INTEGER AS outcome
        FROM odds_snapshots o
        JOIN matches    m  ON o.match_id     = m.match_id
        JOIN bookmakers bk ON o.bookmaker_id = bk.bookmaker_id
        WHERE bk.short_code = '{bookmaker_short_code}'
          AND o.implied_prob_over IS NOT NULL
          AND m.over_25 IS NOT NULL
        """,
        db_path=db_path,
    )

    if raw.empty:
        return {}

    p = raw["implied_prob_over"].values
    y = raw["outcome"].values

    brier = np.mean((p - y) ** 2)
    eps = 1e-9
    ll = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

    return {
        "bookmaker":          bookmaker_short_code,
        "n_matches":          len(raw),
        "mean_implied_prob":  round(float(p.mean()), 4),
        "actual_over25_rate": round(float(y.mean()), 4),
        "avg_overround_bias": round(float((p - y).mean()), 4),
        "brier_score":        round(float(brier), 4),
        "log_loss":           round(float(ll), 4),
    }


# ---------------------------------------------------------------------------
# 4. Team tendency analysis
# ---------------------------------------------------------------------------


def team_over25_rates(db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Over-2.5 rate for each team as home team and away team.

    Returns DataFrame sorted by combined over-2.5 rate descending.
    """
    return run_query(
        """
        WITH home AS (
            SELECT
                ht.name_canonical                        AS team,
                COUNT(*)                                 AS home_matches,
                ROUND(AVG(m.over_25::INTEGER), 4)        AS home_over25_rate,
                ROUND(AVG(m.total_goals), 3)             AS home_avg_goals
            FROM matches m
            JOIN teams ht ON m.home_team_id = ht.team_id
            GROUP BY ht.name_canonical
        ),
        away AS (
            SELECT
                awt.name_canonical                       AS team,
                COUNT(*)                                 AS away_matches,
                ROUND(AVG(m.over_25::INTEGER), 4)        AS away_over25_rate,
                ROUND(AVG(m.total_goals), 3)             AS away_avg_goals
            FROM matches m
            JOIN teams awt ON m.away_team_id = awt.team_id
            GROUP BY awt.name_canonical
        )
        SELECT
            h.team,
            h.home_matches,
            h.home_over25_rate,
            h.home_avg_goals,
            a.away_matches,
            a.away_over25_rate,
            a.away_avg_goals,
            ROUND((h.home_over25_rate + a.away_over25_rate) / 2, 4) AS combined_over25_rate
        FROM home h
        JOIN away  a ON h.team = a.team
        ORDER BY combined_over25_rate DESC
        """,
        db_path=db_path,
    )


def team_season_over25(db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Over-2.5 rate per team per season (home + away combined).
    Useful for tracking whether a team's style changes across seasons.
    """
    return run_query(
        """
        WITH home_rows AS (
            SELECT m.season, ht.name_canonical AS team, m.over_25, m.total_goals
            FROM matches m JOIN teams ht ON m.home_team_id = ht.team_id
        ),
        away_rows AS (
            SELECT m.season, awt.name_canonical AS team, m.over_25, m.total_goals
            FROM matches m JOIN teams awt ON m.away_team_id = awt.team_id
        ),
        combined AS (
            SELECT * FROM home_rows
            UNION ALL
            SELECT * FROM away_rows
        )
        SELECT
            season, team,
            COUNT(*)                            AS matches,
            ROUND(AVG(over_25::INTEGER), 4)     AS over25_rate,
            ROUND(AVG(total_goals), 3)          AS avg_goals
        FROM combined
        GROUP BY season, team
        ORDER BY season, over25_rate DESC
        """,
        db_path=db_path,
    )


def top_bottom_teams(
    n: int = 8,
    db_path: Path = DB_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return top-n and bottom-n teams by combined over-2.5 rate.

    Returns
    -------
    (top_n_df, bottom_n_df)
    """
    rates = team_over25_rates(db_path=db_path)
    return rates.head(n), rates.tail(n)


# ---------------------------------------------------------------------------
# 5. Odds distribution
# ---------------------------------------------------------------------------


def odds_distribution_by_bookmaker(db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Percentile distribution of over-2.5 decimal odds by bookmaker.
    """
    return run_query(
        """
        SELECT
            bk.name                                    AS bookmaker,
            ROUND(MIN(o.odds_over), 3)                 AS min_odds,
            ROUND(PERCENTILE_CONT(0.10) WITHIN GROUP
                  (ORDER BY o.odds_over), 3)           AS p10,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP
                  (ORDER BY o.odds_over), 3)           AS p25,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP
                  (ORDER BY o.odds_over), 3)           AS median,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP
                  (ORDER BY o.odds_over), 3)           AS p75,
            ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP
                  (ORDER BY o.odds_over), 3)           AS p90,
            ROUND(MAX(o.odds_over), 3)                 AS max_odds
        FROM odds_snapshots o
        JOIN bookmakers bk ON o.bookmaker_id = bk.bookmaker_id
        WHERE o.odds_over IS NOT NULL
        GROUP BY bk.name
        ORDER BY median
        """,
        db_path=db_path,
    )


def odds_by_season(db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Average Pinnacle over-2.5 odds and implied probability by season.
    Shows how market pricing has shifted over time.
    """
    return run_query(
        """
        SELECT
            m.season,
            ROUND(AVG(o.odds_over),          3) AS avg_odds_over,
            ROUND(AVG(o.odds_under),         3) AS avg_odds_under,
            ROUND(AVG(o.implied_prob_over),  4) AS avg_implied_prob_over,
            ROUND(AVG(o.margin) * 100,       3) AS avg_margin_pct,
            COUNT(*)                            AS n
        FROM odds_snapshots o
        JOIN matches    m  ON o.match_id     = m.match_id
        JOIN bookmakers bk ON o.bookmaker_id = bk.bookmaker_id
        WHERE bk.short_code = 'PIN'
          AND o.odds_over IS NOT NULL
        GROUP BY m.season
        ORDER BY m.season
        """,
        db_path=db_path,
    )


# ---------------------------------------------------------------------------
# 6. Market consensus / bookmaker agreement
# ---------------------------------------------------------------------------


def bookmaker_prob_divergence(db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    For each match, compute the divergence between Pinnacle and Bet365
    implied over-2.5 probabilities.

    High divergence matches are candidates for diagnostic investigation.
    """
    return run_query(
        """
        WITH pin AS (
            SELECT o.match_id, o.implied_prob_over AS pin_prob
            FROM odds_snapshots o
            JOIN bookmakers bk ON o.bookmaker_id = bk.bookmaker_id
            WHERE bk.short_code = 'PIN'
        ),
        b365 AS (
            SELECT o.match_id, o.implied_prob_over AS b365_prob
            FROM odds_snapshots o
            JOIN bookmakers bk ON o.bookmaker_id = bk.bookmaker_id
            WHERE bk.short_code = 'B365'
        )
        SELECT
            m.season,
            p.match_id,
            ht.name_canonical                                   AS home_team,
            awt.name_canonical                                  AS away_team,
            ROUND(p.pin_prob,  4)                               AS pinnacle_prob,
            ROUND(b.b365_prob, 4)                               AS bet365_prob,
            ROUND(ABS(p.pin_prob - b.b365_prob), 4)             AS abs_divergence,
            m.over_25,
            m.total_goals
        FROM pin p
        JOIN b365   b   ON p.match_id     = b.match_id
        JOIN matches m  ON p.match_id     = m.match_id
        JOIN teams ht   ON m.home_team_id = ht.team_id
        JOIN teams awt  ON m.away_team_id = awt.team_id
        ORDER BY abs_divergence DESC
        """,
        db_path=db_path,
    )


def high_divergence_matches(
    min_divergence: float = 0.05,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """
    Return matches where Pinnacle and Bet365 diverge by >= min_divergence
    in implied over-2.5 probability.

    These are the most interesting cases for diagnostic analysis.
    """
    df = bookmaker_prob_divergence(db_path=db_path)
    return df[df["abs_divergence"] >= min_divergence].copy()


# ---------------------------------------------------------------------------
# 7. Market efficiency quick check
# ---------------------------------------------------------------------------


def pinnacle_efficiency_check(db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Group matches into deciles by Pinnacle's implied over-2.5 probability
    and compute the actual over-2.5 rate in each decile.

    A perfectly efficient market gives a straight diagonal:
        implied_prob ≈ actual_rate in every decile.

    Returns
    -------
    DataFrame: decile, n, mean_implied_prob, actual_over25_rate,
               edge (actual - implied), cumulative_roi
    """
    raw = run_query(
        """
        SELECT o.implied_prob_over, m.over_25::INTEGER AS outcome,
               o.odds_over
        FROM odds_snapshots o
        JOIN matches    m  ON o.match_id     = m.match_id
        JOIN bookmakers bk ON o.bookmaker_id = bk.bookmaker_id
        WHERE bk.short_code = 'PIN'
          AND o.implied_prob_over IS NOT NULL
          AND m.over_25 IS NOT NULL
        ORDER BY o.implied_prob_over
        """,
        db_path=db_path,
    )

    if raw.empty:
        return pd.DataFrame()

    raw["decile"] = pd.qcut(raw["implied_prob_over"], q=10, labels=False, duplicates="drop")

    grouped = raw.groupby("decile").agg(
        n=("outcome", "count"),
        mean_implied_prob=("implied_prob_over", "mean"),
        actual_over25_rate=("outcome", "mean"),
        avg_odds_over=("odds_over", "mean"),
    ).reset_index()

    grouped["edge"] = grouped["actual_over25_rate"] - grouped["mean_implied_prob"]
    grouped["flat_roi"] = grouped["actual_over25_rate"] * grouped["avg_odds_over"] - 1

    return grouped.round(4)
