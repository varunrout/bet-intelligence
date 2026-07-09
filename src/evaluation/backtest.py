"""
Walk-forward backtesting engine for Football Market Intelligence System — Phase 5.

Consumes the pooled out-of-fold predictions produced by
src.modeling.walk_forward.predictions_dataframe() joined with market odds,
and answers the only question that matters in a market with a real price:
would acting on this model have made or lost money, and at what risk?

The engine is deliberately framed as uncertainty-aware decision support,
not a promise of guaranteed edge:

  Selection  — a bet is placed only when the model's fair price beats the
               market price by at least `edge_threshold`
               (edge = p_model × decimal_odds − 1).
  Staking    — flat or fractional-Kelly, delegated to
               src.prescriptive.staking (stake is sized on the CURRENT
               bankroll, so losses compound realistically).
  Accounting — per-bet PnL, bankroll trajectory, realised return on
               turnover, and maximum drawdown from a running peak.
  CLV        — when closing odds are supplied, closing-line value measures
               whether bets beat the market's final (sharpest) price —
               the strongest known leading indicator of real skill.

Design rules
------------
- Bets are processed strictly in chronological order; the bankroll at bet i
  depends only on bets 0..i-1.  No information travels backwards.
- Predictions must be OUT-OF-FOLD (walk-forward), never in-sample.
- The engine never shorts: negative-edge bets are simply not selected.

Public API
----------
  select_bets(preds, edge_threshold, min_odds, max_odds) -> pd.DataFrame
  run_backtest(bets, staking, initial_bankroll, **staking_kwargs)
                                                         -> dict
  backtest_summary(result)                               -> pd.DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.prescriptive.staking import flat_stake, kelly_stake


# ── Bet selection ──────────────────────────────────────────────────────────────


def select_bets(
    preds: pd.DataFrame,
    edge_threshold: float = 0.02,
    min_odds: float = 1.10,
    max_odds: float = 10.0,
) -> pd.DataFrame:
    """
    Apply the selection rule to out-of-fold predictions.

    Parameters
    ----------
    preds          : DataFrame with columns
                       y_prob        — model probability (out-of-fold)
                       y_true        — realised outcome {0, 1}
                       decimal_odds  — market price taken for the bet
                     and optionally
                       closing_odds  — market closing price for CLV
                       test_season   — season identifier
                     Rows must be in chronological order.
    edge_threshold : minimum model edge required to bet, where
                     edge = y_prob × decimal_odds − 1  (default 2%).
    min_odds       : ignore prices below this (near-certainties carry
                     outsized margin and stake little information).
    max_odds       : ignore prices above this (longshot extremes, where
                     both model and market are least reliable).

    Returns
    -------
    Copy of the selected rows with an added 'edge' column, preserving order.
    """
    required = {"y_prob", "y_true", "decimal_odds"}
    missing = required - set(preds.columns)
    if missing:
        raise ValueError(f"select_bets: preds is missing columns {sorted(missing)}")

    out = preds.copy()
    out["edge"] = out["y_prob"] * out["decimal_odds"] - 1.0

    mask = (
        (out["edge"] >= edge_threshold)
        & (out["decimal_odds"] >= min_odds)
        & (out["decimal_odds"] <= max_odds)
    )
    return out.loc[mask].copy()


# ── Backtest loop ──────────────────────────────────────────────────────────────


def run_backtest(
    bets: pd.DataFrame,
    staking: str = "flat",
    initial_bankroll: float = 1000.0,
    **staking_kwargs,
) -> dict:
    """
    Run the chronological backtest over selected bets.

    Parameters
    ----------
    bets             : output of select_bets(), in chronological order.
    staking          : 'flat' or 'kelly'.
    initial_bankroll : starting bankroll (default 1000 units).
    staking_kwargs   : forwarded to the staking rule —
                       flat  : fraction (default 0.01)
                       kelly : fraction (default 0.25), cap (default 0.05)

    Returns
    -------
    dict with keys:
        ledger   : pd.DataFrame — one row per bet with stake, pnl,
                   bankroll_after, drawdown (and clv when closing odds exist)
        summary  : dict — n_bets, hit_rate, total_staked, total_pnl,
                   roi_on_turnover, final_bankroll, bankroll_return,
                   max_drawdown, avg_edge, avg_odds
                   (+ clv_mean, clv_positive_rate when closing odds exist)

    Notes
    -----
    max_drawdown is the largest peak-to-trough fall of the bankroll,
    expressed as a fraction of the peak (0.20 = a 20% fall).
    """
    if staking not in ("flat", "kelly"):
        raise ValueError(f"run_backtest: unknown staking rule '{staking}'")

    bankroll = float(initial_bankroll)
    peak     = bankroll
    rows: list[dict] = []

    has_clv = "closing_odds" in bets.columns

    for _, bet in bets.iterrows():
        p, odds, won = float(bet["y_prob"]), float(bet["decimal_odds"]), bool(bet["y_true"])

        if staking == "flat":
            stake = flat_stake(bankroll, **staking_kwargs)
        else:
            stake = kelly_stake(bankroll, p, odds, **staking_kwargs)

        pnl = stake * (odds - 1.0) if won else -stake
        bankroll += pnl
        peak = max(peak, bankroll)
        drawdown = (peak - bankroll) / peak if peak > 0 else 0.0

        row = {
            "y_prob":         p,
            "decimal_odds":   odds,
            "won":            won,
            "edge":           float(bet["edge"]),
            "stake":          round(stake, 4),
            "pnl":            round(pnl, 4),
            "bankroll_after": round(bankroll, 4),
            "drawdown":       round(drawdown, 4),
        }
        if "test_season" in bet.index:
            row["test_season"] = bet["test_season"]
        if has_clv:
            # CLV: value captured relative to the closing (sharpest) price.
            closing = float(bet["closing_odds"])
            row["clv"] = round(odds / closing - 1.0, 4) if closing > 0 else np.nan
        rows.append(row)

    ledger = pd.DataFrame(rows)

    if ledger.empty:
        summary = {
            "n_bets": 0, "hit_rate": np.nan, "total_staked": 0.0,
            "total_pnl": 0.0, "roi_on_turnover": np.nan,
            "final_bankroll": round(bankroll, 4), "bankroll_return": 0.0,
            "max_drawdown": 0.0, "avg_edge": np.nan, "avg_odds": np.nan,
        }
        return {"ledger": ledger, "summary": summary}

    total_staked = float(ledger["stake"].sum())
    total_pnl    = float(ledger["pnl"].sum())

    summary = {
        "n_bets":          int(len(ledger)),
        "hit_rate":        round(float(ledger["won"].mean()), 4),
        "total_staked":    round(total_staked, 4),
        "total_pnl":       round(total_pnl, 4),
        "roi_on_turnover": round(total_pnl / total_staked, 4) if total_staked > 0 else np.nan,
        "final_bankroll":  round(bankroll, 4),
        "bankroll_return": round(bankroll / initial_bankroll - 1.0, 4),
        "max_drawdown":    round(float(ledger["drawdown"].max()), 4),
        "avg_edge":        round(float(ledger["edge"].mean()), 4),
        "avg_odds":        round(float(ledger["decimal_odds"].mean()), 4),
    }
    if has_clv and "clv" in ledger.columns:
        clv = ledger["clv"].dropna()
        if len(clv):
            summary["clv_mean"]          = round(float(clv.mean()), 4)
            summary["clv_positive_rate"] = round(float((clv > 0).mean()), 4)

    return {"ledger": ledger, "summary": summary}


# ── Reporting ──────────────────────────────────────────────────────────────────


def backtest_summary(result: dict) -> pd.DataFrame:
    """
    Render a run_backtest() result as a one-row DataFrame for display,
    or per-season rows when the ledger carries a 'test_season' column.
    """
    ledger, summary = result["ledger"], result["summary"]

    if ledger.empty or "test_season" not in ledger.columns:
        return pd.DataFrame([summary])

    rows = []
    for season, grp in ledger.groupby("test_season", sort=True):
        staked = float(grp["stake"].sum())
        rows.append(
            {
                "test_season":     season,
                "n_bets":          int(len(grp)),
                "hit_rate":        round(float(grp["won"].mean()), 4),
                "total_staked":    round(staked, 4),
                "total_pnl":       round(float(grp["pnl"].sum()), 4),
                "roi_on_turnover": round(float(grp["pnl"].sum()) / staked, 4) if staked > 0 else np.nan,
            }
        )
    rows.append({"test_season": "ALL", **{k: summary[k] for k in
                 ("n_bets", "hit_rate", "total_staked", "total_pnl", "roi_on_turnover")}})
    return pd.DataFrame(rows)
