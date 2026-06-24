"""
Historical betting backtest utilities.

The backtest layer evaluates settled betting opportunities after a model has
already produced probabilities. It does not train models and it does not decide
which features are valid. Its job is to convert historical predictions into
bets, profit and risk metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.betting.edge import EdgePolicy, add_betting_edges


@dataclass(frozen=True)
class BacktestSummary:
    """Summary metrics for a settled betting strategy."""

    rows: int
    bets: int
    wins: int
    losses: int
    pushes: int
    stake: float
    profit: float
    roi: float
    hit_rate: float
    avg_odds: float
    avg_edge: float
    max_drawdown: float

    def as_dict(self) -> dict[str, float | int]:
        """Return the summary as a plain dictionary."""
        return {
            "rows": self.rows,
            "bets": self.bets,
            "wins": self.wins,
            "losses": self.losses,
            "pushes": self.pushes,
            "stake": round(self.stake, 4),
            "profit": round(self.profit, 4),
            "roi": round(self.roi, 4),
            "hit_rate": round(self.hit_rate, 4),
            "avg_odds": round(self.avg_odds, 4),
            "avg_edge": round(self.avg_edge, 4),
            "max_drawdown": round(self.max_drawdown, 4),
        }


def run_backtest(
    df: pd.DataFrame,
    model_prob_col: str,
    odds_col: str,
    target_col: str,
    policy: EdgePolicy | None = None,
    stake_strategy: str = "flat",
    flat_stake_amount: float = 1.0,
    bankroll: float = 100.0,
    kelly_multiplier: float = 0.25,
    max_bankroll_fraction: float = 0.05,
    sort_col: str | None = None,
) -> pd.DataFrame:
    """
    Run a historical backtest from model probabilities and settled outcomes.

    Parameters
    ----------
    df : DataFrame containing one row per historical candidate bet.
    model_prob_col : Model probability column for the positive event.
    odds_col : Decimal odds column for the same event.
    target_col : Settled binary outcome column. For Over 2.5, this is `over_25`.
    policy : EdgePolicy for selecting bets.
    stake_strategy : `flat` or `kelly`.
    flat_stake_amount : Constant stake for flat staking.
    bankroll : Reference bankroll for Kelly staking.
    kelly_multiplier : Fraction of full Kelly to stake.
    max_bankroll_fraction : Maximum Kelly stake cap.
    sort_col : Optional chronological column for ordering the profit curve.

    Returns
    -------
    DataFrame with edge, EV, bet flag, stake, profit, cumulative profit and
    drawdown columns.
    """
    if sort_col is not None and sort_col not in df.columns:
        raise KeyError(f"Missing sort column: {sort_col}")

    ordered = df.copy()
    if sort_col is not None:
        ordered = ordered.sort_values(sort_col).reset_index(drop=True)

    out = add_betting_edges(
        ordered,
        model_prob_col=model_prob_col,
        odds_col=odds_col,
        target_col=target_col,
        policy=policy,
        stake_strategy=stake_strategy,
        flat_stake_amount=flat_stake_amount,
        bankroll=bankroll,
        kelly_multiplier=kelly_multiplier,
        max_bankroll_fraction=max_bankroll_fraction,
    )

    out["cumulative_profit"] = out["profit"].cumsum()
    out["equity_peak"] = out["cumulative_profit"].cummax().clip(lower=0.0)
    out["drawdown"] = out["equity_peak"] - out["cumulative_profit"]
    return out


def max_drawdown(profits: pd.Series | np.ndarray | list[float]) -> float:
    """Return maximum drawdown from a sequence of realised profits."""
    profit_series = pd.Series(profits, dtype="float64")
    if profit_series.empty:
        return 0.0
    cumulative = profit_series.cumsum()
    peak = cumulative.cummax().clip(lower=0.0)
    drawdown = peak - cumulative
    return float(drawdown.max())


def summarize_backtest(backtest_df: pd.DataFrame) -> BacktestSummary:
    """Compute headline performance metrics for a backtest DataFrame."""
    required = {"bet_flag", "stake", "profit"}
    missing = required - set(backtest_df.columns)
    if missing:
        raise KeyError(f"Missing required backtest columns: {sorted(missing)}")

    rows = int(len(backtest_df))
    bets_df = backtest_df[backtest_df["bet_flag"]].copy()
    bets = int(len(bets_df))

    if bets == 0:
        return BacktestSummary(
            rows=rows,
            bets=0,
            wins=0,
            losses=0,
            pushes=0,
            stake=0.0,
            profit=0.0,
            roi=0.0,
            hit_rate=0.0,
            avg_odds=0.0,
            avg_edge=0.0,
            max_drawdown=0.0,
        )

    stake = float(bets_df["stake"].sum())
    profit = float(bets_df["profit"].sum())
    wins = int((bets_df["profit"] > 0).sum())
    losses = int((bets_df["profit"] < 0).sum())
    pushes = int((bets_df["profit"] == 0).sum())
    roi = profit / stake if stake > 0 else 0.0
    hit_rate = wins / bets if bets > 0 else 0.0

    avg_odds = float(bets_df["decimal_odds"].mean()) if "decimal_odds" in bets_df.columns else 0.0
    if avg_odds == 0.0:
        odds_like = [c for c in bets_df.columns if "odds" in c and pd.api.types.is_numeric_dtype(bets_df[c])]
        avg_odds = float(bets_df[odds_like[0]].mean()) if odds_like else 0.0

    avg_edge = float(bets_df["edge"].mean()) if "edge" in bets_df.columns else 0.0
    dd = max_drawdown(bets_df["profit"])

    return BacktestSummary(
        rows=rows,
        bets=bets,
        wins=wins,
        losses=losses,
        pushes=pushes,
        stake=stake,
        profit=profit,
        roi=roi,
        hit_rate=hit_rate,
        avg_odds=avg_odds,
        avg_edge=avg_edge,
        max_drawdown=dd,
    )


def summarize_by_group(backtest_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Summarise a backtest by season, team, odds band or any other group."""
    if group_col not in backtest_df.columns:
        raise KeyError(f"Missing group column: {group_col}")

    rows: list[dict] = []
    for group_value, group_df in backtest_df.groupby(group_col, dropna=False):
        summary = summarize_backtest(group_df).as_dict()
        summary[group_col] = group_value
        rows.append(summary)

    if not rows:
        return pd.DataFrame()

    ordered_cols = [group_col] + [c for c in rows[0].keys() if c != group_col]
    return pd.DataFrame(rows)[ordered_cols]


def profit_curve(backtest_df: pd.DataFrame) -> pd.DataFrame:
    """Return only bet rows with cumulative stake, profit and drawdown columns."""
    required = {"bet_flag", "stake", "profit"}
    missing = required - set(backtest_df.columns)
    if missing:
        raise KeyError(f"Missing required backtest columns: {sorted(missing)}")

    bets = backtest_df[backtest_df["bet_flag"]].copy().reset_index(drop=True)
    if bets.empty:
        return pd.DataFrame(
            columns=["bet_number", "stake", "profit", "cumulative_profit", "drawdown"]
        )

    bets["bet_number"] = np.arange(1, len(bets) + 1)
    bets["cumulative_stake"] = bets["stake"].cumsum()
    bets["cumulative_profit"] = bets["profit"].cumsum()
    bets["equity_peak"] = bets["cumulative_profit"].cummax().clip(lower=0.0)
    bets["drawdown"] = bets["equity_peak"] - bets["cumulative_profit"]
    return bets
