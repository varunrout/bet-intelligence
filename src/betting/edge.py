"""
Betting edge calculation utilities.

This module converts model probabilities and bookmaker odds into value-betting
signals. It is intentionally model-agnostic: any model can produce the input
probability, and this layer decides whether that probability implies a bet.

Core definitions
----------------
market_prob = 1 / decimal_odds
edge       = model_prob - market_prob
ev         = model_prob * decimal_odds - 1

A positive EV does not guarantee profit. It only means the model probability is
higher than the break-even probability implied by the available price.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EdgePolicy:
    """Configuration for turning probability edges into bet flags."""

    min_edge: float = 0.02
    min_ev: float = 0.0
    min_odds: float = 1.01
    max_odds: float | None = None


def _as_float(value: float | int | np.number, name: str) -> float:
    """Convert a numeric value to float and fail clearly for invalid inputs."""
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric; got {value!r}.") from exc

    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite; got {value!r}.")
    return out


def validate_probability(probability: float, name: str = "probability") -> float:
    """Return a valid probability in [0, 1] or raise ValueError."""
    p = _as_float(probability, name)
    if p < 0.0 or p > 1.0:
        raise ValueError(f"{name} must be between 0 and 1; got {p}.")
    return p


def validate_decimal_odds(decimal_odds: float, name: str = "decimal_odds") -> float:
    """Return valid decimal odds or raise ValueError."""
    odds = _as_float(decimal_odds, name)
    if odds <= 1.0:
        raise ValueError(f"{name} must be greater than 1.0; got {odds}.")
    return odds


def implied_probability(decimal_odds: float) -> float:
    """Convert decimal odds to break-even implied probability."""
    odds = validate_decimal_odds(decimal_odds)
    return 1.0 / odds


def probability_edge(model_probability: float, decimal_odds: float) -> float:
    """Return model probability minus break-even market probability."""
    p = validate_probability(model_probability, "model_probability")
    return p - implied_probability(decimal_odds)


def expected_value(model_probability: float, decimal_odds: float) -> float:
    """
    Return expected profit per 1 unit staked.

    Example: p=0.55 and odds=2.10 gives EV = 0.55 * 2.10 - 1 = 0.155.
    This means expected profit is 0.155 units per 1 unit staked, before any
    practical constraints such as limits, liquidity or execution quality.
    """
    p = validate_probability(model_probability, "model_probability")
    odds = validate_decimal_odds(decimal_odds)
    return p * odds - 1.0


def should_bet(
    model_probability: float,
    decimal_odds: float,
    policy: EdgePolicy | None = None,
) -> bool:
    """Return True when the probability edge and EV pass the supplied policy."""
    policy = policy or EdgePolicy()
    odds = validate_decimal_odds(decimal_odds)
    edge = probability_edge(model_probability, odds)
    ev = expected_value(model_probability, odds)

    if odds < policy.min_odds:
        return False
    if policy.max_odds is not None and odds > policy.max_odds:
        return False
    return edge >= policy.min_edge and ev >= policy.min_ev


def flat_stake(amount: float = 1.0) -> float:
    """Return a constant stake size."""
    stake = _as_float(amount, "amount")
    if stake < 0.0:
        raise ValueError(f"amount must be non-negative; got {stake}.")
    return stake


def kelly_fraction(model_probability: float, decimal_odds: float) -> float:
    """
    Return the full Kelly fraction for a single binary bet.

    Formula:
        f* = (b * p - q) / b

    where b = decimal_odds - 1, p = model probability, and q = 1 - p.
    Negative fractions are clipped to 0 because a no-edge bet should not be
    staked by the long-only betting strategy used in this project.
    """
    p = validate_probability(model_probability, "model_probability")
    odds = validate_decimal_odds(decimal_odds)
    b = odds - 1.0
    q = 1.0 - p
    raw_fraction = (b * p - q) / b
    return max(0.0, raw_fraction)


def fractional_kelly_stake(
    bankroll: float,
    model_probability: float,
    decimal_odds: float,
    fraction: float = 0.25,
    max_bankroll_fraction: float = 0.05,
) -> float:
    """
    Return a capped fractional Kelly stake.

    The cap prevents one extreme probability estimate from creating an
    unrealistic stake. This function does not update bankroll. It only computes
    the stake for the current opportunity.
    """
    bankroll = _as_float(bankroll, "bankroll")
    fraction = _as_float(fraction, "fraction")
    max_bankroll_fraction = _as_float(max_bankroll_fraction, "max_bankroll_fraction")

    if bankroll < 0.0:
        raise ValueError(f"bankroll must be non-negative; got {bankroll}.")
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError(f"fraction must be between 0 and 1; got {fraction}.")
    if max_bankroll_fraction < 0.0 or max_bankroll_fraction > 1.0:
        raise ValueError(
            "max_bankroll_fraction must be between 0 and 1; "
            f"got {max_bankroll_fraction}."
        )

    uncapped = bankroll * kelly_fraction(model_probability, decimal_odds) * fraction
    cap = bankroll * max_bankroll_fraction
    return min(uncapped, cap)


def add_betting_edges(
    df: pd.DataFrame,
    model_prob_col: str,
    odds_col: str,
    target_col: str | None = None,
    policy: EdgePolicy | None = None,
    stake_strategy: str = "flat",
    flat_stake_amount: float = 1.0,
    bankroll: float = 100.0,
    kelly_multiplier: float = 0.25,
    max_bankroll_fraction: float = 0.05,
) -> pd.DataFrame:
    """
    Add edge, EV, bet flag, stake and optional profit columns to a DataFrame.

    Parameters
    ----------
    df : DataFrame with one row per candidate betting event.
    model_prob_col : Column containing model probability for the positive event.
    odds_col : Column containing decimal odds for the same event.
    target_col : Optional realised binary outcome column. If supplied, profit is
        calculated for settled historical bets.
    policy : EdgePolicy controlling bet selection.
    stake_strategy : `flat` or `kelly`.
    flat_stake_amount : Stake used when stake_strategy=`flat`.
    bankroll : Bankroll used for Kelly stake sizing.
    kelly_multiplier : Fraction of full Kelly to use.
    max_bankroll_fraction : Maximum stake as a fraction of bankroll.

    Returns
    -------
    DataFrame with added columns:
        market_prob, edge, ev, bet_flag, stake and optional profit.
    """
    if model_prob_col not in df.columns:
        raise KeyError(f"Missing model probability column: {model_prob_col}")
    if odds_col not in df.columns:
        raise KeyError(f"Missing odds column: {odds_col}")
    if target_col is not None and target_col not in df.columns:
        raise KeyError(f"Missing target column: {target_col}")

    if stake_strategy not in {"flat", "kelly"}:
        raise ValueError("stake_strategy must be one of: 'flat', 'kelly'.")

    policy = policy or EdgePolicy()
    out = df.copy()

    model_probs = pd.to_numeric(out[model_prob_col], errors="coerce")
    odds = pd.to_numeric(out[odds_col], errors="coerce")

    valid = model_probs.between(0.0, 1.0, inclusive="both") & (odds > 1.0)

    out["market_prob"] = np.nan
    out["edge"] = np.nan
    out["ev"] = np.nan
    out["bet_flag"] = False
    out["stake"] = 0.0

    out.loc[valid, "market_prob"] = 1.0 / odds.loc[valid]
    out.loc[valid, "edge"] = model_probs.loc[valid] - out.loc[valid, "market_prob"]
    out.loc[valid, "ev"] = model_probs.loc[valid] * odds.loc[valid] - 1.0

    bet_mask = (
        valid
        & (out["edge"] >= policy.min_edge)
        & (out["ev"] >= policy.min_ev)
        & (odds >= policy.min_odds)
    )
    if policy.max_odds is not None:
        bet_mask = bet_mask & (odds <= policy.max_odds)

    out.loc[bet_mask, "bet_flag"] = True

    if stake_strategy == "flat":
        stake = flat_stake(flat_stake_amount)
        out.loc[bet_mask, "stake"] = stake
    else:
        stakes = []
        for idx in out.index:
            if not bool(bet_mask.loc[idx]):
                stakes.append(0.0)
                continue
            stakes.append(
                fractional_kelly_stake(
                    bankroll=bankroll,
                    model_probability=float(model_probs.loc[idx]),
                    decimal_odds=float(odds.loc[idx]),
                    fraction=kelly_multiplier,
                    max_bankroll_fraction=max_bankroll_fraction,
                )
            )
        out["stake"] = stakes

    if target_col is not None:
        targets = pd.to_numeric(out[target_col], errors="coerce")
        out["profit"] = 0.0
        wins = out["bet_flag"] & (targets == 1)
        losses = out["bet_flag"] & (targets == 0)
        out.loc[wins, "profit"] = out.loc[wins, "stake"] * (odds.loc[wins] - 1.0)
        out.loc[losses, "profit"] = -out.loc[losses, "stake"]

    return out
