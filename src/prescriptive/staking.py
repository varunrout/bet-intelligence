"""
Stake-sizing rules for Football Market Intelligence System — Phase 5.

Converts a model probability and a market price into a stake, treating every
rule as uncertainty-aware decision support rather than a promise of edge.

Two staking families are implemented:

  Flat      — a fixed fraction of the CURRENT bankroll per bet.  Simple,
              robust to model miscalibration, the baseline every fancier
              rule must beat.
  Kelly     — stake proportional to perceived edge (fractional Kelly).
              Full Kelly is famously ruinous under model error, so a
              fraction (default 0.25) and a hard cap are applied.

Design rules
------------
- A stake is NEVER negative and NEVER exceeds `cap` × bankroll.
- If the model sees no positive edge, the Kelly stake is exactly 0.0 —
  the rule declines the bet rather than shorting (no lay betting here).
- All functions are pure: bankroll state lives in the backtest loop,
  not in this module.

Public API
----------
  flat_stake(bankroll, fraction)                    -> float
  kelly_fraction(p_model, decimal_odds)             -> float
  kelly_stake(bankroll, p_model, decimal_odds,
              fraction, cap)                        -> float
"""

from __future__ import annotations


# ── Flat staking ───────────────────────────────────────────────────────────────


def flat_stake(bankroll: float, fraction: float = 0.01) -> float:
    """
    Fixed-fraction stake: always `fraction` of the current bankroll.

    Parameters
    ----------
    bankroll : current bankroll (must be >= 0).
    fraction : fraction of bankroll to stake per bet (default 1%).

    Returns
    -------
    Stake as a float, floored at 0.0.
    """
    if bankroll <= 0.0 or fraction <= 0.0:
        return 0.0
    return float(bankroll * fraction)


# ── Kelly staking ──────────────────────────────────────────────────────────────


def kelly_fraction(p_model: float, decimal_odds: float) -> float:
    """
    Full-Kelly optimal fraction for a binary bet at decimal odds.

    f* = (p·(o − 1) − (1 − p)) / (o − 1)  =  (p·o − 1) / (o − 1)

    Parameters
    ----------
    p_model      : model probability of the outcome, in [0, 1].
    decimal_odds : decimal (European) odds, must be > 1.0.

    Returns
    -------
    The optimal bankroll fraction, floored at 0.0 when the model sees no
    positive edge (p·o <= 1) or inputs are degenerate.
    """
    if decimal_odds <= 1.0 or not (0.0 <= p_model <= 1.0):
        return 0.0
    b = decimal_odds - 1.0
    f = (p_model * decimal_odds - 1.0) / b
    return max(0.0, float(f))


def kelly_stake(
    bankroll: float,
    p_model: float,
    decimal_odds: float,
    fraction: float = 0.25,
    cap: float = 0.05,
) -> float:
    """
    Fractional-Kelly stake with a hard cap, sized on the CURRENT bankroll.

    Parameters
    ----------
    bankroll     : current bankroll (must be >= 0).
    p_model      : model probability of the outcome.
    decimal_odds : decimal odds offered by the market.
    fraction     : Kelly multiplier (0.25 = quarter Kelly, the default) —
                   guards against model error, which full Kelly punishes
                   catastrophically.
    cap          : maximum bankroll fraction allowed on any single bet
                   (default 5%), applied AFTER the Kelly fraction.

    Returns
    -------
    Stake as a float in [0, cap × bankroll].
    """
    if bankroll <= 0.0:
        return 0.0
    f = kelly_fraction(p_model, decimal_odds) * fraction
    f = min(f, cap)
    return float(bankroll * f)
