from __future__ import annotations

import math

import pandas as pd
import pytest

from src.betting.edge import (
    EdgePolicy,
    add_betting_edges,
    expected_value,
    flat_stake,
    fractional_kelly_stake,
    implied_probability,
    kelly_fraction,
    probability_edge,
    should_bet,
)


def test_implied_probability_from_decimal_odds():
    assert implied_probability(2.0) == pytest.approx(0.5)
    assert implied_probability(4.0) == pytest.approx(0.25)


def test_invalid_decimal_odds_raise():
    with pytest.raises(ValueError):
        implied_probability(1.0)
    with pytest.raises(ValueError):
        implied_probability(0.95)


def test_probability_edge():
    assert probability_edge(0.55, 2.0) == pytest.approx(0.05)


def test_expected_value_per_unit_staked():
    assert expected_value(0.55, 2.10) == pytest.approx(0.155)
    assert expected_value(0.45, 2.00) == pytest.approx(-0.10)


def test_invalid_probability_raises():
    with pytest.raises(ValueError):
        expected_value(1.2, 2.0)
    with pytest.raises(ValueError):
        expected_value(-0.1, 2.0)


def test_should_bet_requires_edge_and_ev():
    policy = EdgePolicy(min_edge=0.02, min_ev=0.0)
    assert should_bet(0.55, 2.0, policy) is True
    assert should_bet(0.505, 2.0, policy) is False
    assert should_bet(0.40, 2.0, policy) is False


def test_flat_stake_validation():
    assert flat_stake(2.5) == pytest.approx(2.5)
    with pytest.raises(ValueError):
        flat_stake(-1.0)


def test_kelly_fraction_is_zero_when_no_edge():
    assert kelly_fraction(0.40, 2.0) == pytest.approx(0.0)


def test_kelly_fraction_positive_when_edge_exists():
    # Full Kelly at p=0.55, odds=2.0 is 0.10
    assert kelly_fraction(0.55, 2.0) == pytest.approx(0.10)


def test_fractional_kelly_stake_is_capped():
    stake = fractional_kelly_stake(
        bankroll=100.0,
        model_probability=0.70,
        decimal_odds=2.50,
        fraction=1.0,
        max_bankroll_fraction=0.05,
    )
    assert stake == pytest.approx(5.0)


def test_add_betting_edges_flat_stake_profit():
    df = pd.DataFrame(
        {
            "model_prob": [0.55, 0.51, 0.40],
            "odds_over": [2.0, 2.0, 2.0],
            "over_25": [1, 0, 1],
        }
    )

    out = add_betting_edges(
        df,
        model_prob_col="model_prob",
        odds_col="odds_over",
        target_col="over_25",
        policy=EdgePolicy(min_edge=0.02, min_ev=0.0),
        stake_strategy="flat",
        flat_stake_amount=1.0,
    )

    assert out["bet_flag"].tolist() == [True, False, False]
    assert out["market_prob"].iloc[0] == pytest.approx(0.5)
    assert out["edge"].iloc[0] == pytest.approx(0.05)
    assert out["ev"].iloc[0] == pytest.approx(0.10)
    assert out["stake"].tolist() == [1.0, 0.0, 0.0]
    assert out["profit"].tolist() == [1.0, 0.0, 0.0]


def test_add_betting_edges_handles_invalid_rows_without_crashing():
    df = pd.DataFrame(
        {
            "model_prob": [0.55, math.nan, 1.20],
            "odds_over": [2.0, 2.0, 2.0],
            "over_25": [1, 1, 1],
        }
    )

    out = add_betting_edges(
        df,
        model_prob_col="model_prob",
        odds_col="odds_over",
        target_col="over_25",
    )

    assert out["bet_flag"].tolist() == [True, False, False]
    assert out["stake"].tolist() == [1.0, 0.0, 0.0]


def test_add_betting_edges_kelly_strategy():
    df = pd.DataFrame(
        {
            "model_prob": [0.55],
            "odds_over": [2.0],
            "over_25": [1],
        }
    )

    out = add_betting_edges(
        df,
        model_prob_col="model_prob",
        odds_col="odds_over",
        target_col="over_25",
        stake_strategy="kelly",
        bankroll=100.0,
        kelly_multiplier=0.25,
        max_bankroll_fraction=0.05,
    )

    # Full Kelly is 10%, quarter Kelly is 2.5 units on a 100 unit bankroll.
    assert out["stake"].iloc[0] == pytest.approx(2.5)
    assert out["profit"].iloc[0] == pytest.approx(2.5)
