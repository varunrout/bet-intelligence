from __future__ import annotations

import pandas as pd
import pytest

from src.betting.backtest import (
    max_drawdown,
    profit_curve,
    run_backtest,
    summarize_backtest,
    summarize_by_group,
)
from src.betting.edge import EdgePolicy


def sample_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": ["M1", "M2", "M3", "M4"],
            "season": ["2022/23", "2022/23", "2023/24", "2023/24"],
            "kickoff_utc": ["2022-08-01", "2022-08-02", "2023-08-01", "2023-08-02"],
            "model_prob": [0.55, 0.56, 0.40, 0.60],
            "odds_over": [2.0, 2.0, 2.0, 1.80],
            "over_25": [1, 0, 1, 1],
        }
    )


def test_run_backtest_adds_profit_and_drawdown_columns():
    out = run_backtest(
        sample_predictions(),
        model_prob_col="model_prob",
        odds_col="odds_over",
        target_col="over_25",
        policy=EdgePolicy(min_edge=0.02, min_ev=0.0),
        sort_col="kickoff_utc",
    )

    assert "profit" in out.columns
    assert "cumulative_profit" in out.columns
    assert "drawdown" in out.columns
    assert out["bet_flag"].tolist() == [True, True, False, True]
    assert out["profit"].tolist() == [1.0, -1.0, 0.0, 0.8]
    assert out["cumulative_profit"].iloc[-1] == pytest.approx(0.8)


def test_summarize_backtest_headline_metrics():
    out = run_backtest(
        sample_predictions(),
        model_prob_col="model_prob",
        odds_col="odds_over",
        target_col="over_25",
        policy=EdgePolicy(min_edge=0.02, min_ev=0.0),
    )

    summary = summarize_backtest(out)
    data = summary.as_dict()

    assert data["rows"] == 4
    assert data["bets"] == 3
    assert data["wins"] == 2
    assert data["losses"] == 1
    assert data["stake"] == pytest.approx(3.0)
    assert data["profit"] == pytest.approx(0.8)
    assert data["roi"] == pytest.approx(0.2667)
    assert data["hit_rate"] == pytest.approx(0.6667)


def test_summarize_backtest_handles_no_bets():
    out = run_backtest(
        sample_predictions(),
        model_prob_col="model_prob",
        odds_col="odds_over",
        target_col="over_25",
        policy=EdgePolicy(min_edge=0.50, min_ev=0.50),
    )

    summary = summarize_backtest(out).as_dict()
    assert summary["bets"] == 0
    assert summary["profit"] == 0.0
    assert summary["roi"] == 0.0


def test_summarize_by_group_returns_season_rows():
    out = run_backtest(
        sample_predictions(),
        model_prob_col="model_prob",
        odds_col="odds_over",
        target_col="over_25",
        policy=EdgePolicy(min_edge=0.02, min_ev=0.0),
    )

    grouped = summarize_by_group(out, "season")
    assert set(grouped["season"]) == {"2022/23", "2023/24"}
    assert grouped["bets"].sum() == 3


def test_profit_curve_contains_only_bet_rows():
    out = run_backtest(
        sample_predictions(),
        model_prob_col="model_prob",
        odds_col="odds_over",
        target_col="over_25",
        policy=EdgePolicy(min_edge=0.02, min_ev=0.0),
    )

    curve = profit_curve(out)
    assert len(curve) == 3
    assert curve["bet_number"].tolist() == [1, 2, 3]
    assert curve["cumulative_profit"].iloc[-1] == pytest.approx(0.8)


def test_max_drawdown_from_profit_sequence():
    assert max_drawdown([1.0, -2.0, 0.5, -1.0, 3.0]) == pytest.approx(2.5)


def test_missing_group_column_raises():
    out = run_backtest(
        sample_predictions(),
        model_prob_col="model_prob",
        odds_col="odds_over",
        target_col="over_25",
    )
    with pytest.raises(KeyError):
        summarize_by_group(out, "missing_col")
