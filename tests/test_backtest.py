"""
Tests for the Phase 5 backtesting engine (selection, staking, accounting, CLV).

Every test uses small hand-computable cases so expected values are exact,
in keeping with the leakage-test philosophy of the rest of the suite:
the engine must be provably right on cases a human can verify by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest import backtest_summary, run_backtest, select_bets
from src.prescriptive.staking import flat_stake, kelly_fraction, kelly_stake


# ── Staking rules ──────────────────────────────────────────────────────────────


def test_flat_stake_is_fraction_of_bankroll():
    assert flat_stake(1000.0, fraction=0.01) == pytest.approx(10.0)


def test_flat_stake_never_negative():
    assert flat_stake(-50.0, fraction=0.01) == 0.0
    assert flat_stake(1000.0, fraction=-0.5) == 0.0


def test_kelly_fraction_known_case():
    # p=0.5 at odds 3.0: f* = (0.5*3 - 1) / 2 = 0.25
    assert kelly_fraction(0.5, 3.0) == pytest.approx(0.25)


def test_kelly_declines_negative_edge():
    # p*o = 0.4*2.0 = 0.8 < 1 -> no bet, never a negative (short) stake
    assert kelly_fraction(0.4, 2.0) == 0.0
    assert kelly_stake(1000.0, 0.4, 2.0) == 0.0


def test_kelly_stake_respects_cap():
    # Full Kelly f*=0.25, quarter Kelly = 0.0625, but cap = 0.05 binds
    stake = kelly_stake(1000.0, 0.5, 3.0, fraction=0.25, cap=0.05)
    assert stake == pytest.approx(50.0)


def test_kelly_degenerate_odds():
    assert kelly_fraction(0.9, 1.0) == 0.0
    assert kelly_fraction(0.9, 0.5) == 0.0


# ── Selection rule ─────────────────────────────────────────────────────────────


def _preds() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y_prob":       [0.60, 0.50, 0.30, 0.95, 0.40],
            "y_true":       [1,    0,    1,    1,    0],
            "decimal_odds": [2.00, 2.00, 3.00, 1.05, 12.0],
        }
    )


def test_select_bets_edge_threshold():
    # edges: 0.20, 0.00, -0.10, -0.0025, 3.80
    bets = select_bets(_preds(), edge_threshold=0.02, min_odds=1.10, max_odds=10.0)
    # Row 0 qualifies; row 1 edge 0 < 0.02; row 2 negative; row 3 below
    # min_odds; row 4 above max_odds.
    assert list(bets.index) == [0]
    assert bets.loc[0, "edge"] == pytest.approx(0.20)


def test_select_bets_missing_columns_raise():
    with pytest.raises(ValueError, match="missing columns"):
        select_bets(pd.DataFrame({"y_prob": [0.5]}))


# ── Backtest accounting ────────────────────────────────────────────────────────


def _two_bet_ledger() -> pd.DataFrame:
    # Bet 1: p=0.6 @ 2.0, wins.  Bet 2: p=0.6 @ 2.0, loses.
    return pd.DataFrame(
        {
            "y_prob":       [0.6, 0.6],
            "y_true":       [1,   0],
            "decimal_odds": [2.0, 2.0],
            "edge":         [0.2, 0.2],
        }
    )


def test_backtest_exact_pnl_flat():
    # Flat 1% of CURRENT bankroll: stake1=10 -> win +10 -> bankroll 1010
    # stake2=10.10 -> lose -> bankroll 999.90
    result = run_backtest(_two_bet_ledger(), staking="flat",
                          initial_bankroll=1000.0, fraction=0.01)
    ledger, summary = result["ledger"], result["summary"]
    assert ledger["stake"].tolist() == pytest.approx([10.0, 10.10])
    assert ledger["bankroll_after"].tolist() == pytest.approx([1010.0, 999.90])
    assert summary["final_bankroll"] == pytest.approx(999.90)
    assert summary["n_bets"] == 2
    assert summary["hit_rate"] == pytest.approx(0.5)


def test_bankroll_conservation():
    result = run_backtest(_two_bet_ledger(), staking="flat",
                          initial_bankroll=1000.0, fraction=0.01)
    total_pnl = result["ledger"]["pnl"].sum()
    assert result["summary"]["final_bankroll"] == pytest.approx(1000.0 + total_pnl)


def test_max_drawdown_from_peak():
    # Win first (peak 1010), then lose 10.10: drawdown = 10.10 / 1010
    result = run_backtest(_two_bet_ledger(), staking="flat",
                          initial_bankroll=1000.0, fraction=0.01)
    assert result["summary"]["max_drawdown"] == pytest.approx(10.10 / 1010.0, abs=1e-4)


def test_no_information_travels_backwards():
    # Reversing bet order must change the trajectory (stakes depend on
    # bankroll path), proving the loop is sequential, not vectorised leakage.
    fwd = run_backtest(_two_bet_ledger(), staking="flat",
                       initial_bankroll=1000.0, fraction=0.01)
    rev = run_backtest(_two_bet_ledger().iloc[::-1].reset_index(drop=True),
                       staking="flat", initial_bankroll=1000.0, fraction=0.01)
    assert fwd["ledger"]["stake"].tolist() != rev["ledger"]["stake"].tolist()


def test_empty_selection_is_safe():
    empty = select_bets(_preds(), edge_threshold=10.0)
    result = run_backtest(empty, staking="flat")
    assert result["summary"]["n_bets"] == 0
    assert result["summary"]["final_bankroll"] == pytest.approx(1000.0)


def test_unknown_staking_rule_raises():
    with pytest.raises(ValueError, match="unknown staking rule"):
        run_backtest(_two_bet_ledger(), staking="martingale")


# ── Closing-line value ─────────────────────────────────────────────────────────


def test_clv_computation():
    bets = _two_bet_ledger().assign(closing_odds=[1.8, 2.2])
    result = run_backtest(bets, staking="flat", initial_bankroll=1000.0)
    ledger = result["ledger"]
    # CLV: 2.0/1.8 - 1 = +0.1111 (beat the close); 2.0/2.2 - 1 = -0.0909
    assert ledger["clv"].tolist() == pytest.approx([0.1111, -0.0909], abs=1e-4)
    assert result["summary"]["clv_positive_rate"] == pytest.approx(0.5)


# ── Reporting ──────────────────────────────────────────────────────────────────


def test_backtest_summary_per_season():
    bets = _two_bet_ledger().assign(test_season=["2022/23", "2023/24"])
    result = run_backtest(bets, staking="flat", initial_bankroll=1000.0)
    table = backtest_summary(result)
    assert list(table["test_season"]) == ["2022/23", "2023/24", "ALL"]
    assert table.loc[table["test_season"] == "ALL", "n_bets"].iloc[0] == 2
