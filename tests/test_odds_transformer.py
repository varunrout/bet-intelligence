"""
Tests for OddsTransformer — implied probability, margin, and CLV calculations.

Run with: pytest tests/test_odds_transformer.py -v
"""

from __future__ import annotations

import pytest
import pandas as pd

from src.transform.odds_transformer import (
    compute_clv,
    compute_clv_series,
    compute_margin_three_way,
    compute_margin_two_way,
    enrich_ou_dataframe,
    implied_prob,
    implied_prob_series,
    odds_movement,
    prob_movement,
    remove_margin_three_way,
    remove_margin_two_way,
)


class TestImpliedProb:

    def test_evens(self):
        assert implied_prob(2.0) == pytest.approx(0.5)

    def test_short_odds(self):
        assert implied_prob(1.25) == pytest.approx(0.8)

    def test_long_odds(self):
        assert implied_prob(5.0) == pytest.approx(0.2)

    def test_invalid_odds_raises(self):
        with pytest.raises(ValueError):
            implied_prob(1.0)

    def test_invalid_odds_below_one_raises(self):
        with pytest.raises(ValueError):
            implied_prob(0.5)

    def test_series(self):
        s = pd.Series([2.0, 4.0, 1.5])
        result = implied_prob_series(s)
        assert result.tolist() == pytest.approx([0.5, 0.25, 1.0 / 1.5])

    def test_series_raises_on_invalid(self):
        with pytest.raises(ValueError):
            implied_prob_series(pd.Series([2.0, 1.0, 3.0]))


class TestMarginTwoWay:

    def test_fair_book_has_zero_margin(self):
        # Fair book: 2.0 and 2.0 -> total implied = 1.0 -> margin = 0
        margin = compute_margin_two_way(2.0, 2.0)
        assert margin == pytest.approx(0.0, abs=1e-6)

    def test_typical_bookmaker_margin(self):
        # Bet365-style: 1.85 / 1.85 -> margin ~ 8%
        margin = compute_margin_two_way(1.85, 1.85)
        expected = (1 / 1.85 + 1 / 1.85) - 1
        assert margin == pytest.approx(expected, rel=1e-5)
        assert 0.05 < margin < 0.12

    def test_pinnacle_tight_margin(self):
        # Pinnacle-style: 1.95 / 1.95 -> ~2.6%
        margin = compute_margin_two_way(1.95, 1.95)
        assert 0.01 < margin < 0.04

    def test_asymmetric_market(self):
        margin = compute_margin_two_way(1.80, 2.10)
        assert margin > 0


class TestMarginThreeWay:

    def test_typical_1x2_margin(self):
        # Typical 1X2: H=2.0, D=3.5, A=4.0
        margin = compute_margin_three_way(2.0, 3.5, 4.0)
        assert margin > 0
        assert margin < 0.15

    def test_fair_three_way(self):
        # Probabilities: 0.5, 0.25, 0.25 -> fair odds: 2.0, 4.0, 4.0
        margin = compute_margin_three_way(2.0, 4.0, 4.0)
        assert margin == pytest.approx(0.0, abs=1e-6)


class TestRemoveMargin:

    def test_two_way_probabilities_sum_to_one(self):
        p_over, p_under = remove_margin_two_way(1.85, 1.85)
        assert p_over + p_under == pytest.approx(1.0)

    def test_two_way_fair_odds_unchanged(self):
        p_over, p_under = remove_margin_two_way(2.0, 2.0)
        assert p_over == pytest.approx(0.5)
        assert p_under == pytest.approx(0.5)

    def test_three_way_probabilities_sum_to_one(self):
        p_h, p_d, p_a = remove_margin_three_way(2.0, 3.5, 4.0)
        assert p_h + p_d + p_a == pytest.approx(1.0)

    def test_three_way_all_in_range(self):
        p_h, p_d, p_a = remove_margin_three_way(2.0, 3.5, 4.0)
        for p in [p_h, p_d, p_a]:
            assert 0 < p < 1


class TestOddsMovement:

    def test_shortening(self):
        # Odds fell from 2.0 to 1.7 — outcome became more favoured
        assert odds_movement(2.0, 1.7) == pytest.approx(0.3)

    def test_drifting(self):
        # Odds rose from 2.0 to 2.3 — outcome became less favoured
        assert odds_movement(2.0, 2.3) == pytest.approx(-0.3)

    def test_no_movement(self):
        assert odds_movement(1.9, 1.9) == pytest.approx(0.0)

    def test_prob_movement_shortening(self):
        # Probability increases when odds fall
        pm = prob_movement(2.0, 1.7)
        assert pm > 0


class TestCLV:

    def test_positive_clv(self):
        # Bet at 2.1, market closed at 1.9 — good bet (got better odds)
        clv = compute_clv(2.1, 1.9)
        assert clv > 0

    def test_negative_clv(self):
        # Bet at 1.8, market closed at 2.0 — poor bet (got worse odds)
        clv = compute_clv(1.8, 2.0)
        assert clv < 0

    def test_zero_clv_at_closing(self):
        # Bet exactly at closing -> CLV = 0
        clv = compute_clv(2.0, 2.0)
        assert clv == pytest.approx(0.0)

    def test_invalid_closing_odds_raises(self):
        with pytest.raises(ValueError):
            compute_clv(2.0, 0.5)

    def test_clv_series(self):
        taken = pd.Series([2.1, 1.8, 2.0])
        closing = pd.Series([1.9, 2.0, 2.0])
        result = compute_clv_series(taken, closing)
        assert result.iloc[0] > 0   # positive CLV
        assert result.iloc[1] < 0   # negative CLV
        assert result.iloc[2] == pytest.approx(0.0)


class TestEnrichDataFrame:

    def test_enriches_valid_rows(self):
        df = pd.DataFrame({"over": [1.85, 2.0], "under": [1.95, 2.0]})
        result = enrich_ou_dataframe(df, "over", "under")
        assert "implied_prob_over" in result.columns
        assert "implied_prob_under" in result.columns
        assert "margin" in result.columns
        assert (result["implied_prob_over"] + result["implied_prob_under"]).round(6).eq(1.0).all()

    def test_skips_invalid_rows(self):
        df = pd.DataFrame({"over": [1.85, None, 1.0], "under": [1.95, 1.90, 1.90]})
        result = enrich_ou_dataframe(df, "over", "under")
        # Rows with None or <=1.0 should produce NaN in probability columns
        assert result.loc[1, "implied_prob_over"] != result.loc[1, "implied_prob_over"]  # NaN

    def test_snapshot_type_label(self):
        df = pd.DataFrame({"over": [1.85], "under": [1.95]})
        result = enrich_ou_dataframe(df, "over", "under", snapshot_type="opening")
        assert result["snapshot_type"].iloc[0] == "opening"
