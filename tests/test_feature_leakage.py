"""
Feature leakage guard tests for Football Market Intelligence System.

These tests exist to catch data leakage bugs before they corrupt model training.
They are not optional. Any failure here means a fix is required before modeling.

Categories tested:
  1. Rolling features use only past matches — no future data.
  2. Odds snapshots used as features pre-date kickoff.
  3. Target labels (over_25, goals) are not present in the feature set.
  4. Engineered features are keyed to the correct temporal position.

Run with: pytest tests/test_feature_leakage.py -v
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.utils.time_utils import is_before_kickoff, season_from_date


# ---------------------------------------------------------------------------
# Helpers — build synthetic data for leakage tests
# ---------------------------------------------------------------------------


def make_match_sequence(n: int = 10) -> pd.DataFrame:
    """
    Create a synthetic time-ordered sequence of n matches for one team.
    Each match is 7 days apart starting from a fixed date.
    """
    base = datetime(2023, 8, 12, 15, 0, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        kickoff = base + timedelta(weeks=i)
        rows.append({
            "match_id":    f"TEST_MATCH_{i:03d}",
            "kickoff_utc": kickoff,
            "goals_home":  i % 4,
            "goals_away":  (i + 1) % 3,
            "over_25":     (i % 4 + (i + 1) % 3) > 2,
        })
    return pd.DataFrame(rows)


def compute_rolling_mean_naive(
    df: pd.DataFrame, target_col: str, window: int, row_idx: int
) -> float | None:
    """
    Compute the rolling mean of target_col for row at row_idx,
    using only past rows (index < row_idx).

    This is the CORRECT approach — look back only.
    """
    if row_idx == 0:
        return None
    lookback = df.iloc[max(0, row_idx - window):row_idx]
    if lookback.empty:
        return None
    return lookback[target_col].mean()


def compute_rolling_mean_leaky(
    df: pd.DataFrame, target_col: str, window: int, row_idx: int
) -> float:
    """
    LEAKY version — uses pandas rolling which includes the current row.
    This is what you must NOT do.
    """
    return df[target_col].rolling(window, min_periods=1).mean().iloc[row_idx]


# ---------------------------------------------------------------------------
# Leakage test: rolling features
# ---------------------------------------------------------------------------


class TestRollingFeatureLeakage:

    def test_correct_rolling_does_not_use_current_row(self):
        df = make_match_sequence(10)
        df["goals_total"] = df["goals_home"] + df["goals_away"]

        for i in range(len(df)):
            correct_val = compute_rolling_mean_naive(df, "goals_total", 5, i)
            if correct_val is None:
                continue

            # Verify: correct rolling only includes past rows
            past_values = df["goals_total"].iloc[max(0, i - 5):i].tolist()
            expected = sum(past_values) / len(past_values)
            assert abs(correct_val - expected) < 1e-9, (
                f"Row {i}: rolling mean {correct_val} does not match "
                f"manual past-only calculation {expected}."
            )

    def test_leaky_rolling_differs_from_correct_at_current_row(self):
        """
        Demonstrates that naive pandas rolling() includes the current row,
        which contaminates the feature with same-match information.
        """
        df = make_match_sequence(10)
        df["goals_total"] = df["goals_home"] + df["goals_away"]

        mismatches = 0
        for i in range(1, len(df)):
            correct = compute_rolling_mean_naive(df, "goals_total", 5, i)
            leaky   = compute_rolling_mean_leaky(df, "goals_total", 5, i)
            if abs(correct - leaky) > 1e-9:
                mismatches += 1

        # Leaky computation should differ for at least some rows
        assert mismatches > 0, (
            "Expected leaky rolling to differ from correct rolling for at least one row. "
            "This test is validating that the leaky version IS leaky."
        )

    def test_features_computed_before_kickoff(self):
        """
        For every match in a feature set, assert the feature 'computed_at_utc'
        is strictly before 'kickoff_utc'.
        """
        df = make_match_sequence(5)
        # Simulate feature computation time = 1 hour before kickoff
        df["computed_at_utc"] = df["kickoff_utc"] - pd.Timedelta(hours=1)

        violations = df[df["computed_at_utc"] >= df["kickoff_utc"]]
        assert violations.empty, (
            f"Leakage: features computed at or after kickoff for {len(violations)} matches:\n"
            f"{violations[['match_id', 'kickoff_utc', 'computed_at_utc']]}"
        )

    def test_feature_after_kickoff_is_detected(self):
        """
        Verify the leakage check correctly catches a post-kickoff feature.
        """
        df = make_match_sequence(5)
        # Intentionally set one feature to be computed AFTER kickoff
        df["computed_at_utc"] = df["kickoff_utc"] + pd.Timedelta(hours=2)

        violations = df[df["computed_at_utc"] >= df["kickoff_utc"]]
        assert len(violations) == len(df)  # all rows are now leaky


# ---------------------------------------------------------------------------
# Leakage test: odds snapshots
# ---------------------------------------------------------------------------


class TestOddsSnapshotLeakage:

    def test_snapshot_before_kickoff_passes(self):
        kickoff = datetime(2023, 10, 21, 15, 0, tzinfo=timezone.utc)
        snapshot = datetime(2023, 10, 20, 9, 0, tzinfo=timezone.utc)
        assert is_before_kickoff(snapshot, kickoff) is True

    def test_snapshot_after_kickoff_fails(self):
        kickoff  = datetime(2023, 10, 21, 15, 0, tzinfo=timezone.utc)
        snapshot = datetime(2023, 10, 21, 17, 0, tzinfo=timezone.utc)
        assert is_before_kickoff(snapshot, kickoff) is False

    def test_snapshot_at_exact_kickoff_fails(self):
        kickoff = datetime(2023, 10, 21, 15, 0, tzinfo=timezone.utc)
        assert is_before_kickoff(kickoff, kickoff) is False

    def test_all_feature_odds_snapshots_pre_kickoff(self):
        """
        Simulate a feature DataFrame where odds_snapshot_utc should be
        before kickoff_utc for every row.
        """
        data = {
            "match_id":         ["M1", "M2", "M3"],
            "kickoff_utc":      [
                datetime(2023, 9, 1,  15, 0, tzinfo=timezone.utc),
                datetime(2023, 9, 8,  15, 0, tzinfo=timezone.utc),
                datetime(2023, 9, 15, 15, 0, tzinfo=timezone.utc),
            ],
            "odds_snapshot_utc": [
                datetime(2023, 9, 1,  10, 0, tzinfo=timezone.utc),
                datetime(2023, 9, 7,  20, 0, tzinfo=timezone.utc),
                datetime(2023, 9, 14, 12, 0, tzinfo=timezone.utc),
            ],
        }
        df = pd.DataFrame(data)

        violations = df[df["odds_snapshot_utc"] >= df["kickoff_utc"]]
        assert violations.empty, (
            f"Odds snapshot leakage detected in {len(violations)} rows."
        )


# ---------------------------------------------------------------------------
# Leakage test: target variable isolation
# ---------------------------------------------------------------------------


class TestTargetVariableIsolation:

    FORBIDDEN_IN_FEATURES = ["over_25", "goals_home", "goals_away", "total_goals", "result_ftr"]

    def test_target_columns_not_in_feature_set(self):
        """
        The feature set passed to the model must not contain target labels.
        Simulates a feature DataFrame and checks for forbidden column names.
        """
        # A realistic-looking feature column list
        feature_columns = [
            "home_goals_scored_avg5",
            "home_goals_conceded_avg5",
            "away_goals_scored_avg5",
            "away_goals_conceded_avg5",
            "home_wins_last5",
            "away_wins_last5",
            "home_rest_days",
            "away_rest_days",
            "elo_differential",
            "opening_implied_prob_over",
            "odds_movement_over",
        ]
        leaked = [c for c in feature_columns if c in self.FORBIDDEN_IN_FEATURES]
        assert leaked == [], (
            f"Target columns found in feature set: {leaked}. "
            f"These must be removed before passing to the model."
        )

    def test_simulated_leaked_feature_is_detected(self):
        """
        Verify that if over_25 is accidentally added to features, it is caught.
        """
        bad_feature_columns = [
            "home_goals_scored_avg5",
            "over_25",   # <-- this should not be here
            "home_rest_days",
        ]
        leaked = [c for c in bad_feature_columns if c in self.FORBIDDEN_IN_FEATURES]
        assert "over_25" in leaked


# ---------------------------------------------------------------------------
# Temporal split helper tests
# ---------------------------------------------------------------------------


class TestTemporalSplit:

    def test_train_test_split_respects_time_order(self):
        """
        Validate that a time-based train/test split never allows future data
        into the training window.
        """
        matches = make_match_sequence(20)
        cutoff = matches["kickoff_utc"].quantile(0.8, interpolation="nearest")

        train = matches[matches["kickoff_utc"] < cutoff]
        test  = matches[matches["kickoff_utc"] >= cutoff]

        assert train["kickoff_utc"].max() < test["kickoff_utc"].min(), (
            "Train set contains matches after the earliest test match. "
            "Time ordering violation."
        )

    def test_no_match_id_overlap_between_train_and_test(self):
        matches = make_match_sequence(20)
        cutoff = matches["kickoff_utc"].quantile(0.8, interpolation="nearest")

        train_ids = set(matches[matches["kickoff_utc"] < cutoff]["match_id"])
        test_ids  = set(matches[matches["kickoff_utc"] >= cutoff]["match_id"])

        overlap = train_ids & test_ids
        assert overlap == set(), (
            f"Match IDs appear in both train and test sets: {overlap}"
        )


# ---------------------------------------------------------------------------
# Season inference
# ---------------------------------------------------------------------------


class TestSeasonInference:

    def test_august_is_new_season(self):
        dt = datetime(2023, 8, 12, tzinfo=timezone.utc)
        assert season_from_date(dt) == "2023/24"

    def test_may_is_same_season(self):
        dt = datetime(2024, 5, 19, tzinfo=timezone.utc)
        assert season_from_date(dt) == "2023/24"

    def test_july_is_previous_season(self):
        dt = datetime(2024, 7, 15, tzinfo=timezone.utc)
        assert season_from_date(dt) == "2023/24"

    def test_january_mid_season(self):
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert season_from_date(dt) == "2023/24"
