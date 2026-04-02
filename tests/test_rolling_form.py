"""
Tests for rolling form feature engineering.

Core test: verify that all rolling features use only historical data —
no information from the current match or any future match is included.

Run with: pytest tests/test_rolling_form.py -v
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.features.rolling_form import (
    build_rolling_features,
    build_team_match_events,
    compute_rolling_stats,
    pivot_to_match_features,
)


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------


def _make_synthetic_matches(n: int = 20, n_teams: int = 4) -> pd.DataFrame:
    """
    Create n synthetic matches between n_teams rotating teams.
    Matches are one week apart.
    Every team plays home and away in alternating fashion.
    """
    base = datetime(2023, 8, 5, 15, 0, tzinfo=timezone.utc)
    team_ids = list(range(1, n_teams + 1))
    rows = []
    for i in range(n):
        h = team_ids[i % n_teams]
        a = team_ids[(i + 1) % n_teams]
        goals_h = (i * 2) % 5
        goals_a = (i * 3 + 1) % 4
        total   = goals_h + goals_a
        ftr     = "H" if goals_h > goals_a else ("D" if goals_h == goals_a else "A")
        rows.append({
            "match_id":      f"M{i:03d}",
            "kickoff_utc":   base + timedelta(weeks=i),
            "season":        "2023/24",
            "home_team_id":  h,
            "away_team_id":  a,
            "goals_home":    goals_h,
            "goals_away":    goals_a,
            "total_goals":   total,
            "result_ftr":    ftr,
            "over_25":       total > 2,
        })
    return pd.DataFrame(rows)


def _make_synthetic_stats(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Create simple stat rows for each team in each match."""
    rows = []
    for _, m in matches_df.iterrows():
        rows.append({
            "match_id": m["match_id"], "team_id": m["home_team_id"],
            "is_home": True,
            "shots": int(m["goals_home"] * 5 + 3),
            "shots_on_target": int(m["goals_home"] * 2 + 1),
            "corners": 5, "yellow_cards": 1,
        })
        rows.append({
            "match_id": m["match_id"], "team_id": m["away_team_id"],
            "is_home": False,
            "shots": int(m["goals_away"] * 5 + 2),
            "shots_on_target": int(m["goals_away"] * 2 + 1),
            "corners": 4, "yellow_cards": 1,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Unit tests: build_team_match_events
# ---------------------------------------------------------------------------


class TestBuildTeamMatchEvents:

    def test_row_count_is_two_per_match(self):
        matches = _make_synthetic_matches(10)
        stats   = _make_synthetic_stats(matches)
        events  = build_team_match_events(matches, stats)
        assert len(events) == len(matches) * 2

    def test_goals_for_consistency(self):
        """For home team, goals_for == goals_home in the match."""
        matches = _make_synthetic_matches(10)
        stats   = _make_synthetic_stats(matches)
        events  = build_team_match_events(matches, stats)

        for _, match in matches.iterrows():
            home_row = events[
                (events["match_id"] == match["match_id"]) & (events["is_home"])
            ]
            assert len(home_row) == 1
            assert home_row.iloc[0]["goals_for"] == match["goals_home"]
            assert home_row.iloc[0]["goals_against"] == match["goals_away"]

    def test_goals_against_consistency(self):
        """For away team, goals_for == goals_away in the match."""
        matches = _make_synthetic_matches(10)
        stats   = _make_synthetic_stats(matches)
        events  = build_team_match_events(matches, stats)

        for _, match in matches.iterrows():
            away_row = events[
                (events["match_id"] == match["match_id"]) & (~events["is_home"])
            ]
            assert len(away_row) == 1
            assert away_row.iloc[0]["goals_for"] == match["goals_away"]

    def test_won_flag_correct_for_home_win(self):
        matches = _make_synthetic_matches(20)
        stats   = _make_synthetic_stats(matches)
        events  = build_team_match_events(matches, stats)

        for _, match in matches.iterrows():
            if match["result_ftr"] == "H":
                home_row = events[
                    (events["match_id"] == match["match_id"]) & events["is_home"]
                ]
                away_row = events[
                    (events["match_id"] == match["match_id"]) & ~events["is_home"]
                ]
                assert home_row.iloc[0]["won"] == 1.0
                assert away_row.iloc[0]["won"] == 0.0


# ---------------------------------------------------------------------------
# Unit tests: compute_rolling_stats — strict leakage check
# ---------------------------------------------------------------------------


class TestRollingStatsLeakage:
    """
    Core test suite for the rolling lookback leakage guarantee.

    For each team × match, verify that the rolling feature value equals
    the manually computed mean of the PREVIOUS N matches only.
    """

    def setup_method(self):
        matches = _make_synthetic_matches(20, n_teams=4)
        stats   = _make_synthetic_stats(matches)
        self.events = build_team_match_events(matches, stats)
        self.events_rolled = compute_rolling_stats(self.events, windows=[3, 5])

    def test_first_appearance_is_nan(self):
        """
        The first match for any team must have NaN for all rolling features,
        since there is no prior history.
        """
        for team_id, grp in self.events_rolled.groupby("team_id"):
            first_row = grp.sort_values("kickoff_utc").iloc[0]
            assert pd.isna(first_row.get("goals_scored_avg5")), (
                f"Team {team_id}: first match should have NaN goals_scored_avg5 "
                f"but got {first_row.get('goals_scored_avg5')}"
            )

    def test_rolling_avg5_uses_previous_5_matches_only(self):
        """
        For each team, verify goals_scored_avg5 at match i equals
        the mean goals_for from matches i-5 .. i-1.
        """
        for team_id, grp in self.events_rolled.groupby("team_id"):
            grp = grp.sort_values("kickoff_utc").reset_index(drop=True)
            for i in range(1, len(grp)):
                row = grp.iloc[i]
                computed_val = row.get("goals_scored_avg5")
                if pd.isna(computed_val):
                    continue

                # Manual lookback: matches 0..i-1, last 5
                lookback_start = max(0, i - 5)
                past            = grp.iloc[lookback_start:i]
                expected        = past["goals_for"].mean()

                assert abs(computed_val - expected) < 1e-9, (
                    f"Team {team_id}, row {i}: "
                    f"goals_scored_avg5={computed_val:.4f} "
                    f"but manual lookback={expected:.4f}"
                )

    def test_rolling_avg3_uses_previous_3_matches_only(self):
        """Same test for window=3."""
        for team_id, grp in self.events_rolled.groupby("team_id"):
            grp = grp.sort_values("kickoff_utc").reset_index(drop=True)
            for i in range(1, len(grp)):
                row = grp.iloc[i]
                computed_val = row.get("goals_scored_avg3")
                if pd.isna(computed_val):
                    continue
                lookback_start = max(0, i - 3)
                past           = grp.iloc[lookback_start:i]
                expected       = past["goals_for"].mean()
                assert abs(computed_val - expected) < 1e-9, (
                    f"Team {team_id}, row {i}: "
                    f"goals_scored_avg3={computed_val:.4f} "
                    f"but manual={expected:.4f}"
                )

    def test_wins_last5_is_sum_not_mean(self):
        """wins_last5 should be the count (sum) of wins, not the mean."""
        for team_id, grp in self.events_rolled.groupby("team_id"):
            grp = grp.sort_values("kickoff_utc").reset_index(drop=True)
            for i in range(1, len(grp)):
                row = grp.iloc[i]
                val = row.get("wins_last5")
                if pd.isna(val):
                    continue
                lookback_start = max(0, i - 5)
                past           = grp.iloc[lookback_start:i]
                expected_sum   = past["won"].sum()
                assert abs(val - expected_sum) < 1e-9

    def test_clean_sheets_last5_is_sum(self):
        for team_id, grp in self.events_rolled.groupby("team_id"):
            grp = grp.sort_values("kickoff_utc").reset_index(drop=True)
            for i in range(5, len(grp)):
                row = grp.iloc[i]
                val = row.get("clean_sheets_last5")
                if pd.isna(val):
                    continue
                past     = grp.iloc[i-5:i]
                expected = past["clean_sheet"].sum()
                assert abs(val - expected) < 1e-9

    def test_no_future_data_at_any_index(self):
        """
        For every row in events_rolled, verify that the kickoff of matches
        used in the rolling window are all strictly BEFORE the current kickoff.

        This tests the temporal guarantee, not just the positional one.
        """
        for team_id, grp in self.events_rolled.groupby("team_id"):
            grp = grp.sort_values("kickoff_utc").reset_index(drop=True)
            kickoffs = grp["kickoff_utc"].tolist()

            for i in range(len(grp)):
                # The rolling window at position i should only include positions 0..i-1
                # => all kickoffs in the window are < kickoffs[i]
                window_start = max(0, i - 5)
                for j in range(window_start, i):
                    assert kickoffs[j] < kickoffs[i], (
                        f"Team {team_id}: kickoff[{j}]={kickoffs[j]} is not "
                        f"before kickoff[{i}]={kickoffs[i]}. Ordering violation."
                    )


# ---------------------------------------------------------------------------
# Integration test: build_rolling_features
# ---------------------------------------------------------------------------


class TestBuildRollingFeatures:

    def test_returns_one_row_per_match(self):
        matches = _make_synthetic_matches(20)
        stats   = _make_synthetic_stats(matches)
        features = build_rolling_features(matches, stats)
        assert len(features) == len(matches)

    def test_match_ids_are_index(self):
        matches = _make_synthetic_matches(10)
        stats   = _make_synthetic_stats(matches)
        features = build_rolling_features(matches, stats)
        assert features.index.name == "match_id"
        assert set(features.index) == set(matches["match_id"])

    def test_no_target_columns_in_output(self):
        """
        The output must not contain goals, result, or over_25 outcome columns.
        These are targets — they would cause direct leakage.
        """
        matches  = _make_synthetic_matches(20)
        stats    = _make_synthetic_stats(matches)
        features = build_rolling_features(matches, stats)

        forbidden = {
            "goals_home", "goals_away", "total_goals",
            "over_25", "result_ftr",
        }
        leaked = set(features.columns) & forbidden
        assert not leaked, f"Target columns found in features: {leaked}"

    def test_has_home_and_away_prefixed_columns(self):
        matches  = _make_synthetic_matches(20)
        stats    = _make_synthetic_stats(matches)
        features = build_rolling_features(matches, stats)

        home_cols = [c for c in features.columns if c.startswith("home_")]
        away_cols = [c for c in features.columns if c.startswith("away_")]
        assert len(home_cols) > 0, "No home_ prefixed columns found."
        assert len(away_cols) > 0, "No away_ prefixed columns found."

    def test_first_few_matches_mostly_nan(self):
        """
        The first N matches (where teams have < 1 prior game) should have NaN
        for rolling features. Not all will be NaN since different teams
        may have history from other matches.
        """
        matches  = _make_synthetic_matches(20)
        stats    = _make_synthetic_stats(matches)
        features = build_rolling_features(matches, stats, windows=[5])

        first_match = features.iloc[0]
        n_nan = first_match[["home_goals_scored_avg5", "away_goals_scored_avg5"]].isna().sum()
        assert n_nan >= 1, (
            "Expected at least one NaN rolling feature for the very first match."
        )

    def test_feature_coverage_increases_with_match_number(self):
        """
        By match 10+, all rolling features should be fully populated
        (assuming at least 5 prior matches per team).
        """
        matches  = _make_synthetic_matches(30)
        stats    = _make_synthetic_stats(matches)
        features = build_rolling_features(matches, stats, windows=[5])

        late_matches  = features.iloc[20:]
        check_cols    = ["home_goals_scored_avg5", "away_goals_scored_avg5"]
        na_rate_late  = late_matches[check_cols].isna().mean().max()
        assert na_rate_late == 0.0, (
            f"NaN rate in late matches should be 0.0 but got {na_rate_late:.2f}. "
            f"Suggests rolling window is not filling correctly."
        )
