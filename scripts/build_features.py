"""
CLI entry point for Phase 3 feature engineering.

Usage
-----
    # Build and save features for all EPL matches:
    python scripts/build_features.py

    # Dry run — compute but do not write to DB:
    python scripts/build_features.py --dry-run

    # Specific competition:
    python scripts/build_features.py --competition EPL
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.pipeline import build_features, save_features
from src.utils.config import DB_PATH
from src.utils.db import get_row_count

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pre-match features.")
    parser.add_argument(
        "--competition",
        default="EPL",
        help="Competition short code (default: EPL)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute features but do not write to DB.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only process first N matches (for quick validation).",
    )
    args = parser.parse_args()

    if not args.dry_run and not DB_PATH.exists():
        log.error("Database not found: %s. Run scripts/init_db.py first.", DB_PATH)
        sys.exit(1)

    log.info("=" * 60)
    log.info("Phase 3 — Feature Engineering")
    log.info("Competition : %s", args.competition)
    log.info("Dry run     : %s", args.dry_run)
    log.info("DB path     : %s", DB_PATH)
    log.info("=" * 60)

    features = build_features(competition_key=args.competition, db_path=DB_PATH)

    if args.sample:
        features = features.head(args.sample)
        log.info("Sample mode: using first %d rows.", args.sample)

    # Print summary of computed features
    log.info("Feature matrix: %d rows × %d columns", len(features), len(features.columns))

    # Coverage report
    coverage = features.notna().mean().sort_values()
    log.info("Feature coverage (lowest 10):")
    for col, pct in coverage.head(10).items():
        log.info("  %-45s  %5.1f%%", col, pct * 100)

    if args.dry_run:
        log.info("[DRY RUN] Skipping DB write.")
        log.info("Sample row (first match with ≥5 home matches of history):")
        sample = features[features["home_wins_last5"].notna()]
        if not sample.empty:
            row = sample.iloc[0]
            for col in [
                "home_goals_scored_avg5", "home_goals_conceded_avg5",
                "away_goals_scored_avg5", "away_goals_conceded_avg5",
                "home_wins_last5", "away_wins_last5",
                "home_rest_days", "away_rest_days",
                "home_elo", "away_elo", "elo_differential",
                "pin_implied_prob_over", "b365_implied_prob_over",
                "attack_proxy", "combined_goals_avg5",
            ]:
                val = row.get(col)
                if val is not None and str(val) != "nan":
                    log.info("  %-45s  %s", col, f"{val:.4f}" if isinstance(val, float) else val)
    else:
        n_inserted = save_features(features, db_path=DB_PATH)
        log.info("=" * 60)
        log.info("Feature pipeline complete.")
        log.info("  Rows inserted             : %d", n_inserted)
        log.info("  engineered_features total : %d", get_row_count("engineered_features"))
        log.info("=" * 60)


if __name__ == "__main__":
    main()
