"""
CLI script to fetch EPL xG data from understat.com and write it to the DB.

What it does
------------
1. Fetches match-level xG (home_xg, away_xg) from understat.com for one or
   more EPL seasons using UnderstatLoader.
2. UPDATEs the team_match_stats table: sets xg_for and xg_against on the
   existing home and away rows for each match.
3. Prints a summary of rows updated per season.
4. Reminds you to rebuild the feature matrix so rolling xG features are
   populated in engineered_features.

Usage
-----
    # Fetch all 5 seasons (2019/20 – 2023/24)
    python scripts/fetch_xg.py --all

    # Fetch a single season
    python scripts/fetch_xg.py --season 2022/23

    # Dry run — fetch and display without writing to DB
    python scripts/fetch_xg.py --all --dry-run

After running this script, rebuild the feature matrix to get rolling xG features:
    python scripts/build_features.py --rebuild
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.ingestion.understat_loader import SEASON_YEAR_MAP, UnderstatLoader
from src.utils.config import DB_PATH
from src.utils.db import get_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DB update helper
# ---------------------------------------------------------------------------


def update_xg_in_db(xg_df: pd.DataFrame, season_id: str, dry_run: bool) -> int:
    """
    UPDATE team_match_stats rows with xG values from *xg_df*.

    For each match:
      - home team row: xg_for  = home_xg,  xg_against = away_xg
      - away team row: xg_for  = away_xg,  xg_against = home_xg

    Returns the total number of team_match_stats rows updated (should be
    2 × number of matches).
    """
    if xg_df.empty:
        log.warning("No xG data to write for season %s.", season_id)
        return 0

    if dry_run:
        log.info("[DRY RUN] Would update %d matches for season %s.", len(xg_df), season_id)
        log.info("[DRY RUN] Sample (first 5 rows):\n%s", xg_df.head().to_string(index=False))
        return 0

    with get_connection() as con:
        # Register the xG dataframe as a temporary view
        con.register("_xg_stage", xg_df)

        # Single UPDATE using a FROM join — sets home and away rows in one pass
        con.execute(
            """
            UPDATE team_match_stats AS t
            SET
                xg_for     = CASE WHEN t.is_home THEN x.home_xg ELSE x.away_xg END,
                xg_against = CASE WHEN t.is_home THEN x.away_xg ELSE x.home_xg END
            FROM _xg_stage AS x
            WHERE t.match_id = x.match_id
            """
        )

        # Verify: count how many rows now have xg_for populated for this season
        updated = con.execute(
            """
            SELECT COUNT(*) FROM team_match_stats t
            JOIN matches m ON t.match_id = m.match_id
            WHERE m.season = ?
              AND t.xg_for IS NOT NULL
            """,
            [season_id],
        ).fetchone()[0]

    log.info(
        "Season %s: %d team_match_stats rows now have xg_for populated.",
        season_id, updated,
    )
    return int(updated)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch EPL xG from understat.com and write to team_match_stats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    season_group = parser.add_mutually_exclusive_group(required=True)
    season_group.add_argument(
        "--season",
        metavar="SEASON",
        help="Single season to fetch, e.g. '2022/23'.",
    )
    season_group.add_argument(
        "--all",
        dest="all_seasons",
        action="store_true",
        help="Fetch all supported seasons (2019/20 – 2023/24).",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch data but do not write to the database.",
    )
    args = parser.parse_args()

    # ── Resolve seasons list ──────────────────────────────────────────────
    if args.all_seasons:
        seasons = sorted(SEASON_YEAR_MAP.keys())
    else:
        if args.season not in SEASON_YEAR_MAP:
            log.error(
                "Unknown season '%s'. Supported: %s", args.season, sorted(SEASON_YEAR_MAP)
            )
            sys.exit(1)
        seasons = [args.season]

    log.info("=" * 60)
    log.info("Understat xG Fetcher")
    log.info("Seasons   : %s", seasons)
    log.info("Dry run   : %s", args.dry_run)
    log.info("DB path   : %s", DB_PATH)
    log.info("=" * 60)

    loader = UnderstatLoader()
    total_matches = 0
    total_rows_updated = 0

    for season_id in seasons:
        log.info("─" * 50)
        log.info("Processing season: %s", season_id)

        try:
            xg_df = loader.load_season(season_id)
        except Exception as exc:
            log.error("Failed to fetch season %s: %s", season_id, exc)
            continue

        if xg_df.empty:
            log.warning("No xG data resolved for season %s — skipping DB write.", season_id)
            continue

        total_matches += len(xg_df)
        rows_updated = update_xg_in_db(xg_df, season_id, dry_run=args.dry_run)
        total_rows_updated += rows_updated

    # ── Summary ───────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Understat xG fetch complete.")
    log.info("  Matches with xG resolved   : %d", total_matches)
    log.info("  team_match_stats rows set  : %d", total_rows_updated)
    if args.dry_run:
        log.info("  [DRY RUN] No data written.")
    else:
        log.info("")
        log.info("  Next step: rebuild the feature matrix to populate")
        log.info("  rolling xG features in engineered_features:")
        log.info("")
        log.info("      python scripts/build_features.py --rebuild")
        log.info("")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
