"""
Phase 1 ETL pipeline orchestrator for Football Market Intelligence System.

Runs the full extract-transform-load cycle for one competition's FDCO data:
  1. Download raw CSVs from football-data.co.uk
  2. Parse and normalise matches, odds, and team stats
  3. Resolve team canonical names to team_ids
  4. Insert into DuckDB (skips existing rows on conflict)

This script is safe to re-run — it will skip already-loaded records.
Run scripts/init_db.py first to create the database schema.

Usage
-----
    # Full EPL load (all configured seasons)
    python scripts/run_pipeline.py

    # Single season
    python scripts/run_pipeline.py --season "2023/24"

    # Force re-download of raw CSVs
    python scripts/run_pipeline.py --force-download

    # Dry run — parse and validate but do not insert into DB
    python scripts/run_pipeline.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.ingestion.fdco_loader import FDCOLoader
from src.transform.match_transformer import (
    prepare_matches_for_insert,
    prepare_odds_for_insert,
    prepare_stats_for_insert,
)
from src.transform.team_resolver import TeamResolver
from src.utils.db import get_connection, get_row_count, run_query, upsert_dataframe
from src.utils.config import DB_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def fetch_name_to_id(competition_key: str = "epl") -> dict[str, int]:
    """
    Return a dict mapping canonical team name -> team_id from the teams table.
    """
    df = run_query("SELECT team_id, name_canonical FROM teams")
    if df.empty:
        raise RuntimeError(
            "Teams table is empty. Run scripts/init_db.py first."
        )
    return dict(zip(df["name_canonical"], df["team_id"]))


def fetch_max_id(table: str, id_col: str) -> int:
    """Return the current maximum value of an ID column, or 0 if table is empty."""
    df = run_query(f"SELECT COALESCE(MAX({id_col}), 0) AS max_id FROM {table}")
    return int(df["max_id"].iloc[0])


# ---------------------------------------------------------------------------
# Insert helpers
# ---------------------------------------------------------------------------


def insert_matches(matches_df: pd.DataFrame, dry_run: bool = False) -> int:
    if matches_df.empty:
        return 0
    if dry_run:
        log.info("[DRY RUN] Would insert %d match rows.", len(matches_df))
        return 0

    inserted = upsert_dataframe(matches_df, "matches", conflict_columns=["match_id"])
    log.info("Matches inserted: %d new rows.", inserted)
    return inserted


def insert_odds(odds_df: pd.DataFrame, id_offset: int = 0, dry_run: bool = False) -> int:
    if odds_df.empty:
        return 0

    prepared = prepare_odds_for_insert(odds_df, id_offset=id_offset)

    if dry_run:
        log.info("[DRY RUN] Would insert %d odds snapshot rows.", len(prepared))
        return 0

    inserted = upsert_dataframe(
        prepared, "odds_snapshots", conflict_columns=["snapshot_id"]
    )
    log.info("Odds snapshots inserted: %d new rows.", inserted)
    return inserted


def insert_stats(
    stats_df: pd.DataFrame,
    name_to_id: dict[str, int],
    id_offset: int = 0,
    dry_run: bool = False,
) -> int:
    if stats_df.empty:
        return 0

    prepared = prepare_stats_for_insert(stats_df, name_to_id, id_offset=id_offset)

    if dry_run:
        log.info("[DRY RUN] Would insert %d team stat rows.", len(prepared))
        return 0

    inserted = upsert_dataframe(
        prepared, "team_match_stats", conflict_columns=["stat_id"]
    )
    log.info("Team stats inserted: %d new rows.", inserted)
    return inserted


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    competition_key: str = "epl",
    season_id: str | None = None,
    force_download: bool = False,
    dry_run: bool = False,
) -> None:
    """
    Run the full Phase 1 ETL pipeline.

    Parameters
    ----------
    competition_key  : Competition key from league_config.yaml.
    season_id        : If set, load only this season. Otherwise load all.
    force_download   : Re-download raw CSVs even if already present.
    dry_run          : Parse and validate only — do not insert into DB.
    """
    log.info("=" * 60)
    log.info("Football Market Intelligence — Phase 1 ETL")
    log.info("Competition : %s", competition_key.upper())
    log.info("Season      : %s", season_id or "ALL")
    log.info("Dry run     : %s", dry_run)
    log.info("DB path     : %s", DB_PATH)
    log.info("=" * 60)

    if not dry_run and not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. "
            f"Run scripts/init_db.py first."
        )

    # Fetch team id lookup
    name_to_id = fetch_name_to_id() if not dry_run else {}

    # Initialise loader
    resolver = TeamResolver()
    loader = FDCOLoader(
        competition_key=competition_key,
        resolver=resolver,
    )

    # Determine which seasons to process
    if season_id:
        seasons_to_process = [season_id]
    else:
        seasons_to_process = list(loader.season_codes.keys())

    total_matches = 0
    total_odds = 0
    total_stats = 0

    for sid in seasons_to_process:
        log.info("-" * 40)
        log.info("Processing season: %s", sid)

        try:
            matches_df, odds_df, stats_df = loader.load_season(sid)
        except Exception as e:
            log.error("Failed to load season %s: %s", sid, e)
            continue

        # Validate + prepare matches (skip DB ID resolution in dry-run)
        if dry_run:
            log.info("[DRY RUN] Would process %d matches, %d odds rows, %d stat rows.",
                     len(matches_df), len(odds_df), len(stats_df))
            # Still verify team resolution works (resolver uses YAML, not DB)
            try:
                from src.transform.match_transformer import dedup_matches
                dedup_matches(matches_df)
            except Exception as e:
                log.warning("Dedup check failed: %s", e)
            continue

        try:
            matches_ready = prepare_matches_for_insert(matches_df, name_to_id)
        except ValueError as e:
            log.error("Match preparation failed for %s: %s", sid, e)
            log.error(
                "Check config/team_aliases.yaml for missing FDCO team name mappings."
            )
            continue

        # Get current max IDs for surrogate key generation
        odds_offset   = fetch_max_id("odds_snapshots", "snapshot_id") if not dry_run else 0
        stats_offset  = fetch_max_id("team_match_stats", "stat_id")  if not dry_run else 0

        n_m = insert_matches(matches_ready, dry_run=dry_run)
        n_o = insert_odds(odds_df, id_offset=odds_offset,  dry_run=dry_run)
        n_s = insert_stats(stats_df, name_to_id, id_offset=stats_offset, dry_run=dry_run)

        total_matches += n_m
        total_odds    += n_o
        total_stats   += n_s

    # Summary
    log.info("=" * 60)
    log.info("Pipeline complete.")
    log.info("  Total matches inserted   : %d", total_matches)
    log.info("  Total odds rows inserted : %d", total_odds)
    log.info("  Total stats rows inserted: %d", total_stats)

    if not dry_run:
        log.info("  DB matches table size    : %d", get_row_count("matches"))
        log.info("  DB odds table size       : %d", get_row_count("odds_snapshots"))
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Phase 1 Football Market Intelligence ETL pipeline."
    )
    parser.add_argument(
        "--competition",
        default="epl",
        help="Competition key from league_config.yaml (default: epl).",
    )
    parser.add_argument(
        "--season",
        default=None,
        help="Single season to process (e.g. '2023/24'). Omit for all seasons.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download raw CSVs even if already present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate only. Do not insert into database.",
    )
    args = parser.parse_args()

    run_pipeline(
        competition_key=args.competition,
        season_id=args.season,
        force_download=args.force_download,
        dry_run=args.dry_run,
    )
