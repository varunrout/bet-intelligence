"""
Database initialisation script for Football Market Intelligence System.

Creates all tables in DuckDB if they do not already exist.
Safe to re-run — uses CREATE TABLE IF NOT EXISTS throughout.

Usage
-----
    python scripts/init_db.py
    python scripts/init_db.py --reset   # drops and recreates all tables (DESTRUCTIVE)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb

from src.utils.config import DB_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL statements — executed in dependency order
# ---------------------------------------------------------------------------

DDL_STATEMENTS = [
    # ------------------------------------------------------------------
    # Reference tables
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS bookmakers (
        bookmaker_id   INTEGER PRIMARY KEY,
        name           TEXT    NOT NULL,
        short_code     TEXT,
        is_sharp       BOOLEAN DEFAULT FALSE,
        notes          TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS competitions (
        competition_id    INTEGER PRIMARY KEY,
        name              TEXT    NOT NULL,
        short_code        TEXT,
        country           TEXT,
        tier              INTEGER,
        fdco_code         TEXT,
        api_football_id   INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS teams (
        team_id         INTEGER PRIMARY KEY,
        name_canonical  TEXT    NOT NULL UNIQUE,
        name_fdco       TEXT,
        name_api_fb     TEXT,
        short_code      TEXT,
        competition_id  INTEGER REFERENCES competitions(competition_id)
    )
    """,

    # ------------------------------------------------------------------
    # Core match table
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS matches (
        match_id         TEXT PRIMARY KEY,
        competition_id   INTEGER REFERENCES competitions(competition_id),
        season           TEXT    NOT NULL,
        gameweek         INTEGER,
        kickoff_utc      TIMESTAMPTZ NOT NULL,
        home_team_id     INTEGER REFERENCES teams(team_id),
        away_team_id     INTEGER REFERENCES teams(team_id),
        venue            TEXT,
        goals_home       INTEGER,
        goals_away       INTEGER,
        goals_ht_home    INTEGER,
        goals_ht_away    INTEGER,
        result_ftr       TEXT,
        total_goals      INTEGER,
        over_25          BOOLEAN,
        source           TEXT,
        api_football_id  INTEGER,
        fdco_row_key     TEXT,
        created_at       TIMESTAMPTZ DEFAULT NOW()
    )
    """,

    # ------------------------------------------------------------------
    # Team-level match stats
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS team_match_stats (
        stat_id         INTEGER PRIMARY KEY,
        match_id        TEXT REFERENCES matches(match_id),
        team_id         INTEGER REFERENCES teams(team_id),
        is_home         BOOLEAN,
        shots           INTEGER,
        shots_on_target INTEGER,
        xg_for          DOUBLE,
        xg_against      DOUBLE,
        possession_pct  DOUBLE,
        corners         INTEGER,
        fouls           INTEGER,
        yellow_cards    INTEGER,
        red_cards       INTEGER,
        source          TEXT,
        created_at      TIMESTAMPTZ DEFAULT NOW(),
        UNIQUE (match_id, team_id)
    )
    """,

    # ------------------------------------------------------------------
    # Lineups
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS lineups (
        lineup_id       INTEGER PRIMARY KEY,
        match_id        TEXT REFERENCES matches(match_id),
        team_id         INTEGER REFERENCES teams(team_id),
        player_name     TEXT,
        player_api_id   INTEGER,
        position        TEXT,
        is_starter      BOOLEAN,
        shirt_number    INTEGER,
        source          TEXT,
        created_at      TIMESTAMPTZ DEFAULT NOW()
    )
    """,

    # ------------------------------------------------------------------
    # Injuries / suspensions
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS injuries (
        injury_id        INTEGER PRIMARY KEY,
        match_id         TEXT REFERENCES matches(match_id),
        team_id          INTEGER REFERENCES teams(team_id),
        player_name      TEXT,
        player_api_id    INTEGER,
        reason           TEXT,
        is_confirmed     BOOLEAN,
        reported_at_utc  TIMESTAMPTZ,
        source           TEXT,
        created_at       TIMESTAMPTZ DEFAULT NOW()
    )
    """,

    # ------------------------------------------------------------------
    # Odds snapshots
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS odds_snapshots (
        snapshot_id         INTEGER PRIMARY KEY,
        match_id            TEXT REFERENCES matches(match_id),
        bookmaker_id        INTEGER REFERENCES bookmakers(bookmaker_id),
        market_type         TEXT    NOT NULL,
        snapshot_type       TEXT    NOT NULL,
        snapshot_utc        TIMESTAMPTZ,
        odds_home           DOUBLE,
        odds_draw           DOUBLE,
        odds_away           DOUBLE,
        line                DOUBLE,
        odds_over           DOUBLE,
        odds_under          DOUBLE,
        implied_prob_over   DOUBLE,
        implied_prob_under  DOUBLE,
        margin              DOUBLE,
        source              TEXT,
        created_at          TIMESTAMPTZ DEFAULT NOW(),
        UNIQUE (match_id, bookmaker_id, market_type, snapshot_type)
    )
    """,

    # ------------------------------------------------------------------
    # Pre-computed engineered features (one row per match)
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS engineered_features (
        feature_id                  INTEGER PRIMARY KEY,
        match_id                    TEXT REFERENCES matches(match_id),
        computed_at_utc             TIMESTAMPTZ,

        -- Rolling form: home team (5-match window) -------------------
        home_goals_scored_avg5      DOUBLE,
        home_goals_conceded_avg5    DOUBLE,
        home_shots_avg5             DOUBLE,
        home_shots_on_target_avg5   DOUBLE,
        home_wins_last5             DOUBLE,
        home_clean_sheets_last5     DOUBLE,
        home_over25_last5           DOUBLE,

        -- Rolling form: home team (3-match window) -------------------
        home_goals_scored_avg3      DOUBLE,
        home_goals_conceded_avg3    DOUBLE,

        -- Rolling form: away team (5-match window) -------------------
        away_goals_scored_avg5      DOUBLE,
        away_goals_conceded_avg5    DOUBLE,
        away_shots_avg5             DOUBLE,
        away_shots_on_target_avg5   DOUBLE,
        away_wins_last5             DOUBLE,
        away_clean_sheets_last5     DOUBLE,
        away_over25_last5           DOUBLE,

        -- Rolling form: away team (3-match window) -------------------
        away_goals_scored_avg3      DOUBLE,
        away_goals_conceded_avg3    DOUBLE,

        -- xG (NULL until API-Football data added in Phase T2) --------
        home_xg_for_avg5            DOUBLE,
        home_xg_against_avg5        DOUBLE,
        away_xg_for_avg5            DOUBLE,
        away_xg_against_avg5        DOUBLE,

        -- Derived cross-team features --------------------------------
        combined_goals_avg5         DOUBLE,
        attack_proxy                DOUBLE,

        -- Context features -------------------------------------------
        home_rest_days              INTEGER,
        away_rest_days              INTEGER,
        rest_differential           INTEGER,
        home_matches_in_14_days     INTEGER,
        away_matches_in_14_days     INTEGER,
        is_neutral                  BOOLEAN DEFAULT FALSE,

        -- Elo ratings ------------------------------------------------
        home_elo                    DOUBLE,
        away_elo                    DOUBLE,
        elo_differential            DOUBLE,

        -- Market features (Pinnacle) ---------------------------------
        pin_implied_prob_over       DOUBLE,
        pin_odds_over               DOUBLE,
        pin_odds_under              DOUBLE,
        pin_margin                  DOUBLE,

        -- Market features (Bet365) -----------------------------------
        b365_implied_prob_over      DOUBLE,
        b365_odds_over              DOUBLE,
        b365_odds_under             DOUBLE,
        b365_margin                 DOUBLE,

        -- Market features (aggregate) --------------------------------
        avg_implied_prob_over       DOUBLE,
        max_implied_prob_over       DOUBLE,

        -- Market disagreement signal ---------------------------------
        pin_b365_divergence         DOUBLE,

        -- Legacy/compat columns (kept for Phase 7 CLV work) ---------
        opening_implied_prob_over   DOUBLE,
        closing_implied_prob_over   DOUBLE,
        opening_margin              DOUBLE,
        odds_movement_over          DOUBLE,

        -- Availability (NULL until Phase T2 data) -------------------
        home_key_absences           INTEGER,
        away_key_absences           INTEGER,

        -- Metadata ---------------------------------------------------
        feature_version             TEXT    DEFAULT 'v1',
        UNIQUE (match_id, feature_version)
    )
    """,

    # ------------------------------------------------------------------
    # Model predictions
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS model_predictions (
        prediction_id     INTEGER PRIMARY KEY,
        match_id          TEXT REFERENCES matches(match_id),
        model_name        TEXT    NOT NULL,
        model_version     TEXT,
        market_type       TEXT    NOT NULL,
        predicted_at_utc  TIMESTAMPTZ,
        prob_over         DOUBLE,
        prob_under        DOUBLE,
        fair_odds_over    DOUBLE,
        fair_odds_under   DOUBLE,
        edge_over         DOUBLE,
        edge_under        DOUBLE,
        model_confidence  DOUBLE,
        created_at        TIMESTAMPTZ DEFAULT NOW()
    )
    """,

    # ------------------------------------------------------------------
    # Backtest bet records
    # ------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS backtest_bets (
        bet_id              INTEGER PRIMARY KEY,
        match_id            TEXT REFERENCES matches(match_id),
        prediction_id       INTEGER REFERENCES model_predictions(prediction_id),
        strategy_name       TEXT    NOT NULL,
        market_type         TEXT    NOT NULL,
        side                TEXT    NOT NULL,
        edge_at_bet         DOUBLE,
        odds_taken          DOUBLE,
        bookmaker_id        INTEGER REFERENCES bookmakers(bookmaker_id),
        simulated_at_utc    TIMESTAMPTZ,
        stake_units         DOUBLE,
        bankroll_before     DOUBLE,
        outcome             TEXT,
        pnl_units           DOUBLE,
        bankroll_after      DOUBLE,
        closing_odds        DOUBLE,
        clv                 DOUBLE,
        created_at          TIMESTAMPTZ DEFAULT NOW()
    )
    """,

    # ------------------------------------------------------------------
    # Useful indexes
    # ------------------------------------------------------------------
    "CREATE INDEX IF NOT EXISTS idx_matches_kickoff ON matches(kickoff_utc)",
    "CREATE INDEX IF NOT EXISTS idx_matches_season  ON matches(season)",
    "CREATE INDEX IF NOT EXISTS idx_odds_match      ON odds_snapshots(match_id)",
    "CREATE INDEX IF NOT EXISTS idx_odds_market     ON odds_snapshots(market_type, snapshot_type)",
    "CREATE INDEX IF NOT EXISTS idx_features_match  ON engineered_features(match_id)",
    "CREATE INDEX IF NOT EXISTS idx_preds_match     ON model_predictions(match_id, model_name)",
]

# ---------------------------------------------------------------------------
# Seed data — bookmakers known from football-data.co.uk
# ---------------------------------------------------------------------------

SEED_BOOKMAKERS = [
    (1, "Bet365",         "B365",  False),
    (2, "Pinnacle",       "PIN",   True),
    (3, "BetWin",         "BW",    False),
    (4, "Market Average", "AVG",   False),
    (5, "Market Maximum", "MAX",   False),
]

SEED_COMPETITIONS = [
    (1, "English Premier League", "EPL", "England", 1, "E0", 39),
]


def drop_all_tables(con: duckdb.DuckDBPyConnection) -> None:
    tables = [
        "backtest_bets", "model_predictions", "engineered_features",
        "odds_snapshots", "injuries", "lineups",
        "team_match_stats", "matches",
        "teams", "competitions", "bookmakers",
    ]
    for t in tables:
        con.execute(f"DROP TABLE IF EXISTS {t}")
        log.info("Dropped table: %s", t)


def create_schema(con: duckdb.DuckDBPyConnection) -> None:
    for stmt in DDL_STATEMENTS:
        con.execute(stmt)
    log.info("Schema created / verified.")


def seed_reference_data(con: duckdb.DuckDBPyConnection) -> None:
    for row in SEED_BOOKMAKERS:
        con.execute(
            "INSERT OR IGNORE INTO bookmakers VALUES (?, ?, ?, ?, NULL)",
            list(row),
        )
    log.info("Seeded %d bookmakers.", len(SEED_BOOKMAKERS))

    for row in SEED_COMPETITIONS:
        con.execute(
            "INSERT OR IGNORE INTO competitions VALUES (?, ?, ?, ?, ?, ?, ?)",
            list(row),
        )
    log.info("Seeded %d competitions.", len(SEED_COMPETITIONS))


def seed_teams(con: duckdb.DuckDBPyConnection) -> None:
    """
    Seed teams table from team_aliases.yaml.
    Teams are given integer IDs in alphabetical order of canonical name.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.utils.config import load_team_aliases

    aliases = load_team_aliases()
    sorted_teams = sorted(aliases.items())

    for team_id, (canonical, info) in enumerate(sorted_teams, start=1):
        con.execute(
            """
            INSERT OR IGNORE INTO teams
                (team_id, name_canonical, name_fdco, name_api_fb, short_code, competition_id)
            VALUES (?, ?, ?, ?, ?, 1)
            """,
            [
                team_id,
                canonical,
                info.get("fdco"),
                info.get("api_football"),
                info.get("short_code"),
            ],
        )
    log.info("Seeded %d teams.", len(sorted_teams))


def init_database(reset: bool = False) -> None:
    log.info("Database path: %s", DB_PATH)
    con = duckdb.connect(str(DB_PATH))
    try:
        if reset:
            log.warning("--reset flag set. Dropping all tables.")
            drop_all_tables(con)

        create_schema(con)
        seed_reference_data(con)
        seed_teams(con)
        con.commit()
        log.info("Database initialisation complete.")
    finally:
        con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialise the bet-intelligence DuckDB database.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate all tables. DESTRUCTIVE — loses all data.",
    )
    args = parser.parse_args()
    init_database(reset=args.reset)
