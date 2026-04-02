"""
Football-Data.co.uk (FDCO) ingestion loader.

Responsibilities:
  1. Download raw CSV files from football-data.co.uk.
  2. Parse columns, handle schema variation across seasons.
  3. Resolve team names to canonical form via TeamResolver.
  4. Produce clean DataFrames for matches and odds_snapshots tables.
  5. Persist raw files exactly as downloaded (no modifications to raw/).

FDCO data notes:
  - Odds provided are NOT timestamped. They represent bookmaker prices
    collected at some point before the match — treat as 'closing' for
    Phase 1/2. Opening odds require a separate source (The Odds API).
  - Date format: DD/MM/YYYY.
  - Time column may be absent in older files.
  - Column naming inconsistencies exist across seasons — handled below.

Usage
-----
    loader = FDCOLoader()
    loader.download_all_seasons("epl")
    matches_df, odds_df = loader.load_season("epl", "2023/24")
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
import requests

from src.transform.team_resolver import TeamResolver
from src.transform.odds_transformer import enrich_ou_dataframe
from src.utils.config import (
    RAW_FDCO_DIR,
    get_competition_config,
    get_season_fdco_codes,
    load_league_config,
)
from src.utils.time_utils import parse_fdco_datetime_series

log = logging.getLogger(__name__)

FDCO_BASE_URL = "https://www.football-data.co.uk/mmz4281"

# Columns always expected (subset that must be present for a row to be valid)
REQUIRED_MATCH_COLS = {"HomeTeam", "AwayTeam", "Date", "FTHG", "FTAG", "FTR"}

# Canonical column names for OU 2.5 market — different seasons use different prefixes
OU_COL_VARIANTS: list[tuple[str, str, str]] = [
    ("B365>2.5", "B365<2.5", "B365"),
    ("P>2.5",    "P<2.5",    "Pinnacle"),
    ("Avg>2.5",  "Avg<2.5",  "Market Average"),
    ("Max>2.5",  "Max<2.5",  "Market Maximum"),
    # Some older seasons use slightly different spellings
    ("B365C>2.5", "B365C<2.5", "Bet365 Closing"),
]

# Bookmaker short-code -> bookmaker_id mapping (matches seed data in init_db.py)
BOOKMAKER_ID_MAP = {
    "B365":   1,
    "Bet365": 1,
    "Bet365 Closing": 1,
    "Pinnacle": 2,
    "BetWin": 3,
    "Market Average": 4,
    "Market Maximum": 5,
}


class FDCOLoader:
    """
    Loads and parses football-data.co.uk CSV files.

    Parameters
    ----------
    competition_key : str
        Key in league_config.yaml (e.g. 'epl').
    resolver : TeamResolver | None
        Optional pre-built resolver. One is created automatically if not supplied.
    raw_dir : Path | None
        Override for the raw FDCO storage directory.
    """

    def __init__(
        self,
        competition_key: str = "epl",
        resolver: TeamResolver | None = None,
        raw_dir: Path | None = None,
    ) -> None:
        self.competition_key = competition_key
        self.comp_config = get_competition_config(competition_key)
        self.season_codes = get_season_fdco_codes(competition_key)
        self.fdco_code = self.comp_config["fdco_code"]
        self.competition_id = 1  # EPL seed ID — extend if multi-league
        self.raw_dir = raw_dir or RAW_FDCO_DIR
        self.resolver = resolver or TeamResolver()

        log.info(
            "FDCOLoader initialised for %s (%s). Seasons: %s",
            self.comp_config["name"],
            self.fdco_code,
            list(self.season_codes.keys()),
        )

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def _csv_url(self, fdco_season: str) -> str:
        return f"{FDCO_BASE_URL}/{fdco_season}/{self.fdco_code}.csv"

    def _raw_path(self, fdco_season: str) -> Path:
        return self.raw_dir / f"{self.fdco_code}_{fdco_season}.csv"

    def download_season(self, season_id: str, force: bool = False) -> Path:
        """
        Download a single season CSV file and save to raw/.

        Parameters
        ----------
        season_id : str
            Season string, e.g. '2023/24'.
        force : bool
            If True, re-download even if file already exists.

        Returns
        -------
        Path to the saved file.
        """
        fdco_season = self.season_codes.get(season_id)
        if fdco_season is None:
            raise ValueError(
                f"Season '{season_id}' not found in config for '{self.competition_key}'."
            )

        out_path = self._raw_path(fdco_season)

        if out_path.exists() and not force:
            log.info("Already downloaded: %s (use force=True to re-download).", out_path.name)
            return out_path

        url = self._csv_url(fdco_season)
        log.info("Downloading: %s -> %s", url, out_path.name)

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        out_path.write_bytes(response.content)
        log.info("Saved %d bytes to %s.", len(response.content), out_path)
        return out_path

    def download_all_seasons(self, force: bool = False) -> list[Path]:
        """Download all configured seasons. Returns list of saved file paths."""
        paths = []
        for season_id in self.season_codes:
            try:
                p = self.download_season(season_id, force=force)
                paths.append(p)
            except Exception as e:
                log.error("Failed to download %s: %s", season_id, e)
        return paths

    # ------------------------------------------------------------------
    # Parse raw CSV
    # ------------------------------------------------------------------

    def _read_csv(self, path: Path) -> pd.DataFrame:
        """
        Read a raw FDCO CSV file into a DataFrame.

        Handles common encoding issues and trailing empty rows.
        """
        try:
            df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")

        # FDCO files often have trailing blank rows
        df = df.dropna(subset=["HomeTeam", "AwayTeam", "Date"])
        df = df.reset_index(drop=True)

        log.debug("Read %d rows from %s", len(df), path.name)
        return df

    def _validate_required_columns(self, df: pd.DataFrame, path: Path) -> None:
        missing = REQUIRED_MATCH_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"Required columns missing in {path.name}: {missing}. "
                f"Columns present: {list(df.columns)}"
            )

    # ------------------------------------------------------------------
    # Match extraction
    # ------------------------------------------------------------------

    def _extract_matches(
        self, df: pd.DataFrame, season_id: str
    ) -> pd.DataFrame:
        """
        Extract and normalise match rows from a raw FDCO DataFrame.

        Returns a DataFrame with columns aligned to the matches table schema.
        """
        records = []

        for idx, row in df.iterrows():
            try:
                home_canonical = self.resolver.resolve(
                    str(row["HomeTeam"]).strip(), source="fdco"
                )
                away_canonical = self.resolver.resolve(
                    str(row["AwayTeam"]).strip(), source="fdco"
                )
            except ValueError as e:
                log.error("Row %d in %s: %s", idx, season_id, e)
                continue

            # Parse kickoff datetime
            date_str = str(row["Date"]).strip()
            time_str = str(row.get("Time", "")).strip()

            try:
                kickoff_utc = parse_fdco_datetime_series(
                    pd.Series([date_str]),
                    pd.Series([time_str]),
                ).iloc[0]
            except Exception as e:
                log.error("Row %d: failed to parse datetime '%s %s': %s", idx, date_str, time_str, e)
                continue

            # Build match_id from competition + season + teams + date
            date_slug = kickoff_utc.strftime("%Y%m%d")
            home_code = self.resolver.get_short_code(home_canonical)
            away_code = self.resolver.get_short_code(away_canonical)
            match_id = f"EPL_{season_id.replace('/', '')}_{date_slug}_{home_code}{away_code}"

            goals_home = _safe_int(row.get("FTHG"))
            goals_away = _safe_int(row.get("FTAG"))
            total_goals = (
                goals_home + goals_away
                if goals_home is not None and goals_away is not None
                else None
            )
            over_25 = (total_goals > 2) if total_goals is not None else None

            records.append({
                "match_id":        match_id,
                "competition_id":  self.competition_id,
                "season":          season_id,
                "gameweek":        None,
                "kickoff_utc":     kickoff_utc,
                "home_team_canonical": home_canonical,
                "away_team_canonical": away_canonical,
                "goals_home":      goals_home,
                "goals_away":      goals_away,
                "goals_ht_home":   _safe_int(row.get("HTHG")),
                "goals_ht_away":   _safe_int(row.get("HTAG")),
                "result_ftr":      str(row.get("FTR", "")).strip() or None,
                "total_goals":     total_goals,
                "over_25":         over_25,
                "source":          "fdco",
                "fdco_row_key":    f"{self.fdco_code}_{season_id}_{idx}",
            })

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Odds extraction
    # ------------------------------------------------------------------

    def _extract_odds(
        self, df: pd.DataFrame, matches_df: pd.DataFrame, season_id: str
    ) -> pd.DataFrame:
        """
        Extract Over/Under 2.5 odds from the raw FDCO DataFrame.

        Returns a DataFrame aligned to the odds_snapshots table schema.
        """
        # Build a match_id lookup: (home_canonical, date_slug) -> match_id
        match_lookup: dict[tuple[str, str], str] = {}
        for _, mrow in matches_df.iterrows():
            date_slug = pd.Timestamp(mrow["kickoff_utc"]).strftime("%Y%m%d")
            key = (mrow["home_team_canonical"], date_slug)
            match_lookup[key] = mrow["match_id"]

        records = []

        for idx, row in df.iterrows():
            # Identify match_id for this row
            try:
                home_canonical = self.resolver.resolve(
                    str(row["HomeTeam"]).strip(), source="fdco"
                )
            except ValueError:
                continue

            date_str = str(row["Date"]).strip()
            try:
                kickoff_utc = parse_fdco_datetime_series(
                    pd.Series([date_str]),
                    pd.Series([str(row.get("Time", ""))]),
                ).iloc[0]
            except Exception:
                continue

            date_slug = pd.Timestamp(kickoff_utc).strftime("%Y%m%d")
            match_id = match_lookup.get((home_canonical, date_slug))

            if match_id is None:
                log.warning(
                    "No match_id found for odds row: %s %s", home_canonical, date_slug
                )
                continue

            # Try each OU column variant present in this file
            for over_col, under_col, bm_name in OU_COL_VARIANTS:
                if over_col not in df.columns or under_col not in df.columns:
                    continue

                odds_over = _safe_float(row.get(over_col))
                odds_under = _safe_float(row.get(under_col))

                if odds_over is None or odds_under is None:
                    continue
                if odds_over <= 1.0 or odds_under <= 1.0:
                    continue

                bm_id = BOOKMAKER_ID_MAP.get(bm_name, 4)  # default to Average

                from src.transform.odds_transformer import (
                    compute_margin_two_way,
                    remove_margin_two_way,
                )

                margin = compute_margin_two_way(odds_over, odds_under)
                prob_over, prob_under = remove_margin_two_way(odds_over, odds_under)

                records.append({
                    "match_id":           match_id,
                    "bookmaker_id":       bm_id,
                    "market_type":        "ou25",
                    "snapshot_type":      "closing",   # FDCO does not provide opening
                    "snapshot_utc":       None,         # not timestamped
                    "line":               2.5,
                    "odds_over":          round(odds_over,  4),
                    "odds_under":         round(odds_under, 4),
                    "implied_prob_over":  round(prob_over,  6),
                    "implied_prob_under": round(prob_under, 6),
                    "margin":             round(margin, 6),
                    "source":             "fdco",
                })

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Team stats extraction
    # ------------------------------------------------------------------

    def _extract_team_stats(
        self, df: pd.DataFrame, matches_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract home and away shot/card/corner stats from FDCO CSV.

        Returns a DataFrame aligned to team_match_stats schema.
        """
        match_lookup: dict[tuple[str, str], tuple[str, str, str]] = {}
        for _, mrow in matches_df.iterrows():
            date_slug = pd.Timestamp(mrow["kickoff_utc"]).strftime("%Y%m%d")
            match_lookup[(mrow["home_team_canonical"], date_slug)] = (
                mrow["match_id"],
                mrow["home_team_canonical"],
                mrow["away_team_canonical"],
            )

        records = []
        stat_col_map = {
            "home": {
                "shots":            "HS",
                "shots_on_target":  "HST",
                "corners":          "HC",
                "fouls":            "HF",
                "yellow_cards":     "HY",
                "red_cards":        "HR",
            },
            "away": {
                "shots":            "AS",
                "shots_on_target":  "AST",
                "corners":          "AC",
                "fouls":            "AF",
                "yellow_cards":     "AY",
                "red_cards":        "AR",
            },
        }

        for _, row in df.iterrows():
            try:
                home_canonical = self.resolver.resolve(
                    str(row["HomeTeam"]).strip(), source="fdco"
                )
            except ValueError:
                continue

            date_str = str(row["Date"]).strip()
            try:
                kickoff_utc = parse_fdco_datetime_series(
                    pd.Series([date_str]),
                    pd.Series([str(row.get("Time", ""))]),
                ).iloc[0]
            except Exception:
                continue

            date_slug = pd.Timestamp(kickoff_utc).strftime("%Y%m%d")
            lookup = match_lookup.get((home_canonical, date_slug))
            if lookup is None:
                continue

            match_id, home_team, away_team = lookup

            for side, team_canonical in [("home", home_team), ("away", away_team)]:
                stat_record = {
                    "match_id":  match_id,
                    "team_canonical": team_canonical,
                    "is_home":   side == "home",
                    "source":    "fdco",
                }
                for stat_name, col in stat_col_map[side].items():
                    stat_record[stat_name] = (
                        _safe_int(row.get(col)) if col in df.columns else None
                    )
                records.append(stat_record)

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_season(
        self, season_id: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load one season from raw CSV and return normalised DataFrames.

        Downloads the CSV if not already present.

        Returns
        -------
        matches_df     : aligned to matches table
        odds_df        : aligned to odds_snapshots table
        team_stats_df  : aligned to team_match_stats table
        """
        path = self.download_season(season_id)
        raw_df = self._read_csv(path)
        self._validate_required_columns(raw_df, path)

        log.info("Processing season %s (%d rows).", season_id, len(raw_df))

        matches_df = self._extract_matches(raw_df, season_id)
        log.info("  Extracted %d match records.", len(matches_df))

        odds_df = self._extract_odds(raw_df, matches_df, season_id)
        log.info("  Extracted %d odds snapshot records.", len(odds_df))

        stats_df = self._extract_team_stats(raw_df, matches_df)
        log.info("  Extracted %d team stat records.", len(stats_df))

        return matches_df, odds_df, stats_df

    def load_all_seasons(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all configured seasons and return concatenated DataFrames.

        Returns
        -------
        matches_df, odds_df, stats_df — all seasons combined.
        """
        all_matches, all_odds, all_stats = [], [], []

        for season_id in self.season_codes:
            try:
                m, o, s = self.load_season(season_id)
                all_matches.append(m)
                all_odds.append(o)
                all_stats.append(s)
            except Exception as e:
                log.error("Failed to load season %s: %s", season_id, e)

        matches_df = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()
        odds_df    = pd.concat(all_odds,   ignore_index=True) if all_odds    else pd.DataFrame()
        stats_df   = pd.concat(all_stats,  ignore_index=True) if all_stats   else pd.DataFrame()

        log.info(
            "Loaded all seasons: %d matches, %d odds rows, %d stat rows.",
            len(matches_df), len(odds_df), len(stats_df),
        )
        return matches_df, odds_df, stats_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_int(val) -> int | None:
    try:
        if pd.isna(val):
            return None
        return int(float(val))
    except (TypeError, ValueError):
        return None


def _safe_float(val) -> float | None:
    try:
        if pd.isna(val):
            return None
        f = float(val)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None
