"""
Understat xG data loader for Football Market Intelligence System.

Fetches match-level expected goals (xG) for EPL seasons from understat.com
via their internal JSON API (discovered from the site's JS bundle).

How it works
------------
Understat serves fixture data via a GET endpoint:

    GET https://understat.com/getLeagueData/EPL/{year}

The request requires two headers:
  - ``X-Requested-With: XMLHttpRequest``  (without this the server returns 404)
  - ``Referer: https://understat.com/league/EPL/{year}``

The response is a JSON object with keys ``teams``, ``players``, and ``dates``.
The ``dates`` list contains one record per fixture, each with the structure:

    {
      "isResult": true,
      "h": {"title": "Arsenal"},
      "a": {"title": "Chelsea"},
      "xG": {"h": "1.45", "a": "0.87"},
      "datetime": "2022-08-05 19:00:00"
    }

Usage
-----
    from src.ingestion.understat_loader import UnderstatLoader

    loader = UnderstatLoader()
    xg_df  = loader.load_season("2022/23")
    # Returns DataFrame: match_id (str), home_xg (float), away_xg (float)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

from src.transform.team_resolver import TeamResolver
from src.utils.db import run_query

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Season ID  ->  start-year used in the Understat URL
SEASON_YEAR_MAP: dict[str, int] = {
    "2019/20": 2019,
    "2020/21": 2020,
    "2021/22": 2021,
    "2022/23": 2022,
    "2023/24": 2023,
}

_API_URL = "https://understat.com/getLeagueData/EPL/{year}"
_REFERER_URL = "https://understat.com/league/EPL/{year}"

# Understat's AJAX API requires these headers — without X-Requested-With it returns 404
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.9",
    "X-Requested-With": "XMLHttpRequest",
}

# Polite delay between HTTP requests (seconds)
REQUEST_DELAY: float = 2.5


# ---------------------------------------------------------------------------
# Public loader class
# ---------------------------------------------------------------------------


class UnderstatLoader:
    """
    Fetches EPL match-level xG data from understat.com for one or more seasons.

    Team name resolution is handled by TeamResolver using the 'understat' source
    key defined in config/team_aliases.yaml.

    Parameters
    ----------
    aliases_path : Path, optional
        Override path to team_aliases.yaml.  Defaults to the project-level config.
    """

    def __init__(self, aliases_path: Path | None = None) -> None:
        self._resolver = TeamResolver(aliases_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_season(self, season_id: str) -> pd.DataFrame:
        """
        Fetch xG for all played matches in *season_id* (e.g. ``"2022/23"``).

        Each match is resolved to a DB ``match_id`` by joining Understat team
        names (resolved to canonical form) against the matches table.

        Parameters
        ----------
        season_id : str
            Season in ``"YYYY/YY"`` format, e.g. ``"2019/20"``.

        Returns
        -------
        pd.DataFrame
            Columns: ``match_id`` (str), ``home_xg`` (float), ``away_xg`` (float).
            Rows with unresolvable team names or no DB match are skipped and
            logged as warnings.

        Raises
        ------
        ValueError
            If *season_id* is not in :data:`SEASON_YEAR_MAP`.
        requests.HTTPError
            If the HTTP request fails.
        """
        if season_id not in SEASON_YEAR_MAP:
            raise ValueError(
                f"Unknown season '{season_id}'. "
                f"Supported: {sorted(SEASON_YEAR_MAP)}"
            )

        year = SEASON_YEAR_MAP[season_id]
        log.info("Fetching Understat xG  season=%s  year=%d", season_id, year)

        raw_matches = self._fetch_season_json(year)
        log.info("  %d fixture records returned (includes unplayed).", len(raw_matches))

        parsed = self._parse_matches(raw_matches)
        log.info("  %d played matches with xG data.", len(parsed))

        xg_df = self._resolve_match_ids(parsed, season_id)
        log.info(
            "  %d/%d matches resolved to DB match_ids.",
            len(xg_df), len(parsed),
        )
        return xg_df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_season_json(self, year: int) -> list[dict]:
        """Call Understat's JSON API for the season — returns the 'dates' fixture list."""
        url = _API_URL.format(year=year)
        # Referer must match the league page for the request to be accepted
        headers = {**_HEADERS, "Referer": _REFERER_URL.format(year=year)}
        log.info("GET %s", url)
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        time.sleep(REQUEST_DELAY)
        data = resp.json()
        return data.get("dates", [])

    def _parse_matches(self, raw: list[dict]) -> list[dict]:
        """
        Filter to played fixtures and extract team names + xG values.

        Returns a list of dicts: home_name, away_name, home_xg, away_xg.
        """
        parsed = []
        for m in raw:
            if not m.get("isResult"):
                continue  # scheduled but not yet played

            xg_block = m.get("xG", {})
            try:
                home_xg = float(xg_block.get("h") or 0.0)
                away_xg = float(xg_block.get("a") or 0.0)
            except (ValueError, TypeError):
                log.warning(
                    "Cannot parse xG for match id=%s  home=%s  away=%s — skipping.",
                    m.get("id"), m.get("h", {}).get("title"), m.get("a", {}).get("title"),
                )
                continue

            parsed.append(
                {
                    "home_name": m["h"]["title"],
                    "away_name": m["a"]["title"],
                    "home_xg":   home_xg,
                    "away_xg":   away_xg,
                }
            )
        return parsed

    def _resolve_match_ids(
        self,
        parsed: list[dict],
        season_id: str,
    ) -> pd.DataFrame:
        """
        Join parsed xG rows to ``match_id`` values from the DB.

        Match key: (home_canonical, away_canonical) within the season — each
        pair plays exactly once at home per season so this is always unique.
        """
        # ── Build DB lookup: {(home_canonical, away_canonical): match_id} ──
        db_df = run_query(
            """
            SELECT m.match_id,
                   ht.name_canonical AS home_team,
                   awt.name_canonical AS away_team
            FROM matches m
            JOIN teams ht  ON m.home_team_id = ht.team_id
            JOIN teams awt ON m.away_team_id = awt.team_id
            WHERE m.season = ?
            """,
            params=[season_id],
        )

        if db_df.empty:
            log.warning("No matches in DB for season %s — cannot resolve match_ids.", season_id)
            return pd.DataFrame(columns=["match_id", "home_xg", "away_xg"])

        db_lookup: dict[tuple[str, str], str] = {
            (row.home_team, row.away_team): row.match_id
            for row in db_df.itertuples(index=False)
        }

        # ── Resolve each parsed row ──────────────────────────────────────
        resolved: list[dict] = []
        unresolved_names: set[str] = set()

        for row in parsed:
            # Resolve team names from Understat → canonical
            try:
                home_canon = self._resolver.resolve(row["home_name"], source="understat")
                away_canon = self._resolver.resolve(row["away_name"], source="understat")
            except ValueError as exc:
                unresolved_names.add(str(exc))
                continue

            match_id = db_lookup.get((home_canon, away_canon))
            if match_id is None:
                log.warning(
                    "No DB match for '%s' vs '%s' in season %s — skipping.",
                    home_canon, away_canon, season_id,
                )
                continue

            resolved.append(
                {
                    "match_id": match_id,
                    "home_xg":  row["home_xg"],
                    "away_xg":  row["away_xg"],
                }
            )

        if unresolved_names:
            log.error(
                "%d team name(s) could not be resolved.\n"
                "Add an 'understat' alias to config/team_aliases.yaml for each:\n  %s",
                len(unresolved_names),
                "\n  ".join(sorted(unresolved_names)),
            )

        if not resolved:
            return pd.DataFrame(columns=["match_id", "home_xg", "away_xg"])

        return pd.DataFrame(resolved)
