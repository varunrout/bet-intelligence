"""
Team name resolver for Football Market Intelligence System.

Provides a single source of truth for mapping raw team name strings
(from any data source) to the canonical team names and IDs used in the database.

Design rules:
  - Never silently fail. If a name cannot be resolved, raise ValueError.
  - The resolver is immutable after construction — aliases drive everything.
  - Thread-safe for read operations (no shared mutable state after __init__).
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


class TeamResolver:
    """
    Resolves raw team name strings to canonical names and team IDs.

    Parameters
    ----------
    aliases_path : Path
        Path to team_aliases.yaml. Defaults to config/team_aliases.yaml.

    Examples
    --------
    >>> resolver = TeamResolver()
    >>> resolver.resolve("Man United", source="fdco")
    'Manchester United'
    >>> resolver.resolve("Manchester Utd", source="api_football")
    'Manchester United'
    """

    def __init__(self, aliases_path: Path | None = None) -> None:
        if aliases_path is None:
            aliases_path = (
                Path(__file__).resolve().parents[2] / "config" / "team_aliases.yaml"
            )

        with open(aliases_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        teams: dict[str, dict] = raw.get("teams", {})

        # Build lookup: (source, raw_name) -> canonical_name
        self._lookup: dict[tuple[str, str], str] = {}

        # Also track canonical -> team info for later use
        self._canonical_info: dict[str, dict] = {}

        for canonical, info in teams.items():
            self._canonical_info[canonical] = info

            for source_key in ("fdco", "api_football", "understat"):
                raw_name = info.get(source_key)
                if raw_name:
                    self._lookup[(source_key, raw_name.strip())] = canonical

            # Always allow the canonical name to resolve to itself
            self._lookup[("canonical", canonical.strip())] = canonical
            self._lookup[("fdco", canonical.strip())] = canonical
            self._lookup[("api_football", canonical.strip())] = canonical

        log.debug(
            "TeamResolver loaded: %d canonical teams, %d alias entries.",
            len(self._canonical_info),
            len(self._lookup),
        )

    def resolve(self, raw_name: str, source: str) -> str:
        """
        Resolve a raw team name from a given source to the canonical name.

        Parameters
        ----------
        raw_name : str
            Team name as it appears in the data source.
        source : str
            One of 'fdco', 'api_football', or 'canonical'.

        Returns
        -------
        str
            Canonical team name.

        Raises
        ------
        ValueError
            If the name cannot be resolved. Fix by adding an alias to
            config/team_aliases.yaml.
        """
        key = (source, raw_name.strip())
        result = self._lookup.get(key)

        if result is None:
            raise ValueError(
                f"Cannot resolve team '{raw_name}' from source '{source}'. "
                f"Add an alias entry to config/team_aliases.yaml."
            )

        return result

    def resolve_series(self, raw_names, source: str):
        """
        Resolve a pandas Series or list of raw team names.

        Returns a list of canonical names in the same order.
        Raises ValueError on the first unresolvable name.
        """
        return [self.resolve(name, source) for name in raw_names]

    def get_short_code(self, canonical_name: str) -> str:
        """Return the 3-letter short code for a canonical team name."""
        info = self._canonical_info.get(canonical_name)
        if info is None:
            raise ValueError(f"Unknown canonical team: '{canonical_name}'")
        return info.get("short_code", canonical_name[:3].upper())

    def all_canonical_names(self) -> list[str]:
        """Return sorted list of all canonical team names."""
        return sorted(self._canonical_info.keys())

    def fdco_to_canonical_map(self) -> dict[str, str]:
        """Return a dict of {fdco_name: canonical_name} for all teams."""
        return {
            raw: canonical
            for (source, raw), canonical in self._lookup.items()
            if source == "fdco"
        }
