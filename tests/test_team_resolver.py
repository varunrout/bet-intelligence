"""
Tests for TeamResolver — entity matching and alias resolution.

Run with: pytest tests/test_team_resolver.py -v
"""

from __future__ import annotations

import pytest

from src.transform.team_resolver import TeamResolver


@pytest.fixture(scope="module")
def resolver():
    return TeamResolver()


class TestFDCOResolution:
    """Test resolution from football-data.co.uk names."""

    def test_resolves_man_united(self, resolver):
        assert resolver.resolve("Man United", source="fdco") == "Manchester United"

    def test_resolves_man_city(self, resolver):
        assert resolver.resolve("Man City", source="fdco") == "Manchester City"

    def test_resolves_nottm_forest(self, resolver):
        assert resolver.resolve("Nott'm Forest", source="fdco") == "Nottingham Forest"

    def test_resolves_wolves(self, resolver):
        assert resolver.resolve("Wolves", source="fdco") == "Wolverhampton Wanderers"

    def test_resolves_arsenal(self, resolver):
        assert resolver.resolve("Arsenal", source="fdco") == "Arsenal"

    def test_resolves_west_brom(self, resolver):
        assert resolver.resolve("West Brom", source="fdco") == "West Bromwich Albion"

    def test_resolves_tottenham(self, resolver):
        assert resolver.resolve("Tottenham", source="fdco") == "Tottenham Hotspur"

    def test_resolves_leeds(self, resolver):
        assert resolver.resolve("Leeds", source="fdco") == "Leeds United"

    def test_resolves_leicester(self, resolver):
        assert resolver.resolve("Leicester", source="fdco") == "Leicester City"


class TestUnresolvableRaisesError:
    """Unresolvable names must raise ValueError — never silently pass."""

    def test_raises_on_unknown_name(self, resolver):
        with pytest.raises(ValueError, match="Cannot resolve team"):
            resolver.resolve("Some Unknown FC", source="fdco")

    def test_raises_on_wrong_source(self, resolver):
        # Manchester City's API-Football name may differ from FDCO name
        # but the canonical name should always work
        with pytest.raises(ValueError):
            resolver.resolve("Man City", source="completely_wrong_source_type")

    def test_raises_on_empty_string(self, resolver):
        with pytest.raises(ValueError):
            resolver.resolve("", source="fdco")


class TestCanonicalResolution:
    """Canonical names should resolve to themselves."""

    def test_canonical_resolves_to_self(self, resolver):
        for name in resolver.all_canonical_names():
            result = resolver.resolve(name, source="canonical")
            assert result == name, f"Canonical name '{name}' did not resolve to itself."

    def test_all_canonical_names_non_empty(self, resolver):
        names = resolver.all_canonical_names()
        assert len(names) > 10
        for name in names:
            assert len(name) > 0


class TestShortCodes:
    """Short codes must be present for all teams."""

    def test_short_code_man_city(self, resolver):
        assert resolver.get_short_code("Manchester City") == "MCI"

    def test_short_code_arsenal(self, resolver):
        assert resolver.get_short_code("Arsenal") == "ARS"

    def test_short_code_raises_on_unknown(self, resolver):
        with pytest.raises(ValueError):
            resolver.get_short_code("Fake FC")


class TestFDCOMap:
    """FDCO-to-canonical map covers all expected teams."""

    def test_fdco_map_contains_expected_aliases(self, resolver):
        fdco_map = resolver.fdco_to_canonical_map()
        assert "Man United" in fdco_map
        assert "Man City" in fdco_map
        assert "Nott'm Forest" in fdco_map

    def test_fdco_map_all_values_are_canonical(self, resolver):
        fdco_map = resolver.fdco_to_canonical_map()
        canonical_set = set(resolver.all_canonical_names())
        for raw, canonical in fdco_map.items():
            assert canonical in canonical_set, (
                f"FDCO alias '{raw}' maps to '{canonical}' which is not in canonical set."
            )
