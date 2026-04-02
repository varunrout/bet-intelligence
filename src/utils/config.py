"""
Central configuration loader for Football Market Intelligence System.

All path and environment config flows through here.
No other module should read .env or os.environ directly.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project root — resolved relative to this file so it works regardless of
# where scripts are invoked from.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Directory constants
# ---------------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_FDCO_DIR = RAW_DIR / "fdco"
RAW_API_FOOTBALL_DIR = RAW_DIR / "api_football"
RAW_ODDS_API_DIR = RAW_DIR / "odds_api"
PROCESSED_DIR = DATA_DIR / "processed"
DB_DIR = DATA_DIR / "db"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
REPORTS_DIR = PROJECT_ROOT / "reports"
CONFIG_DIR = PROJECT_ROOT / "config"

# Ensure critical directories exist at import time
for _d in [RAW_FDCO_DIR, RAW_API_FOOTBALL_DIR, RAW_ODDS_API_DIR,
           PROCESSED_DIR, DB_DIR, MODELS_DIR, REPORTS_DIR / "figures"]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DB_PATH = DB_DIR / "bet_intelligence.duckdb"

# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------

API_FOOTBALL_KEY: str = os.getenv("API_FOOTBALL_KEY", "")
ODDS_API_KEY: str = os.getenv("ODDS_API_KEY", "")

# ---------------------------------------------------------------------------
# Config file loaders
# ---------------------------------------------------------------------------


def load_league_config() -> dict:
    """Return the full league_config.yaml as a dict."""
    with open(CONFIG_DIR / "league_config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_team_aliases() -> dict:
    """Return the team_aliases.yaml as a dict keyed by canonical name."""
    with open(CONFIG_DIR / "team_aliases.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("teams", {})


def get_competition_config(competition_key: str = "epl") -> dict:
    """
    Return config block for one competition.

    Parameters
    ----------
    competition_key : str
        Key as defined in league_config.yaml, e.g. 'epl'.
    """
    cfg = load_league_config()
    comps = cfg.get("competitions", {})
    if competition_key not in comps:
        available = list(comps.keys())
        raise KeyError(
            f"Competition '{competition_key}' not found in league_config.yaml. "
            f"Available: {available}"
        )
    return comps[competition_key]


def get_season_fdco_codes(competition_key: str = "epl") -> dict[str, str]:
    """
    Return a mapping of season_id -> fdco_season code.

    Example return: {'2019/20': '1920', '2020/21': '2021', ...}
    """
    comp = get_competition_config(competition_key)
    return {s["id"]: s["fdco_season"] for s in comp["seasons"]}
