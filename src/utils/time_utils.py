"""
Time and date utilities for Football Market Intelligence System.

All datetimes stored in the database are UTC.
All conversions happen here — nowhere else.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd


# UK football is played in Europe/London timezone
_LONDON_TZ = ZoneInfo("Europe/London")

# Default kickoff time when only a date (no time) is available.
# 15:00 London time is the traditional Saturday 3pm slot.
# This is a fallback only — prefer explicit kickoff times wherever possible.
_DEFAULT_KICKOFF_HOUR = 15
_DEFAULT_KICKOFF_MINUTE = 0


def to_utc(dt: datetime) -> datetime:
    """
    Convert any aware datetime to UTC.
    If dt is naive, it is assumed to be in Europe/London timezone.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_LONDON_TZ)
    return dt.astimezone(timezone.utc)


def parse_fdco_datetime(date_str: str, time_str: str | None = None) -> datetime:
    """
    Parse a football-data.co.uk date (and optional time) into UTC datetime.

    football-data.co.uk date format: DD/MM/YYYY
    football-data.co.uk time format: HH:MM

    Parameters
    ----------
    date_str : str
        Date string in DD/MM/YYYY format.
    time_str : str | None
        Time string in HH:MM format, or None to use default kickoff time.

    Returns
    -------
    datetime (UTC, timezone-aware)
    """
    parsed_date = datetime.strptime(date_str.strip(), "%d/%m/%Y")

    if time_str and str(time_str).strip() not in ("", "nan", "NaN"):
        try:
            t = datetime.strptime(str(time_str).strip(), "%H:%M")
            local_dt = parsed_date.replace(
                hour=t.hour,
                minute=t.minute,
                tzinfo=_LONDON_TZ,
            )
        except ValueError:
            local_dt = parsed_date.replace(
                hour=_DEFAULT_KICKOFF_HOUR,
                minute=_DEFAULT_KICKOFF_MINUTE,
                tzinfo=_LONDON_TZ,
            )
    else:
        local_dt = parsed_date.replace(
            hour=_DEFAULT_KICKOFF_HOUR,
            minute=_DEFAULT_KICKOFF_MINUTE,
            tzinfo=_LONDON_TZ,
        )

    return local_dt.astimezone(timezone.utc)


def parse_fdco_datetime_series(
    date_series: pd.Series,
    time_series: pd.Series | None = None,
) -> pd.Series:
    """
    Vectorised version of parse_fdco_datetime for use on DataFrame columns.

    Returns a Series of UTC-aware datetime objects.
    """
    if time_series is None:
        time_series = pd.Series([""] * len(date_series), index=date_series.index)

    return pd.Series(
        [
            parse_fdco_datetime(d, t)
            for d, t in zip(date_series, time_series)
        ],
        index=date_series.index,
        dtype="datetime64[ns, UTC]",
    )


def days_between(earlier_utc: datetime, later_utc: datetime) -> int:
    """
    Return the number of whole days between two UTC datetimes.
    Order matters: later - earlier. Returns negative if earlier > later.
    """
    delta = later_utc - earlier_utc
    return delta.days


def is_before_kickoff(snapshot_utc: datetime, kickoff_utc: datetime) -> bool:
    """
    Return True if snapshot_utc is strictly before kickoff_utc.
    Used in leakage guards.
    """
    return snapshot_utc < kickoff_utc


def season_from_date(dt: datetime) -> str:
    """
    Infer season string (e.g. '2023/24') from a datetime.
    Football seasons in England run Aug–May.
    """
    year = dt.year
    month = dt.month
    if month >= 8:
        return f"{year}/{str(year + 1)[2:]}"
    else:
        return f"{year - 1}/{str(year)[2:]}"
