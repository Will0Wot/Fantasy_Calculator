"""The league's "today".

ESPN fantasy basketball and the NBA schedule run on US Eastern dates (a scoring period is an
Eastern-time day), so game dates are converted to Eastern time in ingest/league.py. "Today"
must use the same clock: hosting servers (e.g. Streamlit Community Cloud) run on UTC, where
the date flips at 8 pm Eastern while games are still being played.
"""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

DEFAULT_TZ = "America/New_York"


def league_tz(cfg=None) -> ZoneInfo:
    tz = DEFAULT_TZ
    if cfg is not None:
        tz = (cfg.get("league", {}) or {}).get("timezone", DEFAULT_TZ)
    return ZoneInfo(tz)


def league_today(cfg=None) -> date:
    """Current date in the league's timezone (Eastern by default), independent of the server clock."""
    return datetime.now(league_tz(cfg)).date()
