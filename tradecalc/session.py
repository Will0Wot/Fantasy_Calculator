"""Load everything the CLI and the Streamlit app need: config, cache, ESPN, injuries -> LeagueState."""
from __future__ import annotations

import sys
from datetime import date

from dotenv import load_dotenv

from tradecalc.config import load_config
from tradecalc.db import connect


def context(cfg_path=None):
    load_dotenv()
    cfg = load_config(cfg_path)
    conn = connect(cfg.path(cfg["data"]["cache_db"]))
    return cfg, conn


def espn_client(cfg, conn):
    from tradecalc.ingest.league import EspnClient

    lg = cfg["league"]
    return EspnClient(conn, lg["league_id"], lg["season"], cfg["data"]["espn_cache_ttl_hours"])


def load_state(force: bool = False, today: date | str | None = None, warn=None):
    """Refresh what is stale (or everything with force=True) and build the LeagueState."""
    from tradecalc.ingest import nba
    from tradecalc.ingest.injuries import fetch_injuries
    from tradecalc.ingest.league import load_league
    from tradecalc.model.values import build_state

    cfg, conn = context()
    lg = cfg["league"]
    interval = cfg["data"]["nba_min_request_interval_s"]
    league = load_league(espn_client(cfg, conn), force=force)
    nba.refresh_season(conn, lg["prev_nba_season"], complete=True, min_interval=interval)
    r = nba.refresh_season(conn, lg["nba_season"], complete=False, min_interval=interval, force=force,
                           first_day=league.first_day)
    if r.get("note") and warn:
        warn(r["note"])
    try:
        injuries = fetch_injuries(conn, force=force)
    except Exception as e:  # the injury report is an enhancement; fall back to fantasy status
        (warn or (lambda m: print(m, file=sys.stderr)))(
            f"ESPN injury report unavailable ({e}); using fantasy injury status only.")
        injuries = []
    if isinstance(today, str):
        today = date.fromisoformat(today)
    return build_state(cfg, conn, league, today, injuries)
