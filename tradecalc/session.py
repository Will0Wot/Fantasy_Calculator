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
    from tradecalc.seed import ensure_seed

    ensure_seed(conn)  # finished seasons ship as files in data/seed/ (the cache itself isn't committed)
    return cfg, conn


def espn_client(cfg, conn):
    from tradecalc.ingest.league import EspnClient

    lg = cfg["league"]
    return EspnClient(conn, lg["league_id"], lg["season"], cfg["data"]["espn_cache_ttl_hours"])


def load_state(force: bool = False, today: date | str | None = None, warn=None, ref=None, creds=None):
    """Refresh what is stale (or everything with force=True) and build the LeagueState.

    ref: platforms.LeagueRef (default: the home league in league.yaml); creds: platforms.Credentials for a
    private league. The league's own settings (scoring, lineup slots, weeks...) come from its platform."""
    from tradecalc.ingest import nba
    from tradecalc.ingest.injuries import fetch_injuries
    from tradecalc.model.values import build_state
    from tradecalc.platforms import UnsupportedLeague, home_ref, league_config
    from tradecalc.platforms import load as load_league

    base, conn = context()
    ref = ref or home_ref(base)
    league = load_league(ref, conn, base, creds, force=force)
    meta = league.meta
    if meta is not None and meta.scoring_type != "points":
        raise UnsupportedLeague(
            f"{meta.name} is a {meta.scoring_type} league. The app models points leagues only so far "
            "(head-to-head points); category and roto leagues are on the roadmap.")
    cfg = league_config(base, meta) if meta is not None else base
    lg = cfg["league"]
    interval = cfg["data"]["nba_min_request_interval_s"]
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
    state = build_state(cfg, conn, league, today, injuries)
    # Second pass: projected minutes and role from the projection engine (depth chart, last
    # season, ESPN projected minutes, this season's games) feed each player's projection.
    try:
        from tradecalc.sim.engine import minutes_projection
        from tradecalc.sim.live import live_world
        from tradecalc.sim.params import PARAMS_PATH

        if PARAMS_PATH.exists():
            world, coaches_now = live_world(conn, cfg, state)
            state = build_state(cfg, conn, league, today, injuries, minutes_proj=minutes_projection(world))
            state.world = (world, coaches_now)
    except Exception as e:  # noqa: BLE001 - projections still work without the minutes view
        (warn or (lambda m: print(m, file=sys.stderr)))(f"Minutes/role projections unavailable ({e}).")
    if meta is not None:
        for n in meta.notes:
            (warn or (lambda m: print(m, file=sys.stderr)))(n)
    return state
