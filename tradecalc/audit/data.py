"""Data and fitted parameters for the multi-season audit.

Protocol: for a test season S, every parameter is fitted on the two seasons before S (and
calibration on S-1 only). Nothing from S is used except as data "known so far" at each
prediction date.
"""
from __future__ import annotations

import pandas as pd

from tradecalc.model.names import norm_name
from tradecalc.sim.history import POS_GROUP, load_history, prev_season

AUDIT_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
TEST_SEASONS = ["2023-24", "2024-25", "2025-26"]


def espn_year(season: str) -> int:
    return int(season[:4]) + 1


def historical_positions(client, years) -> dict[str, str]:
    """Position group (G/F/C) per player from ESPN's pools for past seasons (newest wins)."""
    from tradecalc.ingest.league import ESPN_SLOT

    pos = {}
    for y in sorted(years):
        pool = client.player_pool(limit=600, season=y, league_scoped=False)
        for e in pool.get("players", []):
            p = e["player"]
            ps = [ESPN_SLOT[s] for s in p.get("eligibleSlots", []) if ESPN_SLOT.get(s) in POS_GROUP]
            if ps:
                pos[norm_name(p["fullName"])] = "C" if "C" in ps else POS_GROUP[ps[0]]
    return pos


def audit_history(conn, cfg, client):
    """5-season history with positions from current and historical ESPN pools."""
    from tradecalc.ingest.league import load_league

    league = load_league(client)
    H = load_history(conn, cfg["scoring"], list(league.players.values()), AUDIT_SEASONS, current_season=None)
    hist = historical_positions(client, range(2022, 2027))
    for k, v in hist.items():
        H.pos.setdefault(k, v)
    return H


def coaches_for(conn, client):
    from tradecalc.sim.live import coaches_table

    pro = {t["id"]: t["abbrev"].upper() for t in client.season_info()["settings"]["proTeams"]}
    return coaches_table(conn, pro, list(range(2022, 2027)))


def fit_engine(H, coaches, test_season: str, log=print) -> dict:
    """Projection-engine parameters for testing `test_season`, fitted only on earlier seasons."""
    from tradecalc.sim.params import fit_noise, fit_params
    from tradecalc.sim.validate import fit_calibration

    s1 = prev_season(test_season)
    s0 = prev_season(s1)
    log(f"  fitting engine on {s0} + {s1}")
    p = fit_params(H, coaches, [s0, s1])
    p.update(fit_noise(H, p, s1, s0))
    mondays = [d for d in sorted(H.team_logs[H.team_logs.season == s1].game_date.unique())
               if pd.Timestamp(d).weekday() == 0][4::5]
    p.update(fit_calibration(H, p, s1, s0, mondays))
    p["form_sd"] = 0.18
    return p
