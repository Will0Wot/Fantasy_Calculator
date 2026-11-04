"""A demo league for trying the app without one: 12 teams drafted from ESPN's projections, a points league
with ESPN's default scoring and lineup, real NBA games, and past weeks scored from real box scores.

It runs through the same code as any league, so it's also how the app is tested away from the home league.
"""
from __future__ import annotations

import copy
from datetime import date, timedelta

import numpy as np

from tradecalc.platforms.base import SLOT_ELIGIBILITY, LeagueSettings

DEMO_ID = "demo"
TEAM_NAMES = ["Fast Breakers", "Glass Cleaners", "Pick and Rollers", "Corner Threes", "Rim Protectors",
              "Bench Mob", "Triple Threats", "Full Court Press", "Splash Zone", "Paint Beasts", "Box Outs", "And-Ones"]
ESPN_DEFAULT_SCORING = {"PTS": 1, "FG3M": 1, "FGA": -1, "FGM": 2, "FTA": -1, "FTM": 1, "REB": 1, "AST": 2,
                        "STL": 4, "BLK": 4, "TOV": -2}
SLOTS = {"PG": 1, "SG": 1, "SF": 1, "PF": 1, "C": 1, "G": 1, "F": 1, "UTIL": 3}


def load_demo(ref, conn, cfg, nba, n_teams: int = 12, roster_size: int = 13, seed: int = 7):
    from tradecalc.clock import league_today
    from tradecalc.ingest import nba as nba_logs
    from tradecalc.ingest.league import LeagueData, Team
    from tradecalc.model.names import norm_name
    from tradecalc.model.scoring import add_fp, fantasy_points
    from tradecalc.schedule.calendar import build_weeks
    from tradecalc.sim.forecast_test import round_robin

    n_weeks = int(cfg["league"].get("regular_season_weeks") or 24)
    meta = LeagueSettings(platform="demo", league_id=DEMO_ID, season=int(cfg["league"]["season"]), name="Demo League",
                          num_teams=n_teams, scoring=dict(ESPN_DEFAULT_SCORING), slots=dict(SLOTS),
                          slot_eligibility={k: SLOT_ELIGIBILITY[k] for k in SLOTS}, bench=3, ir=1,
                          regular_season_weeks=n_weeks, moves_per_week=1,
                          notes=["This is a demo league: teams were drafted by the computer from ESPN's projections."])
    players = {eid: copy.copy(p) for eid, p in nba.players.items()}
    value = {eid: fantasy_points(p.projection, meta.scoring) * min(p.projection.get("GP", 0), 82) / 82
             for eid, p in players.items() if p.projection.get("GP") and p.pro_team not in ("FA", "")}
    board = sorted(value, key=lambda e: -value[e])
    rng = np.random.default_rng(seed)
    rosters: list[list[int]] = [[] for _ in range(n_teams)]
    for rnd in range(roster_size):                       # snake draft, each pick one of the best three left
        order = range(n_teams) if rnd % 2 == 0 else range(n_teams - 1, -1, -1)
        for t in order:
            k = min(3, len(board))
            probs = np.array([0.6, 0.25, 0.15][:k])
            rosters[t].append(board.pop(int(rng.choice(k, p=probs / probs.sum()))))
    for p in players.values():
        p.on_team_id, p.lineup_slot, p.pool_status = 0, None, "FREEAGENT"
    teams = {}
    for t, ids in enumerate(rosters, start=1):
        for eid in ids:
            players[eid].on_team_id, players[eid].pool_status = t, "ONTEAM"
        teams[t] = Team(t, TEAM_NAMES[(t - 1) % len(TEAM_NAMES)], ids, waiver_rank=t)

    games = {g["date"] for g in nba.pro_games}
    last_day = max(games)
    weeks = build_weeks(nba.first_day, last_day, n_weeks, games)
    meta.weeks = [(w.number, w.start, w.end) for w in weeks]
    sched = round_robin(n_teams, n_weeks)
    # past weeks scored from real box scores: every rostered player's fantasy points that week
    # (13 players for 10 daily spots rarely leaves points on the bench)
    today = league_today(cfg)
    logs = nba_logs.game_logs(conn, cfg["league"]["nba_season"])
    team_of = {norm_name(players[e].name): t for t, ids in enumerate(rosters, start=1) for e in ids}
    if len(logs):
        logs = add_fp(logs, meta.scoring)
        logs = logs.assign(team_id=logs["player_name"].map(norm_name).map(team_of))
        logs = logs[logs.team_id.notna()]
    matchups = []
    for w, pairs in zip(weeks, sched):
        for h, a in pairs:
            h, a = h + 1, a + 1
            pts = {}
            for side in (h, a):
                if not len(logs) or w.start >= today:
                    pts[side] = {}
                    continue
                wk = logs[(logs.team_id == side) & (logs.game_date >= w.start.isoformat())
                          & (logs.game_date <= min(w.end, today - timedelta(days=1)).isoformat())]
                day = wk.groupby("game_date")["fp"].sum()
                pts[side] = {(date.fromisoformat(str(d)[:10]) - nba.first_day).days + 1: float(v)
                             for d, v in day.items()}
            ht, at = sum(pts[h].values()), sum(pts[a].values())
            done = w.end < today
            matchups.append({"week": w.number, "home": h, "away": a, "home_points": pts[h], "away_points": pts[a],
                             "winner": ("HOME" if ht > at else "AWAY" if at > ht else "TIE") if done else "UNDECIDED",
                             "home_total": ht, "away_total": at})
    return LeagueData({"name": meta.name}, teams, players, matchups, nba.pro_games, nba.pro_teams, nba.first_day, meta)
