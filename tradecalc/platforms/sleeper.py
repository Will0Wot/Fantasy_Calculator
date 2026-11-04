"""Sleeper leagues (public API, no login): https://docs.sleeper.com

  /league/<id>            settings, scoring_settings, roster_positions, season
  /league/<id>/users      team names (a team is a "roster" owned by a user)
  /league/<id>/rosters    players on each roster (reserve = IR), waiver position
  /league/<id>/matchups/<week>   pairings (matchup_id) and points, for every week of the season
  /players/nba            every NBA player (2.5 MB; cached a day)

Weeks: Sleeper's NBA weeks run Monday-Sunday from opening night (week 1 = opening day to the first
Sunday), the same as ESPN's; the regular season ends the week before `playoff_week_start`.
"""
from __future__ import annotations

import json
import time
from datetime import date, timedelta

import requests

from tradecalc.db import cache_get, cache_put
from tradecalc.platforms.base import (SLOT_ELIGIBILITY, LeagueNotFound, LeagueSettings, UnsupportedLeague,
                                      merge_scoring)
from tradecalc.platforms.matching import PlatformPlayer, match_players, stand_in_id

API = "https://api.sleeper.app/v1"

# Sleeper NBA scoring keys -> box-score stats. Keys not listed are reported, not silently dropped.
SLEEPER_SCORING = {
    "pts": "PTS", "reb": "REB", "ast": "AST", "stl": "STL", "blk": "BLK", "to": "TOV", "tov": "TOV",
    "fgm": "FGM", "fga": "FGA", "ftm": "FTM", "fta": "FTA", "tpm": "FG3M", "tpa": "FG3A", "3pm": "FG3M", "3pa": "FG3A",
    "fgmi": {"FGA": 1, "FGM": -1}, "ftmi": {"FTA": 1, "FTM": -1}, "tpmi": {"FG3A": 1, "FG3M": -1},
    "3pmi": {"FG3A": 1, "FG3M": -1}, "oreb": "OREB", "dreb": "DREB", "pf": "PF", "dd": "DD", "td": "TD", "qd": "QD",
}
SLEEPER_SLOTS = {"PG": "PG", "SG": "SG", "SF": "SF", "PF": "PF", "C": "C", "G": "G", "F": "F", "UTIL": "UTIL",
                 "SG/SF": "SG/SF", "G/F": "G/F", "PF/C": "PF/C", "F/C": "F/C"}
SLEEPER_STATUS = {"out": "OUT", "ir": "OUT", "inactive": "OUT", "suspended": "SUSPENSION",
                  "questionable": "DAY_TO_DAY", "doubtful": "DAY_TO_DAY", "day to day": "DAY_TO_DAY",
                  "dtd": "DAY_TO_DAY", "gtd": "DAY_TO_DAY", "probable": "ACTIVE"}


class SleeperClient:
    def __init__(self, conn, ttl_s: float = 3600, session=None):
        self.conn, self.ttl_s = conn, ttl_s
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = "tradecalc/0.1"

    def get(self, path: str, ttl_s: float | None = None, force: bool = False):
        url = f"{API}{path}"
        if not force:
            hit = cache_get(self.conn, url, self.ttl_s if ttl_s is None else ttl_s)
            if hit is not None:
                return hit
        delay = 1.0
        for _ in range(4):
            r = self.session.get(url, timeout=60)
            if r.status_code == 404:
                return None
            if r.status_code < 500 and r.status_code != 429:
                r.raise_for_status()
                data = r.json()
                cache_put(self.conn, url, data)
                return data
            time.sleep(delay)
            delay *= 2
        r.raise_for_status()


def settings_from_sleeper(lg: dict, n_weeks_default: int) -> LeagueSettings:
    sc = {k: float(v) for k, v in (lg.get("scoring_settings") or {}).items() if v}
    parts, ignored = [], {}
    for k, v in sc.items():
        if k in SLEEPER_SCORING:
            parts.append((SLEEPER_SCORING[k], v))
        else:
            ignored[k] = v
    notes = []
    if ignored:
        notes.append("Not modeled: " + ", ".join(f"{k} ({v:+g})" for k, v in sorted(ignored.items())))
    slots: dict[str, int] = {}
    bench = ir = 0
    for pos in lg.get("roster_positions") or []:
        if pos == "BN":
            bench += 1
        elif pos == "IR":
            ir += 1
        elif pos in SLEEPER_SLOTS:
            slots[SLEEPER_SLOTS[pos]] = slots.get(SLEEPER_SLOTS[pos], 0) + 1
        else:
            notes.append(f"Lineup slot {pos} isn't modeled; treated as a utility slot.")
            slots["UTIL"] = slots.get("UTIL", 0) + 1
    st = lg.get("settings") or {}
    playoff = int(st.get("playoff_week_start") or 0)
    weeks = playoff - 1 if playoff > 1 else n_weeks_default
    return LeagueSettings(
        platform="sleeper", league_id=str(lg["league_id"]), season=int(lg.get("season") or 0),
        name=lg.get("name") or f"Sleeper league {lg['league_id']}", num_teams=int(lg.get("total_rosters") or 0),
        scoring_type="points", scoring=merge_scoring(parts), ignored_scoring=ignored, slots=slots,
        slot_eligibility={k: SLOT_ELIGIBILITY.get(k, SLOT_ELIGIBILITY["UTIL"]) for k in slots},
        bench=bench, ir=ir, regular_season_weeks=weeks, tie_value=0.5, moves_per_week=None,
        lineup_lock="daily", playoffs=playoff > 1, notes=notes)


def sleeper_weeks(first_day: date, n_weeks: int) -> list[tuple[int, date, date]]:
    """Week 1 = opening day through the first Sunday, then Monday-Sunday."""
    out, start = [], first_day
    for n in range(1, n_weeks + 1):
        end = start + timedelta(days=6 - start.weekday())
        out.append((n, start, end))
        start = end + timedelta(days=1)
    return out


def load_sleeper(ref, conn, cfg, nba, force: bool = False, client: SleeperClient | None = None):
    """Sleeper league -> ingest.league.LeagueData, built on the NBA layer `nba` (ESPN's pool and schedule)."""
    from tradecalc.clock import league_today
    from tradecalc.ingest.league import LeagueData, Player, Team

    c = client or SleeperClient(conn)
    lg = c.get(f"/league/{ref.league_id}", force=force)
    if not lg:
        raise LeagueNotFound(f"Sleeper has no league {ref.league_id}.")
    if lg.get("sport") not in (None, "nba"):
        raise UnsupportedLeague(f"Sleeper league {ref.league_id} is a {lg.get('sport')} league, not basketball.")
    nba_year = int(str(cfg["league"]["nba_season"])[:4])
    if int(lg.get("season") or nba_year) != nba_year:
        raise UnsupportedLeague(f"Sleeper league {ref.league_id} is from the {lg.get('season')} season. Open this "
                                f"season's league (Sleeper gives each season's league a new ID).")
    meta = settings_from_sleeper(lg, int(cfg["league"].get("regular_season_weeks") or 24))
    meta.weeks = sleeper_weeks(nba.first_day, meta.regular_season_weeks)

    users = {u["user_id"]: u for u in (c.get(f"/league/{ref.league_id}/users", force=force) or [])}
    rosters = c.get(f"/league/{ref.league_id}/rosters", force=force) or []
    sp = c.get("/players/nba", ttl_s=86400, force=force) or {}
    rostered = {pid for r in rosters for pid in (r.get("players") or [])}
    plist = [PlatformPlayer(pid, (sp.get(pid) or {}).get("full_name") or pid, (sp.get(pid) or {}).get("team") or "",
                            (sp.get(pid) or {}).get("fantasy_positions") or [], (sp.get(pid) or {}).get("injury_status") or "")
             for pid in rostered]
    to_espn, missing = match_players(plist, nba.players)

    import copy

    players = {eid: copy.copy(p) for eid, p in nba.players.items()}
    for p in players.values():
        p.on_team_id, p.lineup_slot, p.pool_status = 0, None, "FREEAGENT"
    for pl in missing:                                       # not in ESPN's pool: a stand-in
        eid = stand_in_id(pl.pid)
        to_espn[pl.pid] = eid
        players[eid] = Player(eid, pl.name, (pl.team or "FA").upper(),
                              [x for x in pl.positions if x in ("PG", "SG", "SF", "PF", "C")],
                              SLEEPER_STATUS.get(str(pl.status).lower(), "ACTIVE"))
    if missing:
        meta.notes.append(f"{len(missing)} rostered player(s) aren't in ESPN's player pool, so they have no "
                          f"preseason projection: " + ", ".join(p.name for p in missing[:5]) + ("…" if len(missing) > 5 else ""))
    teams = {}
    for r in rosters:
        tid = int(r["roster_id"])
        u = users.get(r.get("owner_id")) or {}
        name = ((u.get("metadata") or {}).get("team_name") or u.get("display_name") or f"Team {tid}").strip()
        ir = set(r.get("reserve") or [])
        ids = []
        for pid in r.get("players") or []:
            eid = to_espn.get(pid)
            if eid is None:
                continue
            p = players[eid]
            p.on_team_id, p.lineup_slot, p.pool_status = tid, ("IR" if pid in ir else None), "ONTEAM"
            pos = [x for x in ((sp.get(pid) or {}).get("fantasy_positions") or []) if x in ("PG", "SG", "SF", "PF", "C")]
            if pos:
                p.positions = pos                             # eligibility as this league has it
            ids.append(eid)
        teams[tid] = Team(tid, name, ids, (r.get("settings") or {}).get("waiver_position"))

    # every week's pairings and points; the current week's points so far count as scored yesterday
    today = league_today(cfg)
    matchups = []
    for n, start, end in meta.weeks:
        rows = c.get(f"/league/{ref.league_id}/matchups/{n}", ttl_s=3600 if start <= today else 86400, force=force) or []
        by_m: dict = {}
        for row in rows:
            if row.get("matchup_id") is not None:
                by_m.setdefault(row["matchup_id"], []).append(row)
        for pair in by_m.values():
            if len(pair) != 2:
                continue
            h, a = sorted(pair, key=lambda x: x["roster_id"])
            ht, at = float(h.get("points") or 0), float(a.get("points") or 0)
            last = min(end, today - timedelta(days=1))
            period = (last - nba.first_day).days + 1
            done = end < today
            matchups.append({
                "week": n, "home": int(h["roster_id"]), "away": int(a["roster_id"]),
                "home_points": {period: ht} if start < today and ht else {},
                "away_points": {period: at} if start < today and at else {},
                "winner": ("HOME" if ht > at else "AWAY" if at > ht else "TIE") if done else "UNDECIDED",
                "home_total": ht, "away_total": at})
    return LeagueData(lg, teams, players, matchups, nba.pro_games, nba.pro_teams, nba.first_day, meta)
