"""ESPN fantasy basketball league data via ESPN's public JSON API.

Uses raw HTTP rather than the espn-api package: one request per view, cached in SQLite.
Private leagues need ESPN_S2 and SWID in the environment (or a .env file).
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import requests

from tradecalc.db import cache_get, cache_put

BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{season}"
ET = ZoneInfo("America/New_York")  # NBA/ESPN game days are Eastern dates (see tradecalc/clock.py)

# ESPN stat ids -> stats.nba.com box-score column names.
ESPN_STAT = {
    0: "PTS", 1: "BLK", 2: "STL", 3: "AST", 4: "OREB", 5: "DREB", 6: "REB",
    11: "TOV", 13: "FGM", 14: "FGA", 15: "FTM", 16: "FTA", 17: "FG3M", 18: "FG3A",
    40: "MIN", 42: "GP",
}
# ESPN lineup slot ids.
ESPN_SLOT = {
    0: "PG", 1: "SG", 2: "SF", 3: "PF", 4: "C", 5: "G", 6: "F", 7: "SG/SF", 8: "G/F",
    9: "PF/C", 10: "F/C", 11: "UTIL", 12: "BE", 13: "IR",
}
POSITIONS = ("PG", "SG", "SF", "PF", "C")


@dataclass
class Player:
    espn_id: int
    name: str
    pro_team: str
    positions: list[str]
    injury_status: str
    on_team_id: int = 0                 # 0 = free agent
    lineup_slot: str | None = None
    adp: float | None = None
    draft_rank: int | None = None       # ESPN STANDARD draft rank
    projection: dict = field(default_factory=dict)  # season per-game projection, nba column names
    prev_projection: dict = field(default_factory=dict)
    outlook: str = ""                   # ESPN season outlook blurb
    pool_status: str = ""               # FREEAGENT (add now) | WAIVERS (claim) | ONTEAM


@dataclass
class Team:
    team_id: int
    name: str
    roster: list[int]                   # espn player ids
    waiver_rank: int | None = None      # 1 = first claim on a dropped player


@dataclass
class LeagueData:
    settings: dict                      # the platform's raw settings (ESPN's for an ESPN league)
    teams: dict[int, Team]
    players: dict[int, Player]          # rostered + top free agents
    matchups: list[dict]                # {"week", "home", "away"}
    pro_games: list[dict]               # {"game_id", "date", "scoring_period", "home", "away"}
    pro_teams: dict[int, str]           # ESPN pro team id -> abbrev
    first_day: date                     # date of scoring period 1
    meta: object = None                 # platforms.base.LeagueSettings (None: the NBA alone, no league)


class EspnClient:
    def __init__(self, conn: sqlite3.Connection, league_id: int, season: int, ttl_hours: float = 6,
                 cookies: dict | None = None, use_env: bool = True):
        """cookies: {"espn_s2": ..., "SWID": ...} for a private league (a visitor's own, kept only for this
        client). use_env: fall back to ESPN_S2 / SWID from the environment (the home league's)."""
        self.conn = conn
        self.league_id = league_id
        self.season = season
        self.ttl_s = ttl_hours * 3600
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "tradecalc/0.1"
        if not cookies and use_env and os.environ.get("ESPN_S2") and os.environ.get("SWID"):
            cookies = {"espn_s2": os.environ["ESPN_S2"], "SWID": os.environ["SWID"]}
        self.private = bool(cookies)
        # Responses fetched with someone's cookies are cached under a key that includes a hash of them,
        # so a private league's data is never served to a visitor without those cookies.
        self._cred = ""
        if cookies:
            import hashlib

            self.session.cookies.update(cookies)
            self._cred = hashlib.sha256(f"{cookies.get('espn_s2')}|{cookies.get('SWID')}".encode()).hexdigest()[:16]
        self.api_calls = 0

    def _get(self, url: str, params=None, fantasy_filter: dict | None = None, ttl_s=None, force=False):
        key = json.dumps([url, params, fantasy_filter] + ([self._cred] if self._cred else []), sort_keys=True)
        if not force:
            hit = cache_get(self.conn, key, self.ttl_s if ttl_s is None else ttl_s)
            if hit is not None:
                return hit
        headers = {"x-fantasy-filter": json.dumps(fantasy_filter)} if fantasy_filter else {}
        delay = 1.0
        for attempt in range(4):
            self.api_calls += 1
            r = self.session.get(url, params=params, headers=headers, timeout=60)
            if r.status_code in (401, 403):
                from tradecalc.platforms.base import LeagueNotFound

                raise LeagueNotFound(
                    "ESPN says this league is private. Add your espn_s2 and SWID cookies to open it."
                    if not self.private else "ESPN refused these cookies for this league: check espn_s2 and SWID.")
            if r.status_code == 404:
                from tradecalc.platforms.base import LeagueNotFound

                raise LeagueNotFound(f"ESPN has no basketball league {self.league_id} for this season.")
            if r.status_code < 500 and r.status_code != 429:
                r.raise_for_status()
                data = r.json()
                cache_put(self.conn, key, data)
                return data
            time.sleep(delay)
            delay *= 2
        r.raise_for_status()

    def league(self, views: list[str], force=False, season: int | None = None, fantasy_filter=None):
        url = f"{BASE.format(season=season or self.season)}/segments/0/leagues/{self.league_id}"
        return self._get(url, params=[("view", v) for v in views], fantasy_filter=fantasy_filter, force=force)

    def season_info(self, force=False, season: int | None = None):
        # NBA schedule changes rarely; cache for a day.
        return self._get(
            BASE.format(season=season or self.season),
            params={"view": "proTeamSchedules_wl"},
            ttl_s=86400,
            force=force,
        )

    def player_pool(self, limit=600, force=False, season: int | None = None, league_scoped=True):
        """Players sorted by % owned, with season projections only (keeps the payload small)."""
        flt = {
            "players": {
                "limit": limit,
                "sortPercOwned": {"sortPriority": 1, "sortAsc": False},
                "filterStatsForSourceIds": {"value": [1]},
                "filterStatsForSplitTypeIds": {"value": [0]},
            }
        }
        base = BASE.format(season=season or self.season)
        url = f"{base}/segments/0/leagues/{self.league_id}" if league_scoped else f"{base}/segments/0/leaguedefaults/1"
        return self._get(url, params={"view": "kona_player_info"}, fantasy_filter=flt, force=force)


def _projection(player: dict, season: int) -> dict:
    for s in player.get("stats", []):
        if s.get("statSourceId") == 1 and s.get("statSplitTypeId") == 0 and s.get("seasonId") == season:
            avg = s.get("averageStats") or {}
            return {ESPN_STAT[int(k)]: v for k, v in avg.items() if int(k) in ESPN_STAT}
    return {}


def _player_from_espn(p: dict, pro_teams: dict[int, str], season: int) -> Player:
    own = p.get("ownership") or {}
    ranks = p.get("draftRanksByRankType") or {}
    adp = own.get("averageDraftPosition")
    return Player(
        espn_id=p["id"],
        name=p["fullName"],
        pro_team=pro_teams.get(p.get("proTeamId"), "FA"),
        positions=[ESPN_SLOT[s] for s in p.get("eligibleSlots", []) if ESPN_SLOT.get(s) in POSITIONS],
        injury_status=p.get("injuryStatus") or "ACTIVE",
        adp=adp,
        draft_rank=(ranks.get("STANDARD") or {}).get("rank"),
        projection=_projection(p, season),
        prev_projection=_projection(p, season - 1),
        outlook=p.get("seasonOutlook") or "",
    )


def pro_schedule(client: EspnClient, force=False) -> tuple[dict[int, str], list[dict], date]:
    """The NBA schedule from ESPN (no league needed): pro team ids -> abbreviations, every game with
    its Eastern date and ESPN scoring period, and the date of scoring period 1."""
    season_raw = client.season_info(force=force)
    pro_teams = {t["id"]: t["abbrev"].upper() for t in season_raw["settings"]["proTeams"]}
    games, seen = [], set()
    for t in season_raw["settings"]["proTeams"]:
        for sp, gs in (t.get("proGamesByScoringPeriod") or {}).items():
            for g in gs:
                if g["id"] in seen:
                    continue
                seen.add(g["id"])
                games.append({
                    "game_id": g["id"],
                    "date": datetime.fromtimestamp(g["date"] / 1000, ET).date(),
                    "scoring_period": int(sp),
                    "home": pro_teams[g["homeProTeamId"]],
                    "away": pro_teams[g["awayProTeamId"]],
                })
    games.sort(key=lambda g: (g["scoring_period"], g["game_id"]))
    # Scoring periods are consecutive days; derive day 1 from any game.
    g0 = games[0]
    return pro_teams, games, g0["date"] - timedelta(days=g0["scoring_period"] - 1)


def _pool(client: EspnClient, pro_teams: dict, league_scoped: bool, force=False) -> dict[int, Player]:
    pool = client.player_pool(force=force, league_scoped=league_scoped)
    players: dict[int, Player] = {}
    for entry in pool.get("players", []):
        pl = _player_from_espn(entry["player"], pro_teams, client.season)
        pl.on_team_id = entry.get("onTeamId", 0) if league_scoped else 0
        pl.pool_status = (entry.get("status") or "") if league_scoped else "FREEAGENT"
        players[pl.espn_id] = pl
    return players


def nba_layer(client: EspnClient, force=False) -> LeagueData:
    """ESPN's NBA player pool (IDs, positions, injuries, projections, ADP) and schedule, with no fantasy
    league: what other platforms' leagues are built on (platforms/)."""
    pro_teams, games, first_day = pro_schedule(client, force)
    return LeagueData({}, {}, _pool(client, pro_teams, league_scoped=False, force=force), [], games, pro_teams, first_day)


def load_league(client: EspnClient, force=False) -> LeagueData:
    from tradecalc.platforms.espn import settings_from_espn

    raw = client.league(["mSettings", "mTeam", "mRoster", "mMatchup", "mMatchupScore"], force=force)
    pro_teams, games, first_day = pro_schedule(client, force)
    players = _pool(client, pro_teams, league_scoped=True, force=force)

    teams: dict[int, Team] = {}
    for t in raw["teams"]:
        ids = []
        for e in t["roster"]["entries"]:
            pe = e["playerPoolEntry"]["player"]
            pl = players.get(pe["id"]) or _player_from_espn(pe, pro_teams, client.season)
            pl.on_team_id = t["id"]
            pl.lineup_slot = ESPN_SLOT.get(e.get("lineupSlotId"))
            players[pl.espn_id] = pl
            ids.append(pl.espn_id)
        name = t.get("name") or f"{t.get('location', '')} {t.get('nickname', '')}".strip()
        teams[t["id"]] = Team(t["id"], name.strip(), ids, t.get("waiverRank"))

    def _pts(side: dict) -> dict[int, float]:
        return {int(k): float(v) for k, v in (side.get("pointsByScoringPeriod") or {}).items()}

    # points_by_period: fantasy points each team has scored per day (scoring period) so far
    matchups = [
        {"week": m["matchupPeriodId"], "home": m["home"]["teamId"], "away": m.get("away", {}).get("teamId"),
         "home_points": _pts(m["home"]), "away_points": _pts(m.get("away") or {}),
         "winner": m.get("winner", "UNDECIDED"),
         "home_total": float(m["home"].get("totalPoints") or 0), "away_total": float((m.get("away") or {}).get("totalPoints") or 0)}
        for m in raw.get("schedule", [])
    ]
    meta = settings_from_espn(raw, client.league_id, client.season)
    return LeagueData(raw["settings"], teams, players, matchups, games, pro_teams, first_day, meta)


def check_settings_against_config(settings: dict, cfg) -> list[str]:
    """Compare ESPN's settings to league.yaml. Returns a list of mismatch messages."""
    problems = []
    espn_scoring = {
        ESPN_STAT.get(i["statId"], f"espn_stat_{i['statId']}"): i["points"]
        for i in settings["scoringSettings"]["scoringItems"]
    }
    cfg_scoring = {k: float(v) for k, v in cfg["scoring"].items() if v}
    if {k: float(v) for k, v in espn_scoring.items() if v} != cfg_scoring:
        problems.append(f"scoring differs: ESPN={espn_scoring} config={cfg_scoring}")

    counts = {ESPN_SLOT[int(k)]: v for k, v in settings["rosterSettings"]["lineupSlotCounts"].items() if v}
    cfg_counts = dict(cfg["roster"]["slots"])
    cfg_counts.update({"BE": cfg["roster"]["bench"], "IR": cfg["roster"]["ir"]})
    if counts != {k: v for k, v in cfg_counts.items() if v}:
        problems.append(f"roster slots differ: ESPN={counts} config={cfg_counts}")

    weeks = settings["scheduleSettings"]["matchupPeriodCount"]
    if weeks != cfg["league"]["regular_season_weeks"]:
        problems.append(f"weeks differ: ESPN={weeks} config={cfg['league']['regular_season_weeks']}")
    tie = settings["scoringSettings"].get("matchupTieRule")
    if tie != "NONE" and cfg["league"]["tie_value"] == 0.5:
        problems.append(f"ESPN tie rule is {tie}; config assumes ties count 0.5")
    return problems
