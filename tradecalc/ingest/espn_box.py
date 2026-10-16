"""Current-season box scores from ESPN's public site API.

Fallback for when stats.nba.com is unreachable (it often blocks cloud servers). Rows are
written into the same game_logs table, in the same format, so nothing downstream changes:
  - player ids: matched to stats.nba.com ids by name; unmatched players (e.g. rookies who
    have not appeared in nba_api data yet) get -<ESPN id>
  - game ids are prefixed "E" so they never collide with stats.nba.com ids
One scoreboard request per day plus one summary request per finished game.
"""
from __future__ import annotations

import sqlite3
import time
from datetime import date, timedelta

import pandas as pd
import requests

from tradecalc.db import GAME_LOG_COLS, cache_get, cache_put
from tradecalc.model.names import norm_name
from tradecalc.teams import to_nba

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={d}"
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={e}"
_H = {"User-Agent": "tradecalc/0.1"}
API_CALLS = 0


def _get(conn, url: str, ttl_s: float | None):
    global API_CALLS
    hit = cache_get(conn, url, ttl_s)
    if hit is not None:
        return hit
    for attempt in range(3):
        API_CALLS += 1
        r = requests.get(url, headers=_H, timeout=30)
        if r.ok:
            data = r.json()
            cache_put(conn, url, data)
            return data
        time.sleep(1 + attempt)
    r.raise_for_status()


def _split(s: str) -> tuple[float, float]:
    try:
        a, b = s.split("-")
        return float(a), float(b)
    except (ValueError, AttributeError):
        return 0.0, 0.0


def _num(s) -> float:
    try:
        return float(str(s).replace("+", ""))
    except ValueError:
        return 0.0


def game_rows(summary: dict, game_day: date, id_by_name: dict[str, int]) -> list[dict]:
    """Player rows (game_logs format) from one ESPN game summary."""
    comp = summary["header"]["competitions"][0]
    side = {c["team"]["abbreviation"]: c["homeAway"] for c in comp["competitors"]}
    teams = [to_nba(a) for a in side]
    rows = []
    for block in summary.get("boxscore", {}).get("players", []):
        espn_abbr = block["team"]["abbreviation"]
        team = to_nba(espn_abbr)
        opp = next(t for t in teams if t != team)
        matchup = f"{team} vs. {opp}" if side.get(espn_abbr) == "home" else f"{team} @ {opp}"
        for stats in block.get("statistics", []):
            idx = {lab: i for i, lab in enumerate(stats["labels"])}
            for a in stats.get("athletes", []):
                s = a.get("stats") or []
                if a.get("didNotPlay") or not s:
                    continue
                mins = _num(s[idx["MIN"]])
                if mins <= 0:
                    continue
                fgm, fga = _split(s[idx["FG"]])
                fg3m, fg3a = _split(s[idx["3PT"]])
                ftm, fta = _split(s[idx["FT"]])
                name = a["athlete"]["displayName"]
                pid = id_by_name.get(norm_name(name), -int(a["athlete"]["id"]))
                rows.append({
                    "game_id": f"E{summary['header']['id']}", "player_id": pid, "player_name": name, "team": team,
                    "game_date": game_day.isoformat(), "matchup": matchup, "min": mins,
                    "pts": _num(s[idx["PTS"]]), "reb": _num(s[idx["REB"]]), "ast": _num(s[idx["AST"]]),
                    "stl": _num(s[idx["STL"]]), "blk": _num(s[idx["BLK"]]), "tov": _num(s[idx["TO"]]),
                    "fgm": fgm, "fga": fga, "ftm": ftm, "fta": fta, "fg3m": fg3m, "fg3a": fg3a,
                    "oreb": _num(s[idx["OREB"]]), "dreb": _num(s[idx["DREB"]]), "pf": _num(s[idx["PF"]]),
                })
    return rows


def refresh_from_espn(conn: sqlite3.Connection, season: str, first_day: date, today: date | None = None) -> dict:
    """Fill game_logs for `season` from ESPN, for every finished game from the day after the
    last cached ESPN date (or the season start) through yesterday."""
    today = today or date.today()
    last = conn.execute("SELECT MAX(game_date) d FROM game_logs WHERE season=?", (season,)).fetchone()["d"]
    if last:
        # re-fetch the last day only if ESPN wrote it (it may have been partial); days that
        # stats.nba.com already covered are never fetched again from ESPN
        nba_rows = conn.execute("SELECT COUNT(*) c FROM game_logs WHERE season=? AND game_date=? "
                                "AND game_id NOT LIKE 'E%'", (season, last)).fetchone()["c"]
        start = date.fromisoformat(last) + timedelta(days=1 if nba_rows else 0)
        start = max(start, first_day)
    else:
        start = first_day
    names = conn.execute("SELECT player_name, player_id FROM game_logs WHERE player_id > 0 "
                         "GROUP BY player_id").fetchall()
    id_by_name = {norm_name(n): pid for n, pid in names}
    cols = ["season", "game_id", "player_id", "player_name", "team", "game_date", "matchup", *GAME_LOG_COLS]
    added, d = 0, start
    while d < today:
        sb = _get(conn, SCOREBOARD.format(d=d.strftime("%Y%m%d")), None if d < today - timedelta(days=2) else 1800)
        for ev in sb.get("events", []):
            if ev.get("status", {}).get("type", {}).get("name") != "STATUS_FINAL":
                continue
            summ = _get(conn, SUMMARY.format(e=ev["id"]), None)
            rows = game_rows(summ, d, id_by_name)
            conn.executemany(
                f"INSERT OR REPLACE INTO game_logs({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                [(season, *[r[c] for c in cols[1:]]) for r in rows])
            added += len(rows)
        d += timedelta(days=1)
    conn.commit()
    return {"season": season, "fetched": added, "source": "ESPN", "from": str(start)}


def team_logs_from_players(conn: sqlite3.Connection, season: str) -> pd.DataFrame:
    """Team-game totals (LeagueGameLog 'T' format, lower-case) built from player rows."""
    g = pd.read_sql_query("SELECT * FROM game_logs WHERE season=?", conn, params=(season,))
    if g.empty:
        return g
    stat = ["min", "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf"]
    t = g.groupby(["game_id", "team"]).agg({**{c: "sum" for c in stat}, "game_date": "first", "matchup": "first"}).reset_index()
    opp_pts = t.groupby("game_id")["pts"].transform("sum") - t["pts"]
    t["plus_minus"] = t["pts"] - opp_pts
    t["wl"] = (t["plus_minus"] > 0).map({True: "W", False: "L"})
    t["team_abbreviation"] = t["team"]
    t["min"] = t["min"].round()
    return t.drop(columns=["team"])
