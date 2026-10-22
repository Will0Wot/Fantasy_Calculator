"""NBA play-by-play from ESPN's public API, parsed into one row per event.

Used by the possession simulator (sim/possession/): where each player shoots from (5 zones), how
often he makes it, who assists him, rebounds, turnovers and steals, blocks, fouls and free throws.
ESPN works from cloud servers (stats.nba.com does not), so the same source serves past seasons
(backtests) and the live season (daily GitHub Actions run).

Storage: data/pbp/pbp_<season>.parquet (events) and data/pbp/players_<season>.parquet (who played
in each game: ESPN id, name, team, starter, minutes). Raw API responses are not kept.

Court coordinates: ESPN gives x 0-50 ft across the court and y = feet from the basket line, with
the basket at (25, 0); distance = hypot(x - 25, y), which matches the distance stated in the text.
Zones: RA (restricted area, <= 4 ft), PAINT (other shots in the lane: |x - 25| <= 8, y <= 14),
MID (other twos), C3 (corner three: y <= 9 and |x - 25| >= 20), AB3 (other threes).
"""
from __future__ import annotations

import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from tradecalc.teams import to_nba

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={d}"
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={e}"
_H = {"User-Agent": "Mozilla/5.0 (tradecalc play-by-play)"}
ZONES = ("RA", "PAINT", "MID", "C3", "AB3")
NO_COORD = -1_000_000

EVENT_COLS = ["game_id", "season", "date", "seq", "period", "clock", "team", "opp", "etype", "p1", "p2",
              "made", "pts", "three", "dist", "x", "y", "zone", "blocked", "ft_n", "ft_of", "type_id"]


def _get(url: str, tries: int = 4) -> dict | None:
    for k in range(tries):
        try:
            r = requests.get(url, headers=_H, timeout=30)
            if r.ok:
                return r.json()
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(1.5 * (k + 1))
    return None


def _clock(s: str) -> float:
    """Seconds left in the period from ESPN's clock string ('11:32' or '13.0')."""
    s = str(s or "0")
    if ":" in s:
        m, sec = s.split(":")
        return int(m) * 60 + float(sec)
    try:
        return float(s)
    except ValueError:
        return 0.0


def zone_of(three: bool, dist: float | None, x: float | None, y: float | None) -> str | None:
    if three:
        if x is not None and y is not None:
            return "C3" if (y <= 9 and abs(x - 25) >= 20) else "AB3"
        return "AB3"
    if dist is None:
        return None
    if dist <= 4:
        return "RA"
    if x is not None and y is not None:
        return "PAINT" if (abs(x - 25) <= 8 and y <= 14) else "MID"
    return "PAINT" if dist <= 14 else "MID"


_FOOT = re.compile(r"(\d+)-foot")
# turnovers ESPN logs without the word "turnover" (checked against box scores: exact match)
_TOV_WORDS = ("traveling", "palming", "double dribble", "discontinue", "backcourt", "shot clock",
              "offensive goaltending", "3 second", "3-second violation", "step out", "inbound", "lane violation",
              "jump ball violation", "basket from below", "swinging elbows")
_FT = re.compile(r"free throw (\d) of (\d)", re.I)


def parse_game(summary: dict, season: str, day: str) -> tuple[list[dict], list[dict]]:
    """(events, players) for one ESPN game summary."""
    comp = summary["header"]["competitions"][0]
    gid = f"E{summary['header']['id']}"
    abbr = {c["team"]["id"]: to_nba(c["team"]["abbreviation"]) for c in comp["competitors"]}
    ids = list(abbr)
    players = []
    for block in summary.get("boxscore", {}).get("players", []):
        team = to_nba(block["team"]["abbreviation"])
        for stats in block.get("statistics", []):
            idx = {lab: i for i, lab in enumerate(stats.get("labels", []))}
            for a in stats.get("athletes", []):
                if not (a.get("athlete") or {}).get("id"):
                    continue
                s = a.get("stats") or []
                mins = 0.0
                if s and "MIN" in idx:
                    try:
                        mins = float(s[idx["MIN"]])
                    except ValueError:
                        mins = 0.0
                players.append({"game_id": gid, "season": season, "date": day, "team": team,
                                "espn_id": int(a["athlete"]["id"]), "name": a["athlete"]["displayName"],
                                "starter": bool(a.get("starter")), "min": mins})
    events = []
    for k, p in enumerate(summary.get("plays", [])):
        tid = (p.get("team") or {}).get("id")
        team = abbr.get(tid)
        opp = abbr.get(next((i for i in ids if i != tid), None)) if tid else None
        text = p.get("text") or ""
        low = text.lower()
        ttype = (p.get("type") or {}).get("text", "")
        tlow = ttype.lower()
        parts = [int(x["athlete"]["id"]) for x in p.get("participants", []) if x.get("athlete", {}).get("id")]
        c = p.get("coordinate") or {}
        x, y = c.get("x"), c.get("y")
        if x is None or y is None or x < -1000 or y < -1000:
            x = y = None
        e = {"game_id": gid, "season": season, "date": day, "seq": k, "period": (p.get("period") or {}).get("number", 0),
             "clock": _clock((p.get("clock") or {}).get("displayValue")), "team": team, "opp": opp, "etype": "other",
             "p1": parts[0] if parts else None, "p2": parts[1] if len(parts) > 1 else None, "made": None,
             "pts": int(p.get("scoreValue") or 0), "three": None, "dist": None, "x": x, "y": y, "zone": None,
             "blocked": None, "ft_n": None, "ft_of": None, "type_id": int((p.get("type") or {}).get("id") or 0)}
        if "free throw" in tlow or "free throw" in low:
            m = _FT.search(low) or _FT.search(tlow)
            e.update(etype="ft", made=" makes " in f" {low} ", ft_n=int(m.group(1)) if m else None,
                     ft_of=int(m.group(2)) if m else None)
            if "technical" in tlow or "technical" in low:
                e["etype"] = "ft_tech"
        elif p.get("shootingPlay") and "heave" in tlow:
            e.update(etype="heave", made=" makes " in f" {low} ")        # not counted as a shot in box scores
        elif p.get("shootingPlay"):
            m = _FOOT.search(low)
            dist = float(m.group(1)) if m else (math.hypot(x - 25, y) if x is not None else None)
            beyond_arc = dist is not None and (dist >= 23.7 or (x is not None and y is not None
                                                              and abs(x - 25) >= 21.9 and y <= 9))
            three = ("three point" in low or "3-pt" in low or e["pts"] == 3
                     or ("two point" not in low and beyond_arc))
            if dist is None and ("layup" in low or "dunk" in low or "tip" in low):
                dist = 1.0
            blocked = " blocks " in low
            made = (" makes " in f" {low} ") and not blocked
            e.update(etype="shot", made=made, three=three, dist=dist, zone=zone_of(three, dist, x, y), blocked=blocked)
            if blocked:                       # participants: shooter, blocker
                pass
            elif not made:
                e["p2"] = None
        elif "offensive rebound" in tlow:
            e["etype"] = "oreb"
        elif "defensive rebound" in tlow:
            e["etype"] = "dreb"
        elif ("turnover" in tlow or "turnover" in low
              or (any(w in tlow for w in _TOV_WORDS) and "defensive" not in tlow and "technical" not in tlow)):
            e["etype"] = "tov"                # participants: player, stealer (violations like traveling too)
        elif "shooting foul" in tlow:
            e["etype"] = "foul_shoot"         # p1: the defender who fouled
        elif "foul" in tlow:
            e["etype"] = "foul"
        elif "substitution" in tlow:
            e["etype"] = "sub"                # p1 enters, p2 leaves
        elif "jumpball" in tlow or "jump ball" in tlow:
            e["etype"] = "jump"
        elif "end period" in tlow or "end game" in tlow:
            e["etype"] = "end"
        events.append(e)
    return events, players


def season_event_ids(days: list[date], log=print) -> list[tuple[str, str]]:
    """(ESPN event id, date) for every finished regular-season game on the given days."""
    out = []
    with ThreadPoolExecutor(6) as ex:
        boards = list(ex.map(lambda d: (d, _get(SCOREBOARD.format(d=d.strftime("%Y%m%d")))), days))
    for d, sb in boards:
        for ev in (sb or {}).get("events", []):
            st = (ev.get("status") or {}).get("type", {})
            if not st.get("completed"):
                continue
            if (ev.get("season") or {}).get("type") not in (2, None):
                continue
            teams = [to_nba(c["team"]["abbreviation"]) for c in ev["competitions"][0]["competitors"]]
            if all(t and len(t) == 3 for t in teams):
                out.append((ev["id"], d.isoformat()))
    return out


def collect(season: str, days: list[date], out_dir: Path, log=print, workers: int = 6,
            known: set[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and parse every game of `days` not in `known` (game ids 'E<id>'). Returns (events, players)."""
    games = [(e, d) for e, d in season_event_ids(days, log) if f"E{e}" not in (known or set())]
    log(f"{season}: {len(games)} games to fetch")
    evs, pls = [], []

    def one(item):
        e, d = item
        s = _get(SUMMARY.format(e=e))
        if not s or not s.get("plays"):
            return [], []
        try:
            return parse_game(s, season, d)
        except Exception as ex:  # one bad game must not stop a season
            log(f"  game {e}: parse failed ({type(ex).__name__}: {ex})")
            return [], []

    with ThreadPoolExecutor(workers) as ex:
        for i, (ev, pl) in enumerate(ex.map(one, games)):
            evs += ev
            pls += pl
            if (i + 1) % 200 == 0:
                log(f"  {season}: {i + 1}/{len(games)} games")
    return pd.DataFrame(evs, columns=EVENT_COLS), pd.DataFrame(pls)


def path(out_dir: Path, kind: str, season: str) -> Path:
    return out_dir / f"{kind}_{season}.parquet"


def load(out_dir: Path, season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    pe, pp = path(out_dir, "pbp", season), path(out_dir, "players", season)
    return (pd.read_parquet(pe) if pe.exists() else pd.DataFrame(columns=EVENT_COLS),
            pd.read_parquet(pp) if pp.exists() else pd.DataFrame())


def update(season: str, days: list[date], out_dir: Path, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add games not yet stored for `season` and save. Returns the full season."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ev0, pl0 = load(out_dir, season)
    known = set(ev0.game_id.unique()) if len(ev0) else set()
    ev1, pl1 = collect(season, days, out_dir, log, known=known)
    ev = pd.concat([ev0, ev1], ignore_index=True) if len(ev1) else ev0
    pl = pd.concat([pl0, pl1], ignore_index=True) if len(pl1) else pl0
    if len(ev1):
        for col in ("p1", "p2"):
            ev[col] = ev[col].astype("Int64")
        ev.to_parquet(path(out_dir, "pbp", season), index=False)
        pl.to_parquet(path(out_dir, "players", season), index=False)
    log(f"{season}: {ev.game_id.nunique() if len(ev) else 0} games stored")
    return ev, pl
