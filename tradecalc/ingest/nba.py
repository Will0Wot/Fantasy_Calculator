"""NBA box scores from stats.nba.com via nba_api, cached in SQLite.

One LeagueGameLog call returns every player-game for a season (or for a date range),
so a refresh is a single request for games on/after the last cached date. Completed
seasons are marked and never re-fetched.
"""
from __future__ import annotations

import sqlite3
import time
from datetime import date, datetime

import pandas as pd

from tradecalc.db import GAME_LOG_COLS

# Number of HTTP requests made to stats.nba.com in this process (used by tests).
API_CALLS = 0
_last_request = 0.0

REFRESH_MIN_INTERVAL_S = 3600  # current season: skip if refreshed within this window


def _throttled_call(fn, *, min_interval: float, retries: int = 4):
    """Rate-limit and retry a stats.nba.com call (it throttles and times out under load)."""
    global API_CALLS, _last_request
    delay = 2.0
    for attempt in range(retries + 1):
        wait = min_interval - (time.time() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.time()
        API_CALLS += 1
        try:
            return fn()
        except Exception:
            if attempt == retries:
                raise
            time.sleep(delay)
            delay *= 2


# Set once stats.nba.com has failed in this process (typical on cloud hosts): skip it after that.
NBA_BLOCKED = False


def _fetch_league_log(season: str, date_from: date | None, min_interval: float, timeout: int = 60,
                      retries: int = 4) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguegamelog

    kwargs = dict(
        season=season,
        season_type_all_star="Regular Season",
        player_or_team_abbreviation="P",
        timeout=timeout,
    )
    if date_from is not None:
        kwargs["date_from_nullable"] = date_from.strftime("%m/%d/%Y")
    return _throttled_call(
        lambda: leaguegamelog.LeagueGameLog(**kwargs).get_data_frames()[0],
        min_interval=min_interval, retries=retries,
    )


def _store(conn: sqlite3.Connection, season: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = []
    for r in df.itertuples(index=False):
        rec = r._asdict()
        rows.append(
            (
                season, str(rec["GAME_ID"]), int(rec["PLAYER_ID"]), rec["PLAYER_NAME"],
                rec["TEAM_ABBREVIATION"], str(rec["GAME_DATE"])[:10], rec["MATCHUP"],
                *[float(rec[c.upper()] or 0) for c in GAME_LOG_COLS],
            )
        )
    cols = ["season", "game_id", "player_id", "player_name", "team", "game_date", "matchup", *GAME_LOG_COLS]
    conn.executemany(
        f"INSERT OR REPLACE INTO game_logs({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", rows
    )
    conn.commit()
    return len(rows)


def refresh_season(
    conn: sqlite3.Connection,
    season: str,
    *,
    complete: bool,
    min_interval: float = 1.0,
    force: bool = False,
    first_day: date | None = None,
) -> dict:
    """Pull new game logs for `season`. Returns a summary dict.

    complete=True marks a past season: after one full pull it is never fetched again.
    """
    state = conn.execute("SELECT complete, last_fetch FROM season_state WHERE season=?", (season,)).fetchone()
    if state and state["complete"]:
        return {"season": season, "fetched": 0, "skipped": "season complete and cached"}
    if state and not force and state["last_fetch"] and time.time() - state["last_fetch"] < REFRESH_MIN_INTERVAL_S:
        return {"season": season, "fetched": 0, "skipped": "refreshed recently"}

    last = conn.execute("SELECT MAX(game_date) d FROM game_logs WHERE season=?", (season,)).fetchone()["d"]
    # Re-pull the last cached date too, in case it was fetched while games were in progress.
    date_from = datetime.strptime(last, "%Y-%m-%d").date() if last else None
    global NBA_BLOCKED
    if complete:
        df = _fetch_league_log(season, date_from, min_interval)
    else:
        # Current season: fail fast and fall back to ESPN box scores if stats.nba.com is blocked.
        try:
            if NBA_BLOCKED:
                raise ConnectionError("stats.nba.com unreachable earlier in this session")
            df = _fetch_league_log(season, date_from, min_interval, timeout=15, retries=1)
        except Exception as e:  # noqa: BLE001
            NBA_BLOCKED = True
            from tradecalc.ingest.espn_box import refresh_from_espn

            from tradecalc.clock import league_today

            r = refresh_from_espn(conn, season, first_day or league_today())
            conn.execute("INSERT OR REPLACE INTO season_state(season, complete, last_fetch) VALUES (?,?,?)",
                         (season, 0, time.time()))
            conn.commit()
            r["note"] = f"stats.nba.com unavailable ({type(e).__name__}); used ESPN box scores"
            return r
        if len(df) and date_from is not None:
            # stats.nba.com is back: its rows replace any ESPN fallback rows for these dates
            conn.execute("DELETE FROM game_logs WHERE season=? AND game_id LIKE 'E%' AND game_date>=?",
                         (season, str(date_from)))
    n = _store(conn, season, df)
    conn.execute(
        "INSERT OR REPLACE INTO season_state(season, complete, last_fetch) VALUES (?,?,?)",
        (season, int(complete and n > 0), time.time()),
    )
    conn.commit()
    return {"season": season, "fetched": n, "from": str(date_from) if date_from else "season start"}


def game_logs(conn: sqlite3.Connection, season: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM game_logs WHERE season=? ORDER BY player_id, game_date", conn, params=(season,)
    )


def player_game_log(conn: sqlite3.Connection, season: str, player_id: int) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM game_logs WHERE season=? AND player_id=? ORDER BY game_date",
        conn,
        params=(season, player_id),
    )


def find_player_id(conn: sqlite3.Connection, name: str) -> tuple[int, str] | None:
    from rapidfuzz import process, utils

    names = dict(conn.execute("SELECT DISTINCT player_name, player_id FROM game_logs").fetchall())
    if not names:
        return None
    hit = process.extractOne(name, list(names), processor=utils.default_process, score_cutoff=80)
    return (names[hit[0]], hit[0]) if hit else None
