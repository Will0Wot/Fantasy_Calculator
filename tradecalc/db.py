"""SQLite cache: raw HTTP responses (with TTL) and normalized NBA game logs."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS http_cache (
    key TEXT PRIMARY KEY,
    fetched_at REAL NOT NULL,
    body TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS game_logs (
    season TEXT NOT NULL,
    game_id TEXT NOT NULL,
    player_id INTEGER NOT NULL,
    player_name TEXT,
    team TEXT,
    game_date TEXT NOT NULL,
    matchup TEXT,
    min REAL, pts REAL, reb REAL, ast REAL, stl REAL, blk REAL, tov REAL,
    fgm REAL, fga REAL, ftm REAL, fta REAL, fg3m REAL, fg3a REAL, oreb REAL, dreb REAL, pf REAL,
    PRIMARY KEY (game_id, player_id)
);
CREATE INDEX IF NOT EXISTS ix_game_logs_season_date ON game_logs(season, game_date);
CREATE INDEX IF NOT EXISTS ix_game_logs_player ON game_logs(player_id, game_date);
CREATE TABLE IF NOT EXISTS injury_log (
    espn_id INTEGER NOT NULL,
    name TEXT,
    profile TEXT NOT NULL,
    status TEXT,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    return_date TEXT,
    text TEXT,
    PRIMARY KEY (espn_id, profile, first_seen)
);
CREATE TABLE IF NOT EXISTS status_snapshots (
    espn_id INTEGER NOT NULL,
    day TEXT NOT NULL,
    status TEXT NOT NULL,
    PRIMARY KEY (espn_id, day)
);
CREATE TABLE IF NOT EXISTS season_state (
    season TEXT PRIMARY KEY,
    complete INTEGER NOT NULL DEFAULT 0,
    last_fetch REAL
);
"""

GAME_LOG_COLS = [
    "min", "pts", "reb", "ast", "stl", "blk", "tov",
    "fgm", "fga", "ftm", "fta", "fg3m", "fg3a", "oreb", "dreb", "pf",
]


def connect(path: str | Path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def cache_get(conn: sqlite3.Connection, key: str, ttl_s: float | None):
    row = conn.execute("SELECT fetched_at, body FROM http_cache WHERE key=?", (key,)).fetchone()
    if row is None:
        return None
    if ttl_s is not None and time.time() - row["fetched_at"] > ttl_s:
        return None
    return json.loads(row["body"])


def cache_put(conn: sqlite3.Connection, key: str, data) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO http_cache(key, fetched_at, body) VALUES (?,?,?)",
        (key, time.time(), json.dumps(data)),
    )
    conn.commit()
