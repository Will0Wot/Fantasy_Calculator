"""Seed data: finished NBA seasons shipped with the app as compressed files.

The runtime cache (data/cache.sqlite) is local and not committed. What a fresh install
(e.g. Streamlit Community Cloud) needs but cannot reliably download, because stats.nba.com
often blocks cloud servers, is the data of *finished* seasons. That never changes, so it is
exported once per season to data/seed/:

  game_logs_<season>.csv.gz        every player-game (box scores)
  nba_teamlog_<season>.json.gz     team-games with opponent box scores
  nba_playeradv_<season>.json.gz   player ages / usage
  nba_onoff_<season>.json.gz       on/off ratings

On startup, ensure_seed() loads any finished season the local cache does not have yet.
Everything else (ESPN rosters, injuries, depth charts, this season's games) is fetched live.
"""
from __future__ import annotations

import gzip
import json
import sqlite3
import time
from pathlib import Path

import pandas as pd

from tradecalc.db import GAME_LOG_COLS, cache_put

SEED_DIR = Path(__file__).resolve().parent / "data" / "seed"
KINDS = ("teamlog", "playeradv", "onoff")
LOG_COLS = ["season", "game_id", "player_id", "player_name", "team", "game_date", "matchup", *GAME_LOG_COLS]


def seeded_seasons() -> list[str]:
    return sorted(p.name[len("game_logs_"):-len(".csv.gz")] for p in SEED_DIR.glob("game_logs_*.csv.gz"))


def export_seed(conn: sqlite3.Connection, seasons: list[str]) -> list[Path]:
    """Write finished seasons from the local cache to data/seed/."""
    SEED_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for s in seasons:
        g = pd.read_sql_query(f"SELECT {','.join(LOG_COLS)} FROM game_logs WHERE season=? ORDER BY game_date, game_id, player_id",
                              conn, params=(s,))
        if g.empty:
            continue
        path = SEED_DIR / f"game_logs_{s}.csv.gz"
        g.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
        written.append(path)
        for kind in KINDS:
            row = conn.execute("SELECT body FROM http_cache WHERE key=?", (f"nba:{kind}:{s}",)).fetchone()
            if row:
                path = SEED_DIR / f"nba_{kind}_{s}.json.gz"
                with gzip.GzipFile(path, "wb", mtime=0) as f:
                    f.write(row[0].encode())
                written.append(path)
    return written


def ensure_seed(conn: sqlite3.Connection) -> list[str]:
    """Load seeded seasons missing from the local cache; mark them complete so they are never
    re-downloaded. Returns the seasons loaded."""
    loaded = []
    for s in seeded_seasons():
        have = conn.execute("SELECT complete FROM season_state WHERE season=?", (s,)).fetchone()
        if have and have[0]:
            continue
        g = pd.read_csv(SEED_DIR / f"game_logs_{s}.csv.gz", dtype={"game_id": str})
        conn.executemany(f"INSERT OR REPLACE INTO game_logs({','.join(LOG_COLS)}) VALUES ({','.join('?' * len(LOG_COLS))})",
                         g[LOG_COLS].itertuples(index=False, name=None))
        for kind in KINDS:
            path = SEED_DIR / f"nba_{kind}_{s}.json.gz"
            if path.exists():
                with gzip.open(path, "rt") as f:
                    cache_put(conn, f"nba:{kind}:{s}", json.load(f))
        conn.execute("INSERT OR REPLACE INTO season_state(season, complete, last_fetch) VALUES (?,?,?)",
                     (s, 1, time.time()))
        conn.commit()
        loaded.append(s)
    return loaded
