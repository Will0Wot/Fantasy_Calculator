"""NBA player-tracking tables for the possession simulator's "who's guarding him" step.

  defense  LeagueDashPtDefend: for each defender, shots he contested by distance band (closest
           defender at the shot), how many went in, and the shooters' normal make rate on those
           shots. PLUSMINUS < 0 means shooters made fewer than usual (e.g. a rim protector).
  matchups LeagueSeasonMatchups: offensive player x defensive player, partial possessions guarded,
           shots, points, shooting fouls.

stats.nba.com blocks cloud servers, so these are fetched on a home machine and committed as seed
files (data/seed/nba_defense_<season>.csv.gz, nba_matchups_<season>.csv.gz). Backtests only use
seasons before the one being tested (season totals would leak the future).
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

DEFENSE_BANDS = {                      # simulator zone -> tracking distance band
    "RA": "Less Than 6Ft",
    "PAINT": "Less Than 10Ft",
    "MID": "Greater Than 15Ft",
    "C3": "3 Pointers",
    "AB3": "3 Pointers",
}


def fetch_defense(season: str, pause: float = 1.0) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguedashptdefend

    frames = []
    for band in sorted(set(DEFENSE_BANDS.values())):
        d = leaguedashptdefend.LeagueDashPtDefend(season=season, defense_category=band, per_mode_simple="Totals",
                                                  timeout=60).get_data_frames()[0]
        cols = list(d.columns)
        fgm = next(c for c in cols if c.startswith(("FGM", "FG3M", "FG2M")))
        fga = next(c for c in cols if c.startswith(("FGA", "FG3A", "FG2A")))
        norm = next(c for c in cols if c.startswith("NS_"))
        frames.append(pd.DataFrame({"season": season, "band": band, "player_id": d["CLOSE_DEF_PERSON_ID"],
                                    "player_name": d["PLAYER_NAME"], "team": d["PLAYER_LAST_TEAM_ABBREVIATION"],
                                    "gp": d["GP"], "d_fgm": d[fgm], "d_fga": d[fga], "normal_pct": d[norm]}))
        time.sleep(pause)
    return pd.concat(frames, ignore_index=True)


def fetch_matchups(season: str, min_poss: float = 5.0) -> pd.DataFrame:
    from nba_api.stats.endpoints import leagueseasonmatchups

    m = leagueseasonmatchups.LeagueSeasonMatchups(season=season, per_mode_simple="Totals",
                                                  timeout=120).get_data_frames()[0]
    m = m[m["PARTIAL_POSS"] >= min_poss]
    return pd.DataFrame({"season": season, "off_id": m["OFF_PLAYER_ID"], "off_name": m["OFF_PLAYER_NAME"],
                         "def_id": m["DEF_PLAYER_ID"], "def_name": m["DEF_PLAYER_NAME"], "poss": m["PARTIAL_POSS"],
                         "fga": m["MATCHUP_FGA"], "fgm": m["MATCHUP_FGM"], "fg3a": m["MATCHUP_FG3A"],
                         "fg3m": m["MATCHUP_FG3M"], "pts": m["PLAYER_PTS"], "sfl": m["SFL"]})


def seed_paths(seed_dir: Path, season: str) -> tuple[Path, Path]:
    return seed_dir / f"nba_defense_{season}.csv.gz", seed_dir / f"nba_matchups_{season}.csv.gz"


def save(seed_dir: Path, season: str, log=print) -> None:
    pd_, pm = seed_paths(seed_dir, season)
    d = fetch_defense(season)
    d.to_csv(pd_, index=False, compression={"method": "gzip", "mtime": 0})
    m = fetch_matchups(season)
    m.to_csv(pm, index=False, compression={"method": "gzip", "mtime": 0})
    log(f"{season}: defense {len(d):,} rows, matchups {len(m):,} rows")


def load(seed_dir: Path, seasons: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ds, ms = [], []
    for s in seasons:
        pd_, pm = seed_paths(seed_dir, s)
        if pd_.exists():
            ds.append(pd.read_csv(pd_))
        if pm.exists():
            ms.append(pd.read_csv(pm))
    return (pd.concat(ds, ignore_index=True) if ds else pd.DataFrame(),
            pd.concat(ms, ignore_index=True) if ms else pd.DataFrame())
