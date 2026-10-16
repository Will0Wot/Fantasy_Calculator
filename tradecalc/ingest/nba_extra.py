"""Extra NBA data for the projection engine, cached in SQLite (http_cache table).

stats.nba.com: team game logs, player advanced stats (age, usage), player on/off ratings.
ESPN: depth charts and head coaches per team.

Past seasons are cached forever; the current season refreshes after `ttl_hours`.
"""
from __future__ import annotations

import sqlite3
import time

import pandas as pd
import requests

from tradecalc.db import cache_get, cache_put
from tradecalc.ingest.nba import _throttled_call

NBA_TEAM_IDS = {  # stats.nba.com team ids
    "ATL": 1610612737, "BOS": 1610612738, "BKN": 1610612751, "CHA": 1610612766, "CHI": 1610612741,
    "CLE": 1610612739, "DAL": 1610612742, "DEN": 1610612743, "DET": 1610612765, "GSW": 1610612744,
    "HOU": 1610612745, "IND": 1610612754, "LAC": 1610612746, "LAL": 1610612747, "MEM": 1610612763,
    "MIA": 1610612748, "MIL": 1610612749, "MIN": 1610612750, "NOP": 1610612740, "NYK": 1610612752,
    "OKC": 1610612760, "ORL": 1610612753, "PHI": 1610612755, "PHX": 1610612756, "POR": 1610612757,
    "SAC": 1610612758, "SAS": 1610612759, "TOR": 1610612761, "UTA": 1610612762, "WAS": 1610612764,
}


def _cached_df(conn, key: str, fetch, ttl_hours: float | None, min_interval: float = 1.0) -> pd.DataFrame:
    hit = cache_get(conn, key, None if ttl_hours is None else ttl_hours * 3600)
    if hit is not None:
        return pd.DataFrame(hit)
    df = _throttled_call(fetch, min_interval=min_interval)
    cache_put(conn, key, df.to_dict("records"))
    return df


def team_logs(conn: sqlite3.Connection, season: str, current: bool = False) -> pd.DataFrame:
    """One row per team-game, with possessions and the opponent's box score joined in."""
    from nba_api.stats.endpoints import leaguegamelog

    from tradecalc.ingest import nba as _nba

    try:
        if current and _nba.NBA_BLOCKED:
            raise ConnectionError("stats.nba.com blocked")
        df = _cached_df(conn, f"nba:teamlog:{season}", lambda: leaguegamelog.LeagueGameLog(
            season=season, player_or_team_abbreviation="T", timeout=15 if current else 60).get_data_frames()[0],
            6 if current else None)
    except Exception:
        if not current:
            raise
        _nba.NBA_BLOCKED = True
        from tradecalc.ingest.espn_box import team_logs_from_players

        df = team_logs_from_players(conn, season)
    if df.empty:
        return df
    df = df.rename(columns=str.lower)
    df["game_date"] = df["game_date"].astype(str).str[:10]
    df["poss"] = df["fga"] + 0.44 * df["fta"] - df["oreb"] + df["tov"]
    opp = df[["game_id", "team_abbreviation", "pts", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm", "oreb",
              "dreb", "reb", "ast", "stl", "blk", "tov", "poss"]].add_prefix("opp_")
    df = df.merge(opp, left_on="game_id", right_on="opp_game_id")
    df = df[df["team_abbreviation"] != df["opp_team_abbreviation"]].drop(columns="opp_game_id")
    df["team"] = df["team_abbreviation"]
    df["opp"] = df["opp_team_abbreviation"]
    df["home"] = df["matchup"].str.contains("vs.")
    df["pace"] = (df["poss"] + df["opp_poss"]) / 2 * 48 / (df["min"] / 5).clip(lower=48)
    return df


def player_advanced(conn, season: str, current: bool = False) -> pd.DataFrame:
    """Season totals with AGE, USG_PCT etc., one row per player."""
    from nba_api.stats.endpoints import leaguedashplayerstats

    from tradecalc.ingest import nba as _nba

    if current and _nba.NBA_BLOCKED:
        raise ConnectionError("stats.nba.com blocked")
    return _cached_df(conn, f"nba:playeradv:{season}", lambda: leaguedashplayerstats.LeagueDashPlayerStats(
        season=season, measure_type_detailed_defense="Advanced", timeout=60).get_data_frames()[0],
        24 if current else None)


def player_onoff(conn, season: str, current: bool = False) -> pd.DataFrame:
    """On/off defensive and offensive rating for every player on every team (30 requests)."""
    from nba_api.stats.endpoints import teamplayeronoffsummary

    key = f"nba:onoff:{season}"
    hit = cache_get(conn, key, None if not current else 24 * 3600)
    if hit is not None:
        return pd.DataFrame(hit)
    frames = []
    for abbr, tid in NBA_TEAM_IDS.items():
        res = _throttled_call(lambda: teamplayeronoffsummary.TeamPlayerOnOffSummary(
            team_id=tid, season=season, measure_type_detailed_defense="Advanced", timeout=60).get_data_frames(),
            min_interval=1.0)
        on, off = res[1], res[2]
        cols = ["VS_PLAYER_ID", "VS_PLAYER_NAME", "MIN", "DEF_RATING", "OFF_RATING"]
        m = on[cols].merge(off[["VS_PLAYER_ID", "MIN", "DEF_RATING", "OFF_RATING"]],
                           on="VS_PLAYER_ID", suffixes=("_ON", "_OFF"))
        m["TEAM"] = abbr
        frames.append(m)
    df = pd.concat(frames, ignore_index=True)
    cache_put(conn, key, df.to_dict("records"))
    return df


# --- ESPN ---------------------------------------------------------------------------------

_H = {"User-Agent": "tradecalc/0.1"}


def _espn_json(conn, url: str, ttl_hours: float | None):
    hit = cache_get(conn, url, None if ttl_hours is None else ttl_hours * 3600)
    if hit is not None:
        return hit
    for attempt in range(3):
        r = requests.get(url, timeout=30, headers=_H)
        if r.ok:
            data = r.json()
            cache_put(conn, url, data)
            return data
        time.sleep(1 + attempt)
    r.raise_for_status()


def depth_charts(conn, espn_team_ids: dict[int, str], ttl_hours: float = 24) -> pd.DataFrame:
    """ESPN depth charts: one row per (team, position, depth rank, espn athlete id)."""
    rows = []
    for tid, abbr in espn_team_ids.items():
        if tid == 0:
            continue
        d = _espn_json(conn, f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{tid}/depthcharts",
                       ttl_hours)
        for chart in d.get("depthchart", []):
            for pos, v in (chart.get("positions") or {}).items():
                for rank, a in enumerate(v.get("athletes", []), 1):
                    rows.append({"team": abbr, "pos": pos.upper(), "depth": rank, "espn_id": int(a["id"]),
                                 "name": a.get("displayName")})
    return pd.DataFrame(rows)


def head_coaches(conn, espn_team_ids: dict[int, str], espn_seasons: list[int]) -> pd.DataFrame:
    """Head coach per team-season: team, season, coach_id, coach name."""
    rows = []
    for season in espn_seasons:
        for tid, abbr in espn_team_ids.items():
            if tid == 0:
                continue
            base = f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/seasons/{season}/teams/{tid}/coaches"
            try:
                d = _espn_json(conn, base, None)
            except requests.HTTPError:
                continue
            for item in d.get("items", [])[:1]:
                ref = item["$ref"].replace("http://", "https://")
                c = _espn_json(conn, ref, None)
                rows.append({"team": abbr, "season": season, "coach_id": int(c.get("id", 0)),
                             "coach": f"{c.get('firstName', '')} {c.get('lastName', '')}".strip()})
    return pd.DataFrame(rows)
