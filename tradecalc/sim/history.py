"""Historical game logs prepared for the projection engine.

Every model in tradecalc/sim reads `History` "as of" a date: only games before that
date are visible, so backtests are free of look-ahead.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tradecalc.ingest import nba, nba_extra
from tradecalc.model.names import norm_name

SEASONS = ["2023-24", "2024-25", "2025-26", "2026-27"]
POS_GROUP = {"PG": "G", "SG": "G", "SF": "F", "PF": "F", "C": "C"}


def season_of(d: str) -> str:
    y, m = int(d[:4]), int(d[5:7])
    start = y if m >= 8 else y - 1
    return f"{start}-{str(start + 1)[2:]}"


def prev_season(s: str) -> str:
    y = int(s[:4]) - 1
    return f"{y}-{str(y + 1)[2:]}"


@dataclass
class History:
    logs: pd.DataFrame            # player-games (min > 0 only), sorted by date
    team_logs: pd.DataFrame       # team-games with opponent box score and pace
    pos: dict[str, str]           # normalized name -> G / F / C
    age: dict[tuple[int, str], float]  # (player_id, season) -> age
    onoff: pd.DataFrame           # previous-season on/off, all seasons stacked
    scoring: dict
    _cache: dict = field(default_factory=dict, repr=False)

    def before(self, d: str) -> pd.DataFrame:
        """Player-games strictly before date d (YYYY-MM-DD)."""
        key = ("before", d)
        if key not in self._cache:
            i = np.searchsorted(self.logs["game_date"].to_numpy(), d, side="left")
            self._cache[key] = self.logs.iloc[:i]
        return self._cache[key]

    def team_before(self, d: str) -> pd.DataFrame:
        i = np.searchsorted(self.team_logs["game_date"].to_numpy(), d, side="left")
        return self.team_logs.iloc[:i]

    def pos_of(self, name: str) -> str | None:
        return self.pos.get(norm_name(name))


def _b2b_flags(dates: pd.Series) -> np.ndarray:
    d = pd.to_datetime(dates)
    return (d.diff().dt.days == 1).to_numpy()


def load_history(conn, scoring: dict, espn_players=None, seasons=SEASONS, current_season: str | None = None) -> History:
    frames, tframes, ages, onoffs = [], [], {}, []
    for s in seasons:
        cur = s == current_season
        g = nba.game_logs(conn, s)
        if len(g):
            frames.append(g.assign(season=s))
        t = nba_extra.team_logs(conn, s, current=cur) if (cur or len(g)) else pd.DataFrame()
        if len(t):
            t = t.sort_values(["team", "game_date"])
            t["b2b"] = t.groupby("team")["game_date"].transform(_b2b_flags).astype(bool)
            t["season"] = s
            tframes.append(t)
        try:
            adv = nba_extra.player_advanced(conn, s, current=cur)
            for pid, a in zip(adv["PLAYER_ID"], adv["AGE"]):
                ages[(int(pid), s)] = float(a)
        except Exception:
            pass
        if not cur and len(g):
            try:
                oo = nba_extra.player_onoff(conn, s)
                onoffs.append(oo.assign(season=s))
            except Exception:
                pass
    logs = pd.concat(frames, ignore_index=True)
    logs = logs[logs["min"] > 0].copy()
    logs["opp"] = logs["matchup"].str[-3:]
    logs["home"] = logs["matchup"].str.contains("vs.")
    w = scoring
    logs["fp"] = sum(v * logs[k.lower()] for k, v in w.items())
    logs["fg2a"] = logs["fga"] - logs["fg3a"]
    logs["fg2m"] = logs["fgm"] - logs["fg3m"]
    tl = pd.concat(tframes, ignore_index=True)
    b2b = tl.set_index(["game_id", "team"])["b2b"]
    logs = logs.join(b2b, on=["game_id", "team"])
    logs["b2b"] = logs["b2b"].fillna(False).astype(bool)
    logs = logs.sort_values(["game_date", "game_id", "player_id"]).reset_index(drop=True)
    tl = tl.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    pos = {}
    for p in (espn_players or []):
        ps = [x for x in p.positions if x in POS_GROUP]
        if ps:  # C-eligible -> big; else group of the first listed position
            pos[norm_name(p.name)] = "C" if "C" in ps else POS_GROUP[ps[0]]
    onoff = pd.concat(onoffs, ignore_index=True) if onoffs else pd.DataFrame()
    return History(logs, tl, pos, ages, onoff, dict(scoring))
