"""Opponent and pace adjustments.

Team defensive factors: for each stat, what opponents produce per possession against this
team, relative to league average, shrunk toward 1 by K games (fit by split-half, like the
player rates). Percentages (2P%, 3P%, FT%) use attempts as exposure.

Pace: expected possessions = own pace + opponent pace - league pace.

Defender availability: a player's on/off defensive rating difference (last season, shrunk
by minutes) says how much worse the team defends without him. If he sits, the opponent's
points-type stats scale up by share_of_minutes x (DRtg_off - DRtg_on) / league DRtg.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# stat -> (opponent numerator col, exposure col) in team logs ("opp_" = what the opponent did)
DEF_STATS = {
    "fga": ("opp_fga", "opp_poss"), "fta": ("opp_fta", "opp_poss"), "reb": ("opp_reb", "opp_poss"),
    "ast": ("opp_ast", "opp_poss"), "stl": ("opp_stl", "opp_poss"), "blk": ("opp_blk", "opp_poss"),
    "tov": ("opp_tov", "opp_poss"), "fg3a_share": ("opp_fg3a", "opp_fga"),
    "fg2": ("opp_fg2m", "opp_fg2a"), "fg3": ("opp_fg3m", "opp_fg3a"), "ft": ("opp_ftm", "opp_fta"),
}
K_GRID = [2, 5, 10, 20, 40, 80, 160, 320]


def _prep(tl: pd.DataFrame) -> pd.DataFrame:
    t = tl.copy()
    t["opp_fg2m"] = t["opp_fgm"] - t["opp_fg3m"]
    t["opp_fg2a"] = t["opp_fga"] - t["opp_fg3a"]
    return t


def fit_def_k(tl: pd.DataFrame) -> dict[str, float]:
    """Shrinkage (in games) per stat by split-half: odd games predict even games."""
    t = _prep(tl).sort_values("game_date")
    t["half"] = t.groupby("team").cumcount() % 2
    cols = sorted({c for v in DEF_STATS.values() for c in v})
    a = t[t.half == 0].groupby("team")[cols].sum()
    b = t[t.half == 1].groupby("team")[cols].sum()
    na = t[t.half == 0].groupby("team").size()
    out = {}
    for s, (num, exp) in DEF_STATS.items():
        lg = (a[num].sum() + b[num].sum()) / (a[exp].sum() + b[exp].sum())
        fa, fb = a[num] / a[exp] / lg, b[num] / b[exp] / lg
        errs = [(((k + na * fa) / (k + na) - fb) ** 2).sum() for k in K_GRID]
        out[s] = float(K_GRID[int(np.argmin(errs))])
    return out


@dataclass
class DefenseModel:
    factors: pd.DataFrame        # index team, columns DEF_STATS -> multiplier (1 = average)
    pace: pd.Series              # team -> possessions per 48
    league_pace: float
    league_drtg: float


def estimate_defense(tl_season: pd.DataFrame, k: dict[str, float], prev: pd.DataFrame | None = None,
                     carry: float = 0.3) -> DefenseModel:
    """Factors from this season's team games so far; last season's factor is the prior,
    counted as carry x (its games) up to 2K games."""
    cur = _prep(tl_season)
    base = cur if len(cur) else _prep(prev)
    cols = sorted({c for v in DEF_STATS.values() for c in v})
    teams = sorted(set(base["team"]))
    gc = cur.groupby("team")[cols].sum().reindex(teams).fillna(0)
    nc = cur.groupby("team").size().reindex(teams).fillna(0)
    if prev is not None and len(prev):
        p = _prep(prev)
        gp, npv = p.groupby("team")[cols].sum().reindex(teams), p.groupby("team").size().reindex(teams).fillna(0)
    else:
        gp, npv = None, pd.Series(0.0, index=teams)
    pool = pd.concat([cur, _prep(prev)]) if prev is not None and len(prev) else cur
    f = {}
    for s, (num, exp) in DEF_STATS.items():
        lg = pool[num].sum() / pool[exp].sum()
        K = k[s]
        prior = pd.Series(1.0, index=teams)
        strength = pd.Series(float(K), index=teams)
        if gp is not None:
            prev_f = (gp[num] / gp[exp] / lg).fillna(1.0)
            prior = (K + npv * prev_f) / (K + npv)
            strength = K + np.minimum(carry * npv, 2 * K)
        obs = (gc[num] / gc[exp].replace(0, np.nan) / lg).fillna(1.0)
        f[s] = (strength * prior + nc * obs) / (strength + nc)
    pace_src = cur if len(cur) >= 60 else pool
    pace = pace_src.groupby("team")["pace"].mean()
    drtg = 100 * pool["opp_pts"].sum() / pool["opp_poss"].sum()
    return DefenseModel(pd.DataFrame(f), pace, float(pace_src["pace"].mean()), float(drtg))


def defender_impacts(onoff: pd.DataFrame, shrink_min: float = 2000.0) -> pd.Series:
    """player_id -> DRtg points per 100 the team gets worse when he is off the floor
    (positive = good defender), shrunk toward 0 by on-court minutes."""
    if onoff.empty:
        return pd.Series(dtype=float)
    d = onoff.copy()
    diff = d["DEF_RATING_OFF"] - d["DEF_RATING_ON"]
    w = d["MIN_ON"] / (d["MIN_ON"] + shrink_min)
    s = (diff * w).groupby(d["VS_PLAYER_ID"]).mean()
    return s
