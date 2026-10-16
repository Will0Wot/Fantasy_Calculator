"""Per-stat rate model.

Each player's box score is generated from these underlying rates:
  per minute:   fga, fta, reb, ast, stl, blk, tov
  share:        fg3a / fga
  percentages:  fg2 (2PM/2PA), fg3 (3PM/3PA), ft (FTM/FTA)

Each rate is an empirical-Bayes estimate: the player's observed rate this season, shrunk
toward a prior. The prior is his previous-season rate (itself shrunk toward the average
for his position group), weighted by `carry` x last season's exposure. Shrinkage strength
K (in minutes or attempts) is fitted per stat by split-half prediction (fit_k), so a
noisy stat like FT% needs many attempts before it moves, rebounds need few minutes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradecalc.model.names import norm_name

# stat -> (numerator column, exposure column)
RATES = {
    "fga": ("fga", "min"), "fta": ("fta", "min"), "reb": ("reb", "min"), "ast": ("ast", "min"),
    "stl": ("stl", "min"), "blk": ("blk", "min"), "tov": ("tov", "min"),
    "fg3a_share": ("fg3a", "fga"),
    "fg2": ("fg2m", "fg2a"), "fg3": ("fg3m", "fg3a"), "ft": ("ftm", "fta"),
}
K_GRID = {
    "min": [25, 50, 100, 150, 200, 300, 450, 700, 1000, 1500],
    "att": [10, 25, 50, 100, 150, 200, 300, 450, 700, 1000],
}


def _totals(logs: pd.DataFrame) -> pd.DataFrame:
    cols = sorted({c for v in RATES.values() for c in v})
    return logs.groupby("player_id")[cols].sum()


def fit_k(logs: pd.DataFrame, min_exposure: dict | None = None) -> dict[str, float]:
    """Shrinkage K per stat, chosen to best predict a player's even-numbered games from
    his odd-numbered games (exposure-weighted squared error)."""
    g = logs.sort_values("game_date").copy()
    g["half"] = g.groupby("player_id").cumcount() % 2
    a, b = _totals(g[g["half"] == 0]), _totals(g[g["half"] == 1])
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    out = {}
    for stat, (num, exp) in RATES.items():
        ok = (a[exp] > 0) & (b[exp] > 0)
        xa, ea, xb, eb = a.loc[ok, num], a.loc[ok, exp], b.loc[ok, num], b.loc[ok, exp]
        mean = (xa.sum() + xb.sum()) / (ea.sum() + eb.sum())
        target = xb / eb
        grid = K_GRID["min" if exp == "min" else "att"]
        errs = [((((k * mean + xa) / (k + ea)) - target) ** 2 * eb).sum() for k in grid]
        out[stat] = float(grid[int(np.argmin(errs))])
    return out


@dataclass
class RateModel:
    rates: pd.DataFrame          # index player_id, one column per stat
    exposure: pd.DataFrame       # current-season exposure behind each rate
    league: dict[str, dict]      # position group -> stat -> mean


def league_means(logs: pd.DataFrame, pos_of) -> dict[str, dict]:
    t = _totals(logs)
    names = logs.groupby("player_id")["player_name"].last()
    grp = names.map(lambda n: pos_of(n) or "ALL")
    out = {}
    for gname, sub in list(t.groupby(grp)) + [("ALL", t)]:
        out[gname] = {s: sub[n].sum() / max(sub[e].sum(), 1e-9) for s, (n, e) in RATES.items()}
    return out


def estimate_rates(cur: pd.DataFrame, prev: pd.DataFrame, k: dict[str, float], pos_of,
                   carry: float = 0.5) -> RateModel:
    """Rates for every player seen this season or last.

    cur: this season's player-games so far; prev: last season's player-games.
    """
    lg = league_means(prev if len(prev) else cur, pos_of)
    tc, tp = _totals(cur), _totals(prev)
    names = pd.concat([cur, prev]).groupby("player_id")["player_name"].last()
    ids = names.index
    tc, tp = tc.reindex(ids).fillna(0), tp.reindex(ids).fillna(0)
    grp = names.map(lambda n: pos_of(n) or "ALL")
    rates, expo = {}, {}
    for s, (num, exp) in RATES.items():
        base = grp.map(lambda gname: lg.get(gname, lg["ALL"])[s])
        K = k[s]
        prior = (K * base + tp[num]) / (K + tp[exp])           # last season, shrunk to position mean
        strength = K + carry * tp[exp]                          # how much the prior counts this season
        rates[s] = (strength * prior + tc[num]) / (strength + tc[exp])
        expo[s] = tc[exp]
    return RateModel(pd.DataFrame(rates), pd.DataFrame(expo), lg)
