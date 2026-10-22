"""Who's guarding him: defender effects and matchup tendencies for the possession simulator, from
NBA tracking tables (ingest/tracking.py).

  effect(z)   logit change in the shooter's make probability when this defender is the closest
              defender on a shot in zone z: logit(his opponents' make rate) - logit(their normal
              rate), shrunk toward 0 by n / (n + K): rim protection is real and stable (small K),
              three-point "defense" is mostly luck (large K).
  rim         contested shots inside 6 ft per game: who shows up at the rim
  affinity    partial possessions in which defender d guarded shooter o (seasons before the one
              being simulated), for choosing the primary defender on perimeter shots
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradecalc.ingest.tracking import DEFENSE_BANDS
from tradecalc.model.names import norm_name

K_DEF = {"Less Than 6Ft": 250, "Less Than 10Ft": 300, "Greater Than 15Ft": 600, "3 Pointers": 1500}
SEASON_WEIGHTS = (1.0, 0.5)          # most recent season first


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


@dataclass
class DefenseModel:
    effect: pd.DataFrame        # index player key; columns RA, PAINT, MID, C3, AB3 (logit shifts)
    rim: pd.Series              # player key -> contests inside 6 ft per game
    affinity: dict              # (off key, def key) -> partial possessions


def build(defense: pd.DataFrame, matchups: pd.DataFrame, seasons: list[str]) -> DefenseModel:
    """seasons: the seasons to use, most recent first (only seasons before the simulated one)."""
    w = {s: SEASON_WEIGHTS[i] if i < len(SEASON_WEIGHTS) else 0.0 for i, s in enumerate(seasons)}
    if defense is None or defense.empty:
        return DefenseModel(pd.DataFrame(columns=list(DEFENSE_BANDS)), pd.Series(dtype=float), {})
    d = defense[defense.season.isin(w)].copy()
    d["key"] = d.player_name.map(norm_name)
    d["w"] = d.season.map(w)
    d["fgm_w"], d["fga_w"], d["exp_w"] = d.d_fgm * d.w, d.d_fga * d.w, d.normal_pct * d.d_fga * d.w
    agg = d.groupby(["key", "band"])[["fgm_w", "fga_w", "exp_w"]].sum()
    agg["shift"] = _logit(agg.fgm_w / agg.fga_w.clip(lower=1)) - _logit(agg.exp_w / agg.fga_w.clip(lower=1))
    agg["k"] = [K_DEF[b] for b in agg.index.get_level_values("band")]
    agg["shift"] *= agg.fga_w / (agg.fga_w + agg.k)
    shifts = agg["shift"].unstack("band").fillna(0.0)
    eff = pd.DataFrame({z: shifts.get(band, 0.0) for z, band in DEFENSE_BANDS.items()}, index=shifts.index).fillna(0.0)
    rim = d[d.band == "Less Than 6Ft"].assign(pg=lambda x: x.d_fga * x.w, g=lambda x: x.gp * x.w) \
        .groupby("key")[["pg", "g"]].sum()
    rim = (rim.pg / rim.g.clip(lower=1)).rename("rim")
    aff = {}
    if matchups is not None and len(matchups):
        m = matchups[matchups.season.isin(w)]
        for o, dd, p, s in zip(m.off_name, m.def_name, m.poss, m.season):
            k = (norm_name(o), norm_name(dd))
            aff[k] = aff.get(k, 0.0) + p * w[s]
    return DefenseModel(eff, rim, aff)
