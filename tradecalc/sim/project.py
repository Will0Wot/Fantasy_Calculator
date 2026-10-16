"""Expected box score for one player-game: minutes x adjusted per-stat rates.

Shared by the backtest (sim/backtest.py) and the Monte Carlo engine (sim/engine.py).
"""
from __future__ import annotations

import math

import numpy as np

COUNT_STATS = ["fga", "fta", "reb", "ast", "stl", "blk", "tov"]
PCT_STATS = ["fg2", "fg3", "ft"]
USAGE_FOR = {"fga": "fga", "fta": "fta", "tov": "tov", "ast": "ast", "reb": "reb", "stl": "stl", "blk": "blk",
             "fg3a_share": None}


def expected_blowout(margin_sd: float = 16.8, threshold: float = 12.0) -> float:
    """E[max(|margin| - threshold, 0)] for margin ~ Normal(0, sd)."""
    z = threshold / margin_sd
    phi = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    tail = 0.5 * math.erfc(z / math.sqrt(2))
    return 2 * (margin_sd * phi - threshold * tail)


def minutes_mean(exp_min: float, abs_same: float, abs_other: float, b2b: bool, mp: dict,
                 blow: float | None = None) -> float:
    """Expected minutes. blow: realized blowout excess (sim) or None for its expectation."""
    blow = expected_blowout(threshold=mp.get("blowout_threshold", 12)) if blow is None else blow
    m = (exp_min + mp["const"] + mp["abs_min_same"] * abs_same + mp["abs_min_other"] * abs_other
         + mp["b2b"] * b2b)
    if exp_min >= 30:
        m += mp["blow_star"] * blow
    elif exp_min < 20:
        m += mp["blow_bench"] * blow
    return max(m, 0.0)


def minutes_sd(exp_min: float, mp: dict) -> float:
    for edge, sd in sorted(((float(k), v) for k, v in mp["resid_sd"].items())):
        if exp_min <= edge:
            return sd
    return list(mp["resid_sd"].values())[-1]


def adjusted_rates(base: dict, usage_beta: dict, miss: dict, def_f: dict | None, pace_ratio: float,
                   eff_mult: float = 1.0) -> dict:
    """Per-minute counting rates and percentages after teammate, opponent and pace adjustments."""
    r = {}
    for s in COUNT_STATS:
        u = 1 + usage_beta.get(s, 0.0) * miss.get(s, 0.0)
        d = def_f.get(s, 1.0) if def_f else 1.0
        r[s] = base[s] * u * d * pace_ratio
    share_u = 1 + usage_beta.get("fg3a", 0.0) * miss.get("fg3a", 0.0)
    share_fga = 1 + usage_beta.get("fga", 0.0) * miss.get("fga", 0.0)
    r["fg3a_share"] = min(base["fg3a_share"] * share_u / share_fga * (def_f.get("fg3a_share", 1.0) if def_f else 1), 0.95)
    for s in PCT_STATS:
        d = def_f.get(s, 1.0) if def_f else 1.0
        e = eff_mult if s in ("fg2", "fg3") else 1.0
        r[s] = min(base[s] * d * e, 0.99)
    return r


def expected_box(minutes: float, r: dict) -> dict:
    fga = r["fga"] * minutes
    fg3a = fga * r["fg3a_share"]
    fg2a = fga - fg3a
    fg2m, fg3m = fg2a * r["fg2"], fg3a * r["fg3"]
    fta = r["fta"] * minutes
    ftm = fta * r["ft"]
    return {"min": minutes, "fga": fga, "fgm": fg2m + fg3m, "fg3a": fg3a, "fg3m": fg3m, "fta": fta, "ftm": ftm,
            "pts": 2 * fg2m + 3 * fg3m + ftm, "reb": r["reb"] * minutes, "ast": r["ast"] * minutes,
            "stl": r["stl"] * minutes, "blk": r["blk"] * minutes, "tov": r["tov"] * minutes}


def fantasy(box: dict, scoring: dict) -> float:
    return float(sum(w * box[k.lower()] for k, w in scoring.items()))
