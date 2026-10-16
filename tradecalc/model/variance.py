"""Per-game FP standard deviation, shrunk toward a league-typical CV (section 4.5)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_cv_typical(live_prev: pd.DataFrame, top_n: int, min_games: int = 20) -> float:
    """Median coefficient of variation of per-game FP among the `top_n` players by FPPG.

    CV falls steeply with FPPG (bench players ~1.0, stars ~0.35), so it is estimated only
    over the players a league like this one actually rosters: top_n = teams * roster size.
    """
    d = live_prev[live_prev["n"] >= min_games].nlargest(top_n, "live_fppg")
    return float((d["sd_fp"] / d["live_fppg"]).median())


def shrunk_sigma(mu, sd, n, k: float, cv_typical: float) -> np.ndarray:
    """sigma = (k * cv*mu + n * sd) / (k + n); players without games get cv*mu."""
    mu, n = np.asarray(mu, float), np.asarray(n, float)
    sd = np.nan_to_num(np.asarray(sd, float), nan=0.0)
    base = cv_typical * np.maximum(mu, 0)
    w = np.where(n >= 2, n, 0)  # sd undefined for fewer than 2 games
    return (k * base + w * sd) / (k + w)
