"""Shrink live FPPG toward the prior (section 4.4)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def role_change_flags(live: pd.DataFrame, threshold: float, min_games: int) -> pd.Series:
    mpg = live["mpg_season"].replace(0, np.nan)
    return ((live["mpg_last5"] - mpg).abs() / mpg > threshold) & (live["n"] >= min_games)


def blend(
    prior: pd.Series,
    live: pd.DataFrame,
    k: float,
    *,
    role_threshold: float = 0.20,
    role_min_games: int = 5,
    recent_weight: float = 0.5,
    replacement: float | None = None,
) -> pd.DataFrame:
    """blended = (k*prior + n*live) / (k + n), per player.

    prior: prior FPPG indexed like `live` (NaN where unknown -> `replacement`, or live only).
    live: frame from scoring.live_stats (n, live_fppg, mpg_season, mpg_last5, fppg_recent).
    Role change: k is halved and the live term becomes
    (1 - recent_weight)*season FPPG + recent_weight*last-N FPPG.
    """
    df = live.copy()
    df["prior_fppg"] = prior.reindex(df.index)
    if replacement is not None:
        df["prior_fppg"] = df["prior_fppg"].fillna(replacement)
    df["role_change"] = role_change_flags(df, role_threshold, role_min_games)
    df["k_used"] = np.where(df["role_change"], k / 2, k)
    live_term = np.where(
        df["role_change"],
        (1 - recent_weight) * df["live_fppg"] + recent_weight * df["fppg_recent"],
        df["live_fppg"],
    )
    has_prior = df["prior_fppg"].notna()
    blended = (df["k_used"] * df["prior_fppg"].fillna(0) + df["n"] * live_term) / (df["k_used"] * has_prior + df["n"])
    df["blended_fppg"] = blended
    return df


def blend_prior_only(prior: pd.Series, index) -> pd.DataFrame:
    """Players with no games yet: blended = prior."""
    return pd.DataFrame({"prior_fppg": prior.reindex(index), "blended_fppg": prior.reindex(index),
                         "n": 0, "role_change": False})
