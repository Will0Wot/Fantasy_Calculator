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


def blend_minutes(
    prior: pd.Series,
    prior_min: pd.Series,
    live: pd.DataFrame,
    *,
    minutes_k: float,
    rate_k: float,
    recent_weight: float = 0.25,
    role_threshold: float = 0.20,
    role_min_games: int = 5,
    replacement: float | None = None,
    k_fallback: float = 6.0,
) -> pd.DataFrame:
    """FPPG = minutes x fantasy points per minute, each moved from the prior toward this season's
    games at its own speed. Minutes follow a player's role and are believed quickly (the prior is
    worth `minutes_k` games); points per minute are mostly shooting luck and are believed slowly
    (worth `rate_k` games at the prior's minutes). The live minutes give the last 5 games
    `recent_weight`. Players without prior minutes fall back to (k*prior + n*live)/(k+n).

    Tested out of sample on 2022-23..2025-26 against that single-weight blend: lower error in every
    season (3.94 vs 4.03 FPPG), and slow-starting veterans are no longer written off
    (docs/model_audit.md §10).

    prior_min: minutes per game the prior assumes (ESPN's projection, else last season's).
    """
    df = live.copy()
    df["prior_fppg"] = prior.reindex(df.index)
    if replacement is not None:
        df["prior_fppg"] = df["prior_fppg"].fillna(replacement)
    df["role_change"] = role_change_flags(df, role_threshold, role_min_games)
    n = df["n"].astype(float)
    pm = prior_min.reindex(df.index)
    pm = pm.where(pm > 0)
    season_min = df["mpg_season"].where(df["mpg_season"] > 0)
    live_min = (1 - recent_weight) * season_min + recent_weight * df["mpg_last5"].fillna(season_min)
    minutes = (minutes_k * pm + n * live_min) / (minutes_k + n)
    prior_rate = df["prior_fppg"] / pm
    w = n * season_min / pm                         # games, scaled by how many minutes they had
    rate = (rate_k * prior_rate + w * (df["live_fppg"] / season_min).fillna(prior_rate)) / (rate_k + w)
    has_prior = df["prior_fppg"].notna()
    simple = (k_fallback * df["prior_fppg"].fillna(0) + n * df["live_fppg"]) / (k_fallback * has_prior + n)
    split = minutes * rate
    df["blended_fppg"] = split.where(split.notna() & has_prior, simple)
    df["minutes_hat"] = minutes
    return df


def blend_prior_only(prior: pd.Series, index) -> pd.DataFrame:
    """Players with no games yet: blended = prior."""
    return pd.DataFrame({"prior_fppg": prior.reindex(index), "blended_fppg": prior.reindex(index),
                         "n": 0, "role_change": False})
