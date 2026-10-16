"""Fantasy points from box scores, and per-player live stats for the season."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def fantasy_points(box_score_row: Mapping, scoring: Mapping[str, float]) -> float:
    """Apply league weights to one box-score row. Keys are matched case-insensitively."""
    row = {str(k).upper(): v for k, v in dict(box_score_row).items()}
    return float(sum(w * float(row.get(stat.upper(), 0) or 0) for stat, w in scoring.items()))


def add_fp(logs: pd.DataFrame, scoring: Mapping[str, float]) -> pd.DataFrame:
    """Vectorized fantasy_points over a game-log frame (lower-case columns, as stored in SQLite)."""
    out = logs.copy()
    fp = np.zeros(len(out))
    for stat, w in scoring.items():
        fp += w * out[stat.lower()].to_numpy(dtype=float)
    out["fp"] = fp
    return out


def live_stats(logs: pd.DataFrame, scoring: Mapping[str, float], recent_games: int = 10) -> pd.DataFrame:
    """Per-player season stats from game logs (section 4.3).

    Returns one row per player_id with: n, live_fppg, sd_fp, mpg_season, mpg_last5,
    fppg_recent (mean FP over the last `recent_games`), team, player_name.
    Games with 0 minutes (DNP rows, if any) are excluded.
    """
    df = logs if "fp" in logs else add_fp(logs, scoring)
    df = df[df["min"] > 0].sort_values(["player_id", "game_date"])
    g = df.groupby("player_id")
    out = pd.DataFrame({
        "player_name": g["player_name"].last(),
        "team": g["team"].last(),
        "n": g.size(),
        "live_fppg": g["fp"].mean(),
        "sd_fp": g["fp"].std(ddof=1),
        "mpg_season": g["min"].mean(),
        "mpg_last5": g["min"].apply(lambda s: s.tail(5).mean()),
        "fppg_recent": g["fp"].apply(lambda s: s.tail(recent_games).mean()),
        "last_game": g["game_date"].max(),
    })
    return out
