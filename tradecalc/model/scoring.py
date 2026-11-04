"""Fantasy points from box scores, and per-player live stats for the season."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


# Per-game bonuses: how many of points, rebounds, assists, steals and blocks must reach 10.
BONUS_TENS = {"DD": 2, "TD": 3, "QD": 4}
TEN_STATS = ("pts", "reb", "ast", "stl", "blk")


def fantasy_points(box_score_row: Mapping, scoring: Mapping[str, float]) -> float:
    """Apply league weights to one box-score row. Keys are matched case-insensitively. A bonus (DD/TD/QD)
    counts what the row says for it: a 0/1 for one game, a rate for a projection (0 if absent)."""
    row = {str(k).upper(): v for k, v in dict(box_score_row).items()}
    return float(sum(w * float(row.get(stat.upper(), 0) or 0) for stat, w in scoring.items()))


def tens(df: pd.DataFrame) -> np.ndarray:
    """Per game: how many of points, rebounds, assists, steals and blocks reached 10."""
    return sum((df[c].to_numpy(dtype=float) >= 10).astype(int) for c in TEN_STATS if c in df)


def add_fp(logs: pd.DataFrame, scoring: Mapping[str, float]) -> pd.DataFrame:
    """Vectorized fantasy_points over a game-log frame (lower-case columns, as stored in SQLite).
    Double-, triple- and quadruple-double bonuses are scored from each game's box score."""
    out = logs.copy()
    fp = np.zeros(len(out))
    t = None
    for stat, w in scoring.items():
        s = stat.upper()
        if s in BONUS_TENS:
            t = tens(out) if t is None else t
            fp += w * (t >= BONUS_TENS[s])
        else:
            fp += w * out[stat.lower()].to_numpy(dtype=float)
    out["fp"] = fp
    return out


def bonus_rates(logs: pd.DataFrame, scoring: Mapping[str, float], key_of, min_games: int = 10) -> pd.Series:
    """Expected bonus points per game (DD/TD/QD) by player key, from his games in `logs`: what a
    projection of averages (ESPN's, the engine's) leaves out. Empty when the league scores no bonus."""
    bonus = {s.upper(): w for s, w in scoring.items() if s.upper() in BONUS_TENS}
    if not bonus or not len(logs):
        return pd.Series(dtype=float)
    df = logs[logs["min"] > 0]
    t = tens(df)
    pts = sum(w * (t >= BONUS_TENS[s]) for s, w in bonus.items())
    g = pd.DataFrame({"key": df["player_name"].map(key_of).to_numpy(), "b": pts}).groupby("key")["b"]
    return g.mean()[g.size() >= min_games]


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
