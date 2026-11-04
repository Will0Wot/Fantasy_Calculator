"""Per-game fantasy projections from the possession simulator, for the season model.

Every remaining NBA game is simulated (possession by possession) and each player's expected fantasy
points in that game, if he plays, is saved: data/possession/projections_<season>.parquet. The season
model (model/values.py) then projects each game as

    mu(game) = (1 - w) * season-model average + w * this game's simulation,   w = engine.game_weight

so weekly matchups, trades and streaming see each game's opponent, lineup and minutes. On three test
seasons the 50/50 average predicted player-games better than either alone (docs/model_audit.md).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

MIN_PLAYED = 20      # a player-game needs this many simulations with him playing to get a projection
KNOWN_POSS = 150     # play-by-play possessions (about two games) before the simulation's own level is trusted
# Each player's average box score over his simulated games, saved beside the per-game file so a league
# with other scoring can rescale it (fantasy points are linear in these; DD/TD/QD are per-game rates).
MIX_STATS = ["min", "pts", "reb", "ast", "stl", "blk", "tov", "fgm", "fga", "ftm", "fta", "fg3m", "fg3a",
             "oreb", "dreb", "dd", "td", "qd"]


def path(season: str) -> Path:
    from tradecalc.sim.possession import LIVE

    return LIVE / f"projections_{season}.parquet"


def settings() -> tuple[float, int]:
    """(weight on the game simulation, simulations per game) from config engine.game_weight / game_sims."""
    from tradecalc.config import load_config

    eng = load_config().get("engine", {})
    return float(eng.get("game_weight", 0.5)), int(eng.get("game_sims", 300))


def simulate_rest(world, weeks: list[list[tuple]], avail, n: int = 300, seed: int = 0, log=None) -> pd.DataFrame:
    """Simulate the given games, one fantasy week (list of (date, home, away)) at a time, and return
    one row per player-game: key, date, chance he plays, fantasy points if he plays (mean, sd),
    minutes, and how much play-by-play the simulator has on him (possessions)."""
    from tradecalc.sim.possession.engine import simulate_games

    expo = world.possession.prof.players.poss_on
    rows = []
    for i, games in enumerate(weeks):
        if not games:
            continue
        S = simulate_games(world, games, avail, world.possession, n=n, seed=seed + i)
        for (k, d), fp in S.fp.items():
            pl = S.played[(k, d)]
            if pl.sum() < MIN_PLAYED:
                continue
            x, mk = fp[pl], world.possession.key(k)
            box = S.box_mean[(k, d)]
            rows.append({"key": k, "date": d.isoformat(), "p_play": float(pl.mean()), "fp": float(x.mean()),
                         "fp_sd": float(x.std()), "poss": float(expo.get(mk, 0.0)) if mk else 0.0,
                         **{s_: float(box.get(s_, 0.0)) for s_ in MIX_STATS}})
        if log:
            log(f"  week {i + 1}/{len(weeks)}: {len(games)} games")
    return pd.DataFrame(rows, columns=["key", "date", "p_play", "fp", "fp_sd", "poss"] + MIX_STATS)


def project(world, state, coaches=None, n: int = 300, seed: int = 0, log=None) -> pd.DataFrame:
    """Every game left this season (live): see simulate_rest."""
    from tradecalc.sim.engine import live_avail
    from tradecalc.teams import to_nba

    weeks = []
    for w in state.cal.weeks:
        days = {d for d in w.days() if d >= state.today}
        weeks.append(sorted({(g["date"], to_nba(g["home"]), to_nba(g["away"])) for g in state.league.pro_games
                             if g["date"] in days}))
    return simulate_rest(world, weeks, live_avail(world, state, coaches), n=n, seed=seed, log=log)


def mix_path(season: str) -> Path:
    return path(season).with_name(f"projections_{season}_mix.parquet")


def save(df: pd.DataFrame, season: str, asof: date, scoring: dict | None = None) -> Path:
    """Only what the season model reads, compressed (~70 KB a season; the file is committed daily), plus
    each player's average simulated box score (a few KB) and the scoring the per-game points use."""
    import json

    p = path(season)
    p.parent.mkdir(parents=True, exist_ok=True)
    slim = df[["key", "date"]].assign(fp=df.fp.round(1).astype("float32"), poss=df.poss.round().astype("int32"),
                                      asof=asof.isoformat())
    slim.to_parquet(p, index=False, compression="zstd")
    have = [s_ for s_ in MIX_STATS if s_ in df]
    if scoring and have:
        mix = df.groupby("key")[have].mean().astype("float32").round(3).reset_index()
        mix["scoring"] = json.dumps(dict(scoring), sort_keys=True)
        mix.to_parquet(mix_path(season), index=False, compression="zstd")
    return p


def load(season: str) -> pd.DataFrame | None:
    p = path(season)
    return pd.read_parquet(p) if p.exists() else None


def rescale(df: pd.DataFrame, season: str, scoring: dict) -> pd.DataFrame | None:
    """The per-game file under another league's scoring: each player's points scaled by his box-score mix
    (his simulated average scored the new way over the old way). None when it can't be done (no mix
    file yet); the same scoring returns df unchanged."""
    import json

    from tradecalc.model.scoring import fantasy_points

    p = mix_path(season)
    if not p.exists():
        return None
    mix = pd.read_parquet(p)
    if not len(mix):
        return None
    old = json.loads(mix["scoring"].iloc[0])
    if {k.upper(): float(v) for k, v in old.items()} == {k.upper(): float(v) for k, v in scoring.items()}:
        return df
    rows = mix.drop(columns="scoring").set_index("key")
    ratio = {}
    for k, r in rows.iterrows():
        box = {s_.upper(): v for s_, v in r.items()}
        a, b = fantasy_points(box, scoring), fantasy_points(box, old)
        if b > 0 and a > 0:
            ratio[k] = a / b
    out = df[df.key.isin(ratio)].copy()
    out["fp"] = out["fp"] * out["key"].map(ratio)
    return out


def apply(players: dict, season: str, today: date, weight: float, key_of, scoring: dict | None = None) -> int:
    """Per-game projections from this season's saved simulations (see apply_df), rescaled to `scoring`
    when the league scores differently from the saved file."""
    df = load(season)
    if df is not None and scoring:
        df = rescale(df, season, scoring)
    return apply_df(players, df, today, weight, key_of) if df is not None else 0


def apply_df(players: dict, df: pd.DataFrame, today: date, weight: float, key_of) -> int:
    """Give each player a per-game projection for the simulated games still ahead:
    (1 - weight) * season-model average + weight * that game's simulation. Returns how many players
    got one."""
    if weight <= 0 or not len(df):
        return 0
    df = df[df.date >= today.isoformat()]
    if "poss" not in df:
        df = df.assign(poss=float(KNOWN_POSS))
    by_key = {k: (dict(zip(g.date, g.fp)), float(g.poss.iloc[0])) for k, g in df.groupby("key")}
    done = 0
    for p in players.values():
        games, poss = by_key.get(key_of(p.name), ({}, KNOWN_POSS))
        if not games:
            continue
        lam = min(1.0, poss / KNOWN_POSS)
        if lam < 1:
            # the simulator barely knows him (a rookie's first games): it only has a position average,
            # so his level phases in from the season model's; the game-to-game pattern is the simulation's
            avg = float(np.mean(list(games.values())))
            level = lam * avg + (1 - lam) * p.mu
            games = {d: fp * level / avg for d, fp in games.items()} if avg > 0 else {}
        p.mu_by_date = {date.fromisoformat(d): (1 - weight) * p.mu + weight * fp for d, fp in games.items()
                        if np.isfinite(fp)}
        done += 1
    return done
