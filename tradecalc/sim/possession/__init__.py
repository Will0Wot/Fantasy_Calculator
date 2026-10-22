"""Possession-level game simulator (RotoBot-style). See engine.py for the game model, profiles.py for
how players are described, defense.py for who's guarding whom.

model_for(season, asof, pos_of) builds the model as of a date:
  this season's play-by-play before `asof` (weight 1) + earlier seasons (profiles.PREV_WEIGHTS), and defender
  tables from the two seasons before `season` (never the simulated season: no peeking).
Play-by-play counts come from raw play-by-play when it is on this machine (data/pbp/), otherwise
from committed count files: data/seed/pbp_counts_<season>_* (finished seasons) and
data/possession/counts_<season>_* (the live season, refreshed daily by `tradecalc possession`).
"""
from __future__ import annotations

from pathlib import Path

from tradecalc.config import load_config

ROOT = Path(__file__).resolve().parents[2]
PBP_DIR = ROOT / "data" / "pbp"
SEED = ROOT / "data" / "seed"
LIVE = ROOT / "data" / "possession"
THIN_POSS = 500      # under this many (weighted) possessions in this + last season, older seasons are added


def enabled() -> bool:
    return load_config().get("engine", {}).get("mode", "box") == "possession"


def sims() -> int:
    return int(load_config().get("engine", {}).get("sims", 5000))


def season_counts(season: str, before: str | None = None):
    """Counts for `season` (only events before `before` if given), or None."""
    from tradecalc.sim.possession.profiles import Counts, counts, load_events

    E, PL = load_events(PBP_DIR, season, before)
    if len(E):
        return counts(E, PL)
    for base in (LIVE / f"counts_{season}", SEED / f"pbp_counts_{season}"):
        c = Counts.load(base)
        if c is not None:
            if before is not None and getattr(c, "asof", None) and c.asof > before:
                continue
            return c
    return None


def tracking_seasons(season: str) -> list[str]:
    from tradecalc.ingest.tracking import seed_paths
    from tradecalc.sim.history import prev_season

    out, s = [], season
    for _ in range(4):
        s = prev_season(s)
        if seed_paths(SEED, s)[0].exists():
            out.append(s)
        if len(out) == 2:
            break
    return out


_CACHE: dict = {}


def model_for(season: str, asof: str | None, pos_of, prev: str | None = None, cal: dict | None = None,
              back: tuple | None = None):
    """PossessionModel as of `asof` (None: everything so far). pos_of(name key) -> 'G' | 'F' | 'C'.
    back: weights of the seasons before (default profiles.PREV_WEIGHTS). Cached per (season, date,
    calibration, weights)."""
    import json

    import pandas as pd

    from tradecalc.ingest import tracking
    from tradecalc.sim.history import prev_season
    from tradecalc.sim.possession.defense import build as build_defense
    from tradecalc.sim.possession.engine import PossessionModel
    from tradecalc.sim.possession.profiles import PREV_WEIGHTS, Counts, from_counts

    prev = prev or prev_season(season)
    back = tuple(back if back is not None else PREV_WEIGHTS)
    cal = cal if cal is not None else load_config().get("engine", {}).get("calibration")
    key = (season, asof, prev, json.dumps(cal or {}, sort_keys=True), back)
    if key in _CACHE:
        return _CACHE[key]
    windows = []
    cur = season_counts(season, before=asof)
    if cur is not None:
        windows.append((cur, 1.0))
    s = prev
    for w in back:
        pc = season_counts(s)
        if pc is not None:
            windows.append((pc, w))
        s = prev_season(s)
    # players with little play-by-play in those windows (a missed season, a rookie's first games): add
    # their own older seasons at the same halving weights (own data beats a position average)
    expo = pd.Series(dtype=float)
    for c, w in windows:
        expo = expo.add(c.players.poss_on * w, fill_value=0)
    w = back[-1] if back else 1.0
    for _ in range(2):
        w, oc, s = w / 2, season_counts(s), prev_season(s)
        thin = [k for k in oc.players.index if expo.get(k, 0.0) < THIN_POSS] if oc is not None and windows else []
        if thin:
            ts_ = set(thin)
            windows.append((Counts(oc.players.loc[thin], oc.pairs[oc.pairs.passer.isin(ts_) | oc.pairs.shooter.isin(ts_)],
                                   oc.teams.iloc[:0], {k: oc.names.get(k, k) for k in thin}), w))
            expo = expo.add(oc.players.loc[thin].poss_on * w, fill_value=0)
    model = None
    if windows:
        prof = from_counts(windows, pos_of)
        ts = tracking_seasons(season)
        d, m = tracking.load(SEED, ts)
        model = PossessionModel(prof, build_defense(d, m, ts), cal)
    if len(_CACHE) >= 16:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = model
    return model
