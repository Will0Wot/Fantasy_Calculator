"""Lineup simulator (section 5.2): how many games each rostered player actually starts.

Daily lock: each day, fill the starting slots from players with a game that day to
maximize expected FP (a * mu), respecting position eligibility (max-weight bipartite
matching via linear_sum_assignment). Weekly lock: one set of starters for the week
maximizing sum(a * games * mu). Bench games score nothing.

Availability can vary by game (an OUT player returns mid-week), so besides g (starts)
the result carries sum_a = sum of a over started games and sum_var = sum of a*(1-a),
which is what the matchup math needs.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class WeekResult:
    ids: list[int]
    g: np.ndarray        # starts per player
    sum_a: np.ndarray    # sum over started games of P(plays)
    sum_var: np.ndarray  # sum over started games of a*(1-a)
    game_player: np.ndarray | None = None  # per started game: index into ids (for Monte Carlo)
    game_a: np.ndarray | None = None       # per started game: P(plays)

    def starts(self) -> dict[int, int]:
        return {i: int(x) for i, x in zip(self.ids, self.g)}


def slot_list(cfg) -> list[str]:
    return [s for s, n in cfg["roster"]["slots"].items() for _ in range(n)]


def eligibility_matrix(positions: list[tuple[str, ...]], slots: list[str], slot_elig: dict) -> np.ndarray:
    return np.array([[any(pos in slot_elig[s] for pos in ps) for s in slots] for ps in positions], dtype=bool)


def best_lineup(values: np.ndarray, elig: np.ndarray) -> np.ndarray:
    """Boolean mask of players started, maximizing total value subject to slot eligibility."""
    started = np.zeros(len(values), dtype=bool)
    cand = np.flatnonzero(values > 0)
    if len(cand) == 0:
        return started
    w = np.where(elig[cand], values[cand, None], 0.0)
    rows, cols = linear_sum_assignment(-w)
    started[cand[rows[w[rows, cols] > 0]]] = True
    return started


def simulate_week(roster: list[int], week, state, from_day: date | None = None) -> WeekResult:
    cfg = state.cfg
    slots = slot_list(cfg)
    ps = [state.players[i] for i in roster]
    elig = eligibility_matrix([p.positions for p in ps], slots, cfg["roster"]["slot_eligibility"])
    mu = np.array([p.mu for p in ps])
    days = [d for d in week.days() if from_day is None or d >= from_day]
    # a[i, t]: P(player i plays on day t); 0 if his team has no game.
    a = np.array([[state.a_for(p, d) if state.cal.team_plays(p.pro_team, d) else 0.0 for d in days] for p in ps])
    a = a.reshape(len(ps), len(days))
    n = len(ps)
    g, sum_a, sum_var = np.zeros(n), np.zeros(n), np.zeros(n)
    cap = cfg["lineup"].get("weekly_games_cap")

    if cfg["lineup"]["lock"] == "weekly":
        started = best_lineup((a * mu[:, None]).sum(axis=1), elig)
        plays = (a > 0) & started[:, None]
        g = plays.sum(axis=1).astype(float)
        sum_a = np.where(plays, a, 0).sum(axis=1)
        sum_var = np.where(plays, a * (1 - a), 0).sum(axis=1)
        gp, gt = np.nonzero(plays)
        return WeekResult(list(roster), g, sum_a, sum_var, gp, a[gp, gt])
    else:
        used = 0.0
        gp_list, ga_list = [], []
        for t in range(len(days)):
            if cap is not None and used >= cap:
                break
            st = best_lineup(a[:, t] * mu, elig)
            idx = np.flatnonzero(st)
            if cap is not None and used + len(idx) > cap:
                # Partial day at the cap: keep the highest-value starters.
                idx = idx[np.argsort(-(a[idx, t] * mu[idx]))][: int(cap - used)]
            used += len(idx)
            g[idx] += 1
            sum_a[idx] += a[idx, t]
            sum_var[idx] += a[idx, t] * (1 - a[idx, t])
            gp_list.extend(idx)
            ga_list.extend(a[idx, t])
    return WeekResult(list(roster), g, sum_a, sum_var, np.array(gp_list, dtype=int), np.array(ga_list))
