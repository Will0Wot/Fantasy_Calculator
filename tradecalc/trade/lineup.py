"""Lineup simulator (section 5.2): how many games each rostered player actually starts.

Daily lock: each day, fill the starting slots from players with a game that day to
maximize expected FP (a * mu), respecting position eligibility (max-weight bipartite
matching via linear_sum_assignment). Weekly lock: one set of starters for the week
maximizing sum(a * games * mu). Bench games score nothing.

Availability can vary by game (an OUT player returns mid-week), so besides g (starts)
the result carries sum_a = sum of a over started games and sum_var = sum of a*(1-a),
which is what the matchup math needs.

Fill-ins (daily lock, lineup.fill_in = psi): a manager who learns before lock that a starter
is out benches him and starts the next-best player with a game. psi is the share of absences
known before lock (1 = every absence is announced in time). Each day players with a game are
ranked by projection; with S slots usable that day, player i starts if fewer than S players
ranked above him look available, which is computed exactly by a Poisson-binomial recursion:
  P(i starts and plays) = a_i * P(#(higher-ranked who look available) < S)
Without fill-ins (the audit found this undervalues depth, see docs/model_audit.md) a started
player who sits leaves his slot empty.
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
    sum_free: np.ndarray | None = None     # fill-in model: expected starts if available (sum over days)
    plan: list | None = None               # fill-in model, per day: (ranked indices, slots, a, psi) for Monte Carlo
    var_mu: float | None = None            # fill-in model: exact variance of sum(mu x starts-and-plays)

    def starts(self) -> dict[int, int]:
        return {i: int(x) for i, x in zip(self.ids, self.g)}


def slot_list(cfg) -> list[str]:
    return [s for s, n in cfg["roster"]["slots"].items() for _ in range(n)]


def eligibility_matrix(positions: list[tuple[str, ...]], slots: list[str], slot_elig: dict) -> np.ndarray:
    key = tuple(slots)
    cache = _ELIG_CACHE.setdefault(key, {})
    rows = []
    for ps in positions:
        r = cache.get(ps)
        if r is None:
            r = cache[ps] = [any(pos in slot_elig[s] for pos in ps) for s in slots]
        rows.append(r)
    return np.array(rows, dtype=bool).reshape(len(positions), len(slots))


_ELIG_CACHE: dict = {}
_SLOTS_CACHE: dict = {}


def usable_slots(elig: np.ndarray) -> int:
    """Largest number of these players that can start at once (cached by eligibility pattern)."""
    key = (elig.shape, np.packbits(elig).tobytes())
    s = _SLOTS_CACHE.get(key)
    if s is None:
        if len(_SLOTS_CACHE) > 200_000:
            _SLOTS_CACHE.clear()
        s = _SLOTS_CACHE[key] = int(best_lineup(np.ones(len(elig)), elig).sum()) if len(elig) else 0
    return s


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

    psi = cfg["lineup"].get("fill_in")
    if cfg["lineup"]["lock"] != "weekly" and psi is not None and cap is None:
        return _fill_in_week(roster, a, mu, elig, float(psi))

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


def _fill_in_week(roster, a: np.ndarray, mu: np.ndarray, elig: np.ndarray, psi: float) -> WeekResult:
    """Fill-in lineups for every day of the week at once (vectorized over days).

    Players are processed best-first. State per day: k = how many higher-ranked players look
    available at lock (k = S_t means "all S_t usable slots taken"), carrying the first two
    moments of the projected points accumulated so far, so the day's variance is exact."""
    n, days = a.shape
    has = a > 0                                                  # n x days: team has a game (and not ruled out for sure)
    S = np.array([usable_slots(elig[has[:, t]]) for t in range(days)], dtype=int)
    order = np.argsort(-mu, kind="stable")
    R = np.where(has, 1.0 - psi * (1.0 - a), 0.0)                # looks available at lock
    # Nights with no more players than usable slots: everyone with a game starts (closed form).
    occ, q, free = R.copy(), a.copy(), has.astype(float)
    crowded = np.flatnonzero(has.sum(axis=0) > S)
    calm = np.setdiff1d(np.arange(days), crowded)
    var_mu = float((mu[:, None] ** 2 * a[:, calm] * (1 - a[:, calm])).sum())
    if len(crowded):
        ac, hc, Rc, Sc = a[:, crowded], has[:, crowded], R[:, crowded], S[crowded]
        K = int(Sc.max())
        k = np.arange(K + 2)
        below = (k[None, :] < Sc[:, None]).astype(float)         # nights x k: a slot is still free
        at_S = (k[None, :] == Sc[:, None]).astype(float)         # nights x k: all slots taken (absorbing)
        X0, X1, X2 = (np.zeros((len(crowded), K + 2)) for _ in range(3))
        X0[:, 0] = 1.0                 # P(k), E[points; k], E[points^2; k]
        for i in order:
            if not hc[i].any():
                continue
            at, r, c = ac[i][:, None], Rc[i][:, None], float(mu[i])
            f = (X0 * below).sum(axis=1)
            free[i, crowded], occ[i, crowded], q[i, crowded] = f * hc[i], Rc[i] * f, ac[i] * f
            # move up one k with prob r (looks available); if he plays and a slot is free he adds c points
            add1 = at * below * X0
            Y0, Y1, Y2 = X0 * r, X1 * r + c * add1, X2 * r + at * below * (2 * c * X1) + c * c * add1
            n0, n1, n2 = X0 * (1 - r) + Y0 * at_S, X1 * (1 - r) + Y1 * at_S, X2 * (1 - r) + Y2 * at_S
            n0[:, 1:] += (Y0 * below)[:, :-1]
            n1[:, 1:] += (Y1 * below)[:, :-1]
            n2[:, 1:] += (Y2 * below)[:, :-1]
            X0, X1, X2 = n0, n1, n2
        var_mu += float(np.maximum(X2.sum(axis=1) - X1.sum(axis=1) ** 2, 0.0).sum())
    g, sum_a = occ.sum(axis=1), q.sum(axis=1)
    sum_var = (free * a * (1 - a)).sum(axis=1)
    plan = []
    for t in range(days):
        o = order[has[order, t]]
        if len(o):
            plan.append((o, int(S[t]), a[o, t].copy(), psi))
    gp = np.flatnonzero(sum_a > 0)
    return WeekResult(list(roster), g, sum_a, sum_var, gp, sum_a[gp], sum_free=free.sum(axis=1), plan=plan, var_mu=var_mu)
