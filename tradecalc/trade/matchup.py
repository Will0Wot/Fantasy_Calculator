"""Weekly matchup math (section 6).

M_team = sum over started games of a * mu
V_team = sum over started games of (a * sigma^2 + a(1-a) * mu^2)
P(win) = Phi((M_you - M_opp) / sqrt(V_you + V_opp)); E[wins] sums P(win) over remaining weeks.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
from scipy.stats import norm

from tradecalc.trade.lineup import WeekResult, simulate_week


@dataclass(frozen=True)
class WeekDist:
    M: float
    V: float
    lineup: WeekResult


def _from_day(week, today: date | None):
    return today if today is not None and week.start <= today <= week.end else None


def week_dist(roster, week, state, today: date | None = None) -> WeekDist:
    """Team distribution for one week; cached by (roster, week, start day)."""
    cache = state.__dict__.setdefault("_week_cache", {})
    fd = _from_day(week, today)
    key = (tuple(sorted(roster)), week.number, fd)
    hit = cache.get(key)
    if hit is not None:
        return hit
    res = simulate_week(list(key[0]), week, state, from_day=fd)
    ps = [state.players[i] for i in res.ids]
    mu = np.array([p.mu for p in ps])
    sig = np.array([p.sigma for p in ps])
    M = float(mu @ res.sum_a)
    V = float((sig ** 2) @ res.sum_a + (mu ** 2) @ res.sum_var)
    d = WeekDist(M, V, res)
    cache[key] = d
    return d


def p_win(M_you: float, V_you: float, M_opp: float, V_opp: float) -> float:
    s = np.sqrt(V_you + V_opp)
    if s == 0:
        return 1.0 if M_you > M_opp else 0.0 if M_you < M_opp else 0.5
    return float(norm.cdf((M_you - M_opp) / s))


def season_outlook(team_id: int, rosters: dict[int, list[int]], state, today: date | None = None) -> dict:
    """Per remaining week: opponent, M, V, P(win). Plus totals: E[wins], mean M."""
    today = state.today if today is None else today
    weeks = []
    for w in state.cal.remaining_weeks(today):
        opp = state.cal.opponents.get(team_id, {}).get(w.number)
        me = week_dist(rosters[team_id], w, state, today)
        bank_me = state.banked(team_id, w)[0] if hasattr(state, "banked") else 0.0
        if bank_me:
            me = WeekDist(me.M + bank_me, me.V, me.lineup)
        if opp is None:
            weeks.append({"week": w.number, "opp": None, "M": me.M, "V": me.V, "p": None,
                          "starts": int(me.lineup.g.sum())})
            continue
        them = week_dist(rosters[opp], w, state, today)
        bank_them = state.banked(opp, w)[0] if hasattr(state, "banked") else 0.0
        if bank_them:
            them = WeekDist(them.M + bank_them, them.V, them.lineup)
        weeks.append({"week": w.number, "opp": opp, "M": me.M, "V": me.V, "M_opp": them.M, "V_opp": them.V,
                      "p": p_win(me.M, me.V, them.M, them.V), "starts": int(me.lineup.g.sum())})
    played = [x for x in weeks if x["p"] is not None]
    return {
        "weeks": weeks,
        "e_wins": sum(x["p"] for x in played),
        "pts_per_week": float(np.mean([x["M"] for x in weeks])) if weeks else 0.0,
        "starts": sum(x["starts"] for x in weeks),
    }


# --- Monte Carlo cross-check (section 6.4) -------------------------------------------

def _sim_team(d: WeekDist, state, n: int, rng: np.random.Generator) -> np.ndarray:
    res = d.lineup
    if res.game_player is None or len(res.game_player) == 0:
        return np.zeros(n)
    ps = [state.players[res.ids[i]] for i in res.game_player]
    mu = np.array([p.mu for p in ps])
    sig = np.maximum(np.array([p.sigma for p in ps]), 1e-6)
    a = res.game_a
    # Per-game FP ~ Gamma with mean mu and sd sigma: non-negative, matches the first two moments.
    shape, scale = (mu / sig) ** 2, sig ** 2 / np.maximum(mu, 1e-9)
    fp = rng.gamma(shape, scale, size=(n, len(mu))) * (mu > 0)
    plays = rng.random((n, len(mu))) < a
    return (fp * plays).sum(axis=1)


def p_win_monte_carlo(d_you: WeekDist, d_opp: WeekDist, state, n: int = 10000, seed: int | None = 0,
                      tie_value: float = 0.5) -> float:
    rng = np.random.default_rng(seed)
    you, opp = _sim_team(d_you, state, n, rng), _sim_team(d_opp, state, n, rng)
    return float(((you > opp) + tie_value * (you == opp)).mean())


def season_outlook_mc(team_id, rosters, state, n=10000, seed=0, today=None) -> dict:
    """Same as season_outlook but with simulated win probabilities (--precise)."""
    out = season_outlook(team_id, rosters, state, today)
    today = state.today if today is None else today
    tie = state.cfg["league"]["tie_value"]
    for x in out["weeks"]:
        if x["opp"] is None:
            continue
        w = state.cal.weeks[x["week"] - 1]
        x["p"] = p_win_monte_carlo(week_dist(rosters[team_id], w, state, today),
                                   week_dist(rosters[x["opp"]], w, state, today), state, n, seed, tie)
    out["e_wins"] = sum(x["p"] for x in out["weeks"] if x["p"] is not None)
    return out
