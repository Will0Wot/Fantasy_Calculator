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
    V: float              # effective variance used for win probabilities (= V_games + tau^2 U + rho R)
    lineup: WeekResult
    V_games: float = 0.0  # game-to-game noise only (independent games, known player levels)
    U: float = 0.0        # sum over players of (expected starts x mu x s_i)^2: uncertainty in each player's level
    R: float = 0.0        # extra variance if a player's games in the week are absent together (clustered)


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
    if res.q_day is not None and res.mu_day is not None:
        pts_i = (res.q_day * res.mu_day).sum(axis=1)               # each game at its own projection
    else:
        pts_i = mu * res.sum_a
    M = float(pts_i.sum())
    m = state.cfg["model"]
    if res.sum_free is not None:
        # Fill-ins: when a starter sits, the team loses his projection minus the fill-in's, so
        # absence variance uses that marginal value (replacement = best player outside the top slots).
        n_slots = int(sum(state.cfg["roster"]["slots"].values()))
        srt = np.sort(mu)[::-1]
        val = np.maximum(mu - (srt[n_slots] if len(srt) > n_slots else 0.0), 0.0)
        g, sa = res.sum_free, res.sum_a
    else:
        val, g, sa = mu, res.g, res.sum_a
    if res.var_mu is not None:
        V_games = float((sig ** 2) @ res.sum_a + res.var_mu)
    else:
        V_games = float((sig ** 2) @ res.sum_a + (mu ** 2) @ res.sum_var)
    # Two variance sources the game-by-game formula ignores (found by the model audit):
    #  U: a player's true level is uncertain (projection error), and that error is shared by all
    #     his games in the week; it shrinks as he plays games: s_i = sqrt(k / (k + n_i)).
    #  R: missed games cluster (multi-game absences): extra variance g(g-1) a(1-a) mu^2.
    k = m["k"]
    n = np.array([float(getattr(p, "n", 0) or 0) for p in ps])
    s = np.sqrt(k / (k + n))
    U = float(((pts_i * s) ** 2).sum())
    abar = np.divide(sa, g, out=np.zeros_like(sa), where=g > 0)
    R = float((val ** 2 * g * np.maximum(g - 1, 0) * abar * (1 - abar)).sum())
    V = V_games + m.get("proj_uncertainty", 0.0) ** 2 * U + m.get("absence_cluster", 0.0) * R
    d = WeekDist(M, V, res, V_games, U, R)
    cache[key] = d
    return d


def p_win(M_you: float, V_you: float, M_opp: float, V_opp: float, shrink: float = 1.0) -> float:
    """P(you outscore the opponent) = Phi(shrink * (M_you - M_opp) / sqrt(V_you + V_opp)).

    shrink < 1 corrects projected gaps between teams that are systematically too wide (real
    scores regress toward the middle); fitted by the model audit, see config model.matchup_shrink."""
    s = np.sqrt(V_you + V_opp)
    if s == 0:
        return 1.0 if M_you > M_opp else 0.0 if M_you < M_opp else 0.5
    return float(norm.cdf(shrink * (M_you - M_opp) / s))


def matchup_p(d_you: "WeekDist", d_opp: "WeekDist", cfg) -> float:
    return p_win(d_you.M, d_you.V, d_opp.M, d_opp.V, cfg["model"].get("matchup_shrink", 1.0))


def season_outlook(team_id: int, rosters: dict[int, list[int]], state, today: date | None = None) -> dict:
    """Per remaining week: opponent, M, V, P(win). Plus totals: E[wins], mean M."""
    today = state.today if today is None else today
    weeks = []
    for w in state.cal.remaining_weeks(today):
        opp = state.cal.opponents.get(team_id, {}).get(w.number)
        me = week_dist(rosters[team_id], w, state, today)
        bank_me = state.banked(team_id, w)[0] if hasattr(state, "banked") else 0.0
        if bank_me:
            me = WeekDist(me.M + bank_me, me.V, me.lineup, me.V_games, me.U, me.R)
        if opp is None:
            weeks.append({"week": w.number, "opp": None, "M": me.M, "V": me.V, "p": None,
                          "starts": int(round(float(me.lineup.g.sum())))})
            continue
        them = week_dist(rosters[opp], w, state, today)
        bank_them = state.banked(opp, w)[0] if hasattr(state, "banked") else 0.0
        if bank_them:
            them = WeekDist(them.M + bank_them, them.V, them.lineup, them.V_games, them.U, them.R)
        weeks.append({"week": w.number, "opp": opp, "M": me.M, "V": me.V, "M_opp": them.M, "V_opp": them.V,
                      "p": matchup_p(me, them, state.cfg), "starts": int(round(float(me.lineup.g.sum())))})
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
    if res.plan is not None:
        return _sim_team_fill_in(res, state, n, rng)
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


def _sim_team_fill_in(res, state, n: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate each day: who plays, who is benched before lock, and who fills in."""
    ps = [state.players[i] for i in res.ids]
    mu = np.array([p.mu for p in ps])
    sig = np.maximum(np.array([p.sigma for p in ps]), 1e-6)
    total = np.zeros(n)
    for order, S, a, psi, *day_mu in res.plan:
        u = rng.random((n, len(order)))
        plays = u < a
        looks = u < 1.0 - psi * (1.0 - a)          # not known to be out before lock
        starts = plays & (np.cumsum(looks, axis=1) <= S)
        m, s = (day_mu[0] if day_mu else mu[order]), sig[order]
        shape, scale = (m / s) ** 2, s ** 2 / np.maximum(m, 1e-9)
        fp = rng.gamma(shape, scale, size=(n, len(order))) * (m > 0)
        total += (fp * starts).sum(axis=1)
    return total


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
