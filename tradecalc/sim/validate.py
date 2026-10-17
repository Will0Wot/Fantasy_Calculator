"""Calibration and validation of the simulator on past weeks.

Both functions replay real weeks: build the world as of a Monday, simulate the next 7 days
with each player's actual availability forced (so only minutes and box-score randomness are
tested), and compare to what happened.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd

from tradecalc.model.names import norm_name
from tradecalc.sim.engine import build_world_history, simulate_games


def _replay(H, params, asof: str, season: str, prev: str, n: int, seed: int = 1, days: int = 7):
    W = build_world_history(H, params, asof, season, prev)
    end = (date.fromisoformat(asof) + timedelta(days=days)).isoformat()
    tl = H.team_logs[(H.team_logs.game_date >= asof) & (H.team_logs.game_date < end) & H.team_logs.home]
    games = [(date.fromisoformat(d), h, a) for d, h, a in zip(tl.game_date, tl.team, tl.opp)]
    act = H.logs[(H.logs.game_date >= asof) & (H.logs.game_date < end)].copy()
    act["key"] = act.player_name.map(norm_name)
    act["d"] = act.game_date.map(date.fromisoformat)
    A = act.drop_duplicates(["key", "d"]).set_index(["key", "d"])
    S = simulate_games(W, games, lambda k, t, d: 1.0 if (k, d) in A.index else 0.0, n=n, seed=seed)
    return W, S, A, act


def fit_calibration(H, params: dict, season: str, prev: str, mondays: list[str], n: int = 1000) -> dict:
    """rotation_minutes_share: share of team minutes played by the simulated rotation.
    rate_calibration: actual / simulated fantasy points per minute (global scale)."""
    shares, sums, fp_act, fp_sim, min_act = [], [], 0.0, 0.0, 0.0
    p = dict(params, rate_calibration=1.0)
    for m in mondays:
        W, S, A, act = _replay(H, p, m, season, prev, n)
        rot = {k for ks in W.rotation.values() for k in ks}
        sums += [W.players.loc[ks, "exp_min"].sum() for ks in W.rotation.values()]
        tm = act.groupby(["game_id", "team"])["min"].sum()
        rm = act[act.key.isin(rot)].groupby(["game_id", "team"])["min"].sum()
        shares.append((rm / tm).mean())
        for kd, bm in S.box_mean.items():
            if kd in A.index and S.played[kd].all():
                a = A.loc[kd]
                fp_act += a["fp"]
                fp_sim += bm["fp"] / max(bm["min"], 1e-9) * a["min"]
                min_act += a["min"]
    return {"rotation_minutes_share": round(float(np.mean(shares)), 4),
            "rotation_exp_min_sum": round(float(np.mean(sums)), 1),
            "rate_calibration": round(float(fp_act / fp_sim), 4)}


def coverage(H, params: dict, season: str, prev: str, mondays: list[str], n: int = 4000) -> dict:
    """Per player-game: share of actual FP inside the simulated 50% / 80% intervals, and bias.
    Per player-week: same for weekly totals (sum over the week's games)."""
    rows = []
    for m in mondays:
        W, S, A, _ = _replay(H, params, m, season, prev, n)
        for (k, d), fp in S.fp.items():
            if (k, d) in A.index and S.played[(k, d)].all():
                q = np.quantile(fp, [0.1, 0.25, 0.5, 0.75, 0.9])
                rows.append({"week": m, "key": k, "actual": A.loc[(k, d)]["fp"], "mean": fp.mean(),
                             **dict(zip(["q10", "q25", "q50", "q75", "q90"], q)), "_s": fp})
    R = pd.DataFrame(rows)
    wk = []
    for (m, k), g in R.groupby(["week", "key"]):
        tot = np.sum(np.stack(g["_s"].to_list()), axis=0)
        q = np.quantile(tot, [0.1, 0.25, 0.75, 0.9])
        a = g["actual"].sum()
        wk.append({"in80": q[0] <= a <= q[3], "in50": q[1] <= a <= q[2], "err": tot.mean() - a})
    WK = pd.DataFrame(wk)
    return {
        "player_games": len(R),
        "game_in80": float(((R.actual >= R.q10) & (R.actual <= R.q90)).mean()),
        "game_in50": float(((R.actual >= R.q25) & (R.actual <= R.q75)).mean()),
        "game_below_median": float((R.actual < R.q50).mean()),
        "game_bias": float((R["mean"] - R.actual).mean()),
        "game_mae": float((R["mean"] - R.actual).abs().mean()),
        "player_weeks": len(WK),
        "week_in80": float(WK.in80.mean()), "week_in50": float(WK.in50.mean()),
        "week_bias": float(WK.err.mean()),
    }


def history_avail(H, W, params, asof: str, season: str):
    """Backtest availability: a player's games-played rate this season so far (shrunk to 0.85
    over 10 team games), times the rest-model factor on back-to-backs. No injury report."""
    from tradecalc.sim.engine import rest_factor

    L = H.logs[(H.logs.season == season) & (H.logs.game_date < asof)]
    tg = H.team_logs[(H.team_logs.season == season) & (H.team_logs.game_date < asof)].groupby("team").size()
    apps = L.assign(key=L.player_name.map(norm_name)).groupby(["key", "team"]).size()
    first = L.assign(key=L.player_name.map(norm_name)).groupby(["key", "team"])["game_date"].min()
    rate = {}
    for (k, t), a in apps.items():
        team_games = H.team_logs[(H.team_logs.team == t) & (H.team_logs.season == season)
                                 & (H.team_logs.game_date >= first[(k, t)]) & (H.team_logs.game_date < asof)].shape[0]
        rate[(k, t)] = (10 * 0.85 + a) / (10 + max(team_games, a))
    P = W.players

    from tradecalc.config import load_config

    acal = load_config()["model"].get("availability_calibration")

    def avail(k, t, d):
        base = min(rate.get((k, t), 0.85), 0.97)
        if acal:  # same calibration as the live model (raw rates are too extreme)
            base = acal["in_intercept"] + acal["in_slope"] * base
        row = P.loc[k] if k in P.index else None
        b2b = d in W.b2b_days.get(t, set())
        f = rest_factor(params, row["age"] if row is not None else 26, row["exp_min"] if row is not None else 20, b2b)
        return float(np.clip(base * f, 0, 1))
    return avail


def matchup_calibration(H, params: dict, season: str, prev: str, mondays: list[str], matchups_per_week: int = 40,
                        roster_size: int = 13, pool_size: int = 200, n: int = 2000, seed: int = 7, slots_cfg=None):
    """Random fantasy matchups on past weeks: predicted P(win) vs what actually happened.

    Rosters: random draws from the top `pool_size` players by expected FP as of the Monday.
    Actual result: real box scores; each day the lineup is set the same way the simulator
    does (best expected FP among players who actually played)."""
    from tradecalc.sim.engine import fantasy_week
    from tradecalc.trade.lineup import best_lineup, eligibility_matrix

    rng = np.random.default_rng(seed)
    POS = {"G": ("PG", "SG"), "F": ("SF", "PF"), "C": ("C",)}
    slots, slot_elig = slots_cfg
    out = []
    for m in mondays:
        W = build_world_history(H, params, m, season, prev)
        avail = history_avail(H, W, params, m, season)
        start = date.fromisoformat(m)
        days = [start + timedelta(i) for i in range(7)]
        end = (start + timedelta(days=7)).isoformat()
        tl = H.team_logs[(H.team_logs.game_date >= m) & (H.team_logs.game_date < end) & H.team_logs.home]
        games = [(date.fromisoformat(d), h, a) for d, h, a in zip(tl.game_date, tl.team, tl.opp)]
        S = simulate_games(W, games, avail, n=n, seed=int(rng.integers(1 << 30)))
        act = H.logs[(H.logs.game_date >= m) & (H.logs.game_date < end)].copy()
        act["key"] = act.player_name.map(norm_name)
        act["d"] = act.game_date.map(date.fromisoformat)
        A = act.drop_duplicates(["key", "d"]).set_index(["key", "d"])["fp"]
        weekly_exp = defaultdict(float)
        for (k, d), bm in S.box_mean.items():
            weekly_exp[k] += bm["fp"] * S.played[(k, d)].mean()
        pool = sorted(weekly_exp, key=lambda k: -weekly_exp[k])[:pool_size]
        elig = {k: POS.get(W.players.loc[k, "pos"], ("SF", "PF")) for k in pool}
        for _ in range(matchups_per_week):
            pick = rng.choice(pool, 2 * roster_size, replace=False)
            ra, rb = list(pick[:roster_size]), list(pick[roster_size:])
            res = fantasy_week(S, {"A": ra, "B": rb}, days, slots, elig, slot_elig, n)
            actual = {}
            for lab, keys in (("A", ra), ("B", rb)):
                E = eligibility_matrix([elig[k] for k in keys], slots, slot_elig)
                tot = 0.0
                for d in days:
                    idx = [i for i, k in enumerate(keys) if (k, d) in A.index]
                    if not idx:
                        continue
                    prio = np.array([S.box_mean.get((keys[i], d), {"fp": 0})["fp"] for i in idx])
                    st = best_lineup(prio, E[idx])
                    tot += sum(A[(keys[i], d)] for c, i in enumerate(idx) if st[c])
                actual[lab] = tot
            out.append({"week": m, "p_win": res.p_win + res.p_tie / 2, "won": float(actual["A"] > actual["B"]),
                        "exp_A": res.totals["A"].mean(), "act_A": actual["A"],
                        "exp_B": res.totals["B"].mean(), "act_B": actual["B"]})
    return pd.DataFrame(out)
