"""Backtest of the projection engine on a past season, using only data available before each game.

For every test date, every player-game that day is predicted from:
  - per-stat rates estimated from games before that date (this season + last season prior),
  - as-of expected minutes, teammates absent that game (as an injury report would show),
  - opponent defensive factors and pace as of that date, absent opposing defenders.
Compared against the trade calculator's FPPG approach (this season's FPPG shrunk toward last
season's with k games) and against the engine given actual minutes (isolates rate error).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradecalc.sim.context import build_context
from tradecalc.sim.defense import defender_impacts, estimate_defense
from tradecalc.sim.project import adjusted_rates, expected_box, fantasy, minutes_mean
from tradecalc.sim.rates import estimate_rates


def run(H, params: dict, test_season: str, prev_season: str, every: int = 4, start: str | None = None,
        k_baseline: float = 6.0, carry: float = 0.5, prev2_season: str | None = None):
    logs = H.logs
    cur_all, prev_logs = logs[logs.season == test_season], logs[logs.season == prev_season]
    tl = H.team_logs
    tl_cur, tl_prev = tl[tl.season == test_season], tl[tl.season == prev_season]
    C = build_context(logs[logs.season.isin([prev_season, test_season])], tl, H.pos_of, H.age)
    C = C[C.season == test_season]
    impacts = defender_impacts(H.onoff[H.onoff.season == prev_season]) if len(H.onoff) else pd.Series(dtype=float)
    prev_mpg = prev_logs.groupby("player_id")["min"].mean()
    prev_fp = prev_logs.groupby("player_id")["fp"].mean()
    mp, ub, scoring = params["minutes"], params["usage_beta"], H.scoring

    dates = sorted(cur_all["game_date"].unique())
    start = start or dates[min(len(dates) - 1, 25)]
    dates = [d for d in dates if d >= start][::every]
    out = []
    for d in dates:
        before = cur_all[cur_all.game_date < d]
        rm = estimate_rates(before, prev_logs, params["rate_k"], H.pos_of, carry=carry,
                            prev2=logs[logs.season == prev2_season] if prev2_season else None)
        dm = estimate_defense(tl_cur[tl_cur.game_date < d], params["def_k"], tl_prev)
        lg_rates = rm.league["ALL"]
        cur_fp = before.groupby("player_id")["fp"].agg(["mean", "size"])
        day = C[C.game_date == d]
        absent = day[~day.played]
        for _, r in day[day.played].iterrows():
            pid = r.player_id
            base = rm.rates.loc[pid].to_dict() if pid in rm.rates.index else dict(lg_rates)
            exp_min = r.exp_min if r.exp_min == r.exp_min else prev_mpg.get(pid, 12.0)
            opp = logs.loc[(logs.game_id == r.game_id) & (logs.team != r.team), "team"]
            opp = opp.iat[0] if len(opp) else None
            def_f = dm.factors.loc[opp].to_dict() if opp in dm.factors.index else None
            pace_ratio = ((dm.pace.get(r.team, dm.league_pace) + dm.pace.get(opp, dm.league_pace) - dm.league_pace)
                          / dm.pace.get(r.team, dm.league_pace))
            opp_abs = absent[(absent.game_id == r.game_id) & (absent.team == opp)]
            eff = 1 + sum(impacts.get(p, 0.0) * (m / 48) for p, m in zip(opp_abs.player_id, opp_abs.exp_min.fillna(0))) / dm.league_drtg
            miss = {s: r[f"miss_{s}"] for s in ub}
            m_hat = minutes_mean(exp_min, r.abs_min_same, r.abs_min_other, r.b2b, mp)
            m_blind = minutes_mean(exp_min, 0, 0, r.b2b, mp)
            rates = adjusted_rates(base, ub, miss, def_f, pace_ratio, eff)
            rates_blind = adjusted_rates(base, ub, {}, def_f, pace_ratio)
            rates_plain = adjusted_rates(base, ub, {}, None, 1.0)
            cf = cur_fp.loc[pid] if pid in cur_fp.index else None
            n = cf["size"] if cf is not None else 0
            pf = prev_fp.get(pid)
            if pf is None and n == 0:
                base_fp = np.nan
            elif pf is None:
                base_fp = cf["mean"]
            else:
                base_fp = (k_baseline * pf + n * (cf["mean"] if n else 0)) / (k_baseline + n)
            out.append({
                "date": d, "player_id": pid, "apps": r.apps, "actual_fp": logs.loc[(logs.game_id == r.game_id) & (logs.player_id == pid), "fp"].iat[0],
                "actual_min": r["min"], "exp_min": exp_min, "min_hat": m_hat,
                "baseline": base_fp,
                "engine": fantasy(expected_box(m_hat, rates), scoring),
                "engine_no_absence_info": fantasy(expected_box(m_blind, rates_blind), scoring),
                "engine_no_matchup": fantasy(expected_box(m_hat, adjusted_rates(base, ub, miss, None, 1.0)), scoring),
                "engine_rates_x_actual_min": fantasy(expected_box(r["min"], rates), scoring),
                "plain_rates_x_actual_min": fantasy(expected_box(r["min"], rates_plain), scoring),
            })
    return pd.DataFrame(out)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["baseline"])
    cols = ["baseline", "engine", "engine_no_absence_info", "engine_no_matchup",
            "engine_rates_x_actual_min", "plain_rates_x_actual_min"]
    res = {c: {"MAE": (d[c] - d.actual_fp).abs().mean(), "bias": (d[c] - d.actual_fp).mean(),
               "RMSE": np.sqrt(((d[c] - d.actual_fp) ** 2).mean())} for c in cols}
    t = pd.DataFrame(res).T
    t.loc["minutes: engine", ["MAE", "bias"]] = [(d.min_hat - d.actual_min).abs().mean(), (d.min_hat - d.actual_min).mean()]
    t.loc["minutes: recent avg only", ["MAE", "bias"]] = [(d.exp_min - d.actual_min).abs().mean(), (d.exp_min - d.actual_min).mean()]
    t.attrs["n"] = len(d)
    return t
