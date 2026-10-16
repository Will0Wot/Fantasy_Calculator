"""Fit every projection-engine parameter from historical seasons and save to config/sim_params.yaml.

  rate_k        per-stat shrinkage (minutes / attempts)               sim/rates.py
  def_k         per-stat team-defense shrinkage (games)               sim/defense.py
  usage_beta    rate multiplier per unit of missing teammate share    1 + beta * miss_share
  minutes       expected-minutes adjustments and residual spread
  rest          logistic model of a healthy player sitting one game
  coach         per-coach rest rate and rotation concentration (shrunk)
  depth_minutes typical minutes by depth-chart rank
  dispersion    negative-binomial size per counting stat (fit in sim/backtest.py)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tradecalc.sim.context import USAGE_STATS, build_context
from tradecalc.sim.defense import fit_def_k
from tradecalc.sim.rates import fit_k

PARAMS_PATH = Path(__file__).resolve().parent.parent / "config" / "sim_params.yaml"
COACH_REST_K = 400      # B2B player-games of shrinkage for a coach's rest rate (weakly persistent)
COACH_CONC_K = 40       # team-games of shrinkage for rotation concentration (persistent, r~0.7)


def _wls_slope(y, x, w):
    import statsmodels.api as sm

    m = sm.WLS(y, sm.add_constant(x), weights=w).fit()
    return float(m.params.iloc[1]), float(m.bse.iloc[1])


def fit_params(H, coaches: pd.DataFrame | None, seasons: list[str]) -> dict:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    logs = H.logs[H.logs.season.isin(seasons)]
    tl = H.team_logs[H.team_logs.season.isin(seasons)]
    p: dict = {"fit_seasons": seasons}

    ks = [fit_k(logs[logs.season == s]) for s in seasons]
    p["rate_k"] = {k: float(np.mean([d[k] for d in ks])) for k in ks[0]}
    dks = [fit_def_k(tl[tl.season == s]) for s in seasons]
    p["def_k"] = {k: float(np.mean([d[k] for d in dks])) for k in dks[0]}

    C = build_context(logs, H.team_logs, H.pos_of, H.age)
    P = C[C.played & C.in_rotation & (C["min"] >= 8)].copy()
    p["usage_beta"] = {}
    for s in USAGE_STATS:
        d = P[P[f"base_{s}"] > 0.005]
        y = (d[s] / d["min"]) / d[f"base_{s}"] - 1
        p["usage_beta"][s] = round(_wls_slope(y, d[[f"miss_{s}"]], d["min"] * d[f"base_{s}"])[0], 4)

    P["dmin"] = P["min"] - P["exp_min"]
    P["blow"] = np.maximum(P.margin.abs() - 12, 0)
    P["blow_star"] = P.blow * (P.exp_min >= 30)
    P["blow_bench"] = P.blow * (P.exp_min < 20)
    X = sm.add_constant(P[["abs_min_same", "abs_min_other", "b2b", "blow_star", "blow_bench"]].astype(float))
    m = sm.OLS(P.dmin, X).fit()
    resid_sd = P.assign(r=m.resid).groupby(pd.cut(P.exp_min, [0, 15, 20, 25, 30, 35, 48]), observed=True)["r"].std()
    p["minutes"] = {k: round(float(v), 4) for k, v in m.params.items()}
    p["minutes"]["resid_sd"] = {str(int(iv.right)): round(float(v), 2) for iv, v in resid_sd.items()}
    p["minutes"]["blowout_threshold"] = 12

    R = C[C.in_rotation].sort_values(["player_id", "game_date"]).copy()
    g = R.groupby(["player_id", "season", "team"])
    R["prev_played"], R["next_played"] = g["played"].shift(1), g["played"].shift(-1)
    Hh = R[(R.prev_played == True) & (R.next_played == True)].copy()  # noqa: E712
    Hh["sat"] = (~Hh.played).astype(int)
    Hh["age"] = Hh.age.fillna(Hh.age.median())
    Hh["age28"] = np.maximum(Hh.age - 28, 0)
    Hh["load"] = Hh.exp_min / 30
    Hh["b2b"] = Hh.b2b.astype(int)
    lm = smf.logit("sat ~ b2b + age28 + b2b:age28 + load + b2b:load", Hh).fit(disp=0)
    p["rest"] = {k: round(float(v), 4) for k, v in lm.params.items()}
    lg_b2b_rate = float(Hh[Hh.b2b == 1].sat.mean())
    p["rest"]["league_b2b_rate"] = round(lg_b2b_rate, 4)

    # coach tendencies (team-season -> coach)
    p["coach"] = {}
    if coaches is not None and len(coaches):
        cm = coaches.set_index(["team", "season"])["coach"].to_dict()
        Hh["coach"] = [cm.get((t, s)) for t, s in zip(Hh.team, Hh.season)]
        tg = logs.groupby(["season", "team", "game_id"])["min"].apply(
            lambda m: np.sort(m.to_numpy())[-5:].sum() / m.sum())
        conc = tg.groupby(level=[0, 1]).agg(["mean", "size"]).reset_index()
        conc["coach"] = [cm.get((t, s)) for t, s in zip(conc.team, conc.season)]
        lg_conc = float(tg.mean())
        p["coach_league"] = {"b2b_rest": round(lg_b2b_rate, 4), "top5_share": round(lg_conc, 4)}
        for c, sub in Hh[Hh.b2b == 1].groupby("coach"):
            n, r = len(sub), sub.sat.mean()
            p["coach"].setdefault(c, {})["b2b_rest"] = round(float((COACH_REST_K * lg_b2b_rate + n * r) / (COACH_REST_K + n)), 4)
        for c, sub in conc.groupby("coach"):
            n = sub["size"].sum()
            v = (sub["mean"] * sub["size"]).sum() / n
            p["coach"].setdefault(c, {})["top5_share"] = round(float((COACH_CONC_K * lg_conc + n * v) / (COACH_CONC_K + n)), 4)

    # typical minutes by depth: rank players within team-season by mpg (>= 10 games)
    last = logs[logs.season == seasons[-1]]
    mpg = last.groupby(["team", "player_id"]).agg(mpg=("min", "mean"), gp=("min", "size")).reset_index()
    mpg = mpg[mpg.gp >= 10].sort_values(["team", "mpg"], ascending=[True, False])
    mpg["rank"] = mpg.groupby("team").cumcount() + 1
    p["depth_minutes"] = {int(r): round(float(v), 1) for r, v in mpg.groupby("rank")["mpg"].median().items() if r <= 15}
    return p


def save_params(p: dict, path: Path = PARAMS_PATH) -> None:
    path.write_text("# Fitted by `tradecalc sim-fit` (tradecalc/sim/params.py). Do not hand-edit lightly.\n"
                    + yaml.safe_dump(p, sort_keys=False, allow_unicode=True))


def load_params(path: Path = PARAMS_PATH) -> dict:
    return yaml.safe_load(path.read_text())


def fit_noise(H, params: dict, season: str, prev: str) -> dict:
    """Randomness the simulator needs, fitted on one season:
      dispersion   negative-binomial size r per counting stat, given expected count
                   (rate x actual minutes):  Var = mu + mu^2 / r
      pace_sd      game-to-game pace deviation from expectation, as a fraction
      margin_sd    final-margin spread around the expected margin
    """
    from tradecalc.sim.rates import estimate_rates

    logs = H.logs[H.logs.season == season]
    rm = estimate_rates(logs, H.logs[H.logs.season == prev], params["rate_k"], H.pos_of)
    d = logs.join(rm.rates, on="player_id", rsuffix="_rate")
    disp = {}
    for s in ["fga", "fta", "reb", "ast", "stl", "blk", "tov"]:
        mu = (d[f"{s}_rate"] if f"{s}_rate" in d else d[s + "_rate"]) * d["min"]
        y = d[s]
        excess = ((y - mu) ** 2).sum() - mu.sum()
        disp[s] = round(float((mu ** 2).sum() / excess), 2) if excess > 0 else 1000.0
    tl = H.team_logs[H.team_logs.season == season].copy()
    lg = tl["pace"].mean()
    tp = tl.groupby("team")["pace"].mean()
    exp_pace = tl["team"].map(tp) + tl["opp"].map(tp) - lg
    pace_sd = float(((tl["pace"] - exp_pace) / exp_pace).std())
    # expected margin from season net ratings (points per 100 x pace)
    net = tl.groupby("team").apply(lambda g: 100 * (g.pts.sum() - g.opp_pts.sum()) / g.poss.sum(), include_groups=False)
    exp_margin = (tl["team"].map(net) - tl["opp"].map(net)) / 100 * exp_pace + np.where(tl["home"], 1.5, -1.5)
    margin_sd = float((tl["plus_minus"] - exp_margin).std())
    return {"dispersion": disp, "pace_sd": round(pace_sd, 4), "margin_sd": round(margin_sd, 2), "home_edge": 1.5}
