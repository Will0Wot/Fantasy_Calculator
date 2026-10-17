"""Multi-season audit: every component tested out of sample on past seasons.

Run with `tradecalc audit`. Results go to config/audit.json and the app's Model audit tab.
"""
from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from tradecalc.audit.data import TEST_SEASONS, coaches_for, espn_year, fit_engine
from tradecalc.model.names import norm_name
from tradecalc.sim.history import prev_season


def _brier(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def calibration_table(p, y, bins=(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0)) -> list[dict]:
    d = pd.DataFrame({"p": p, "y": y})
    d["bin"] = pd.cut(d.p, bins, include_lowest=True)
    g = d.groupby("bin", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")).reset_index()
    return [{"bin": f"{b.left:.0%}–{b.right:.0%}", "n": int(n), "predicted": float(pr), "actual": float(a)}
            for b, n, pr, a in zip(g.bin, g.n, g.predicted, g.actual)]


def calibration_slope(p, y) -> float:
    """Logistic recalibration slope: 1 = honest; < 1 = overconfident (too extreme)."""
    import statsmodels.api as sm

    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    x = np.log(p / (1 - p))
    try:
        m = sm.GLM(np.asarray(y, float), sm.add_constant(x), family=sm.families.Binomial()).fit()
        return float(m.params[1])
    except Exception:
        return float("nan")


# 1. season model: rest-of-season FPPG ------------------------------------------------------

def test_season_model_fppg(H, ctx, k_grid=(2, 4, 6, 9, 12, 15, 20), n_grid=(5, 10, 20, 30)) -> dict:
    L = ctx.logs.sort_values("d")
    L = L.assign(g=L.groupby("key").cumcount())
    tot = L.groupby("key").size()
    rows = []
    for N in n_grid:
        keys = tot[tot >= N + 10].index
        sub = L[L.key.isin(keys)]
        early = sub[sub.g < N].groupby("key")["fp"].mean()
        rest = sub[sub.g >= N].groupby("key")["fp"].mean()
        prior_proj = pd.Series({k: ctx.proj_fp.get(k, ctx.prev_fppg.get(k, np.nan)) for k in keys}).dropna()
        prior_last = ctx.prev_fppg.reindex(keys).dropna()
        idx = prior_proj.index.intersection(prior_last.index)
        res = {"N": N, "players": int(len(idx)),
               "preseason (ESPN projection)": float((prior_proj[idx] - rest[idx]).abs().mean()),
               "last season FPPG": float((prior_last[idx] - rest[idx]).abs().mean()),
               "this season so far": float((early[idx] - rest[idx]).abs().mean())}
        for k in k_grid:
            res[f"blend k={k}"] = float((((k * prior_proj[idx] + N * early[idx]) / (k + N)) - rest[idx]).abs().mean())
        rows.append(res)
    df = pd.DataFrame(rows)
    blend = [c for c in df.columns if c.startswith("blend")]
    best = df[blend].mean().idxmin()
    return {"table": df.to_dict("records"), "best_k": int(best.split("=")[1])}


# 2-3. engine point projections and ranges --------------------------------------------------

def test_engine_points(H, params, season) -> dict:
    from tradecalc.sim.backtest import run, summarize

    df = run(H, params, season, prev_season(season), every=6)
    s = summarize(df)
    out = {k: {c: (None if pd.isna(v) else float(v)) for c, v in row.items()} for k, row in s.to_dict("index").items()}
    d = df.dropna(subset=["baseline"])
    # error by stage of season (games played so far)
    d = d.assign(stage=pd.cut(d.apps, [-1, 5, 15, 30, 100], labels=["0-5 games", "6-15", "16-30", "31+"]))
    by_stage = d.groupby("stage", observed=True).apply(
        lambda g: pd.Series({"n": len(g), "engine": (g.engine - g.actual_fp).abs().mean(),
                             "baseline": (g.baseline - g.actual_fp).abs().mean()}), include_groups=False).reset_index()
    return {"summary": out, "n": int(len(d)), "by_stage": by_stage.assign(stage=by_stage.stage.astype(str)).to_dict("records")}


def test_engine_ranges(H, params, season) -> dict:
    from tradecalc.sim.validate import coverage

    mondays = [d for d in sorted(H.team_logs[H.team_logs.season == season].game_date.unique())
               if pd.Timestamp(d).weekday() == 0][4::4]
    return coverage(H, params, season, prev_season(season), mondays, n=2000)


# 4. availability ---------------------------------------------------------------------------

def test_availability(H, params, season) -> dict:
    from tradecalc.sim.context import build_context
    from tradecalc.sim.engine import rest_factor

    prev = prev_season(season)
    logs = H.logs[H.logs.season.isin([prev, season])]
    C = build_context(logs, H.team_logs, H.pos_of, H.age)
    C = C[(C.season == season) & C.in_rotation].copy()
    P = H.logs[H.logs.season == prev]
    prev_rate = (P.groupby("player_id").size() / 82).clip(upper=0.95)
    tl = H.team_logs[H.team_logs.season == season].sort_values("game_date")
    tl["n_before"] = tl.groupby("team").cumcount()
    tg = tl.set_index(["game_id", "team"])["n_before"]
    C["team_games"] = [tg.get((g, t), 0) for g, t in zip(C.game_id, C.team)]
    C = C[C.team_games >= 10]
    pr = C.player_id.map(prev_rate).fillna(0.85)
    rate = ((10 * pr + C.apps) / (10 + C.team_games)).clip(upper=0.97)
    from tradecalc.config import load_config

    acal = load_config()["model"].get("availability_calibration")
    C["p_naive"] = rate
    from tradecalc.model.availability import in_season_calibration

    cal_rate = [in_season_calibration(r, acal, m) for r, m in zip(rate, C.exp_min)] if acal else rate
    C["p_model"] = [float(np.clip(r * rest_factor(params, a if a == a else 26, m if m == m else 20, b), 0, 1))
                    for r, a, m, b in zip(cal_rate, C.age, C.exp_min, C.b2b)]
    C["p_raw_rest"] = [float(np.clip(r * rest_factor(params, a if a == a else 26, m if m == m else 20, b), 0, 1))
                       for r, a, m, b in zip(rate, C.age, C.exp_min, C.b2b)]
    y = C.played.astype(float)
    b2b = C.b2b.astype(bool)
    return {"n": int(len(C)), "brier_model": _brier(C.p_model, y), "brier_naive": _brier(C.p_naive, y),
            "brier_raw_rest": _brier(C.p_raw_rest, y), "calibration_raw": calibration_table(C.p_raw_rest, y, bins=(0, .5, .7, .8, .85, .9, .95, 1.0)),
            "brier_b2b_model": _brier(C.p_model[b2b], y[b2b]), "brier_b2b_naive": _brier(C.p_naive[b2b], y[b2b]),
            "played_rate": float(y.mean()), "played_rate_b2b": float(y[b2b].mean()),
            "calibration": calibration_table(C.p_model, y, bins=(0, .5, .7, .8, .85, .9, .95, 1.0))}


# 5. engine weekly matchups -----------------------------------------------------------------

def test_engine_matchups(H, params, cfg, season, per_week=40) -> dict:
    from tradecalc.sim.validate import matchup_calibration
    from tradecalc.trade.lineup import slot_list

    mondays = [d for d in sorted(H.team_logs[H.team_logs.season == season].game_date.unique())
               if pd.Timestamp(d).weekday() == 0][3::3]
    R = matchup_calibration(H, params, season, prev_season(season), mondays, matchups_per_week=per_week, n=1500,
                            slots_cfg=(slot_list(cfg), cfg["roster"]["slot_eligibility"]))
    return {"n": int(len(R)), "brier": _brier(R.p_win, R.won), "slope": calibration_slope(R.p_win, R.won),
            "calibration": calibration_table(R.p_win, R.won), "points_bias": float((R.exp_A - R.act_A).mean()),
            "_raw": R}


# 6. league forecast ------------------------------------------------------------------------

def test_league_forecast(H, params, client, season, n_leagues=10) -> dict:
    from tradecalc.sim import forecast_test as FT

    out = FT.run(H, params, client, season, espn_year(season), n_leagues=n_leagues, n=400, log=lambda m: None)
    m = FT.metrics(out["results"])
    res = out["results"]
    return {"metrics": m.to_dict("records"),
            "rows": res[["league", "checkpoint", "team", "exp_wins", "p10", "p90", "p_first", "avg_rank",
                         "actual_wins", "actual_rank"]].round(3).to_dict("records")}


# 7. season model matchups + trades ---------------------------------------------------------

def summarize_season_model(r: dict) -> dict:
    m = r["matchups"]
    mm, ss = m[m.kind == "matchup"], m[m.kind == "season"]
    t = r["trades"]
    out = {"matchups": {"n": int(len(mm)), "brier": _brier(mm.p, mm.won), "slope": calibration_slope(mm.p, mm.won),
                        "calibration": calibration_table(mm.p, mm.won),
                        "by_checkpoint": [{"checkpoint": int(c), "brier": _brier(g.p, g.won),
                                           "slope": calibration_slope(g.p, g.won)} for c, g in mm.groupby("checkpoint")]},
           "season_wins": [{"checkpoint": int(c), "mae": float((g.exp_wins - g.actual_wins).abs().mean()),
                            "corr": float(spearmanr(g.exp_wins, g.actual_wins)[0])} for c, g in ss.groupby("checkpoint")]}
    if len(t):
        import statsmodels.api as sm

        x = np.r_[t.pred_you, t.pred_partner]
        y = np.r_[t.real_you, t.real_partner]
        fit = sm.OLS(y, sm.add_constant(x)).fit()
        acc = t[t.verdict == "ACCEPT"]
        dec = t[t.verdict == "DECLINE"]
        big = np.abs(x) >= 0.5
        out["trades"] = {
            "n": int(len(t)), "corr": float(pearsonr(x, y)[0]), "slope": float(fit.params[1]),
            "slope_se": float(fit.bse[1]),
            "sign_hit": float(np.mean(np.sign(x[big]) == np.sign(y[big]))) if big.any() else float("nan"),
            "corr_naive_ros": float(pearsonr(t.d_ros, t.real_you)[0]),
            "accept_n": int(len(acc)), "accept_real_mean": float(acc.real_you.mean()) if len(acc) else float("nan"),
            "accept_pred_mean": float(acc.pred_you.mean()) if len(acc) else float("nan"),
            "decline_real_mean": float(dec.real_you.mean()) if len(dec) else float("nan"),
            "both_gain_pred": float(((t.pred_you > 0.1) & (t.pred_partner > 0.1)).mean()),
            "sum_pred": float((t.pred_you + t.pred_partner).mean()), "sum_real": float((t.real_you + t.real_partner).mean()),
            "by_shape": [{"shape": s, "n": int(len(g)), "pred": float(g.pred_you.mean()), "real": float(g.real_you.mean())}
                         for s, g in t.groupby("shape")],
            "points": [{"pred": float(a), "real": float(b), "shape": s} for a, b, s in zip(t.pred_you, t.real_you, t["shape"])],
        }
    return out


# --- matchup shrink (beta), fitted rolling-origin ---------------------------------------------

def fit_shrink(matchups: pd.DataFrame) -> float:
    """beta in P(win) = Phi(beta * z), z = (M_you - M_opp) / sqrt(V_you + V_opp): probit slope with
    no intercept of real results on the model's z-scores. M and V do not depend on beta, so matchups
    scored under any beta can be reused."""
    import statsmodels.api as sm

    mm = matchups[matchups.kind == "matchup"]
    z = ((mm.M - mm.M_opp) / np.sqrt(mm.V + mm.V_opp)).to_numpy()
    fam = sm.families.Binomial(link=sm.families.links.Probit())
    return float(sm.GLM(mm.won.to_numpy(float), z[:, None], family=fam).fit().params[0])


def _with_shrink(cfg, beta: float):
    import copy

    c = copy.deepcopy(cfg)
    c["model"]["matchup_shrink"] = round(beta, 3)
    return c


FIT_ONLY_SEASONS = ["2022-23"]       # earliest season with a season before it in the audit data


# --- orchestration --------------------------------------------------------------------------

def run_audit(conn, cfg, client, seasons=TEST_SEASONS, H=None, log=print, quick=False, anchor=True,
              rolling_shrink=True) -> dict:
    """rolling_shrink: before each test season, refit the matchup shrink beta on every earlier
    season (2022-23 onward) and test with that; report["matchup_shrink"]["all"] is the fit on
    every season, the value to use live."""
    from tradecalc.audit.data import audit_history
    from tradecalc.audit.hist_state import run_season_model_tests, season_ctx

    H = H or audit_history(conn, cfg, client)
    coaches = coaches_for(conn, client)
    report = {"run_on": date.today().isoformat(), "seasons": {}}
    seen = []                                         # season-model matchups from earlier seasons
    if rolling_shrink:
        for s in FIT_ONLY_SEASONS:
            log(f"== {s} (fit only: matchup shrink) ==")
            seen.append(run_season_model_tests(H, client, cfg, s, n_leagues=3 if quick else 5, trade_checkpoints=(),
                                               log=lambda m: None)["matchups"])
        report["matchup_shrink"] = {}
    for s in seasons:
        t0 = time.time()
        log(f"== {s} ==")
        params = fit_engine(H, coaches, s, log)
        ctx = season_ctx(H, client, s)
        if anchor:
            params["_proj_fp"] = dict(ctx.proj_fp)   # ESPN preseason projections (known before the season)
        r = {"fit_on": [prev_season(prev_season(s)), prev_season(s)]}
        log("  season model: rest-of-season FPPG"); r["season_fppg"] = test_season_model_fppg(H, ctx)
        log("  engine: point projections"); r["engine_points"] = test_engine_points(H, params, s)
        log("  engine: ranges"); r["engine_ranges"] = test_engine_ranges(H, params, s)
        log("  availability"); r["availability"] = test_availability(H, params, s)
        log("  engine: weekly matchups")
        em = test_engine_matchups(H, params, cfg, s, per_week=20 if quick else 40)
        em.pop("_raw")
        r["engine_matchups"] = em
        log("  season model: matchups + trade counterfactuals")
        cfg_s = cfg
        if rolling_shrink:
            beta = fit_shrink(pd.concat(seen, ignore_index=True))
            report["matchup_shrink"][s] = beta
            cfg_s = _with_shrink(cfg, beta)
            log(f"    matchup shrink fitted on earlier seasons: {beta:.3f}")
        smt = run_season_model_tests(H, client, cfg_s, s, n_leagues=3 if quick else 5, n_trades=25 if quick else 40,
                                     log=lambda m: None)
        seen.append(smt["matchups"])
        r["season_model"] = summarize_season_model(smt)
        log("  league forecast")
        r["league_forecast"] = test_league_forecast(H, params, client, s, n_leagues=6 if quick else 10)
        r["seconds"] = round(time.time() - t0, 1)
        report["seasons"][s] = r
        log(f"  done in {r['seconds']:.0f}s")
    if rolling_shrink:
        report["matchup_shrink"]["all"] = fit_shrink(pd.concat(seen, ignore_index=True))
        log(f"matchup shrink fitted on every season (use live): {report['matchup_shrink']['all']:.3f}")
    return report


AUDIT_PATH = Path(__file__).resolve().parents[1] / "config" / "audit.json"
BASELINE_PATH = Path(__file__).resolve().parents[1] / "config" / "audit_baseline.json"


def baseline_config(cfg):
    """The model as it was before the audit: raw availability rates, no extra variance terms,
    no fill-ins, no ESPN anchor in the engine."""
    import copy

    b = copy.deepcopy(cfg)
    for k in ("availability_calibration", "espn_anchor"):
        b["model"].pop(k, None)
    b["model"].update(matchup_shrink=1.0, proj_uncertainty=0.0, absence_cluster=0.0)
    b["lineup"]["fill_in"] = None
    return b


def load(path=AUDIT_PATH) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save(report: dict, path) -> None:
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    path.write_text(json.dumps(report, default=default))
