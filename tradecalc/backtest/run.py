"""Backtest of the FPPG model (section 8).

Out of sample: the ADP->FPPG curve is fit on `fit_season` and applied to `test_season`
draft positions.

Caveat: ESPN keeps past seasons' STANDARD draft ranks, but they are revised during the
season, so they leak end-of-season information. When the ADP CSVs are ESPN draft ranks
(rank_type == espn_draft_rank) the curve prior is reported for reference only, and k is
chosen with ESPN's preseason projection as the prior. Drop genuine historical ADP CSVs
(e.g. FantasyPros) in place of those files and the curve+projection prior is used instead. For each N in n_grid, every player with at least N + min_rest_games
games is "observed" through his first N games, and each method predicts his FPPG over
the rest of the season. Error is MAE in FPPG.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradecalc.ingest import adp as adp_mod
from tradecalc.ingest import nba
from tradecalc.model.blend import blend
from tradecalc.model.prior import fit_adp_curve, prior_table
from tradecalc.model.scoring import add_fp, fantasy_points, live_stats
from tradecalc.model.names import norm_name
from tradecalc.model.variance import estimate_cv_typical


def _ensure_adp(client, path, espn_season):
    if not path.exists():
        pool = client.player_pool(limit=600, season=espn_season, league_scoped=False)
        adp_mod.export_espn_adp(pool, path, use_rank=True)
    return adp_mod.load_adp(path)


def _espn_projections(client, espn_season, scoring) -> dict[str, float]:
    from tradecalc.ingest.league import _projection

    pool = client.player_pool(limit=600, season=espn_season, league_scoped=False)
    out = {}
    for e in pool.get("players", []):
        proj = _projection(e["player"], espn_season)
        if proj.get("GP"):
            out[e["player"]["fullName"]] = fantasy_points(proj, scoring)
    return out


def prepare(cfg, conn, client):
    bt, m = cfg["backtest"], cfg["model"]
    interval = cfg["data"]["nba_min_request_interval_s"]
    for s in (bt["fit_season"], bt["test_season"]):
        nba.refresh_season(conn, s, complete=True, min_interval=interval)
    scoring = cfg["scoring"]
    fit_logs = add_fp(nba.game_logs(conn, bt["fit_season"]), scoring)
    test_logs = add_fp(nba.game_logs(conn, bt["test_season"]), scoring)
    test_logs = test_logs[test_logs["min"] > 0].sort_values(["player_id", "game_date"])

    fit_adp = _ensure_adp(client, cfg.path(bt["fit_adp_csv"]), bt["fit_espn_season"])
    curve = fit_adp_curve(fit_adp, live_stats(fit_logs, scoring))
    test_adp = _ensure_adp(client, cfg.path(bt["test_adp_csv"]), bt["test_espn_season"])
    leaky = is_leaky(fit_adp) or is_leaky(test_adp)
    proj = _espn_projections(client, bt["test_espn_season"], scoring)
    # Projection-only prior; players without a projection get the curve's tail value.
    proj_prior = pd.Series({norm_name(k): v for k, v in proj.items()}).groupby(level=0).first()
    priors = {
        "curve" + (" (leaky ESPN rank)" if leaky else ""): prior_table(test_adp, curve)["prior_fppg"],
        "curve+proj" + (" (leaky ESPN rank)" if leaky else ""): prior_table(test_adp, curve, proj)["prior_fppg"],
        "proj": proj_prior,
    }
    blend_prior = "proj" if leaky else "curve+proj"
    r = cfg["roster"]
    rostered = cfg["league"]["num_teams"] * (sum(r["slots"].values()) + r["bench"])
    cv = estimate_cv_typical(live_stats(test_logs, scoring), rostered)
    return test_logs, curve, priors, blend_prior, proj, cv


def is_leaky(adp_df: pd.DataFrame) -> bool:
    return "rank_type" in adp_df and (adp_df["rank_type"] == "espn_draft_rank").any()


def run(cfg, conn, client, verbose=True):
    bt, m = cfg["backtest"], cfg["model"]
    scoring = cfg["scoring"]
    test_logs, curve, priors, blend_prior, proj, cv = prepare(cfg, conn, client)
    test_logs["gnum"] = test_logs.groupby("player_id").cumcount()
    totals = test_logs.groupby("player_id").size()
    keys = test_logs.groupby("player_id")["player_name"].first().map(norm_name)

    rows = []
    for N in bt["n_grid"]:
        ids = totals[totals >= N + bt["min_rest_games"]].index
        sub = test_logs[test_logs["player_id"].isin(ids)]
        early, rest = sub[sub["gnum"] < N], sub[sub["gnum"] >= N]
        live = live_stats(early, scoring, recent_games=m["role_change_recent_games"])
        actual = rest.groupby("player_id")["fp"].mean()
        res = {"N": N, "players": len(ids)}
        for pname, pr in priors.items():
            p = pr.reindex(keys.loc[live.index]).fillna(curve.tail_value).set_axis(live.index)
            res[f"prior[{pname}]"] = (p - actual).abs().mean()
            if pname == blend_prior:
                res["live only"] = (live["live_fppg"] - actual).abs().mean()
                for k in bt["k_grid"]:
                    b = blend(p, live, k, role_threshold=m["role_change_threshold"],
                              role_min_games=m["role_change_min_games"],
                              recent_weight=m["role_change_recent_weight"])
                    res[f"blend k={k}"] = (b["blended_fppg"] - actual).abs().mean()
                    b0 = blend(p, live, k, role_threshold=np.inf)
                    res[f"blend k={k} (no role)"] = (b0["blended_fppg"] - actual).abs().mean()
        rows.append(res)
    table = pd.DataFrame(rows).set_index("N")
    blend_cols = [f"blend k={k}" for k in bt["k_grid"]]
    best_k = int(table[blend_cols].mean().idxmin().split("=")[1])
    return table, best_k, cv, curve, proj, blend_prior


def format_table(table: pd.DataFrame) -> str:
    t = table.copy()
    t.loc["mean"] = t.mean()
    t["players"] = t["players"].round().astype(int)
    return t.T.to_string(float_format=lambda v: f"{v:.2f}")
