from datetime import date

import numpy as np
import pandas as pd
import pytest

from tradecalc.sim.defense import DefenseModel
from tradecalc.sim.engine import World, _p_rest, rest_factor, simulate_games, trim_rotation
from tradecalc.sim.project import expected_blowout
from tradecalc.sim.rates import fit_k

SCORING = {"PTS": 1, "REB": 1, "AST": 2, "STL": 4, "BLK": 4, "TOV": -2, "FGM": 2, "FGA": -1, "FTM": 1, "FTA": -1, "FG3M": 1}
PARAMS = {
    "minutes": {"const": 0.0, "abs_min_same": 0.05, "abs_min_other": 0.025, "b2b": 0.0, "blow_star": -0.3,
                "blow_bench": 0.1, "resid_sd": {"15": 5.0, "30": 5.0, "48": 4.5}, "blowout_threshold": 12},
    "usage_beta": {"fga": 0.35, "fta": 0.45, "tov": 0.4, "ast": 0.33, "reb": 0.1, "stl": 0.08, "blk": 0.05, "fg3a": 0.2},
    "dispersion": {"fga": 1000, "fta": 3.6, "reb": 100, "ast": 60, "stl": 23, "blk": 55, "tov": 1000},
    "rest": {"Intercept": -1.77, "b2b": -0.83, "age28": 0.039, "b2b:age28": 0.059, "load": -1.73, "b2b:load": 1.70,
             "league_b2b_rate": 0.078},
    "pace_sd": 0.04, "margin_sd": 13.6, "home_edge": 1.5, "rotation_minutes_share": 0.94, "rate_calibration": 1.0,
}
RATE = {"fga": 0.4, "fta": 0.1, "reb": 0.18, "ast": 0.1, "stl": 0.03, "blk": 0.02, "tov": 0.05,
        "fg3a_share": 0.4, "fg2": 0.53, "fg3": 0.36, "ft": 0.78}


def _world(exp_mins=(34, 33, 32, 30, 28, 24, 20, 18, 14, 10)):
    rows = []
    for t in ("AAA", "BBB"):
        for i, m in enumerate(exp_mins):
            rows.append({"key": f"{t}{i}", "name": f"{t}{i}", "team": t, "pos": "GFFCG"[i % 5], "age": 26,
                         "exp_min": float(m), **RATE})
    P = pd.DataFrame(rows).set_index("key")
    rot = {t: [f"{t}{i}" for i in range(len(exp_mins))] for t in ("AAA", "BBB")}
    dm = DefenseModel(pd.DataFrame(), pd.Series({"AAA": 100.0, "BBB": 100.0}), 100.0, 112.0)
    return World(date(2026, 11, 2), P, rot, dm, {"AAA": 0.0, "BBB": 0.0}, {}, PARAMS, SCORING)


def test_box_scores_are_internally_consistent():
    W = _world()
    S = simulate_games(W, [(date(2026, 11, 2), "AAA", "BBB")], lambda k, t, d: 1.0, n=3000, seed=1)
    # Recompute FP from a sampled box: run the sampler directly through the module
    from tradecalc.sim.engine import _sample_box
    rng = np.random.default_rng(0)
    mins = np.full(3000, 32.0)
    miss = {s: np.zeros(3000) for s in ["fga", "fta", "tov", "ast", "reb", "stl", "blk", "fg3a"]}
    b = _sample_box(rng, pd.Series(RATE), mins, miss, PARAMS["usage_beta"], {}, np.ones(3000), np.ones(3000),
                    PARAMS["dispersion"], 3000)
    assert np.all(b["pts"] == 2 * (b["fgm"] - b["fg3m"]) + 3 * b["fg3m"] + b["ftm"])
    assert np.all((b["fg3m"] <= b["fg3a"]) & (b["fg3a"] <= b["fga"]) & (b["fgm"] <= b["fga"]) & (b["ftm"] <= b["fta"]))
    assert len(S.fp) == 20


def test_team_minutes_respect_total_and_cap():
    W = _world()
    S = simulate_games(W, [(date(2026, 11, 2), "AAA", "BBB")], lambda k, t, d: 1.0, n=2000, seed=2)
    d = date(2026, 11, 2)
    # box_mean holds per-player mean minutes; team mean total should be ~ 241 * 0.94
    tot = sum(S.box_mean[(f"AAA{i}", d)]["min"] for i in range(10))
    assert tot == pytest.approx(241 * 0.94, rel=0.02)
    assert max(S.box_mean[(f"AAA{i}", d)]["min"] for i in range(10)) <= 48


def test_star_absence_raises_teammates_usage_and_minutes():
    W = _world()
    d = date(2026, 11, 2)
    full = simulate_games(W, [(d, "AAA", "BBB")], lambda k, t, dd: 1.0, n=6000, seed=3)
    out = simulate_games(W, [(d, "AAA", "BBB")], lambda k, t, dd: 0.0 if k == "AAA0" else 1.0, n=6000, seed=3)
    k = ("AAA5", d)  # a bench guard
    assert out.box_mean[k]["min"] > full.box_mean[k]["min"]
    assert out.box_mean[k]["fga"] / out.box_mean[k]["min"] > full.box_mean[k]["fga"] / full.box_mean[k]["min"]
    assert not out.played[("AAA0", d)].any()


def test_rest_factor_back_to_back():
    old_b2b = rest_factor(PARAMS, 35, 32, True)
    old_rest = rest_factor(PARAMS, 35, 32, False)
    assert old_b2b < 1 < old_rest
    assert _p_rest(PARAMS, 35, 32, True) > _p_rest(PARAMS, 22, 32, True)


def test_trim_rotation_cuts_bench_first():
    m = np.array([34.0, 32, 30, 28, 26, 22, 20, 18, 16, 14, 12, 10, 8])
    t = trim_rotation(m, 240)
    assert t.sum() == pytest.approx(240)
    assert np.allclose(t[:9], m[:9])          # top of the rotation untouched
    assert t[-1] == 0 and t[-2] == 0


def test_expected_blowout_matches_simulation():
    x = np.random.default_rng(0).normal(0, 13.6, 400_000)
    assert expected_blowout(13.6, 12) == pytest.approx(np.maximum(np.abs(x) - 12, 0).mean(), rel=0.02)


def test_fit_k_prefers_more_shrinkage_for_noisy_stats():
    rng = np.random.default_rng(0)
    rows = []
    for p in range(150):
        true_stl = max(rng.normal(0.035, 0.004), 0.005)
        true_reb = max(rng.normal(0.18, 0.06), 0.02)
        for g in range(40):
            m = 25
            fga = rng.poisson(0.4 * m)
            fg3a = rng.binomial(fga, 0.4)
            rows.append({"player_id": p, "game_date": f"2025-{g:03d}", "min": m, "fga": fga, "fg3a": fg3a,
                         "fta": rng.poisson(2.5), "reb": rng.poisson(true_reb * m), "ast": rng.poisson(2.5),
                         "stl": rng.poisson(true_stl * m), "blk": rng.poisson(0.5), "tov": rng.poisson(1.3),
                         "fg2m": rng.binomial(fga - fg3a, 0.53), "fg2a": fga - fg3a,
                         "fg3m": rng.binomial(fg3a, 0.36), "ftm": 2})
    k = fit_k(pd.DataFrame(rows))
    assert k["stl"] > k["reb"]
