"""Adversarial and property tests: things that must hold whatever the data, and inputs
designed to break the model. Each returns {name, idea, passed, detail}; run_all() feeds the
Model audit page and tests/test_breaks.py runs them in CI.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from tradecalc.tests.conftest import make_player, make_state
from tradecalc.trade.evaluate import Trade, apply_trade, evaluate
from tradecalc.trade.matchup import p_win, season_outlook, week_dist

WEEK = [date(2026, 10, 26) + timedelta(i) for i in range(7)]
ALL = ["PG", "SG", "SF", "PF", "C"]


def _league(n_weeks=2, mus_a=None, mus_b=None, games=3, fa=True):
    mus_a = mus_a or [30 - i for i in range(13)]
    mus_b = mus_b or [30 - i for i in range(13)]
    ps = [make_player(i + 1, 1, "AAA", ALL, m) for i, m in enumerate(mus_a)]
    ps += [make_player(i + 101, 2, "BBB", ALL, m) for i, m in enumerate(mus_b)]
    if fa:
        ps += [make_player(900 + i, 0, "CCC", ALL, 22 - i) for i in range(3)]
    gd = {}
    for w in range(n_weeks):
        for d in range(games):
            gd[WEEK[0] + timedelta(7 * w + d)] = {"AAA", "BBB", "CCC"}
    opp = {1: {w + 1: 2 for w in range(n_weeks)}, 2: {w + 1: 1 for w in range(n_weeks)}}
    return make_state(ps, gd, n_weeks=n_weeks, opponents=opp)


def _res(name, idea, passed, detail):
    return {"name": name, "idea": idea, "passed": bool(passed), "detail": detail}


def better_player_never_hurts():
    st = _league()
    base = season_outlook(1, st.rosters, st)["e_wins"]
    worst = min(st.rosters[1], key=lambda i: st.players[i].mu)
    st2 = _league()
    st2.players[worst].mu += 15
    st2.players[worst].ros_value = st2.players[worst].mu * st2.players[worst].avail.base * st2.players[worst].games_remaining
    up = season_outlook(1, st2.rosters, st2)["e_wins"]
    return _res("Better player never lowers your odds", "Upgrade one player by 15 points/game, nothing else changes",
                up >= base - 1e-9, f"expected wins {base:.3f} → {up:.3f}")


def identical_swap_is_zero():
    st = _league(fa=False)
    tr = Trade(1, 2, give=(1,), get=(101,))      # both 30 FPPG, same schedule and positions
    r = evaluate(tr, st)
    ok = abs(r.you.d_wins) < 1e-6 and abs(r.partner.d_wins) < 1e-6
    return _res("Swapping identical players changes nothing", "Two clones (same points, schedule, positions) traded",
                ok, f"Δ you {r.you.d_wins:+.4f}, Δ partner {r.partner.d_wins:+.4f}")


def two_team_zero_sum():
    st = _league(mus_b=[40 - i for i in range(13)], fa=False)
    r = evaluate(Trade(1, 2, give=(1, 2), get=(101, 102)), st)
    return _res("Head-to-head is zero-sum", "Two-team league, even trade, no free agents: your gain = their loss",
                abs(r.you.d_wins + r.partner.d_wins) < 1e-6, f"Δ you {r.you.d_wins:+.3f}, Δ partner {r.partner.d_wins:+.3f}")


def reverse_trade_undoes():
    st = _league(mus_b=[38 - i for i in range(13)], fa=False)
    tr = Trade(1, 2, give=(3,), get=(104,))
    r = evaluate(tr, st)
    new, _ = apply_trade(tr, st.rosters, st)
    st.rosters = new
    back = evaluate(Trade(1, 2, give=(104,), get=(3,)), st)
    ok = abs(r.you.d_wins + back.you.d_wins) < 1e-6
    return _res("Undoing a trade restores the odds", "Make a trade, then trade back", ok,
                f"Δ {r.you.d_wins:+.3f} then {back.you.d_wins:+.3f}")


def all_injured_team():
    st = _league()
    for i in st.rosters[1]:
        from tradecalc.model.availability import Availability

        st.players[i].avail = Availability(0.9, out_games=99)
    out = season_outlook(1, st.rosters, st)
    ok = all(w["p"] < 0.01 for w in out["weeks"]) and out["pts_per_week"] < 1
    return _res("A team of injured players can't win", "Every player out for the season",
                ok, f"points/week {out['pts_per_week']:.1f}, win chances {[round(w['p'], 3) for w in out['weeks']]}")


def impossible_lineup():
    ps = [make_player(i + 1, 1, "AAA", ["C"], 30) for i in range(13)]
    ps += [make_player(i + 101, 2, "BBB", ALL, 20) for i in range(13)]
    st = make_state(ps, {d: {"AAA", "BBB"} for d in WEEK}, opponents={1: {1: 2}, 2: {1: 1}})
    d = week_dist(st.rosters[1], st.cal.weeks[0], st)
    starts = d.lineup.g.sum()
    return _res("Lineup rules are enforced", "13 centers: only C + 3 UTIL can start each day",
                starts == 4 * 7, f"{int(starts)} starts in 7 days (max {4 * 7})")


def empty_week():
    st = _league(games=0)
    out = season_outlook(1, st.rosters, st)
    ok = all(w["p"] == 0.5 and w["M"] == 0 for w in out["weeks"])
    return _res("A week with no games is a coin flip, not a crash", "No NBA games scheduled (All-Star style)",
                ok, f"points {[w['M'] for w in out['weeks']]}, win chances {[w['p'] for w in out['weeks']]}")


def no_variance():
    return _res("Zero variance doesn't divide by zero", "Both teams perfectly predictable",
                p_win(100, 0, 90, 0) == 1.0 and p_win(90, 0, 100, 0) == 0.0 and p_win(5, 0, 5, 0) == 0.5,
                f"p_win(100 vs 90) = {p_win(100, 0, 90, 0)}, equal = {p_win(5, 0, 5, 0)}")


def scale_invariance():
    a = p_win(1000, 120 ** 2, 950, 110 ** 2)
    b = p_win(2000, 240 ** 2, 1900, 220 ** 2)
    return _res("Doubling every score changes no probability", "Scoring weights x2",
                abs(a - b) < 1e-12, f"{a:.4f} vs {b:.4f}")


def lopsided():
    st = _league(mus_a=[60] * 13, mus_b=[10] * 13, fa=False)
    out = season_outlook(1, st.rosters, st)
    return _res("A superteam is (almost) certain to win", "13 x 60 FPPG vs 13 x 10 FPPG",
                all(w["p"] > 0.999 for w in out["weeks"]), f"win chances {[round(w['p'], 4) for w in out['weeks']]}")


def hot_start():
    from tradecalc.config import load_config
    from tradecalc.model.blend import blend
    import pandas as pd

    k = load_config()["model"]["k"]
    live = pd.DataFrame({"n": [1], "live_fppg": [80.0], "mpg_season": [30.0], "mpg_last5": [30.0],
                         "fppg_recent": [80.0]}, index=[1])
    mu = float(blend(pd.Series({1: 20.0}), live, k)["blended_fppg"].iloc[0])
    return _res("One huge game doesn't make a star", "Projected 20 FPPG, then scores 80 in his first game",
                mu < 30, f"projection moves 20 → {mu:.1f} (k={k})")


def free_agent_not_phantom():
    """2-for-1 'consolidation': value must come only from the real free agent signed."""
    st = _league(mus_b=[30 - i for i in range(13)])
    worst_two = sorted(st.rosters[1], key=lambda i: st.players[i].mu)[:2]
    their_worst = min(st.rosters[2], key=lambda i: st.players[i].mu)
    r = evaluate(Trade(1, 2, give=tuple(worst_two), get=(their_worst,)), st)
    # benchmark: same roster change done without a trade (drop worst, add best FA)
    st2 = _league(mus_b=[30 - i for i in range(13)])
    fa = max(st2.free_agents(), key=lambda p: p.ros_value)
    new = [i for i in st2.rosters[1] if i not in worst_two] + [their_worst + 1000, fa.espn_id]
    clone = make_player(their_worst + 1000, 1, st2.players[their_worst].pro_team, ALL, st2.players[their_worst].mu)
    st2.players[clone.espn_id] = clone
    clone.ros_value = st.players[their_worst].ros_value
    alt = season_outlook(1, {**st2.rosters, 1: new}, st2)["e_wins"] - season_outlook(1, st2.rosters, st2)["e_wins"]
    return _res("2-for-1 trades create no phantom value", "Gain must match doing the same roster change without a trade",
                abs(r.you.d_wins - alt) < 0.05, f"trade {r.you.d_wins:+.3f} vs same change without trading {alt:+.3f}")


def run_all(H=None) -> list[dict]:
    tests = [better_player_never_hurts, identical_swap_is_zero, two_team_zero_sum, reverse_trade_undoes,
             all_injured_team, impossible_lineup, empty_week, no_variance, scale_invariance, lopsided, hot_start,
             free_agent_not_phantom]
    out = []
    for t in tests:
        try:
            out.append(t())
        except Exception as e:  # a crash is a failure, not an abort
            out.append(_res(t.__name__, (t.__doc__ or "").strip(), False, f"crashed: {type(e).__name__}: {e}"))
    if H is not None:
        out += leakage_tests(H)
    return out


def leakage_tests(H) -> list[dict]:
    """Corrupt all data after the prediction date; predictions must not change."""
    import copy

    from tradecalc.sim.engine import build_world_history

    res = []
    import yaml

    from tradecalc.sim.params import PARAMS_PATH

    params = yaml.safe_load(PARAMS_PATH.read_text())
    asof, season, prev = "2026-01-05", "2025-26", "2024-25"
    w1 = build_world_history(H, params, asof, season, prev)
    H2 = copy.copy(H)
    H2.logs = H.logs.copy()
    fut = H2.logs.game_date >= asof
    H2.logs.loc[fut, ["pts", "reb", "ast", "min", "fp"]] = 999.0
    H2.team_logs = H.team_logs.copy()
    H2.team_logs.loc[H2.team_logs.game_date >= asof, ["pts", "opp_pts", "pace"]] = 999.0
    H2._cache = {}
    w2 = build_world_history(H2, params, asof, season, prev)
    same = (w1.players[["exp_min", "fga", "reb"]].round(9).equals(w2.players[["exp_min", "fga", "reb"]].round(9))
            and w1.defense.factors.round(9).equals(w2.defense.factors.round(9)))
    res.append(_res("No peeking at the future (engine)", "Overwrite every box score after the prediction date with 999s",
                    same, "projections identical" if same else "projections changed: future data leaked"))
    return res
