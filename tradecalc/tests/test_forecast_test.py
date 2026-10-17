import numpy as np
import pytest

from tradecalc.sim.engine import SeasonLayer
from tradecalc.sim.forecast_test import greedy_lineup, play_weeks, round_robin, standings


def test_round_robin_everyone_plays_once_per_week():
    sched = round_robin(12, 24)
    assert len(sched) == 24
    for week in sched:
        teams = [t for pair in week for t in pair]
        assert sorted(teams) == list(range(12))
    # each pair meets twice in 22 weeks (11 rounds x 2)
    pairs = [tuple(sorted(p)) for w in sched[:22] for p in w]
    assert all(pairs.count(p) == 2 for p in set(pairs))


def test_greedy_lineup_respects_slots():
    # 5 guards, 1 center active: guards fill 3 guard slots + 2 of 3 UTIL, center takes C
    groups = ["G"] * 5 + ["C"]
    act = np.ones((1, 6), dtype=bool)
    st = greedy_lineup(act, np.arange(6, 0, -1, dtype=float), groups)
    assert st.sum() == 6
    # 8 guards: 3 guard slots + 3 UTIL = 6 start, the 2 lowest sit
    st = greedy_lineup(np.ones((1, 8), dtype=bool), np.arange(8, 0, -1, dtype=float), ["G"] * 8)
    assert st.sum() == 6 and not st[0, 6] and not st[0, 7]


def test_standings_wins_then_points():
    wins = np.array([[10, 12, 12]])
    pts = np.array([[900, 800, 850]])
    assert list(standings(wins, pts)[0]) == [3, 2, 1]


def test_play_weeks_counts_ties_as_half():
    scores = np.array([[[100, 100]], [[90, 80]]]).reshape(2, 2)  # 2 weeks x 2 teams
    w, p = play_weeks(scores, [[(0, 1)], [(0, 1)]])
    assert list(w[0]) == [1.5, 0.5]


def test_season_layer_matches_measured_absence_share_and_form():
    rng = np.random.default_rng(0)
    L = SeasonLayer([f"p{i}" for i in range(200)], 24, 400, rng)
    out_share = 1 - L.on[:, 8:, :].mean()          # after burn-in (everyone starts healthy)
    assert out_share == pytest.approx(0.175, abs=0.02)
    assert L.talent.mean() == pytest.approx(1.0, abs=0.01)
    assert np.log(L.talent).std() == pytest.approx(0.30, abs=0.02)
    # form uncertainty shrinks with games played
    L2 = SeasonLayer(["a"], 2, 4000, rng, {"a": 60})
    assert np.log(L2.talent).std() < 0.2
