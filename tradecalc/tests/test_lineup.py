from tradecalc.tests.conftest import make_player, make_state
from tradecalc.trade.lineup import simulate_week

# Starting slots from league.yaml: PG SG SF PF C G F UTIL UTIL UTIL (10 starters).


def test_position_eligibility_three_centers(week_dates):
    # Three C-only players and nothing else: C slot + 3 UTIL -> all three start.
    # Add two more C-only players: still only 4 C-eligible starter slots (C + 3 UTIL).
    ps = [make_player(i, 1, "AAA", ["C"], 40 - i) for i in range(1, 6)]
    st = make_state(ps, {week_dates[0]: {"AAA", "ZZZ"}})
    g = simulate_week([p.espn_id for p in ps], st.cal.weeks[0], st).starts()
    assert sum(g.values()) == 4
    assert g[5] == 0  # the lowest-mu center sits


def test_eligibility_prefers_assignment_that_starts_more_value(week_dates):
    # PG/SG guard and a PG-only guard plus 3 SG-only: greedy could waste the G slot.
    ps = [make_player(1, 1, "AAA", ["PG"], 30), make_player(2, 1, "AAA", ["SG"], 29),
          make_player(3, 1, "AAA", ["SG"], 28), make_player(4, 1, "AAA", ["SG"], 27),
          make_player(5, 1, "AAA", ["SG"], 26), make_player(6, 1, "AAA", ["SG"], 25),
          make_player(7, 1, "AAA", ["SG"], 24)]
    st = make_state(ps, {week_dates[0]: {"AAA", "ZZZ"}})
    g = simulate_week([p.espn_id for p in ps], st.cal.weeks[0], st).starts()
    # SG-eligible slots: SG, G, UTIL x3 = 5; PG slot goes to P1. So 6 start, P7 sits.
    assert sum(g.values()) == 6 and g[7] == 0


def test_crowded_night_benches_lowest(week_dates):
    # 12 players all play on Monday; only 10 slots.
    pos = [["PG"], ["SG"], ["SF"], ["PF"], ["C"], ["PG"], ["SF"], ["C"], ["PF"], ["SG"], ["PF"], ["SF"]]
    ps = [make_player(i + 1, 1, "AAA", p, 50 - i) for i, p in enumerate(pos)]
    # Second team plays Tuesday: one player on it.
    ps.append(make_player(13, 1, "BBB", ["C"], 10))
    st = make_state(ps, {week_dates[0]: {"AAA", "XXX"}, week_dates[1]: {"BBB", "YYY"}})
    res = simulate_week([p.espn_id for p in ps], st.cal.weeks[0], st)
    g = res.starts()
    assert sum(g[i] for i in range(1, 13)) == 10
    assert g[11] == 0 and g[12] == 0  # two lowest-mu Monday players benched
    assert g[13] == 1                 # low-mu player still starts on an empty Tuesday


def test_out_player_not_started_and_returns_midweek(week_dates):
    ps = [make_player(1, 1, "AAA", ["PG"], 40, a=0.9, out_games=2)]
    st = make_state(ps, {d: {"AAA", "ZZZ"} for d in week_dates[:4]}, cfg_overrides={"lineup.fill_in": None})
    res = simulate_week([1], st.cal.weeks[0], st)
    assert res.g[0] == 2 and abs(res.sum_a[0] - 1.8) < 1e-9
    # with fill-ins he only takes a slot on the days he is known to play (benched when ruled out)
    st = make_state(ps, {d: {"AAA", "ZZZ"} for d in week_dates[:4]})
    res = simulate_week([1], st.cal.weeks[0], st)
    assert abs(res.g[0] - 1.8) < 1e-9 and abs(res.sum_a[0] - 1.8) < 1e-9


def test_fill_in_bench_player_starts_when_starter_out(week_dates):
    # 11 guards for 10 slots: the 11th only plays when someone ahead of him sits
    ps = [make_player(i, 1, "AAA", ["PG", "SG", "SF", "PF", "C"], 40 - i, a=0.8) for i in range(1, 12)]
    st = make_state(ps, {week_dates[0]: {"AAA", "ZZZ"}})
    res = simulate_week([p.espn_id for p in ps], st.cal.weeks[0], st)
    from scipy.stats import binom

    assert abs(res.sum_a[-1] - 0.8 * binom.cdf(9, 10, 0.8)) < 1e-9     # P(<10 of the 10 above play)
    assert abs(res.sum_a[:10].sum() - 8.0) < 1e-9                     # the top 10 always start when they play
    no_fill = simulate_week([p.espn_id for p in ps], st.cal.weeks[0],
                            make_state(ps, {week_dates[0]: {"AAA", "ZZZ"}}, cfg_overrides={"lineup.fill_in": None}))
    assert no_fill.sum_a[-1] == 0


def test_from_day_only_counts_remaining_days(week_dates):
    ps = [make_player(1, 1, "AAA", ["PG"], 40)]
    st = make_state(ps, {d: {"AAA", "ZZZ"} for d in week_dates})
    assert simulate_week([1], st.cal.weeks[0], st, from_day=week_dates[4]).g[0] == 3


def test_weekly_games_cap(week_dates):
    ps = [make_player(i, 1, "AAA", ["PG", "SG", "SF", "PF", "C"], 40 - i) for i in range(1, 11)]
    games = {d: {"AAA", "ZZZ"} for d in week_dates}
    st = make_state(ps, games, cfg_overrides={"lineup.weekly_games_cap": 25})
    res = simulate_week([p.espn_id for p in ps], st.cal.weeks[0], st)
    assert res.g.sum() == 25
    g = res.starts()
    # Days 1-2 start all 10; day 3 starts only the 5 best.
    assert g[1] == 3 and g[10] == 2


def test_weekly_lock_picks_one_lineup(week_dates):
    # 9 non-C players (4 games x 20 = 80 each) fill all 9 non-C slots; the C slot goes to
    # AAA's center (4 x 30 = 120). BBB's center plays once (50), alone on Friday.
    ps = [make_player(1, 1, "AAA", ["C"], 30), make_player(2, 1, "BBB", ["C"], 50)]
    ps += [make_player(i, 1, "CCC", ["PG", "SG", "SF", "PF"], 20) for i in range(3, 12)]
    games = {d: {"AAA", "CCC"} for d in week_dates[:4]}
    games[week_dates[4]] = {"BBB", "QQQ"}
    ids = [p.espn_id for p in ps]
    weekly = make_state(ps, games, cfg_overrides={"lineup.lock": "weekly"})
    g = simulate_week(ids, weekly.cal.weeks[0], weekly).starts()
    assert g[1] == 4 and g[2] == 0  # weekly: BBB center is benched all week
    daily = make_state(ps, games)
    g = simulate_week(ids, daily.cal.weeks[0], daily).starts()
    assert g[2] == 1                # daily: he starts on his empty Friday


def test_fill_in_matches_brute_force():
    """Expected points and variance of the fill-in recursion equal full enumeration."""
    import itertools

    import numpy as np

    from tradecalc.trade.lineup import _fill_in_week

    rng = np.random.default_rng(3)
    for _ in range(5):
        n, days, slots = 6, 2, 3
        a = rng.uniform(0, 1, (n, days)) * (rng.random((n, days)) < 0.8)
        mu = rng.uniform(5, 50, n)
        psi = float(rng.choice([1.0, 0.6]))
        res = _fill_in_week(list(range(n)), a, mu, np.ones((n, slots), dtype=bool), psi)
        m1 = var = 0.0
        for t in range(days):
            idx = [i for i in np.argsort(-mu, kind="stable") if a[i, t] > 0]
            S = min(slots, len(idx))
            e1 = e2 = 0.0
            for states in itertools.product(range(3), repeat=len(idx)):   # plays / surprise scratch / benched
                pr, k, pts = 1.0, 0, 0.0
                for i, s in zip(idx, states):
                    pr *= (a[i, t], (1 - psi) * (1 - a[i, t]), psi * (1 - a[i, t]))[s]
                    if s < 2:
                        pts += mu[i] if (k < S and s == 0) else 0.0
                        k += 1
                e1 += pr * pts
                e2 += pr * pts * pts
            m1 += e1
            var += e2 - e1 ** 2
        assert abs(float(mu @ res.sum_a) - m1) < 1e-9 and abs(res.var_mu - var) < 1e-6
