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
    st = make_state(ps, {d: {"AAA", "ZZZ"} for d in week_dates[:4]})
    res = simulate_week([1], st.cal.weeks[0], st)
    assert res.g[0] == 2 and abs(res.sum_a[0] - 1.8) < 1e-9


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
