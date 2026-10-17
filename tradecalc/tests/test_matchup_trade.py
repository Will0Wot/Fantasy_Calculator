import numpy as np
import pytest

from tradecalc.tests.conftest import make_player, make_state
from tradecalc.trade.evaluate import Trade, apply_trade, evaluate
from tradecalc.trade.matchup import p_win, p_win_monte_carlo, season_outlook, week_dist

POS = [["PG"], ["SG"], ["SF"], ["PF"], ["C"], ["PG", "SG"], ["SF", "PF"], ["PF", "C"],
       ["SG", "SF"], ["C"], ["PG"], ["SF"], ["PF"]]


def _random_league(rng, teams=(1, 2), pros=12):
    pro = [f"T{i}" for i in range(pros)]
    players, pid = [], 0
    for t in teams:
        for pos in POS:
            pid += 1
            mu = rng.uniform(15, 50)
            players.append(make_player(pid, t, pro[rng.integers(pros)], pos, mu, sigma=0.42 * mu,
                                       a=rng.uniform(0.7, 0.95)))
    return players, pro


def _random_schedule(rng, pro, start_dates):
    games = {}
    for d in start_dates:
        games[d] = set(rng.choice(pro, size=rng.integers(4, 11), replace=False))
    return games


def test_normal_matches_monte_carlo_on_20_matchups(week_dates):
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(20):
        players, pro = _random_league(rng)
        st = make_state(players, _random_schedule(rng, pro, week_dates))
        w = st.cal.weeks[0]
        d1, d2 = week_dist(st.rosters[1], w, st), week_dist(st.rosters[2], w, st)
        p_norm = p_win(d1.M, d1.V, d2.M, d2.V)
        p_mc = p_win_monte_carlo(d1, d2, st, n=40000, seed=int(rng.integers(1 << 30)))
        diffs.append(abs(p_norm - p_mc))
    assert max(diffs) < 0.02, f"max |normal - MC| = {max(diffs):.4f}"


def test_week_distribution_formula(week_dates):
    p = make_player(1, 1, "AAA", ["PG"], 40, sigma=10, a=0.8)
    st = make_state([p], {d: {"AAA", "Z"} for d in week_dates[:3]})
    d = week_dist([1], st.cal.weeks[0], st)
    assert d.M == pytest.approx(3 * 0.8 * 40)
    assert d.V == pytest.approx(3 * (0.8 * 100 + 0.8 * 0.2 * 1600))


def _two_team_state(week_dates, n_weeks=1):
    ps = [make_player(i, 1, "AAA", ["PG", "SG", "SF", "PF", "C"], 30 - i) for i in range(1, 14)]
    ps += [make_player(i, 2, "BBB", ["PG", "SG", "SF", "PF", "C"], 50 - i) for i in range(21, 34)]
    ps += [make_player(91, 0, "CCC", ["C"], 22), make_player(92, 0, "CCC", ["PG"], 18)]
    games = {d: {"AAA", "BBB", "CCC"} for d in week_dates[:3]}
    opp = {1: {1: 2}, 2: {1: 1}}
    return make_state(ps, games, n_weeks=n_weeks, opponents=opp)


def test_uneven_trade_two_for_one(week_dates):
    st = _two_team_state(week_dates)
    trade = Trade(you=1, partner=2, give=(1, 2), get=(21,))
    new, moves = apply_trade(trade, st.rosters, st)
    assert len(new[1]) == len(st.rosters[1]) and len(new[2]) == len(st.rosters[2])
    # You receive fewer -> sign the best free agent (by ROS value).
    assert moves[1] == ([], [91]) and 91 in new[1]
    # Partner receives more -> drops lowest-ROS player (one of the incoming low guys or their own).
    lowest = min(st.rosters[2] + [1, 2], key=lambda i: st.players[i].ros_value if i != 21 else 1e9)
    assert moves[2][0] == [lowest] and lowest not in new[2]


def test_uneven_trade_one_for_two_free_agents_not_shared(week_dates):
    st = _two_team_state(week_dates)
    # Both sides lose a player net? Not possible in a 2-team trade; check FA pool is consumed once.
    new, moves = apply_trade(Trade(1, 2, give=(1,), get=(21, 22)), st.rosters, st)
    assert moves[2][1] == [91] and moves[1][1] == []
    assert len(new[1]) == 13 and len(new[2]) == 13


def test_head_to_head_week_uses_partner_post_trade_roster(week_dates):
    st = _two_team_state(week_dates)
    trade = Trade(1, 2, give=(13,), get=(21,))  # you get partner's best for your worst
    r = evaluate(trade, st)
    # In a 1-week season vs the partner, your gain must be exactly the partner's loss.
    assert r.you.d_wins > 0 and r.you.d_wins == pytest.approx(-r.partner.d_wins)
    new, _ = apply_trade(trade, st.rosters, st)
    assert r.you.wins_after == pytest.approx(season_outlook(1, new, st)["e_wins"])
    assert r.verdict in ("ACCEPT", "FAIR-ISH")
    assert any("FPPG" in d for d in r.drivers)


def test_trading_ir_player_does_not_open_active_spot(week_dates):
    st = _two_team_state(week_dates)
    ir_guy = make_player(50, 1, "AAA", ["SF"], 35, a=0.6, out_games=20)
    ir_guy.injury_status = "OUT"
    st.players[50] = ir_guy
    st.rosters[1].append(50)
    st.ir = {1: {50}, 2: set()}
    # 2-for-1 where one of the two you give is your IR player: your active count is unchanged,
    # so no free agent is signed. Partner takes him into their empty IR slot: no drop needed.
    new, moves = apply_trade(Trade(1, 2, give=(50, 13), get=(21,)), st.rosters, st)
    assert moves[1] == ([], []) and moves[2] == ([], [])
    assert 50 in new[2] and 21 in new[1]


def test_banked_points_only_count_completed_days_this_week(week_dates):
    from datetime import timedelta

    st = _two_team_state(week_dates)
    st.cal.first_day = week_dates[0]                        # scoring period 1 = Monday
    st.league.matchups = [{"week": 1, "home": 1, "away": 2,
                           "home_points": {1: 100.0, 2: 80.0, 3: 90.0}, "away_points": {1: 70.0, 2: 60.0}}]
    st.today = week_dates[2]                                 # Wednesday: Mon + Tue are done
    assert st.banked(1, st.cal.weeks[0]) == (180.0, week_dates[1])
    assert st.banked(2, st.cal.weeks[0]) == (130.0, week_dates[1])
    before = season_outlook(1, st.rosters, st)["weeks"][0]
    st.league.matchups[0]["home_points"] = {}
    without = season_outlook(1, st.rosters, st)["weeks"][0]
    assert before["M"] == pytest.approx(without["M"] + 180.0)


def test_explanation_summary_bullets_and_weekly_change(week_dates):
    from tradecalc.trade.explain import explain

    st = _two_team_state(week_dates)
    r = evaluate(Trade(1, 2, give=(13,), get=(21,)), st)
    st.real_today = st.today
    e = explain(r, st)
    assert e.verdict == r.verdict
    assert e.paragraph.startswith({"ACCEPT": "Accept", "DECLINE": "Decline", "FAIR-ISH": "Close call"}[r.verdict])
    assert any(b.startswith("Production:") for b in e.bullets)
    w = e.this_week
    assert w["delta"] == pytest.approx((round(w["after"] * 100) - round(w["before"] * 100)) / 100)
    assert len(e.signs) == len(e.bullets)
    # a losing trade for you reads as a decline led by what hurts you
    r2 = evaluate(Trade(1, 2, give=(1,), get=(33,)), st)
    e2 = explain(r2, st)
    if r2.verdict == "DECLINE":
        assert e2.signs[0] == -1


def test_league_forecast_probabilities_and_last_place(week_dates):
    from tradecalc.trade.standings import forecast, last_place_line

    st = _two_team_state(week_dates)
    for i in st.rosters[2]:  # make team 2 much stronger
        st.players[i].mu += 15
    fc = forecast(st, n=4000, seed=1)
    assert list(fc["team_id"]) == [2, 1]
    assert fc["p_first"].sum() == pytest.approx(1.0) and fc["p_last"].sum() == pytest.approx(1.0)
    assert fc["exp_wins"].sum() == pytest.approx(1.0)          # one week, one matchup
    assert last_place_line(fc) == "Team1 will be eating poop"


def test_league_forecast_counts_finished_weeks(week_dates):
    from datetime import timedelta

    from tradecalc.trade.standings import forecast

    st = _two_team_state(week_dates)
    st.league.matchups = [{"week": 1, "home": 1, "away": 2, "winner": "HOME", "home_total": 900.0, "away_total": 800.0,
                           "home_points": {}, "away_points": {}}]
    st.today = week_dates[-1] + timedelta(days=1)                # week 1 is over
    fc = forecast(st, n=100, seed=1)
    r1 = fc.set_index("team_id").loc[1]
    assert r1["record"] == "1-0" and r1["p_first"] == 1.0
