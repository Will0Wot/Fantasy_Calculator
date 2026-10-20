from datetime import timedelta

from tradecalc.tests.conftest import make_player, make_state
from tradecalc.trade.matchup import week_dist
from tradecalc.trade.streaming import plan, timeline_dist

ALL = ["PG", "SG", "SF", "PF", "C"]
ROSTER_10 = {"roster.bench": 0, "roster.ir": 0}        # 10 active spots, so no open spot to fill


def league(fa_week2_games: int, b_week2_games: int = 4, injured_core: bool = False, ir_slots: int = 0,
           b_mu: float = 25):
    """Team 1: 9 core players who play every day + bench player B (b_mu/game, 2 games in week 1).
    Free agent S (22/game) plays 4 games in week 1. Team 2 is a similar opponent."""
    from datetime import date

    mon = date(2026, 10, 26)
    days = [mon + timedelta(i) for i in range(14)]
    games = {d: {"CORE"} for d in days}
    for i in (0, 2):
        games[days[i]].add("BBB")
    for i in range(b_week2_games):
        games[days[7 + i]].add("BBB")
    for i in range(4):
        games[days[i]].add("SSS")
    for i in range(fa_week2_games):
        games[days[7 + i]].add("SSS")
    ps = [make_player(i, 1, "CORE", ALL, 40) for i in range(1, 10)]
    ps.append(make_player(10, 1, "BBB", ALL, b_mu))
    ps += [make_player(100 + i, 2, "CORE", ALL, 41) for i in range(10)]
    ps.append(make_player(500, 0, "SSS", ALL, 22))
    if injured_core:
        ps[0].injury_status = "OUT"
        ps[0].avail = type(ps[0].avail)(1.0, out_games=99)
    st = make_state(ps, games, n_weeks=2, opponents={1: {1: 2, 2: 2}, 2: {1: 1, 2: 1}},
                    cfg_overrides={**ROSTER_10, "roster.ir": ir_slots})
    return st


def test_timeline_with_one_segment_is_the_week():
    st = league(4)
    w = st.cal.weeks[0]
    M, V = timeline_dist(st, w, [(w.start, st.rosters[1])])
    d = week_dist(st.rosters[1], w, st)
    assert abs(M - d.M) < 1e-9 and abs(V - d.V) < 1e-9


def test_streams_a_free_agent_with_more_games_when_he_is_also_better_later():
    st = league(fa_week2_games=4, b_mu=18)              # S (22/game) beats B (18/game) both weeks
    pl = plan(st, 1, st.cal.weeks[0], max_moves=1)
    assert len(pl.moves) == 1
    m = pl.moves[0]
    assert (m.kind, m.out, m.add, m.day) == ("drop", 10, 500, st.cal.weeks[0].start)
    assert m.gain_pts > 40 and pl.p_after > pl.p_before and m.d_future > 0


def test_waits_to_drop_until_after_the_dropped_players_game_when_that_is_better():
    st = league(fa_week2_games=4)                        # B (25/game) plays Monday; S (22) plays Mon-Thu
    m = plan(st, 1, st.cal.weeks[0], max_moves=1).moves[0]
    assert m.add == 500 and m.day == st.cal.weeks[0].start + timedelta(1)     # pick up Tuesday


def test_no_one_week_rental_that_costs_more_later():
    # S plays 4 games this week and none next week; B plays 2 now and 4 next week
    st = league(fa_week2_games=0, b_week2_games=4)
    pl = plan(st, 1, st.cal.weeks[0], max_moves=1)
    assert not pl.moves, [(m.kind, m.out, m.add, m.d_week, m.d_future) for m in pl.moves]


def test_injured_player_goes_to_ir_instead_of_dropping_someone():
    st = league(fa_week2_games=4, injured_core=True, ir_slots=1)
    pl = plan(st, 1, st.cal.weeks[0], max_moves=1)
    assert pl.moves and pl.moves[0].kind == "ir" and pl.moves[0].out == 1 and pl.moves[0].add == 500
