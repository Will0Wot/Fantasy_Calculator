from datetime import timedelta

from tradecalc.tests.conftest import make_player, make_state
from tradecalc.trade.matchup import week_dist
from tradecalc.trade.streaming import plan, timeline_dist

ALL = ["PG", "SG", "SF", "PF", "C"]
ROSTER_10 = {"roster.bench": 0, "roster.ir": 0}        # 10 active spots, so no open spot to fill


def league(fa_week2_games: int, b_week2_games: int = 4, injured_core: bool = False, ir_slots: int = 0,
           b_mu: float = 25, filler_mu: float = 26, filler_week2_games: int = 2):
    """Team 1: 9 core players who play every day + bench player B (b_mu/game, 2 games in week 1).
    Free agent S (22/game) plays 4 games in week 1. Team 2 is a similar opponent. Three more free
    agents (filler_mu/game, idle in week 1) set the level of the waiver wire."""
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
    for i in range(filler_week2_games):
        games[days[7 + i]].add("ZZZ")
    ps = [make_player(i, 1, "CORE", ALL, 40) for i in range(1, 10)]
    ps.append(make_player(10, 1, "BBB", ALL, b_mu))
    ps += [make_player(100 + i, 2, "CORE", ALL, 41) for i in range(10)]
    ps.append(make_player(500, 0, "SSS", ALL, 22))
    ps += [make_player(600 + i, 0, "ZZZ", ALL, filler_mu) for i in range(3)]
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
    st = league(fa_week2_games=4, b_mu=18, b_week2_games=0)    # S (22/game) beats B (18/game) both weeks
    pl = plan(st, 1, st.cal.weeks[0], max_moves=1)
    assert len(pl.moves) == 1
    m = pl.moves[0]
    assert (m.kind, m.out, m.add, m.day) == ("drop", 10, 500, st.cal.weeks[0].start)
    assert m.gain_pts > 40 and pl.p_after > pl.p_before and m.d_future > 0


def test_waits_to_drop_until_after_the_dropped_players_game_when_that_is_better():
    st = league(fa_week2_games=4, b_week2_games=0)       # B (25/game) plays Monday; S (22) plays Mon-Thu
    m = plan(st, 1, st.cal.weeks[0], max_moves=1).moves[0]
    assert m.add == 500 and m.day == st.cal.weeks[0].start + timedelta(1)     # pick up Tuesday


def test_no_one_week_rental_that_costs_more_later():
    # S plays 4 games this week and none next week; B plays 2 now and 4 next week. The waiver wire is
    # strong enough (fillers 26/game, 7 games next week) that B could be dropped, but not for S.
    st = league(fa_week2_games=0, b_week2_games=4, filler_week2_games=7)
    pl = plan(st, 1, st.cal.weeks[0], max_moves=1)
    assert all(m.add != 500 for m in pl.moves), [(m.kind, m.out, m.add, m.d_week, m.d_future) for m in pl.moves]


def test_injured_player_goes_to_ir_instead_of_dropping_someone():
    st = league(fa_week2_games=4, injured_core=True, ir_slots=1)
    pl = plan(st, 1, st.cal.weeks[0], max_moves=1)
    assert pl.moves and pl.moves[0].kind == "ir" and pl.moves[0].out == 1 and pl.moves[0].add == 500


def test_never_drops_a_player_a_rival_would_claim():
    # B scores 50 a game but is hurt and will miss most games: dropping him would add points this
    # week, but he's far better than any free agent, so a rival would claim him
    st = league(fa_week2_games=4, b_mu=50)
    b = st.players[10]
    b.avail = type(b.avail)(0.3, out_games=3)
    b.ros_value = b.mu * 0.3 * 2                        # lowest rest-of-season value on the roster
    pl = plan(st, 1, st.cal.weeks[0], max_moves=1)
    assert 10 in pl.protected
    assert all(m.out != 10 for m in pl.moves + pl.options)
