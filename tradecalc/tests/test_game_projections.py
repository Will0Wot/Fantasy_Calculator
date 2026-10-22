"""Per-game projections (possession simulator -> season model): averaging, weekly points, who starts."""
from datetime import timedelta

import pandas as pd

from tradecalc.sim.possession import forecast
from tradecalc.tests.conftest import make_player, make_state
from tradecalc.trade.lineup import simulate_week
from tradecalc.trade.matchup import week_dist


def test_mu_on_falls_back_to_the_season_average(week_dates):
    p = make_player(1, 1, "AAA", ["PG"], 20)
    p.mu_by_date = {week_dates[0]: 30.0}
    assert p.mu_on(week_dates[0]) == 30.0 and p.mu_on(week_dates[1]) == 20.0


def test_apply_averages_the_season_estimate_and_the_game_simulation(tmp_path, monkeypatch, week_dates):
    d0, d1 = week_dates[0], week_dates[1]
    pd.DataFrame({"key": ["p1", "p1", "p1", "nobody"],
                  "date": [(d0 - timedelta(1)).isoformat(), d0.isoformat(), d1.isoformat(), d0.isoformat()],
                  "fp": [99.0, 30.0, 12.0, 50.0]}).to_parquet(tmp_path / "proj.parquet")
    monkeypatch.setattr(forecast, "path", lambda season: tmp_path / "proj.parquet")
    p1, p2 = make_player(1, 1, "AAA", ["PG"], 20), make_player(2, 1, "AAA", ["SG"], 25)
    n = forecast.apply({1: p1, 2: p2}, "2026-27", d0, 0.5, key_of=str.lower)
    assert n == 1
    assert p1.mu_by_date == {d0: 25.0, d1: 16.0}        # yesterday's game is ignored
    assert p2.mu_by_date == {}                          # no simulation: season average every game
    assert forecast.apply({1: p1}, "2026-27", d0, 0.0, key_of=str.lower) == 0


def test_rookie_keeps_the_season_level_and_takes_the_game_pattern(tmp_path, monkeypatch, week_dates):
    d0, d1 = week_dates[0], week_dates[1]
    pd.DataFrame({"key": ["p1", "p1"], "date": [d0.isoformat(), d1.isoformat()], "fp": [12.0, 18.0],
                  "poss": [0.0, 0.0]}).to_parquet(tmp_path / "proj.parquet")
    monkeypatch.setattr(forecast, "path", lambda season: tmp_path / "proj.parquet")
    p1 = make_player(1, 1, "AAA", ["PG"], 30)
    forecast.apply({1: p1}, "2026-27", d0, 0.5, key_of=str.lower)
    assert p1.mu_by_date == {d0: 27.0, d1: 33.0}        # 30 x (0.8, 1.2) averaged 50/50 with 30
    # after about one game (75 possessions) his simulated level (15) counts half
    pd.DataFrame({"key": ["p1", "p1"], "date": [d0.isoformat(), d1.isoformat()], "fp": [12.0, 18.0],
                  "poss": [75.0, 75.0]}).to_parquet(tmp_path / "proj.parquet")
    forecast.apply({1: p1}, "2026-27", d0, 0.5, key_of=str.lower)
    assert p1.mu_by_date == {d0: 24.0, d1: 28.5}        # level 22.5 x (0.8, 1.2) averaged with 30


def test_week_points_use_each_games_projection(week_dates):
    ps = [make_player(1, 1, "AAA", ["PG"], 20)]
    st = make_state(ps, {week_dates[0]: {"AAA", "ZZZ"}, week_dates[1]: {"AAA", "YYY"}})
    base = week_dist([1], st.cal.weeks[0], st).M
    ps[0].mu_by_date = {week_dates[0]: 30.0}
    st.__dict__.pop("_week_cache", None)
    assert abs(base - 40.0) < 1e-9
    assert abs(week_dist([1], st.cal.weeks[0], st).M - 50.0) < 1e-9


def test_good_matchup_moves_a_bench_player_into_the_lineup(week_dates):
    # 11 players with a game on Monday, 10 slots: the lowest projection sits...
    ps = [make_player(i, 1, "AAA", ["PG", "SG", "SF", "PF", "C"], 40 - i) for i in range(1, 12)]
    st = make_state(ps, {week_dates[0]: {"AAA", "ZZZ"}})
    assert simulate_week([p.espn_id for p in ps], st.cal.weeks[0], st).starts()[11] == 0
    # ...unless his game projection that night is better than the 10th man's
    ps[10].mu_by_date = {week_dates[0]: 35.0}
    g = simulate_week([p.espn_id for p in ps], st.cal.weeks[0], st).starts()
    assert g[11] == 1 and g[10] == 0
