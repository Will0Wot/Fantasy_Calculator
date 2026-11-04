"""Sanity rules for player values: injured players, prior minutes, unsigned free agents.

Each test is a case found in the live league (Nov 2026) where a number was off and a recommendation
followed from it, e.g. "drop Jimmy Butler (39 a game last season) for a 25-a-game streamer"."""
from types import SimpleNamespace

import pandas as pd
import pytest

from tradecalc.config import load_config
from tradecalc.model.values import _prior_minutes, _record_fppg, _track_record
from tradecalc.sim.engine import STATS, minutes_projection
from tradecalc.tests.conftest import make_player, make_state
from tradecalc.trade.streaming import available_free_agents, best_healthy_fa

RATES = {"fga": 0.4, "fta": 0.12, "reb": 0.2, "ast": 0.15, "stl": 0.03, "blk": 0.02, "tov": 0.06,
         "fg3a_share": 0.35, "fg2": 0.55, "fg3": 0.36, "ft": 0.8}


def _world(rows):
    players = pd.DataFrame([{**RATES, **r} for r in rows]).set_index("key")
    return SimpleNamespace(players=players, params={"rate_calibration": 1.0}, scoring=load_config()["scoring"])


def test_injured_player_is_valued_at_his_healthy_minutes():
    """The depth chart lists an injured player last, so his minutes today are a backup's (Butler: 12).
    His value once back uses the minutes he plays healthy."""
    mp = minutes_projection(_world([{"key": "star", "exp_min": 12.0, "healthy_min": 32.0, "depth": 4},
                                    {"key": "regular", "exp_min": 30.0, "depth": 1}]))
    assert mp.loc["star", "fppg_healthy"] == pytest.approx(mp.loc["star", "fppg_min"] * 32 / 12, rel=1e-6)
    assert mp.loc["regular", "fppg_healthy"] == pytest.approx(mp.loc["regular", "fppg_min"])   # no healthy view


def _league(players):
    return SimpleNamespace(players={i: SimpleNamespace(name=n, projection=pr) for i, (n, pr) in enumerate(players)})


def test_prior_minutes_match_the_minutes_behind_the_prior():
    """Joan Beringer: no ESPN projection, so his prior is the engine's 20.5 at 18.3 minutes. Dividing
    by last season's 7.9 garbage-time minutes made him a 2.6-points-a-minute player (26 a game)."""
    league = _league([("Joan Beringer", {}), ("Mikal Bridges", {"GP": 82, "MIN": 33.6})])
    ptab = pd.DataFrame({"proj_fppg": [float("nan"), 32.8]}, index=["joan beringer", "mikal bridges"])
    mp = pd.DataFrame({"minutes": [18.3, 26.9], "healthy_min": [18.3, 33.0]}, index=["joan beringer", "mikal bridges"])
    prev = pd.DataFrame({"player_name": ["Joan Beringer"] * 40, "min": [7.9] * 40})
    pm = _prior_minutes(league, ptab, mp, set(), prev)
    assert pm["joan beringer"] == pytest.approx(18.3)                 # the engine's minutes, not last season's
    assert pm["mikal bridges"] == pytest.approx((33.6 + 26.9) / 2)    # ESPN's piece and the engine's piece
    hurt = _prior_minutes(league, ptab, mp, {"mikal bridges"}, prev)
    assert hurt["mikal bridges"] == pytest.approx((33.6 + 33.0) / 2)  # injured: the engine's healthy minutes


def test_track_record_uses_the_last_real_season():
    """Haliburton missed 2025-26: his record is 2024-25. A 10-game cameo doesn't replace a full season."""
    def logs(name, n, fp):
        return pd.DataFrame({"player_name": [name] * n, "min": [30.0] * n, "fp": [fp] * n})
    prev = pd.concat([logs("A", 70, 30.0), logs("B", 5, 10.0)])
    prev2 = pd.concat([logs("A", 60, 25.0), logs("B", 74, 44.0)])
    t = _track_record(prev, prev2, {}, "2025-26")
    assert t["a"] == {"last_fppg": 30.0, "last_games": 70, "last_season": "2025-26"}
    assert t["b"] == {"last_fppg": 44.0, "last_games": 74, "last_season": "2024-25"}
    rec = _record_fppg(prev, prev2, {})
    assert rec["a"] == 30.0 and rec["b"] == 44.0


def test_unsigned_players_are_not_free_agents_you_can_stream():
    """Kevin Porter Jr. (no NBA team, 0 games left, 31.7 projected) was setting the 'best free agent' bar."""
    from datetime import date, timedelta

    days = [date(2026, 10, 26) + timedelta(i) for i in range(7)]
    games = {d: {"AAA"} for d in days}
    ps = [make_player(1, 1, "AAA", ["PG"], 30), make_player(2, 0, "AAA", ["PG"], 20),
          make_player(3, 0, "FA", ["PG"], 32)]
    st = make_state(ps, games)
    pool = available_free_agents(st)
    assert pool == [2] and best_healthy_fa(st, pool) == 20


def test_rate_names_cover_the_engine():
    assert set(RATES) == set(STATS)
