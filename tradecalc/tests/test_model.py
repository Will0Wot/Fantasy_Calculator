import numpy as np
import pandas as pd
import pytest

from tradecalc.model.availability import Availability, availability, games_played_rate
from tradecalc.model.blend import blend
from tradecalc.model.names import norm_name
from tradecalc.model.prior import fit_adp_curve, prior_table
from tradecalc.model.variance import shrunk_sigma


def _live(n, fppg, mpg=30, mpg5=30, recent=None):
    return pd.DataFrame({"n": [n], "live_fppg": [fppg], "mpg_season": [mpg], "mpg_last5": [mpg5],
                         "fppg_recent": [recent if recent is not None else fppg]}, index=[1])


def test_blend_formula():
    b = blend(pd.Series({1: 40.0}), _live(12, 30.0), k=12)
    assert b.loc[1, "blended_fppg"] == pytest.approx(35.0)
    assert not b.loc[1, "role_change"]


def test_blend_role_change_halves_k_and_weights_recent():
    # mpg up 50% over last 5 -> flagged; k 12 -> 6; live term = 0.5*30 + 0.5*40 = 35
    b = blend(pd.Series({1: 20.0}), _live(6, 30.0, mpg=20, mpg5=30, recent=40.0), k=12)
    assert b.loc[1, "role_change"]
    assert b.loc[1, "blended_fppg"] == pytest.approx((6 * 20 + 6 * 35) / 12)


def test_blend_needs_min_games_for_role_change():
    b = blend(pd.Series({1: 20.0}), _live(4, 30.0, mpg=20, mpg5=30), k=12)
    assert not b.loc[1, "role_change"]


def test_blend_without_prior_is_live():
    b = blend(pd.Series(dtype=float), _live(3, 30.0), k=12)
    assert b.loc[1, "blended_fppg"] == pytest.approx(30.0)


def test_adp_curve_monotone_and_prior_average():
    ranks = np.arange(1, 201)
    adp = pd.DataFrame({"name": [f"P{i}" for i in ranks], "adp": ranks, "adp_rank": ranks})
    rng = np.random.default_rng(0)
    actual = pd.DataFrame({"player_name": adp["name"], "live_fppg": 60 * np.exp(-ranks / 80) + 10 + rng.normal(0, 3, 200),
                           "n": 60})
    curve = fit_adp_curve(adp, actual)
    v = curve(ranks)
    assert np.all(np.diff(v) <= 1e-9)
    t = prior_table(adp.head(3), curve, {"P1": 100.0, "Rookie X": 30.0})
    assert t.loc["p1", "prior_fppg"] == pytest.approx((curve(1)[0] + 100) / 2)
    assert t.loc["rookie x", "curve_fppg"] == pytest.approx(curve.tail_value)


def test_shrunk_sigma():
    s = shrunk_sigma([30, 30], [5, np.nan], [10, 0], k=10, cv_typical=0.5)
    assert s[0] == pytest.approx((10 * 15 + 10 * 5) / 20)
    assert s[1] == pytest.approx(15)


def test_availability_rules():
    out = availability("OUT", 0.9, cap=0.95, dtd_prob=0.5, out_default_games=10)
    assert [out.for_game(j) for j in (0, 9, 10)] == [0.0, 0.0, 0.9]
    out_known = availability("OUT", 0.9, cap=0.95, dtd_prob=0.5, out_default_games=10, games_until_return=2)
    assert out_known.for_game(2) == 0.9
    dtd = availability("DAY_TO_DAY", 0.99, cap=0.95, dtd_prob=0.5, out_default_games=10)
    assert dtd.for_game(0) == 0.5 and dtd.for_game(1) == 0.95
    assert availability("ACTIVE", None, cap=0.95, dtd_prob=0.5, out_default_games=10).base == 0.85


def test_games_played_rate_ignores_seasons_not_in_league():
    s1 = pd.DataFrame({"player_id": [1] * 41, "min": 30})
    s2 = pd.DataFrame({"player_id": [1] * 82 + [2] * 70, "min": 30})
    r = games_played_rate([s1, s2])
    assert r[1] == pytest.approx(123 / 164) and r[2] == pytest.approx(70 / 82)


def test_norm_name():
    assert norm_name("Nikola Jokić") == norm_name("Nikola Jokic")
    assert norm_name("Jimmy Butler III") == "jimmy butler"
    assert norm_name("Shai Gilgeous-Alexander") == "shai gilgeous alexander"
