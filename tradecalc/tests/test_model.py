import numpy as np
import pandas as pd
import pytest

from tradecalc.model.availability import Availability, availability, games_played_rate
from tradecalc.model.blend import blend, blend_minutes
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
    # no ADP: the projection alone, not dragged toward the curve's tail
    assert np.isnan(t.loc["rookie x", "curve_fppg"]) and t.loc["rookie x", "prior_fppg"] == pytest.approx(30.0)


def test_undrafted_adp_carries_no_information():
    """ESPN puts every undrafted player near the last pick: hundreds share ADPs of 130-150, so
    their order says nothing and must not be read off the curve as ranks 150-600."""
    drafted = list(np.arange(1.0, 131.0))
    bunched = list(np.linspace(130.5, 149.5, 300))
    adp = pd.DataFrame({"name": [f"P{i}" for i in range(430)], "adp": drafted + bunched})
    adp["adp_rank"] = adp["adp"].rank(method="first").astype(int)
    actual = pd.DataFrame({"player_name": adp["name"], "live_fppg": np.linspace(55, 8, 430), "n": 60})
    curve = fit_adp_curve(adp, actual)
    t = prior_table(adp, curve, {"P400": 27.0})
    assert t.loc["p100", "curve_fppg"] == pytest.approx(curve(101)[0])      # a real pick keeps its value
    assert np.isnan(t.loc["p400", "curve_fppg"])                            # ADP 148, "rank" 401: no value
    assert t.loc["p400", "prior_fppg"] == pytest.approx(27.0)               # his projection stands


def test_minutes_blend_keeps_a_slumping_veteran_near_his_level():
    """33 FPPG at 33 minutes, then 7 games at 14 FPPG in 26 minutes: the minutes drop is mostly
    believed, the shooting slump mostly isn't. The single-weight blend (k=6) would say 22.2."""
    b = blend_minutes(pd.Series({1: 33.0}), pd.Series({1: 33.0}), _live(7, 14.0, mpg=26, mpg5=26),
                      minutes_k=6, rate_k=20, recent_weight=0.25)
    minutes = (6 * 33 + 7 * 26) / 13
    rate = (20 * 1.0 + (7 * 26 / 33) * (14 / 26)) / (20 + 7 * 26 / 33)
    assert b.loc[1, "blended_fppg"] == pytest.approx(minutes * rate)
    assert 25 < b.loc[1, "blended_fppg"] < 27


def test_minutes_blend_agrees_with_a_prior_that_was_right():
    b = blend_minutes(pd.Series({1: 30.0}), pd.Series({1: 30.0}), _live(10, 30.0, mpg=30, mpg5=30),
                      minutes_k=6, rate_k=20)
    assert b.loc[1, "blended_fppg"] == pytest.approx(30.0)


def test_minutes_blend_believes_a_new_role_quickly():
    # bench player (20 FPPG in 20 min) becomes a starter: 34 min, same efficiency
    b = blend_minutes(pd.Series({1: 20.0}), pd.Series({1: 20.0}), _live(8, 34.0, mpg=34, mpg5=34),
                      minutes_k=6, rate_k=20)
    assert b.loc[1, "minutes_hat"] == pytest.approx((6 * 20 + 8 * 34) / 14)
    assert b.loc[1, "blended_fppg"] == pytest.approx((6 * 20 + 8 * 34) / 14)    # rate unchanged at 1.0


def test_minutes_blend_without_prior_minutes_falls_back():
    b = blend_minutes(pd.Series({1: 40.0}), pd.Series(dtype=float), _live(12, 30.0), minutes_k=6, rate_k=20,
                      k_fallback=12)
    assert b.loc[1, "blended_fppg"] == pytest.approx(35.0)
    b = blend_minutes(pd.Series(dtype=float), pd.Series(dtype=float), _live(3, 30.0), minutes_k=6, rate_k=20)
    assert b.loc[1, "blended_fppg"] == pytest.approx(30.0)


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


def test_in_season_calibration_forms():
    from tradecalc.model.availability import calibrated_rate, in_season_calibration

    lin = {"pre_intercept": 0.735, "pre_slope": 0.047, "in_intercept": 0.524, "in_slope": 0.378, "prior_games": 10}
    assert abs(in_season_calibration(0.8, lin) - (0.524 + 0.378 * 0.8)) < 1e-12
    assert abs(in_season_calibration(0.8, lin, minutes=10) - in_season_calibration(0.8, lin)) < 1e-12  # linear ignores minutes
    lg = dict(lin, in_logit=[1.158, 0.362, 0.827])
    a_lo, a_hi = in_season_calibration(0.5, lg), in_season_calibration(0.95, lg)
    assert 0 < a_lo < a_hi < 1                                                     # monotone in the rate
    assert in_season_calibration(0.9, lg, minutes=34) > in_season_calibration(0.9, lg, minutes=14)  # starters play more
    assert calibrated_rate(0.9, 0, 0, lg) == calibrated_rate(0.9, 0, 0, lin)       # preseason form unchanged


def test_healthy_player_off_the_injury_report_likely_plays_his_next_game():
    av = availability("ACTIVE", 0.8, cap=0.95, dtd_prob=0.5, out_default_games=10, healthy_now=True)
    assert av.for_game(0) == pytest.approx(1 - 0.2 * 0.37)                 # next game: 93%, not 80%
    assert av.for_game(1) == pytest.approx(1 - 0.2 * (1 - 0.63 * 0.65))
    assert abs(av.for_game(30) - 0.8) < 1e-6                                # fades to the season rate
    assert av.for_game(0, rest=0.9) == pytest.approx(1 - 0.28 * 0.37)       # rest days are reported too
    # without a current injury report (backtests) nothing changes
    assert availability("ACTIVE", 0.8, cap=0.95, dtd_prob=0.5, out_default_games=10).for_game(0) == 0.8
