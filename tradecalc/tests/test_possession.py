"""Possession simulator: bookkeeping, lineups follow minutes, defenders matter, zones parse."""
import numpy as np
import pytest

from tradecalc.ingest.pbp import zone_of
from tradecalc.sim.possession.engine import DEFAULT_CAL, SI, PossessionModel, TeamGame, _Side, simulate_batch


def team(n=2000, minutes=(36, 34, 33, 32, 30, 24, 20, 18, 13), rim=0.5, eff_ra=0.0, seed=0):
    p = len(minutes)
    zp = np.tile([0.3, 0.2, 0.1, 0.1, 0.3], (p, 1))
    fgl = np.log(np.array([0.66, 0.44, 0.42, 0.39, 0.36]) / (1 - np.array([0.66, 0.44, 0.42, 0.39, 0.36])))
    eff = np.zeros((p, 5))
    eff[:, 0] = eff_ra
    M = np.tile(np.array(minutes, dtype=float), (n, 1))
    return TeamGame([f"p{i}" for i in range(p)], M, np.linspace(0.32, 0.12, p), np.full(p, 0.1), zp,
                    np.tile(fgl, (p, 1)), np.tile([0.6, 0.5, 0.45, 0.95, 0.8], (p, 1)), np.full(p, 0.05),
                    np.full(p, 0.78), np.full(p, 0.03), np.full(p, 0.05), np.full(p, 0.015), np.full(p, 0.012),
                    np.full(p, 0.05), np.full(p, 0.14), eff, np.full(p, rim), np.array(["G"] * 4 + ["F"] * 3 + ["C"] * 2),
                    np.ones((p, p)) - np.eye(p), np.full(p, 0.25), 1.0, np.ones(5))


def model():
    m = PossessionModel.__new__(PossessionModel)
    m.cal = dict(DEFAULT_CAL)
    m.blk_ref = 0.06
    m.affinity = lambda off, de: np.ones((len(off.keys), len(de.keys)))
    return m


def test_box_scores_add_up():
    rng = np.random.default_rng(1)
    (h, a), = simulate_batch(model(), [(team(), team(), 98.0, None, None)], 2000, rng)
    for s in (h, a):
        assert (s[..., SI["fgm"]] <= s[..., SI["fga"]]).all() and (s[..., SI["fg3m"]] <= s[..., SI["fg3a"]]).all()
        assert (s[..., SI["ftm"]] <= s[..., SI["fta"]]).all()
        pts = 2 * (s[..., SI["fgm"]] - s[..., SI["fg3m"]]) + 3 * s[..., SI["fg3m"]] + s[..., SI["ftm"]]
        assert np.array_equal(pts, s[..., SI["pts"]])
        assert (s[..., SI["ast"]].sum(axis=1) <= s[..., SI["fgm"]].sum(axis=1)).all()
    tot = h[..., SI["pts"]].sum(axis=1)
    assert 95 < tot.mean() < 135                      # a realistic NBA score
    assert (h[..., SI["pts"]].sum(axis=1) != a[..., SI["pts"]].sum(axis=1)).all()   # overtime breaks ties


def test_lineups_follow_minutes():
    tg = team(n=4000)
    side = _Side([tg], [None], [np.ones((9, 9))], 4000, 9, np.random.default_rng(2), 1.5)
    on = np.mean([side.new_stint(np.random.default_rng(i).random(4000)) or side.on.mean(axis=0)
                  for i in range(30)], axis=0)
    expected = tg.M[0] / 48 * 5 / (tg.M[0] / 48).sum()
    assert np.allclose(on, expected, atol=0.02)
    assert np.allclose(side.on.sum(axis=1), 5)        # always exactly five on the floor


def test_rim_protector_lowers_opponents_make_rate_at_the_rim():
    def rim_fg(eff_ra):
        rng = np.random.default_rng(3)
        (h, _), = simulate_batch(model(), [(team(n=3000), team(n=3000, eff_ra=eff_ra), 98.0, None, None)], 3000, rng)
        return h[..., SI["fgm"]].sum() / h[..., SI["fga"]].sum()
    assert rim_fg(-0.5) < rim_fg(0.0) - 0.02


@pytest.mark.parametrize("three,dist,x,y,zone", [
    (False, 1.0, 25, 1, "RA"), (False, 9.0, 27, 8, "PAINT"), (False, 16.0, 12, 10, "MID"),
    (True, 22.0, 3, 2, "C3"), (True, 25.0, 25, 25, "AB3"), (True, 26.0, None, None, "AB3")])
def test_zones(three, dist, x, y, zone):
    assert zone_of(three, dist, x, y) == zone
