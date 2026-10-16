from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from tradecalc.ingest.injuries import Injury
from tradecalc.model.availability import availability, games_played_rate
from tradecalc.model.injury import HistoryEntry, classify, injury_view, load_profiles, long_absences

IC = load_profiles()
TODAY = date(2026, 10, 20)


@pytest.mark.parametrize("type_text,text,expected", [
    ("Right Knee Surgery", "underwent surgery to repair a torn ACL", "acl"),
    ("Right Knee Surgery", "will miss four weeks", "meniscus"),
    ("Achilles Surgery", "", "achilles"),
    ("Left Ankle", "rolled his ankle", "ankle_sprain"),
    ("Back", "", "back_other"),
    ("Foot", "expected back in two weeks", "other"),          # 'back' in prose is not a back injury
    ("", "Haliburton is returning after a year recovering from a torn Achilles", "achilles"),
    ("", "head coach says he will play", "other"),             # 'head' in prose is not a concussion
    ("Right Hamstring Strain", "", "hamstring"),
    ("Right Foot Fracture", "", "foot_fracture"),
])
def test_classify(type_text, text, expected):
    assert classify(type_text, text, IC) == expected


def _player(status="ACTIVE", outlook=""):
    return SimpleNamespace(injury_status=status, outlook=outlook)


def test_current_major_injury_view():
    rep = Injury(1, "X", "OUT", date(2027, 1, 2), "Right Knee Surgery", "torn ACL", "2026-10-02")
    v = injury_view(_player("OUT"), rep, [], IC, TODAY, TODAY)
    p = IC.profiles["acl"]
    assert v.current == "acl" and v.rtp == p.rtp and v.fppg_mult == p.year1["fppg"]
    # Ramp-up window starts on the return date.
    assert (date(2027, 1, 2), date(2027, 2, 1), p.ramp) in v.modifiers


def test_outlook_history_returns_at_season_start():
    v = injury_view(_player(outlook="sat out last season with a torn Achilles"), None, [], IC, TODAY, TODAY)
    p = IC.profiles["achilles"]
    assert [e.profile for e in v.history] == ["achilles"]
    assert v.fppg_mult == p.year1["fppg"] and v.sigma_mult == p.year1["sigma"]


def test_year_two_effects_after_a_year():
    e = HistoryEntry("acl", TODAY - timedelta(days=400), "manual")
    v = injury_view(_player(), None, [e], IC, TODAY, TODAY)
    assert v.fppg_mult == IC.profiles["acl"].year2["fppg"]


def test_long_absence_dropped_when_explained():
    gen = HistoryEntry("long_absence", None, "game logs")
    rep = Injury(1, "X", "OUT", date(2026, 12, 1), "Foot Fracture", "", "")
    assert injury_view(_player("OUT"), rep, [gen], IC, TODAY, TODAY).history == []
    named = HistoryEntry("meniscus", None, "manual")
    assert [e.profile for e in injury_view(_player(), None, [gen, named], IC, TODAY, TODAY).history] == ["meniscus"]
    assert [e.profile for e in injury_view(_player(), None, [gen], IC, TODAY, TODAY).history] == ["long_absence"]


def test_availability_return_window_and_rtp():
    av = availability("OUT", 0.9, cap=0.95, dtd_prob=0.5, out_default_games=10,
                      return_date=date(2026, 11, 19), today=TODAY, window=(0.3, 4, 30), rtp=0.8)
    # 30 days out -> 9-day window after the return date.
    assert av.for_game(0, date(2026, 11, 18)) == 0.0
    assert 0 < av.for_game(0, date(2026, 11, 22)) < 0.9 * 0.8
    assert av.for_game(0, date(2026, 12, 31)) == pytest.approx(0.9 * 0.8)


def test_typical_days_used_without_return_date():
    av = availability("OUT", 0.9, cap=0.95, dtd_prob=0.5, out_default_games=10, today=TODAY, typical_days=42)
    assert av.for_game(0, TODAY + timedelta(days=41)) == 0.0
    assert av.for_game(0, TODAY + timedelta(days=90)) == pytest.approx(0.9)


def test_dtd_until_return_date():
    av = availability("DAY_TO_DAY", 0.9, cap=0.95, dtd_prob=0.5, out_default_games=10,
                      return_date=TODAY + timedelta(days=5), today=TODAY)
    assert av.for_game(3, TODAY + timedelta(days=4)) == 0.5
    assert av.for_game(4, TODAY + timedelta(days=5)) == pytest.approx(0.9)


def _logs(rows):
    return pd.DataFrame(rows, columns=["player_id", "player_name", "team", "game_date", "min"])


def test_long_absences_counts_against_current_team_after_trade():
    # Team A plays days 1-40, team B plays days 1-40. Player plays A days 1-10, traded,
    # plays B days 12-40: no absence. Player 2 misses B's days 11-35 (25 games).
    days = [f"2026-01-{d:02d}" if d <= 31 else f"2026-02-{d - 31:02d}" for d in range(1, 41)]
    rows = [(9, "fill A", "A", d, 30) for d in days] + [(8, "fill B", "B", d, 30) for d in days]
    rows += [(1, "Traded", "A", d, 30) for d in days[:10]] + [(1, "Traded", "B", d, 30) for d in days[11:]]
    rows += [(2, "Hurt", "B", d, 30) for d in days[:10] + days[35:]]
    out = long_absences(_logs(rows), 20)
    assert 1 not in out and out[2][0] == 25


def test_games_played_rate_excludes_modeled_absence():
    s = pd.DataFrame({"player_id": [1] * 50, "min": 30})
    assert games_played_rate([s])[1] == pytest.approx(50 / 82)
    assert games_played_rate([s], excluded={1: 30})[1] == pytest.approx(50 / 52)
