"""Platform layer: league links, ESPN settings, per-league config, other leagues' scoring."""
from datetime import date

import pandas as pd
import pytest

from tradecalc.config import load_config
from tradecalc.model.scoring import add_fp, bonus_rates
from tradecalc.platforms import Credentials, LeagueRef, league_config, parse_league_input
from tradecalc.platforms.base import merge_scoring
from tradecalc.platforms.espn import settings_from_espn


@pytest.mark.parametrize("text,platform,expected", [
    ("https://fantasy.espn.com/basketball/league?leagueId=1331882803", None, LeagueRef("espn", "1331882803")),
    ("fantasy.espn.com/basketball/team?leagueId=42&teamId=3&seasonId=2027", None, LeagueRef("espn", "42", 2027)),
    ("https://sleeper.com/leagues/1180313591236747264/team", None, LeagueRef("sleeper", "1180313591236747264")),
    ("https://sleeper.app/leagues/1180313591236747264", None, LeagueRef("sleeper", "1180313591236747264")),
    ("https://basketball.fantasysports.yahoo.com/nba/98765/4", None, LeagueRef("yahoo", "98765")),
    (" 1331882803 ", "espn", LeagueRef("espn", "1331882803")),
    ("466.l.98765", "yahoo", LeagueRef("yahoo", "98765")),
])
def test_league_links_and_ids(text, platform, expected):
    assert parse_league_input(text, platform) == expected


@pytest.mark.parametrize("text,platform,msg", [
    ("", "espn", "Enter a league ID"),
    ("1331882803", None, "Pick which site"),
    ("https://fantasy.espn.com/football/league?leagueId=1", None, "not an ESPN fantasy basketball"),
    ("https://fantasy.espn.com/basketball/league", None, "no leagueId"),
    ("https://example.com/league/1", None, "Paste a league link"),
    ("abc", "sleeper", "is a number"),
])
def test_league_input_errors_say_what_to_do(text, platform, msg):
    with pytest.raises(ValueError, match=msg):
        parse_league_input(text, platform)


def test_espn_swid_gets_its_braces():
    assert Credentials("s2", "ABC-123").espn_cookies() == {"espn_s2": "s2", "SWID": "{ABC-123}"}
    assert Credentials("s2", "").espn_cookies() is None


def _espn_raw(**over):
    s = {"name": "Other League", "size": 10,
         "scoringSettings": {"scoringType": "H2H_POINTS", "matchupTieRule": "NONE",
                             "scoringItems": [{"statId": 0, "points": 1}, {"statId": 6, "points": 1.2},
                                              {"statId": 3, "points": 1.5}, {"statId": 23, "points": -0.5},
                                              {"statId": 37, "points": 2}, {"statId": 10, "points": -2}]},
         "rosterSettings": {"lineupSlotCounts": {"0": 1, "1": 1, "4": 2, "8": 1, "11": 2, "12": 4, "13": 2},
                            "lineupLocktimeType": "INDIVIDUAL_GAME"},
         "scheduleSettings": {"matchupPeriodCount": 20, "matchupPeriodLength": 1, "playoffTeamCount": 4},
         "acquisitionSettings": {"matchupAcquisitionLimit": -1}}
    for k, v in over.items():
        s[k] = {**s[k], **v} if isinstance(v, dict) else v
    return {"settings": s, "status": {}}


def test_espn_settings_translate_to_box_score_scoring():
    m = settings_from_espn(_espn_raw(), 42, 2027)
    # FG missed -0.5 = FGA -0.5, FGM +0.5; technical fouls aren't modeled and are listed
    assert m.scoring == {"PTS": 1, "REB": 1.2, "AST": 1.5, "FGA": -0.5, "FGM": 0.5, "DD": 2}
    assert m.ignored_scoring == {"technical fouls": -2} and any("technical fouls" in n for n in m.notes)
    assert m.slots == {"PG": 1, "SG": 1, "C": 2, "G/F": 1, "UTIL": 2} and (m.bench, m.ir) == (4, 2)
    assert m.slot_eligibility["G/F"] == ["PG", "SG", "SF", "PF"]
    assert (m.regular_season_weeks, m.moves_per_week, m.playoffs, m.scoring_type) == (20, None, True, "points")


def test_espn_category_league_is_recognized():
    m = settings_from_espn(_espn_raw(scoringSettings={"scoringType": "H2H_MOST_CATEGORIES"}), 42, 2027)
    assert m.scoring_type == "categories" and any("only points leagues" in n for n in m.notes)


def test_league_config_takes_the_leagues_own_settings():
    base = load_config()
    m = settings_from_espn(_espn_raw(), 42, 2027)
    cfg = league_config(base, m)
    assert cfg["scoring"] == m.scoring and cfg["roster"]["slots"] == m.slots and cfg["roster"]["bench"] == 4
    assert cfg["league"]["league_id"] == "42" and cfg["league"]["my_team_id"] is None
    assert cfg["league"]["season"] == base["league"]["season"]          # NBA data season unchanged
    assert cfg["streaming"]["moves_per_week"] == 4 and cfg["streaming"]["moves_unlimited"]
    assert base["scoring"] != m.scoring                                  # the shared config is untouched
    home = settings_from_espn(_espn_raw(), base["league"]["league_id"], 2027)
    assert league_config(base, home)["league"]["my_team_id"] == base["league"]["my_team_id"]


def test_merge_scoring_combines_missed_shots():
    assert merge_scoring([("FGA", -1), ({"FGA": 1, "FGM": -1}, -1), ("FGM", 2)]) == {"FGA": -2, "FGM": 3}


def _games(rows):
    cols = ["player_name", "min", "pts", "reb", "ast", "stl", "blk"]
    return pd.DataFrame([dict(zip(cols, r)) for r in rows])


def test_double_and_triple_double_bonuses_come_from_each_box_score():
    g = _games([("A", 30, 12, 11, 3, 1, 0), ("A", 30, 20, 10, 10, 0, 0), ("A", 30, 30, 5, 5, 0, 0)])
    fp = add_fp(g, {"PTS": 1, "DD": 2, "TD": 5})["fp"].tolist()
    assert fp == [12 + 2, 20 + 2 + 5, 30]                     # a triple-double is also a double-double
    rate = bonus_rates(g, {"PTS": 1, "DD": 2, "TD": 5}, str.lower, min_games=1)
    assert rate["a"] == pytest.approx((2 + 7 + 0) / 3)
    assert bonus_rates(g, {"PTS": 1}, str.lower).empty      # no bonus in the scoring: nothing to add


def test_projections_rescale_to_another_leagues_scoring(tmp_path, monkeypatch):
    from tradecalc.sim.possession import forecast

    monkeypatch.setattr(forecast, "path", lambda season: tmp_path / f"projections_{season}.parquet")
    home = {"PTS": 1, "REB": 1, "AST": 2}
    df = pd.DataFrame({"key": ["a", "a"], "date": ["2026-11-04", "2026-11-05"], "fp": [40.0, 44.0],
                       "poss": [500, 500], "pts": [20.0, 22.0], "reb": [10.0, 10.0], "ast": [5.0, 6.0],
                       "dd": [0.4, 0.4]})
    forecast.save(df, "2026-27", date(2026, 11, 3), scoring=home)
    same = forecast.rescale(forecast.load("2026-27"), "2026-27", home)
    assert same["fp"].tolist() == [40.0, 44.0]
    other = forecast.rescale(forecast.load("2026-27"), "2026-27", {"PTS": 1, "REB": 1.2, "AST": 1.5, "DD": 5})
    mix = (21 + 1.2 * 10 + 1.5 * 5.5 + 5 * 0.4) / (21 + 10 + 2 * 5.5)
    assert other["fp"].tolist() == pytest.approx([40 * mix, 44 * mix], rel=1e-3)
