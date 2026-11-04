"""Sleeper adapter on recorded-shape responses (no network): settings, rosters, IR, weeks, results."""
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from tradecalc.config import load_config
from tradecalc.ingest.league import LeagueData, Player
from tradecalc.platforms import LeagueRef, league_config
from tradecalc.platforms.base import LeagueNotFound, UnsupportedLeague
from tradecalc.platforms.sleeper import load_sleeper, settings_from_sleeper

LEAGUE = {
    "league_id": "1180313591236747264", "name": "Hoops", "sport": "nba", "season": "2026", "total_rosters": 4,
    "status": "in_season",
    "settings": {"playoff_week_start": 4},
    "scoring_settings": {"pts": 1, "reb": 1.2, "ast": 1.5, "stl": 3, "blk": 3, "to": -1, "dd": 1.5, "td": 3,
                         "bonus_pt_40p": 2, "tf": -1},
    "roster_positions": ["PG", "SG", "G", "SF", "PF", "F", "C", "UTIL", "UTIL", "BN", "BN", "IR"],
}
USERS = [{"user_id": "u1", "display_name": "ann", "metadata": {"team_name": "Ann's Team"}},
         {"user_id": "u2", "display_name": "bob", "metadata": {}}]
ROSTERS = [{"roster_id": 1, "owner_id": "u1", "players": ["s1", "s2", "s9"], "reserve": ["s2"], "settings": {"waiver_position": 3}},
           {"roster_id": 2, "owner_id": "u2", "players": ["s3"], "settings": {"waiver_position": 1}},
           {"roster_id": 3, "owner_id": None, "players": ["s4"], "settings": {}},
           {"roster_id": 4, "owner_id": None, "players": [], "settings": {}}]
SP = {"s1": {"full_name": "Nikola Jokić", "team": "DEN", "fantasy_positions": ["C"]},
      "s2": {"full_name": "Jimmy Butler III", "team": "GSW", "fantasy_positions": ["SF", "PF"], "injury_status": "Out"},
      "s3": {"full_name": "Stephen Curry", "team": "GSW", "fantasy_positions": ["PG"]},
      "s4": {"full_name": "Kevin Durant", "team": "HOU", "fantasy_positions": ["SF", "PF"]},
      "s9": {"full_name": "Deep Bench Guy", "team": "UTA", "fantasy_positions": ["SG"], "injury_status": "Questionable"}}
OPEN = date(2026, 10, 20)                               # Tuesday


def _matchups(week):
    pts = {1: (1100.0, 1000.0, 900.0, 950.0), 2: (500.0, 400.0, 0.0, 0.0), 3: (0, 0, 0, 0)}[week]
    return [{"roster_id": r, "matchup_id": 1 if r <= 2 else 2, "points": pts[r - 1]} for r in (1, 2, 3, 4)]


class FakeClient:
    def __init__(self, league=LEAGUE):
        self.league = league

    def get(self, path, ttl_s=None, force=False):
        if path.endswith("/users"):
            return USERS
        if path.endswith("/rosters"):
            return ROSTERS
        if "/matchups/" in path:
            return _matchups(int(path.rsplit("/", 1)[1]))
        if path == "/players/nba":
            return SP
        return self.league


def _nba():
    pool = {1: Player(1, "Nikola Jokic", "DEN", ["C"], "ACTIVE"), 2: Player(2, "Jimmy Butler III", "GS", ["SF", "PF"], "OUT"),
            3: Player(3, "Stephen Curry", "GS", ["PG"], "ACTIVE"), 4: Player(4, "Kevin Durant", "HOU", ["PF"], "ACTIVE"),
            5: Player(5, "Free Agent", "BOS", ["C"], "ACTIVE")}
    return LeagueData({}, {}, pool, [], [{"game_id": 1, "date": OPEN, "scoring_period": 1, "home": "DEN", "away": "GS"}],
                      {}, OPEN)


@pytest.fixture
def league(monkeypatch):
    monkeypatch.setattr("tradecalc.clock.league_today", lambda cfg=None: date(2026, 11, 3))   # week 3, Tuesday
    return load_sleeper(LeagueRef("sleeper", LEAGUE["league_id"]), None, load_config(), _nba(), client=FakeClient())


def test_sleeper_settings():
    m = settings_from_sleeper(LEAGUE, 24)
    assert m.scoring == {"PTS": 1, "REB": 1.2, "AST": 1.5, "STL": 3, "BLK": 3, "TOV": -1, "DD": 1.5, "TD": 3}
    assert m.ignored_scoring == {"bonus_pt_40p": 2, "tf": -1} and "bonus_pt_40p (+2)" in m.notes[0]
    assert m.slots == {"PG": 1, "SG": 1, "G": 1, "SF": 1, "PF": 1, "F": 1, "C": 1, "UTIL": 2} and (m.bench, m.ir) == (2, 1)
    assert (m.regular_season_weeks, m.playoffs, m.moves_per_week) == (3, True, None)


def test_sleeper_league_maps_to_espn_players(league):
    assert {t: tm.name for t, tm in league.teams.items()} == {1: "Ann's Team", 2: "bob", 3: "Team 3", 4: "Team 4"}
    assert league.teams[1].roster[:2] == [1, 2] and league.teams[1].waiver_rank == 3
    assert league.players[2].lineup_slot == "IR" and league.players[2].on_team_id == 1
    assert league.players[1].positions == ["C"] and league.players[5].on_team_id == 0
    stand_in = league.teams[1].roster[2]
    assert stand_in < 0 and league.players[stand_in].name == "Deep Bench Guy"
    assert league.players[stand_in].injury_status == "DAY_TO_DAY"
    assert any("aren't in ESPN's player pool" in n for n in league.meta.notes)


def test_sleeper_weeks_and_results(league):
    assert league.meta.weeks[:3] == [(1, OPEN, date(2026, 10, 25)), (2, date(2026, 10, 26), date(2026, 11, 1)),
                                     (3, date(2026, 11, 2), date(2026, 11, 8))]
    w1 = [m for m in league.matchups if m["week"] == 1]
    assert {(m["home"], m["away"], m["winner"]) for m in w1} == {(1, 2, "HOME"), (3, 4, "AWAY")}
    # week 2 is over; week 3 has started: its points so far sit on yesterday (Nov 2 = scoring period 14)
    w3 = next(m for m in league.matchups if m["week"] == 3 and m["home"] == 1)
    assert w3["winner"] == "UNDECIDED" and w3["home_points"] == {}          # 0 points so far: nothing banked


def test_sleeper_league_config(league):
    cfg = league_config(load_config(), league.meta)
    assert cfg["scoring"]["DD"] == 1.5 and cfg["roster"]["slots"]["UTIL"] == 2 and cfg["league"]["my_team_id"] is None


def test_wrong_sport_or_season_is_explained():
    with pytest.raises(UnsupportedLeague, match="football"):
        load_sleeper(LeagueRef("sleeper", "1"), None, load_config(), _nba(), client=FakeClient({**LEAGUE, "sport": "nfl"} | {"sport": "football"}))
    with pytest.raises(UnsupportedLeague, match="2025 season"):
        load_sleeper(LeagueRef("sleeper", "1"), None, load_config(), _nba(), client=FakeClient({**LEAGUE, "season": "2025"}))
    with pytest.raises(LeagueNotFound):
        load_sleeper(LeagueRef("sleeper", "1"), None, load_config(), _nba(), client=SimpleNamespace(get=lambda *a, **k: None))
