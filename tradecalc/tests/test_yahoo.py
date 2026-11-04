"""Yahoo adapter on documented-shape XML (no network): sign-in state, settings, rosters, scoreboards."""
import xml.etree.ElementTree as ET
from datetime import date

import pytest

from tradecalc.config import load_config
from tradecalc.ingest.league import LeagueData, Player
from tradecalc.platforms import LeagueRef
from tradecalc.platforms.base import UnsupportedLeague
from tradecalc.platforms.yahoo import (load_yahoo, settings_from_yahoo, sign_state, verify_state)

NS = 'xmlns="http://fantasysports.yahooapis.com/fantasy/v2/base.rng"'
OAUTH = {"client_id": "id", "client_secret": "secret", "redirect_uri": "https://app.example/"}


def _settings_xml(scoring_type="headpoint"):
    return f"""<fantasy_content {NS}><league>
      <league_key>466.l.98765</league_key><league_id>98765</league_id><name>Yahoo Hoops</name>
      <num_teams>2</num_teams><scoring_type>{scoring_type}</scoring_type><season>2026</season>
      <start_week>1</start_week><end_week>3</end_week>
      <settings><scoring_type>{scoring_type}</scoring_type><uses_playoff>1</uses_playoff>
        <playoff_start_week>3</playoff_start_week><max_weekly_adds>4</max_weekly_adds>
        <roster_positions>
          <roster_position><position>PG</position><count>1</count></roster_position>
          <roster_position><position>G</position><count>1</count></roster_position>
          <roster_position><position>C</position><count>2</count></roster_position>
          <roster_position><position>Util</position><count>2</count></roster_position>
          <roster_position><position>BN</position><count>3</count></roster_position>
          <roster_position><position>IL</position><count>2</count></roster_position>
        </roster_positions>
        <stat_categories><stats>
          <stat><stat_id>12</stat_id><display_name>PTS</display_name></stat>
          <stat><stat_id>15</stat_id><display_name>REB</display_name></stat>
          <stat><stat_id>16</stat_id><display_name>AST</display_name></stat>
          <stat><stat_id>17</stat_id><display_name>ST</display_name></stat>
          <stat><stat_id>19</stat_id><display_name>TO</display_name></stat>
          <stat><stat_id>23</stat_id><display_name>TECH</display_name></stat>
        </stats></stat_categories>
        <stat_modifiers><stats>
          <stat><stat_id>12</stat_id><value>1</value></stat><stat><stat_id>15</stat_id><value>1.2</value></stat>
          <stat><stat_id>16</stat_id><value>1.5</value></stat><stat><stat_id>17</stat_id><value>3</value></stat>
          <stat><stat_id>19</stat_id><value>-1</value></stat><stat><stat_id>23</stat_id><value>-2</value></stat>
        </stats></stat_modifiers>
      </settings></league></fantasy_content>"""


ROSTERS = f"""<fantasy_content {NS}><league><teams>
  <team><team_key>466.l.98765.t.1</team_key><team_id>1</team_id><name>Splash</name><waiver_priority>2</waiver_priority>
    <roster><players>
      <player><player_key>466.p.1</player_key><name><full>Nikola Jokic</full></name><editorial_team_abbr>Den</editorial_team_abbr>
        <eligible_positions><position>C</position><position>Util</position></eligible_positions>
        <selected_position><position>C</position></selected_position></player>
      <player><player_key>466.p.2</player_key><name><full>Jimmy Butler III</full></name><editorial_team_abbr>GS</editorial_team_abbr>
        <status>INJ</status><eligible_positions><position>SF</position><position>PF</position></eligible_positions>
        <selected_position><position>IL</position></selected_position></player>
    </players></roster></team>
  <team><team_key>466.l.98765.t.2</team_key><team_id>2</team_id><name>Dunks</name><waiver_priority>1</waiver_priority>
    <roster><players>
      <player><player_key>466.p.3</player_key><name><full>Stephen Curry</full></name><editorial_team_abbr>GS</editorial_team_abbr>
        <eligible_positions><position>PG</position></eligible_positions><selected_position><position>BN</position></selected_position></player>
    </players></roster></team>
</teams></league></fantasy_content>"""


def _scoreboard(week):
    ws, we, status, p1, p2 = {1: ("2026-10-20", "2026-10-25", "postevent", 900.5, 1000.0),
                              2: ("2026-10-26", "2026-11-01", "postevent", 1100.0, 1100.0)}[week]
    winner = "<winner_team_key>466.l.98765.t.2</winner_team_key>" if p2 > p1 else ""
    tied = 1 if p1 == p2 else 0
    return f"""<fantasy_content {NS}><league><scoreboard><matchups><matchup>
      <week>{week}</week><week_start>{ws}</week_start><week_end>{we}</week_end><status>{status}</status>
      <is_tied>{tied}</is_tied>{winner}
      <teams><team><team_key>466.l.98765.t.1</team_key><team_id>1</team_id><team_points><total>{p1}</total></team_points></team>
             <team><team_key>466.l.98765.t.2</team_key><team_id>2</team_id><team_points><total>{p2}</total></team_points></team></teams>
    </matchup></matchups></scoreboard></league></fantasy_content>"""


class FakeYahoo:
    def __init__(self, scoring_type="headpoint"):
        self.scoring_type = scoring_type

    def get(self, path):
        if path.endswith("/settings"):
            return ET.fromstring(_settings_xml(self.scoring_type))
        if path.endswith("/teams/roster"):
            return ET.fromstring(ROSTERS)
        return ET.fromstring(_scoreboard(int(path.rsplit("=", 1)[1])))


def _nba():
    pool = {1: Player(1, "Nikola Jokić", "DEN", ["C"], "ACTIVE"), 2: Player(2, "Jimmy Butler III", "GS", ["SF"], "OUT"),
            3: Player(3, "Stephen Curry", "GS", ["PG"], "ACTIVE"), 4: Player(4, "Free Agent", "BOS", ["C"], "ACTIVE")}
    return LeagueData({}, {}, pool, [], [], {}, date(2026, 10, 20))


def test_signed_state_round_trips_and_rejects_tampering():
    s = sign_state(OAUTH, {"league": "98765"})
    assert verify_state(OAUTH, s)["league"] == "98765"
    body, sig = s.rsplit(".", 1)
    assert verify_state(OAUTH, body + "." + "0" * 32) is None
    assert verify_state({**OAUTH, "client_secret": "other"}, s) is None
    assert verify_state(OAUTH, "garbage") is None


def test_yahoo_settings():
    m = settings_from_yahoo(ET.fromstring(_settings_xml()).find("{http://fantasysports.yahooapis.com/fantasy/v2/base.rng}league"))
    assert m.scoring == {"PTS": 1, "REB": 1.2, "AST": 1.5, "STL": 3, "TOV": -1} and m.ignored_scoring == {"TECH": -2}
    assert m.slots == {"PG": 1, "G": 1, "C": 2, "UTIL": 2} and (m.bench, m.ir) == (3, 2)
    assert (m.regular_season_weeks, m.moves_per_week, m.playoffs, m.num_teams) == (2, 4, True, 2)


def test_yahoo_league_loads(monkeypatch):
    monkeypatch.setattr("tradecalc.clock.league_today", lambda cfg=None: date(2026, 11, 3))
    lg = load_yahoo(LeagueRef("yahoo", "98765"), None, load_config(), _nba(), None, client=FakeYahoo())
    assert {t: (tm.name, tm.roster, tm.waiver_rank) for t, tm in lg.teams.items()} == {1: ("Splash", [1, 2], 2),
                                                                                        2: ("Dunks", [3], 1)}
    assert lg.players[2].lineup_slot == "IR" and lg.players[1].positions == ["C"] and lg.players[4].on_team_id == 0
    assert [(m["week"], m["winner"]) for m in lg.matchups] == [(1, "AWAY"), (2, "TIE")]
    assert lg.meta.weeks == [(1, date(2026, 10, 20), date(2026, 10, 25)), (2, date(2026, 10, 26), date(2026, 11, 1))]


def test_yahoo_category_league_is_explained():
    with pytest.raises(UnsupportedLeague, match="categories league"):
        load_yahoo(LeagueRef("yahoo", "98765"), None, load_config(), _nba(), None, client=FakeYahoo("head"))
