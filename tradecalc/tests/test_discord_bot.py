"""Discord bot data and weekly post: consistent odds, Discord's size limits, once-a-week posting."""
import json
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from tradecalc import discord_bot as DB
from tradecalc.tests.conftest import make_player, make_state
from tradecalc.trade.standings import result_of

ET = ZoneInfo("America/New_York")


def fixture_snapshot() -> dict:
    """Four teams, three weeks; week 1 is over (one result still UNDECIDED at ESPN), week 2 starts today.
    Also writes the Worker's test fixture: see discord-bot/test/."""
    import pandas as pd

    start = date(2026, 10, 26)
    days = [start + timedelta(i) for i in range(21)]
    pros = ["AAA", "BBB", "CCC", "DDD"]
    games = {d: {pros[i % 4], pros[(i + 1) % 4]} for i, d in enumerate(days)}
    ps = [make_player(10 * t + k, t, pros[t - 1], [pos], 30 - 3 * t + k)
          for t in range(1, 5) for k, pos in enumerate(["PG", "SG", "SF", "PF", "C"])]
    opponents = {1: {1: 2, 2: 2, 3: 3}, 2: {1: 1, 2: 1, 3: 4}, 3: {1: 4, 2: 4, 3: 1}, 4: {1: 3, 2: 3, 3: 2}}
    st = make_state(ps, games, n_weeks=3, opponents=opponents)
    st.today = date(2026, 11, 2)
    for t, team in st.league.teams.items():
        team.name = {1: "Inevitable", 2: "Not_Eating*Poop", 3: "Ball Knowers", 4: "Ballers Club"}[t]
    st.league.settings = {"name": "Test League"}
    st.league.matchups = [
        {"week": 1, "home": 1, "away": 2, "home_total": 1100.0, "away_total": 1000.0, "winner": "HOME"},
        {"week": 1, "home": 3, "away": 4, "home_total": 900.0, "away_total": 950.0, "winner": "UNDECIDED"},
    ]
    pred = pd.DataFrame({"kind": "pre-week", "week": 1, "team_id": [1, 2, 3, 4], "win_prob": [0.3, 0.7, 0.6, 0.4]})
    return DB.snapshot(st, n_forecast=2000, now=datetime(2026, 11, 2, 8, 17, tzinfo=ET), predictions=pred)


@pytest.fixture
def snap():
    return fixture_snapshot()


def test_odds_are_consistent_both_ways(snap):
    n = len(snap["team_ids"])
    for wk, m in snap["odds"].items():
        for i in range(n):
            assert m[i][i] is None
            for j in range(n):
                if i != j:
                    assert 0 <= m[i][j] <= 1 and abs(m[i][j] + m[j][i] - 1) < 2e-4
    # the stronger team (higher projection) is favored
    i1, i4 = snap["team_ids"].index(1), snap["team_ids"].index(4)
    assert snap["proj"]["2"]["1"]["mean"] > snap["proj"]["2"]["4"]["mean"] and snap["odds"]["2"][i1][i4] > 0.5


def test_snapshot_is_json_and_has_every_piece_the_bot_reads(snap):
    s = json.loads(json.dumps(snap))
    assert s["current_week"] == 2 and s["weeks"][0]["label"] == "Week 1 (Oct 26–Nov 1)"
    assert [w["status"] for w in s["weeks"]] == ["final", "in_progress", "upcoming"]
    assert sorted(map(sorted, s["schedule"]["2"])) == [[1, 2], [3, 4]]
    assert [r["rank"] for r in s["forecast"]] == [1, 2, 3, 4]
    assert s["as_of_label"] == "Mon Nov 2, 8:17 AM EST"
    assert set(s["text"]["matchups"]) == {"2", "3"} and set(s["text"]["results"]) == {"1"}
    # ESPN's UNDECIDED settled by the scores, and marked as the upset it was
    assert "😮 **Ballers Club** 950 – 900 Ball Knowers (upset: had 40% to win)" in s["text"]["results"]["1"]


def test_tables_fit_a_phone_and_names_cannot_break_formatting(snap):
    for key in ("standings", "forecast"):
        table = snap["text"][key].split("```")[1]
        assert all(len(line) <= 40 for line in table.strip("\n").split("\n"))
        assert "`" not in table                                # backticks would end the code block
    assert "Not\\_Eating\\*Poop" in snap["text"]["matchups"]["2"]  # markdown escaped outside code blocks


def test_standings_show_the_record_each_team_was_supposed_to_have(snap):
    rows = {r["team_id"]: r for r in snap["standings"]}
    assert [r["team_id"] for r in snap["standings"]] == [1, 4, 2, 3]      # wins, then points
    # team 1 was a 30% underdog and won: supposed to be 0-1, actually 1-0
    assert (rows[1]["record"], rows[1]["exp_record"], rows[1]["vs_expected"]) == ("1-0", "0-1", 1.0)
    assert (rows[2]["record"], rows[2]["exp_record"], rows[2]["vs_expected"]) == ("0-1", "1-0", -1.0)
    assert rows[2]["surprises"] == [{"week": 1, "result": "lost", "p": 0.7, "opp": 1}]
    assert rows[1]["exp_wins"] == 0.3 and rows[3]["allplay_wins"] == 0.0 and abs(rows[4]["allplay_wins"] - 1 / 3) < 0.01
    text = snap["text"]["standings"]
    assert " 1. Inevitable       1-0   0-1  +1  1.1k" in text
    assert "**Week 1: supposed to win, didn't**" in text
    assert "😬 **Not\\_Eating\\*Poop** (70% to win) lost to **Inevitable**" in text


def test_weekly_post_respects_discord_limits_and_never_pings(snap):
    p = DB.weekly_post(snap)
    assert p["allowed_mentions"] == {"parse": []}
    assert len(p["content"]) <= 2000 and len(p["embeds"]) <= 10
    assert all(len(e["description"]) <= 4096 for e in p["embeds"])
    assert sum(len(e["title"]) + len(e["description"]) + len(e.get("footer", {}).get("text", "")) for e in p["embeds"]) <= 6000
    assert [e["title"] for e in p["embeds"]] == ["Week 1 (Oct 26–Nov 1): results",
                                                "Standings after week 1: actual vs expected record",
                                                "Projected final standings before week 2", "Week 2 (Nov 2–8): matchups"]
    assert "will be eating poop" in p["embeds"][2]["description"]


def test_posts_once_per_week_at_its_start(snap, tmp_path):
    posted = tmp_path / "posted.json"
    assert DB.post_due(snap, posted)                          # first day of week 2
    DB.mark_posted(snap, posted)
    assert not DB.post_due(snap, posted)                      # already posted
    mid = dict(snap, as_of="2026-11-05T08:17-05:00")
    assert not DB.post_due(mid, tmp_path / "none.json")       # mid-week: wait for next week
    opening = dict(snap, current_week=1)
    assert DB.post_due(dict(opening, as_of="2026-10-25T08:17-04:00"), tmp_path / "none.json")   # day before opening
    assert not DB.post_due(dict(opening, as_of="2026-10-24T08:17-04:00"), tmp_path / "none.json")


def test_finished_week_without_espn_winner_is_decided_by_points():
    assert result_of({"winner": "AWAY", "home_total": 900, "away_total": 800}) == "AWAY"   # ESPN's call stands
    assert result_of({"winner": "UNDECIDED", "home_total": 1102, "away_total": 1072}) == "HOME"
    assert result_of({"winner": "UNDECIDED", "home_total": 1000, "away_total": 1000}) == "TIE"
    assert result_of({"winner": "UNDECIDED", "home_total": 0, "away_total": 0}) is None   # no scores yet


def test_percentages_read_well():
    assert [DB.pct(x) for x in (0.001, 0.4, 0.996, None)] == ["<1%", "40%", ">99%", "–"]
