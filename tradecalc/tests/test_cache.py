"""SQLite cache: a second refresh makes no API calls for games already cached."""
import pandas as pd
import pytest

from tradecalc.db import connect
from tradecalc.ingest import nba


def _fake_log(dates):
    return pd.DataFrame([{
        "GAME_ID": f"g{i}", "PLAYER_ID": 1, "PLAYER_NAME": "A", "TEAM_ABBREVIATION": "X",
        "GAME_DATE": d, "MATCHUP": "X vs. Y", **{c.upper(): 1 for c in nba.GAME_LOG_COLS},
    } for i, d in enumerate(dates)])


@pytest.fixture
def conn(tmp_path):
    return connect(tmp_path / "c.sqlite")


def test_completed_season_never_refetched(conn, monkeypatch):
    calls = []
    monkeypatch.setattr(nba, "_fetch_league_log", lambda s, d, i, **kw: calls.append(d) or _fake_log(["2026-01-01"]))
    nba.refresh_season(conn, "2025-26", complete=True, min_interval=0)
    r = nba.refresh_season(conn, "2025-26", complete=True, min_interval=0, force=True)
    assert len(calls) == 1 and r["fetched"] == 0


def test_current_season_only_requests_new_dates(conn, monkeypatch):
    calls = []

    def fetch(season, date_from, interval, **kw):
        calls.append(date_from)
        return _fake_log(["2026-11-01", "2026-11-02"]) if date_from is None else _fake_log([])

    monkeypatch.setattr(nba, "_fetch_league_log", fetch)
    nba.refresh_season(conn, "2026-27", complete=False, min_interval=0)
    # Immediate second run: no request at all.
    assert nba.refresh_season(conn, "2026-27", complete=False, min_interval=0)["skipped"]
    # Forced later run: requests only from the last cached date onward.
    nba.refresh_season(conn, "2026-27", complete=False, min_interval=0, force=True)
    assert [str(c) if c else None for c in calls] == [None, "2026-11-02"]
    assert len(nba.game_logs(conn, "2026-27")) == 2


def test_current_season_falls_back_to_espn_when_stats_nba_blocked(conn, monkeypatch):
    import tradecalc.ingest.espn_box as eb

    def blocked(*a, **kw):
        raise TimeoutError("blocked")

    used = []
    monkeypatch.setattr(nba, "_fetch_league_log", blocked)
    monkeypatch.setattr(nba, "NBA_BLOCKED", False)
    monkeypatch.setattr(eb, "refresh_from_espn", lambda c, s, d: used.append((s, d)) or {"season": s, "fetched": 0})
    r = nba.refresh_season(conn, "2026-27", complete=False, min_interval=0, force=True)
    assert used and "ESPN" in r["note"] and nba.NBA_BLOCKED


def test_espn_game_rows_parse_box_score():
    from datetime import date

    from tradecalc.ingest.espn_box import game_rows

    summ = {"header": {"id": "1", "competitions": [{"competitors": [
        {"team": {"abbreviation": "GS"}, "homeAway": "home"}, {"team": {"abbreviation": "NY"}, "homeAway": "away"}]}]},
        "boxscore": {"players": [{"team": {"abbreviation": "GS"}, "statistics": [{
            "labels": ["MIN", "PTS", "FG", "3PT", "FT", "REB", "AST", "TO", "STL", "BLK", "OREB", "DREB", "PF", "+/-"],
            "athletes": [{"athlete": {"displayName": "Stephen Curry", "id": "3975"},
                          "stats": ["34", "31", "10-20", "6-12", "5-5", "4", "6", "3", "2", "0", "0", "4", "2", "+9"]},
                         {"athlete": {"displayName": "Bench Guy", "id": "9"}, "didNotPlay": True, "stats": []}]}]}]}}
    rows = game_rows(summ, date(2026, 11, 2), {"stephen curry": 201939})
    assert len(rows) == 1
    r = rows[0]
    assert (r["player_id"], r["team"], r["matchup"], r["fgm"], r["fga"], r["fg3m"], r["ftm"], r["pts"]) == \
        (201939, "GSW", "GSW vs. NYK", 10, 20, 6, 5, 31)
