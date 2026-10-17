from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from tradecalc import tracking


class Week(SimpleNamespace):
    def days(self):
        return [self.start + timedelta(i) for i in range((self.end - self.start).days + 1)]


WEEKS = [Week(number=1, start=date(2026, 10, 20), end=date(2026, 10, 25)),
         Week(number=2, start=date(2026, 10, 26), end=date(2026, 11, 1))]


def fake_state(today, matchups=()):
    return SimpleNamespace(real_today=today, today=today, cal=SimpleNamespace(weeks=WEEKS),
                           cfg={"league": {"tie_value": 0.5}}, league=SimpleNamespace(matchups=list(matchups)))


def snap(week, kind, as_of, p):
    w = WEEKS[week - 1]
    m = pd.DataFrame([{"week": week, "week_start": w.start.isoformat(), "week_end": w.end.isoformat(), "kind": kind,
                       "as_of": as_of, "team_id": t, "team": f"T{t}", "opp_id": 3 - t, "opp": f"T{3 - t}",
                       "proj_points": 1000.0, "proj_sd": 150.0, "banked_points": 0.0, "win_prob": p if t == 1 else 1 - p,
                       "model": "x"} for t in (1, 2)])
    pl = pd.DataFrame([{"week": week, "week_start": w.start.isoformat(), "week_end": w.end.isoformat(), "as_of": as_of,
                        "team_id": 1, "team": "T1", "espn_id": 7, "player": "Nikola Jokic", "pro_team": "DEN",
                        "status": "ACTIVE", "team_games": 3, "proj_fppg": 60.0, "proj_games": 2.8,
                        "proj_points": 168.0, "proj_points_counted": 168.0, "model": "x"}]) if kind == "pre-week" else pd.DataFrame()
    st = pd.DataFrame([{"before_week": week, "as_of": as_of, "team_id": t, "team": f"T{t}", "exp_wins": 12.0}
                       for t in (1, 2)]) if kind == "pre-week" else pd.DataFrame()
    return {"matchups": m, "players": pl, "standings": st}


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "ROOT", tmp_path)
    return tmp_path


def test_pre_week_prediction_is_replaced_until_the_week_starts_then_frozen(store, monkeypatch):
    def run(today, snaps):
        monkeypatch.setattr(tracking, "snapshot", lambda state: {
            k: pd.concat([s[k] for s in snaps], ignore_index=True) for k in ("matchups", "players", "standings")})
        tracking.save("S", tracking.record(fake_state(today), "S", log=lambda m: None))
        return tracking.load("S")

    run(date(2026, 10, 17), [snap(1, "pre-week", "2026-10-17", 0.40)])
    t = run(date(2026, 10, 19), [snap(1, "pre-week", "2026-10-19", 0.45)])        # newer pre-week replaces
    pre1 = t["matchups"][(t["matchups"].week == 1) & (t["matchups"].kind == "pre-week")]
    assert len(pre1) == 2 and set(pre1.as_of) == {"2026-10-19"} and pre1.win_prob.iloc[0] == 0.45
    # week 1 under way: in-week rows for week 1 (one per day), pre-week rows for week 2; week 1 pre-week frozen
    run(date(2026, 10, 22), [snap(1, "in-week", "2026-10-22", 0.60), snap(2, "pre-week", "2026-10-22", 0.50)])
    t = run(date(2026, 10, 22), [snap(1, "in-week", "2026-10-22", 0.65), snap(2, "pre-week", "2026-10-22", 0.55)])
    m = t["matchups"]
    assert m[(m.week == 1) & (m.kind == "pre-week")].win_prob.iloc[0] == 0.45        # untouched
    assert len(m[(m.week == 1) & (m.kind == "in-week")]) == 2                          # same-day rerun replaced
    t = run(date(2026, 10, 23), [snap(1, "in-week", "2026-10-23", 0.70)])
    assert len(t["matchups"][(t["matchups"].week == 1) & (t["matchups"].kind == "in-week")]) == 4  # one per day
    assert len(t["players"]) == 2 and len(t["standings"]) == 4


def test_settle_fills_results_for_finished_weeks_only():
    tables = {k: pd.concat([snap(1, "pre-week", "2026-10-19", 0.6)[k], snap(2, "pre-week", "2026-10-25", 0.5)[k]],
                           ignore_index=True) for k in ("matchups", "players", "standings")}
    results = [{"week": 1, "home": 1, "away": 2, "winner": "HOME", "home_total": 1100.0, "away_total": 950.0},
               {"week": 2, "home": 1, "away": 2, "winner": "UNDECIDED", "home_total": 300.0, "away_total": 200.0}]
    logs = pd.DataFrame({"player_name": ["Nikola Jokić"] * 3, "game_date": ["2026-10-21", "2026-10-23", "2026-10-27"],
                         "min": [35, 0, 33], "fp": [70.0, 0.0, 55.0]})
    out = tracking.settle(fake_state(date(2026, 10, 27), results), tables, logs, log=lambda m: None)
    m = out["matchups"].set_index(["week", "team_id"])
    assert m.loc[(1, 1), "actual_points"] == 1100 and m.loc[(1, 1), "won"] == 1 and m.loc[(1, 2), "won"] == 0
    assert pd.isna(m.loc[(2, 1), "actual_points"])                                   # week 2 not finished
    p = out["players"].set_index("week")
    assert p.loc[1, "games_played"] == 1 and p.loc[1, "actual_points"] == 70 and pd.isna(p.loc[2, "actual_points"])
    sm = tracking.summary(out)
    assert sm.set_index("measure").loc["predicted favourite won", "value"] == 1.0


def test_excel_has_a_sheet_per_table():
    import openpyxl
    from io import BytesIO

    tables = snap(1, "pre-week", "2026-10-19", 0.6)
    wb = openpyxl.load_workbook(BytesIO(tracking.excel(tables)))
    assert wb.sheetnames == ["Summary", "Matchups", "Players", "Standings"]
