import pandas as pd
import pytest

from tradecalc.config import load_config
from tradecalc.model.scoring import add_fp, fantasy_points, live_stats

SCORING = load_config()["scoring"]

# Real 2025-26 box scores (stats.nba.com), with FP hand-calculated under league scoring:
# PTS+REB+2AST+4STL+4BLK-2TOV+2FGM-FGA+FTM-FTA+3PM
BOX_SCORES = [
    # name, date, pts, reb, ast, stl, blk, tov, fgm, fga, ftm, fta, fg3m, expected FP
    ("Victor Wembanyama", "2025-10-22", 40, 15, 1, 1, 3, 0, 15, 21, 9, 11, 1, 81),
    ("Jarrett Allen", "2025-10-22", 4, 4, 1, 2, 0, 1, 1, 4, 2, 4, 0, 12),
    ("Derrick White", "2026-04-09", 8, 3, 1, 1, 0, 0, 2, 10, 4, 4, 0, 11),
    ("Payton Pritchard", "2026-04-10", 21, 3, 10, 1, 0, 1, 7, 14, 2, 2, 5, 51),
    ("Nikola Jokic", "2025-12-25", 56, 16, 15, 0, 2, 5, 15, 21, 22, 23, 4, 112),
    ("Anthony Edwards", "2025-12-04", 11, 6, 4, 1, 3, 8, 3, 6, 3, 3, 2, 27),
]
COLS = ["pts", "reb", "ast", "stl", "blk", "tov", "fgm", "fga", "ftm", "fta", "fg3m"]


@pytest.mark.parametrize("box", BOX_SCORES, ids=[b[0] for b in BOX_SCORES])
def test_fantasy_points_hand_calculated(box):
    row = dict(zip(COLS, box[2:13]))
    assert fantasy_points(row, SCORING) == box[13]


def test_fantasy_points_upper_case_keys_and_missing_stats():
    assert fantasy_points({"PTS": 10, "AST": 3}, SCORING) == 16


def test_vectorized_matches_scalar():
    df = pd.DataFrame([dict(zip(COLS, b[2:13])) for b in BOX_SCORES])
    assert add_fp(df, SCORING)["fp"].tolist() == [b[13] for b in BOX_SCORES]


def test_live_stats():
    rows = []
    for i, m in enumerate([30, 30, 30, 30, 30, 20, 20, 20, 20, 20]):
        rows.append(dict(player_id=1, player_name="A", team="X", game_date=f"2026-11-{i + 1:02d}",
                         min=m, **{c: 0 for c in COLS if c != "pts"}, pts=10 + i))
    rows.append(dict(player_id=1, player_name="A", team="X", game_date="2026-11-20", min=0,
                     **{c: 0 for c in COLS}))  # DNP is ignored
    s = live_stats(pd.DataFrame(rows), SCORING).loc[1]
    assert s.n == 10
    assert s.live_fppg == pytest.approx(14.5)
    assert s.mpg_season == 25 and s.mpg_last5 == 20
    assert s.fppg_recent == pytest.approx(14.5)
