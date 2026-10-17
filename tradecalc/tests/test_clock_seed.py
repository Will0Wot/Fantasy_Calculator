from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from tradecalc import clock
from tradecalc.db import connect


def test_league_today_uses_eastern_not_server_clock(monkeypatch):
    # 01:30 UTC on Nov 3 is still Nov 2 in New York: games that night belong to Nov 2
    fixed = datetime(2026, 11, 3, 1, 30, tzinfo=timezone.utc)

    class FakeDT(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed.astimezone(tz) if tz else fixed

    monkeypatch.setattr(clock, "datetime", FakeDT)
    assert clock.league_today().isoformat() == "2026-11-02"
    assert clock.league_today({"league": {"timezone": "UTC"}}).isoformat() == "2026-11-03"


def test_seed_roundtrip(tmp_path, monkeypatch):
    import tradecalc.seed as seed

    monkeypatch.setattr(seed, "SEED_DIR", tmp_path / "seed")
    src = connect(tmp_path / "a.sqlite")
    row = ("2024-25", "0022400001", 1, "A Player", "BOS", "2024-10-22", "BOS vs. NYK") + (1.0,) * 16
    src.execute(f"INSERT INTO game_logs({','.join(seed.LOG_COLS)}) VALUES ({','.join('?' * len(seed.LOG_COLS))})", row)
    src.execute("INSERT INTO http_cache VALUES (?,?,?)", ("nba:teamlog:2024-25", 0, '[{"x": 1}]'))
    src.execute("INSERT INTO season_state VALUES ('2024-25', 1, 0)")
    src.commit()
    seed.export_seed(src, ["2024-25"])
    dst = connect(tmp_path / "b.sqlite")
    assert seed.ensure_seed(dst) == ["2024-25"]
    got = pd.read_sql_query("SELECT * FROM game_logs", dst)
    assert len(got) == 1 and got.game_id.iat[0] == "0022400001" and got.pts.iat[0] == 1.0
    assert dst.execute("SELECT complete FROM season_state WHERE season='2024-25'").fetchone()[0] == 1
    assert seed.ensure_seed(dst) == []      # already loaded: nothing to do
