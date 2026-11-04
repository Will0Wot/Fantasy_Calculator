"""The whole app on the demo league (real NBA data, no login), as a first-time visitor would use it.

Slow and needs the network (ESPN and NBA data), so it only runs with TRADECALC_APP_SMOKE=1: in CI on
pull requests and main, or locally before a release."""
import os
from pathlib import Path

import pytest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
pytestmark = pytest.mark.skipif(not os.environ.get("TRADECALC_APP_SMOKE"), reason="set TRADECALC_APP_SMOKE=1")


def test_demo_league_first_visit_then_every_tab():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=1200)
    at.query_params.update({"platform": "demo", "league": "demo"})
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Pick your team" in i.value for i in at.info)              # asked for a team first
    at.selectbox(key="my_team").select(2).run()
    assert not at.exception, [e.value for e in at.exception]
    assert dict(at.query_params) == {"platform": "demo", "league": "demo", "team": "2"}
    assert len(at.tabs) >= 13
    assert any("Demo League" in c.value for c in at.sidebar.caption)


def test_a_bad_league_link_explains_itself():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params.update({"platform": "espn", "league": "not-a-number"})
    at.run()
    assert not at.exception
    assert any("isn't valid" in w.value for w in at.warning)
