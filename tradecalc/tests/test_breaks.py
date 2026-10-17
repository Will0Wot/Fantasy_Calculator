"""The audit's adversarial tests (tradecalc/audit/breaks.py) must all hold."""
import pytest

from tradecalc.audit import breaks


@pytest.mark.parametrize("res", breaks.run_all(), ids=lambda r: r["name"])
def test_break(res):
    assert res["passed"], f"{res['name']}: {res['detail']}"
