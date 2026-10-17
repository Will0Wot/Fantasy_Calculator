from tradecalc import help_text as H
from tradecalc.config import load_config


def test_help_text_renders_with_and_without_validation():
    cfg = load_config()
    val = {"run_on": "2026-10-16", "n_player_games": 100,
           "points": {"baseline": {"MAE": 10.0, "RMSE": 13.0}, "engine": {"MAE": 9.7, "RMSE": 12.3},
                      "engine_rates_x_actual_min": {"MAE": 7.7}},
           "coverage": {"game_in80": 0.79, "game_in50": 0.5, "week_in80": 0.8, "week_in50": 0.51, "game_bias": 0.0},
           "matchups": {"n": 10, "brier": 0.2, "buckets": [{"range": "(0.4, 0.6]", "n": 10, "predicted": 0.5, "actual": 0.5}]}}
    for md in (H.matchup(cfg), H.simulate({"form_sd": 0.18}), H.simulate(None), H.trade_builder(cfg),
               H.trade_finder(cfg), H.players(cfg), H.injuries(cfg), H.help_overview(cfg), H.help_trades(cfg),
               H.help_accuracy(cfg, val), H.help_accuracy(cfg, None), H.help_glossary()):
        assert isinstance(md, str) and len(md) > 100
    assert "9.70" in H.help_accuracy(cfg, val) and "sim-validate" in H.help_accuracy(cfg, None)
    assert f"{cfg['trade']['verdict_accept']:.2f}" in H.trade_builder(cfg)


def test_methodology_math_renders_in_streamlit():
    """Streamlit's markdown only treats $$ as a display block when the delimiters are on their own
    lines; `$$a = ...` spanning lines breaks the rest of the page (it did, from section 1.4 on)."""
    import re
    from pathlib import Path

    text = (Path(__file__).resolve().parents[2] / "docs" / "methodology.md").read_text()
    for line in text.splitlines():
        if "$$" in line:
            assert line.strip() == "$$", f"display math delimiter not on its own line: {line[:60]}"
    body = re.sub(r"\n\$\$\n.*?\n\$\$\n", "\n", text, flags=re.S)       # inline math must pair up per paragraph
    for para in body.split("\n\n"):
        assert para.replace("\\$", "").count("$") % 2 == 0, f"unbalanced $ in: {para[:80]}"
