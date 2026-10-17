"""The app's Model audit tab: before/after scoreboard, the written report, every test by season,
and the break tests. Data comes from config/audit.json (final model) and
config/audit_baseline.json (the original model, same tests, same mock leagues)."""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

# (label, how to read it, getter) -- getters return None when a run lacks the number
_GOOD = {"lower": "lower is better", "one": "1.00 = honest", "higher": "higher is better", "zero": "0 = unbiased",
         "80": "80% = honest"}


def _g(r, *path):
    for p in path:
        if r is None:
            return None
        if isinstance(p, int):
            r = r[p] if isinstance(r, list) and len(r) > p else None
        else:
            r = r.get(p) if isinstance(r, dict) else None
    return r


def _lf(r, cp, key):
    for m in _g(r, "league_forecast", "metrics") or []:
        if m["checkpoint"] == cp:
            return m.get(key)
    return None


METRICS = [
    ("Weekly win chances (trade calculator): Brier", "lower", lambda r: _g(r, "season_model", "matchups", "brier")),
    ("Weekly win chances (trade calculator): calibration slope", "one", lambda r: _g(r, "season_model", "matchups", "slope")),
    ("Trades: predicted vs real change in wins, slope", "one", lambda r: _g(r, "season_model", "trades", "slope")),
    ("Trades: predicted vs real change in wins, correlation", "higher", lambda r: _g(r, "season_model", "trades", "corr")),
    ("Trades rated ACCEPT: predicted gain (wins)", None, lambda r: _g(r, "season_model", "trades", "accept_pred_mean")),
    ("Trades rated ACCEPT: real gain (wins)", "higher", lambda r: _g(r, "season_model", "trades", "accept_real_mean")),
    ("Trades rated DECLINE: real gain (wins)", None, lambda r: _g(r, "season_model", "trades", "decline_real_mean")),
    ("Rest-of-season wins after week 6: error (wins)", "lower", lambda r: _g(r, "season_model", "season_wins", 1, "mae")),
    ("Plays tonight? (availability): Brier", "lower", lambda r: _g(r, "availability", "brier_model")),
    ("Simulator: player points per game, error", "lower", lambda r: _g(r, "engine_points", "summary", "engine", "MAE")),
    ("Simulator: player points per game, bias", "zero", lambda r: _g(r, "engine_points", "summary", "engine", "bias")),
    ("Simulator: player-weeks inside the 80% range", "80", lambda r: _g(r, "engine_ranges", "week_in80")),
    ("Simulator: weekly matchups, Brier", "lower", lambda r: _g(r, "engine_matchups", "brier")),
    ("Simulator: weekly matchups, calibration slope", "one", lambda r: _g(r, "engine_matchups", "slope")),
    ("League forecast preseason: rank correlation", "higher", lambda r: _lf(r, "preseason", "rank_corr")),
    ("League forecast preseason: ESPN-projection ranking (benchmark)", None, lambda r: _lf(r, "preseason", "rank_corr_naive")),
    ("League forecast after week 6: rank correlation", "higher", lambda r: _lf(r, "after week 6", "rank_corr")),
    ("League forecast after week 6: teams inside the 80% wins range", "80", lambda r: _lf(r, "after week 6", "in_80pct_range")),
]


def scoreboard(final: dict, base: dict | None) -> pd.DataFrame:
    rows = []
    for label, good, get in METRICS:
        row = {"Metric": label, "Goal": _GOOD.get(good, "")}
        for s, r in final["seasons"].items():
            b = _g(base, "seasons", s)
            ob, nf = get(b) if b else None, get(r)
            row[f"{s} original"] = ob
            row[f"{s} final"] = nf
        rows.append(row)
    return pd.DataFrame(rows)


def _calib_chart(final_rows, base_rows=None, title=""):
    d = [dict(x, model="final") for x in final_rows or []]
    d += [dict(x, model="original") for x in base_rows or []]
    if not d:
        return None
    df = pd.DataFrame(d)
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color="gray").encode(x="x", y="y")
    base = alt.Chart(df).encode(
        x=alt.X("predicted:Q", title="Predicted", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        y=alt.Y("actual:Q", title="Actually happened", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=alt.Color("model:N", scale=alt.Scale(domain=["original", "final"], range=["#bbbbbb", "#1f77b4"])))
    # size only on the points: on a line mark it sets the stroke width and breaks the layout
    pts = base.mark_line() + base.mark_circle(opacity=0.9).encode(
        size=alt.Size("n:Q", legend=None, scale=alt.Scale(range=[20, 300])),
        tooltip=["model", "bin", "n", alt.Tooltip("predicted:Q", format=".0%"), alt.Tooltip("actual:Q", format=".0%")])
    return (diag + pts).properties(height=300, title=title)


def _trade_chart(points):
    df = pd.DataFrame(points)
    if df.empty or "pred" not in df:
        return None
    rng = np.random.default_rng(0)
    df["real (jittered)"] = df["real"] + rng.uniform(-0.15, 0.15, len(df))
    lim = float(max(df["pred"].abs().max(), 1.0))
    diag = alt.Chart(pd.DataFrame({"x": [-lim, lim], "y": [-lim, lim]})).mark_line(strokeDash=[4, 4], color="gray").encode(x="x", y="y")
    sc = alt.Chart(df).mark_circle(opacity=0.45).encode(
        x=alt.X("pred:Q", title="Predicted change in your wins"),
        y=alt.Y("real (jittered):Q", title="Real change in your wins"),
        color=alt.Color("shape:N", title="Trade"), tooltip=["shape", alt.Tooltip("pred:Q", format="+.2f"), "real"])
    fit = alt.Chart(df).transform_regression("pred", "real").mark_line(color="#ff7f0e", strokeWidth=2).encode(
        x="pred:Q", y="real:Q")                                   # least-squares fit; the dashed line is "perfect"
    return (diag + sc + fit).properties(height=340)


def _fmt(v, kind="f2"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return {"f2": f"{v:.2f}", "f3": f"{v:.3f}", "pct": f"{v:.0%}", "f1": f"{v:.1f}"}[kind]


def render(st, final: dict | None, base: dict | None, report_md: str | None) -> None:
    if not final:
        st.info("No audit results yet. Run `tradecalc audit --baseline` to create config/audit.json.")
        return
    seasons = list(final["seasons"])
    st.markdown(f"""
Every part of the model was re-run on **{len(seasons)} past seasons ({", ".join(seasons)})** as if it were
that season: parameters were fitted only on earlier seasons, and predictions only used what was known on the
prediction date. Then the predictions were compared with what really happened in the box scores.
**Original** = the model before this audit; **final** = the model after the fixes (same tests, same mock leagues).
*Run on {final.get('run_on', '?')}. Re-run with `tradecalc audit --baseline`.*
""")
    st.subheader("Scoreboard")
    sb = scoreboard(final, base)
    num = [c for c in sb.columns if c not in ("Metric", "Goal")]
    st.dataframe(sb.style.format({c: lambda v: _fmt(v, "f3") for c in num}, na_rep="–"),
                 hide_index=True, use_container_width=True)
    st.caption("Brier score: average squared error of a probability (0.25 = always saying 50%). Calibration slope: "
               "1 means a 70% call wins 70% of the time; below 1 the model is overconfident, above 1 too timid. "
               "Trade numbers come from random trades in mock leagues, replayed with real box scores.")

    if report_md:
        with st.expander("📄 Full audit report: what works, what doesn't, what changed", expanded=False):
            st.markdown(report_md)

    st.subheader("Every test, by season")
    s = st.radio("Season", seasons, horizontal=True, index=len(seasons) - 1, key="audit_season")
    r = final["seasons"][s]
    b = _g(base, "seasons", s)
    st.caption(f"Fitted on {', '.join(r.get('fit_on', []))}; tested on {s}.")

    t1, t2, t3, t4, t5, t6 = st.tabs(["Weekly win chances", "Trades", "Availability", "Simulator", "Season & league",
                                      "Player projections"])
    with t1:
        m = _g(r, "season_model", "matchups") or {}
        c1, c2, c3 = st.columns(3)
        c1.metric("Matchups tested", f"{m.get('n', 0):,}")
        c2.metric("Brier (final)", _fmt(m.get("brier"), "f3"),
                  delta=None if not b else f"{m.get('brier', 0) - _g(b, 'season_model', 'matchups', 'brier'):+.3f} vs original",
                  delta_color="inverse")
        c3.metric("Calibration slope (final)", _fmt(m.get("slope")),
                  delta=None if not b else f"original {_fmt(_g(b, 'season_model', 'matchups', 'slope'))}", delta_color="off")
        ch = _calib_chart(m.get("calibration"), _g(b, "season_model", "matchups", "calibration"),
                          "Predicted vs real win rate")
        if ch is not None:
            st.altair_chart(ch, use_container_width=True)
        bc = m.get("by_checkpoint") or []
        if bc:
            st.dataframe(pd.DataFrame(bc).rename(columns={"checkpoint": "predicted at week"}).round(3), hide_index=True)
        beta = _g(final, "matchup_shrink", s)
        st.caption("Each dot: matchups grouped by predicted win chance; on the dashed line, a 70% call wins 70% "
                   "of the time. Mock 12-team leagues drafted from ESPN's preseason projections; every remaining matchup is "
                   "predicted at weeks 0, 6 and 12 and compared with the real box-score result."
                   + (f" Matchup shrink β for this season, fitted on earlier seasons only: {beta:.2f}." if beta else ""))
    with t2:
        tr = _g(r, "season_model", "trades") or {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Random trades", f"{tr.get('n', 0):,}")
        c2.metric("Slope (1 = right size)", _fmt(tr.get("slope")),
                  delta=None if not b else f"original {_fmt(_g(b, 'season_model', 'trades', 'slope'))}", delta_color="off")
        c3.metric("Correlation", _fmt(tr.get("corr")),
                  delta=None if not b else f"original {_fmt(_g(b, 'season_model', 'trades', 'corr'))}", delta_color="off")
        c4.metric("Benchmark: total-value correlation", _fmt(tr.get("corr_naive_ros")))
        ch = _trade_chart(tr.get("points") or [])
        if ch is not None:
            st.altair_chart(ch, use_container_width=True)
        st.markdown(f"""
- Trades the calculator rated **ACCEPT** ({tr.get('accept_n', 0)}): predicted **{_fmt(tr.get('accept_pred_mean'))}**
  extra wins, really gained **{_fmt(tr.get('accept_real_mean'))}**.
- Trades rated **DECLINE**: really changed wins by **{_fmt(tr.get('decline_real_mean'))}**.
- Both teams predicted to gain > 0.1 wins in {_fmt(tr.get('both_gain_pred'), 'pct')} of trades.
""")
        if tr.get("by_shape"):
            st.dataframe(pd.DataFrame(tr["by_shape"]).rename(columns={"pred": "predicted Δ wins", "real": "real Δ wins"}).round(2),
                         hide_index=True)
        st.caption("Each random trade is replayed with real box scores: the same forced drops and free-agent "
                   "signings the calculator assumes, then real weekly scores and real head-to-head results.")
    with t3:
        a = r.get("availability") or {}
        c1, c2, c3 = st.columns(3)
        c1.metric("Player-games", f"{a.get('n', 0):,}")
        c2.metric("Brier (final)", _fmt(a.get("brier_model"), "f3"),
                  delta=None if not b else f"original {_fmt(_g(b, 'availability', 'brier_model'), 'f3')}", delta_color="off")
        c3.metric("Benchmark: everyone plays at the league rate", _fmt(a.get("brier_naive"), "f3"))
        st.caption("Grey: the original model used raw games-played rates as probabilities. Blue: calibrated.")
        ch = _calib_chart(a.get("calibration"), a.get("calibration_raw"),
                          "Chance to play a game: predicted vs real")
        if ch is not None:
            st.altair_chart(ch, use_container_width=True)
    with t4:
        ep = _g(r, "engine_points", "summary") or {}
        if ep:
            st.markdown("**Points per game, by model version** (mean absolute error, bias, RMSE)")
            st.dataframe(pd.DataFrame(ep).T.round(2), use_container_width=True)
        bs = _g(r, "engine_points", "by_stage")
        if bs:
            st.dataframe(pd.DataFrame(bs).round(2), hide_index=True, use_container_width=True)
        rg = r.get("engine_ranges") or {}
        c1, c2, c3 = st.columns(3)
        c1.metric("Games inside 80% range", _fmt(rg.get("game_in80"), "pct"))
        c2.metric("Weeks inside 80% range", _fmt(rg.get("week_in80"), "pct"))
        c3.metric("Weeks inside 50% range", _fmt(rg.get("week_in50"), "pct"))
        em = r.get("engine_matchups") or {}
        ch = _calib_chart(em.get("calibration"), _g(b, "engine_matchups", "calibration"),
                          f"Simulator: predicted vs real win rate ({em.get('n', 0)} matchups)")
        if ch is not None:
            st.altair_chart(ch, use_container_width=True)
    with t5:
        sw = _g(r, "season_model", "season_wins")
        if sw:
            st.markdown("**Expected rest-of-season wins vs real** (trade calculator's season math)")
            st.dataframe(pd.DataFrame(sw).rename(columns={"checkpoint": "predicted at week", "mae": "error (wins)",
                                                          "corr": "rank correlation"}).round(2), hide_index=True)
        lf = _g(r, "league_forecast", "metrics")
        if lf:
            st.markdown("**League forecast** (simulator, 10 mock leagues)")
            st.dataframe(pd.DataFrame(lf).set_index("checkpoint").round(3).T, use_container_width=True)
    with t6:
        sf = _g(r, "season_fppg", "table")
        if sf:
            st.markdown("**Rest-of-season fantasy points per game: error (MAE) after N games**, by method")
            st.dataframe(pd.DataFrame(sf).round(2), hide_index=True, use_container_width=True)
            st.caption(f"Best blend weight k in this season: {_g(r, 'season_fppg', 'best_k')}. "
                       "The model blends ESPN's preseason projection with this season's average as (k·prior + n·live)/(k+n).")

    st.subheader("Trying to break the model")
    br = final.get("breaks") or []
    if br:
        df = pd.DataFrame(br)
        st.markdown(f"**{int(df.passed.sum())} of {len(df)} passed.** Each test feeds the model an input designed to "
                    "expose a bug (impossible lineups, clones, zero-sum checks, looking into the future).")
        df["result"] = np.where(df.passed, "✅ pass", "❌ FAIL")
        st.dataframe(df[["result", "name", "idea", "detail"]].rename(columns={"name": "test", "idea": "what it does",
                                                                              "detail": "outcome"}),
                     hide_index=True, use_container_width=True)
