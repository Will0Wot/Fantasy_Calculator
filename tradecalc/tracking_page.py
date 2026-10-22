"""The app's Season tracker tab: the model's predictions, frozen before each week, next to what
really happened (tracking/<season>/*.csv, written by `tradecalc track`)."""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from tradecalc import tracking


def render(st, season: str) -> None:
    t = tracking.load(season)
    m, p, s = t["matchups"], t["players"], t["standings"]
    st.markdown(f"""
Every day the model's predictions are written down **before** the games are played: each team's
projected points and win chance for the coming week, every player's projection, and the league forecast.
When a week ends, the real scores are filled in next to them. At the end of the season this is the
scorecard. Predictions made before a week starts are never changed afterwards.
""")
    if m.empty:
        st.info("No predictions recorded yet. Run `tradecalc track` (it runs automatically once a day).")
        return
    pre = m[m.kind == "pre-week"].copy()
    graded = pre[pre.get("won").notna()] if "won" in pre else pre.iloc[0:0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last snapshot", str(m.as_of.max()))
    c2.metric("Weeks predicted", int(pre.week.nunique()))
    c3.metric("Weeks graded", int(graded.week.nunique()))
    if len(graded):
        fav = graded[graded.win_prob != 0.5]
        c4.metric("Favourite won", f"{((fav.win_prob > 0.5) == (fav.won > 0.5)).mean():.0%}",
                  help="Share of team-weeks where the team the model favoured won.")
    st.download_button("⬇️ Download Excel workbook (summary + all predictions and results)", tracking.excel(t),
                       file_name=f"model_vs_actual_{season}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    sm = tracking.summary(t)
    if len(sm):
        st.subheader("How close so far")
        st.dataframe(sm.style.format({"value": lambda v: f"{v:.3f}" if isinstance(v, float) and abs(v) < 10 else
                                      f"{v:,.1f}" if isinstance(v, float) else v}), hide_index=True,
                     width="stretch")

    st.subheader("Weekly matchups: prediction vs result")
    weeks = sorted(pre.week.unique())
    wk = st.selectbox("Week", weeks, index=len(weeks) - 1, format_func=lambda w: f"Week {w}", key="trk_week")
    g = pre[pre.week == wk].copy()
    cols = ["team", "opp", "proj_points", "proj_sd", "win_prob"]
    names = {"team": "Team", "opp": "Opponent", "proj_points": "Projected pts", "proj_sd": "± (1 s.d.)",
             "win_prob": "Win chance", "actual_points": "Actual pts", "opp_actual_points": "Opponent actual",
             "won": "Won", "as_of": "Predicted on"}
    if "actual_points" in g:
        cols += ["actual_points", "opp_actual_points", "won"]
    g = g[cols + ["as_of"]].rename(columns=names).sort_values("Win chance", ascending=False)
    st.dataframe(g.style.format({"Projected pts": "{:,.0f}", "± (1 s.d.)": "{:,.0f}", "Win chance": "{:.0%}",
                                 "Actual pts": "{:,.0f}", "Opponent actual": "{:,.0f}", "Won": "{:.1f}"}, na_rep="–"),
                 hide_index=True, width="stretch")

    if len(graded):
        a, b = st.columns(2)
        lim = [float(min(graded.proj_points.min(), graded.actual_points.min())) * 0.95,
               float(max(graded.proj_points.max(), graded.actual_points.max())) * 1.05]
        diag = alt.Chart(pd.DataFrame({"x": lim, "y": lim})).mark_line(strokeDash=[4, 4], color="gray").encode(x="x", y="y")
        pts = alt.Chart(graded).mark_circle(size=60, opacity=0.7).encode(
            x=alt.X("proj_points:Q", title="Projected points", scale=alt.Scale(domain=lim)),
            y=alt.Y("actual_points:Q", title="Actual points", scale=alt.Scale(domain=lim)),
            color=alt.Color("week:O", title="Week"), tooltip=["week", "team", "proj_points", "actual_points"])
        a.altair_chart((diag + pts).properties(height=300, title="Team points: projected vs actual"),
                       width="stretch")
        bins = pd.cut(graded.win_prob, [0, .2, .4, .6, .8, 1.0], include_lowest=True)
        cal = graded.groupby(bins, observed=True).agg(predicted=("win_prob", "mean"), actual=("won", "mean"),
                                                      n=("won", "size")).reset_index(drop=True)
        cd = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color="gray").encode(x="x", y="y")
        cp = alt.Chart(cal).mark_line(point=True).encode(
            x=alt.X("predicted:Q", title="Predicted win chance", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            y=alt.Y("actual:Q", title="Actually won", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            tooltip=["n", alt.Tooltip("predicted:Q", format=".0%"), alt.Tooltip("actual:Q", format=".0%")])
        b.altair_chart((cd + cp).properties(height=300, title="Win chances: predicted vs actual"),
                       width="stretch")
        st.caption("On the dashed line the model is exactly right; with few weeks graded the dots will bounce around.")

    if len(s):
        st.subheader("League forecast over the season")
        piv = s.pivot_table(index="before_week", columns="team", values="exp_wins")
        st.line_chart(piv, height=300)
        st.caption("Each team's projected final wins, as predicted before each week. "
                   + ("Final wins are filled in when the season ends." if "final_wins" not in s else ""))

    if len(p):
        st.subheader("Players: projection vs actual")
        pw = sorted(p.week.unique())
        pk = st.selectbox("Week ", pw, index=len(pw) - 1, format_func=lambda w: f"Week {w}", key="trk_pweek")
        q = p[p.week == pk].copy()
        keep = ["team", "player", "pro_team", "status", "proj_fppg", "proj_games", "proj_points"]
        if "actual_points" in q:
            keep += ["games_played", "actual_fppg", "actual_points"]
            q["miss"] = q.proj_points - q.actual_points
            keep.append("miss")
        st.dataframe(q[keep].rename(columns={"team": "Fantasy team", "player": "Player", "pro_team": "NBA",
                                             "status": "Status", "proj_fppg": "Proj FPPG", "proj_games": "Exp. games",
                                             "proj_points": "Proj pts", "games_played": "Games",
                                             "actual_fppg": "Actual FPPG", "actual_points": "Actual pts",
                                             "miss": "Proj − actual"})
                     .style.format({"Proj FPPG": "{:.1f}", "Exp. games": "{:.1f}", "Proj pts": "{:.0f}",
                                    "Games": "{:.0f}", "Actual FPPG": "{:.1f}", "Actual pts": "{:.0f}",
                                    "Proj − actual": "{:+.0f}"}, na_rep="–"),
                     hide_index=True, width="stretch", height=360)
    with st.expander("Raw files"):
        for name, df in t.items():
            if len(df):
                st.download_button(f"{name}.csv ({len(df):,} rows)", df.to_csv(index=False), file_name=f"{name}_{season}.csv",
                                   key=f"trk_csv_{name}")
