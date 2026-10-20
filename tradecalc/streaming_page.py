"""The app's Streaming tab: this week's pickup plan for any team (tradecalc/trade/streaming.py)."""
from __future__ import annotations

import pandas as pd

from tradecalc.trade.streaming import describe, league_summary, plan, streamer_table


def _pct(p):
    return "–" if p is None else f"{p:.0%}"


def render(st, state, ME: int, team_ids: list[int], team_label: dict) -> None:
    P = state.players
    cfg = state.cfg
    if st.session_state.get("stream_for") not in team_ids or st.session_state.get("stream_me") != ME:
        st.session_state.stream_for = ME           # follow "Your team" when it changes
        st.session_state.stream_me = ME
    weeks = [w for w in state.cal.weeks if w.end >= state.today][:2]
    if not weeks:
        st.info("The regular season is over.")
        return
    c0, c1, c2 = st.columns([3, 3, 2])
    team = c0.selectbox("Team", team_ids, key="stream_for", format_func=team_label.get)
    week = c1.selectbox("Week", weeks, format_func=lambda w: f"Week {w.number}: {w.start:%b %d} – {w.end:%b %d}"
                        + (" (this week)" if w.start <= state.today <= w.end else ""))
    limit = int(cfg.get("streaming", {}).get("moves_per_week", 1))
    moves = c2.number_input("Pickups to plan", 1, 4, limit, help=f"League limit read from ESPN: {limit} per matchup.")

    with st.spinner("Trying every drop, pickup and day…"):
        pl = plan(state, team, week, max_moves=int(moves))
    opp = state.team_name(pl.opp) if pl.opp else None
    start = f"from {pl.start:%a %b %d}" if pl.start > week.start else "for the whole week"
    st.caption(f"Plan {start}{f' vs {opp}' if opp else ''}. Roster: {pl.open_spots} open spot(s), "
               f"{pl.free_ir} free IR slot(s).")

    m1, m2, m3 = st.columns(3)
    m1.metric("Win chance this week", _pct(pl.p_after),
              None if pl.p_after is None else f"{round((pl.p_after - pl.p_before) * 100):+d}% vs no moves")
    m2.metric("Projected points this week", f"{pl.M_after:,.0f}", f"{pl.M_after - pl.M_before:+.0f}")
    m3.metric("Later weeks (expected wins)", f"{pl.future_d_wins:+.2f}",
              help="Change in expected wins after this week if you keep every move.")

    if not pl.moves:
        st.success("No pickup helps enough this week: your roster is already the best use of your spots.")
    else:
        st.subheader("The plan")
        for m in pl.moves:
            add = P[m.add]
            line = f"**{m.day:%A %b %d}: {describe(state, m)}.** {add.name} ({add.pro_team}, {add.mu:.1f} pts/game) "
            line += f"has {m.add_games} game{'s' if m.add_games != 1 else ''} from that day."
            if m.kind == "drop":
                out = P[m.out]
                line += (f" {out.name} has {m.out_games} game{'s' if m.out_games != 1 else ''} left from then"
                         + (" (you keep his earlier games by waiting)." if m.day > pl.start else "."))
            line += (f" This week: {m.gain_pts:+.0f} pts, win chance {_pct(m.p_after)} ({m.d_week * 100:+.0f}%)."
                     f" Later weeks if kept: {m.d_future:+.2f} wins.")
            if m.d_future < -0.05:
                line += " ⚠️ A one-week rental: worth it for this matchup, but consider swapping back next week."
            st.markdown("- " + line)

    if pl.options:
        st.subheader("Other good moves (if your first choice is taken)")
        rows = [{"Day": f"{m.day:%a %b %d}", "Move": describe(state, m), "Games": m.add_games,
                 "Extra pts": m.gain_pts, "Win chance": m.p_after, "This week": m.d_week, "Later weeks": m.d_future}
                for m in pl.options[:8]]
        st.dataframe(pd.DataFrame(rows).style.format({"Extra pts": "{:+.0f}", "Win chance": _pct,
                                                      "This week": lambda v: f"{v * 100:+.0f}%",
                                                      "Later weeks": "{:+.2f}"}),
                     hide_index=True, use_container_width=True)

    st.subheader("Best free agents this week")
    tbl = streamer_table(state, week)
    if len(tbl):
        st.dataframe(tbl.drop(columns=["id"]).style.format({"FPPG": "{:.1f}", "Exp. pts this week": "{:.0f}",
                                                            "Rest-of-season value": "{:,.0f}"}),
                     hide_index=True, use_container_width=True, height=320)
        st.caption("Exp. pts = points per game × the chance he plays each remaining game this week. "
                   "Rest-of-season value = his expected fantasy points for the rest of the season.")

    if st.toggle("Show every team's best move this week"):
        with st.spinner("Planning for all 12 teams…"):
            ls = league_summary(state, week, max_moves=int(moves))
        st.dataframe(ls.style.format({"Extra pts": "{:+.0f}", "Win chance now": _pct, "With moves": _pct,
                                      "Long-term (wins)": "{:+.2f}"}), hide_index=True, use_container_width=True)
        st.caption("Each plan assumes nobody else moves first; popular free agents go to whoever adds them first.")
