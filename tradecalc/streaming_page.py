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
    m3.metric("Rest of season (expected wins)", f"{pl.d_future + pl.d_rival:+.2f}",
              help=f"Keeping the moves: {pl.d_future:+.2f} from your own roster, {pl.d_rival:+.2f} from rivals "
                   "claiming the players you drop.")

    if not pl.moves:
        st.success("No pickup helps enough this week: every move either adds too little now or costs more later.")
    else:
        st.subheader("The plan")
        for m in pl.moves:
            st.markdown("- " + _move_text(state, m, pl.start))

    if pl.drop_costs:
        st.subheader("What would dropping each bench player cost?")
        st.markdown(
            "A drop isn't free. A dropped player goes on waivers, and **if he's better than every free agent "
            "out there, a rival will claim him**: you can't get him back, and the team that gets him is tougher in "
            "the weeks you play them. The planner weighs three things for every drop, in expected wins:\n"
            "1. **This week:** the extra games from the pickup.\n"
            "2. **Later weeks:** the rest of the season with the free agent instead of the player you dropped.\n"
            "3. **Rival pickup:** if he'd be the best player on waivers, the team with the best waiver priority "
            "who'd want him claims him, and you face a stronger opponent in your remaining games against them. "
            "If free agents just as good are already out there, nobody gains by claiming him, so this is zero.\n\n"
            "A drop is only suggested when the total is positive.")
        rows = []
        for m in pl.drop_costs:
            c = m.claim
            rows.append({"Drop": P[m.out].name, "Best pickup for it": f"{P[m.add].name} ({m.day:%a})",
                         "This week": m.d_week, "Later weeks": m.d_future,
                         "Claimed by a rival": (f"{c.p:.0%}: {state.team_name(c.team)} (#{c.rank} on waivers)"
                                                if c.team else "unlikely"),
                         "Rival pickup": c.cost, "Total": m.total})
        st.dataframe(pd.DataFrame(rows).style.format({"This week": lambda v: f"{v * 100:+.0f}%",
                                                      "Later weeks": "{:+.2f}", "Rival pickup": "{:+.2f}",
                                                      "Total": "{:+.2f}"})
                     .map(lambda v: f"color: {'#16a34a' if v > 0 else '#dc2626'}", subset=["Total"]),
                     hide_index=True, width="stretch")
        st.caption("This week = change in this week's win chance. The other columns are changes in expected wins over "
                   "the rest of the season. Waiver order is ESPN's (1 = first claim).")

    if len(pl.options) > 1:
        st.subheader("Other moves that help (if your first choice is taken)")
        rows = [{"Day": f"{m.day:%a %b %d}", "Move": describe(state, m), "Games": m.add_games,
                 "Extra pts": m.gain_pts, "Win chance": m.p_after, "This week": m.d_week,
                 "Later weeks": m.d_future, "Rival pickup": m.claim.cost, "Total": m.total}
                for m in pl.options[1:8]]
        st.dataframe(pd.DataFrame(rows).style.format({"Extra pts": "{:+.0f}", "Win chance": _pct,
                                                      "This week": lambda v: f"{v * 100:+.0f}%",
                                                      "Later weeks": "{:+.2f}", "Rival pickup": "{:+.2f}",
                                                      "Total": "{:+.2f}"}),
                     hide_index=True, width="stretch")

    st.subheader("Best free agents this week")
    tbl = streamer_table(state, week)
    if len(tbl):
        st.dataframe(tbl.drop(columns=["id"]).style.format({"FPPG": "{:.1f}", "Exp. pts this week": "{:.0f}",
                                                            "Rest-of-season value": "{:,.0f}"}),
                     hide_index=True, width="stretch", height=320)
        st.caption("Exp. pts = points per game × the chance he plays each remaining game this week. "
                   "Rest-of-season value = his expected fantasy points for the rest of the season.")

    if st.toggle("Show every team's best move this week"):
        with st.spinner("Planning for all 12 teams…"):
            ls = league_summary(state, week, max_moves=int(moves))
        st.dataframe(ls.style.format({"Extra pts": "{:+.0f}", "Win chance now": _pct, "With moves": _pct,
                                      "Later weeks": "{:+.2f}", "Rival pickups": "{:+.2f}"}),
                     hide_index=True, width="stretch")
        st.caption("Each plan assumes nobody else moves first; popular free agents go to whoever adds them first.")


def _move_text(state, m, start) -> str:
    P = state.players
    add = P[m.add]
    line = (f"**{m.day:%A %b %d}: {describe(state, m)}.** {add.name} ({add.pro_team}, {add.mu:.1f} pts/game) has "
            f"{m.add_games} game{'s' if m.add_games != 1 else ''} from that day.")
    if m.kind == "drop":
        out = P[m.out]
        line += (f" {out.name} has {m.out_games} game{'s' if m.out_games != 1 else ''} left from then"
                 + (" (waiting keeps his earlier games)." if m.day > start else "."))
        c = m.claim
        if c.team is not None:
            vs = (f"you play them in week{'s' if len(c.weeks) != 1 else ''} {', '.join(map(str, c.weeks))}"
                  if c.weeks else "you don't play them again")
            line += (f" A rival would likely claim {out.name} ({c.p:.0%}): {state.team_name(c.team)}, "
                     f"#{c.rank} on waivers; {vs} ({c.cost:+.2f} wins).")
        else:
            line += f" Nobody is likely to claim {out.name}: free agents just as good are available."
    line += (f" This week: {m.gain_pts:+.0f} pts, win chance {_pct(m.p_after)} ({m.d_week * 100:+.0f}%)."
             f" Later weeks: {m.d_future:+.2f} wins. **Total: {m.total:+.2f} expected wins.**")
    return line

