"""Streamlit app: matchup, trade builder, trade finder, player values, injuries.

Run:  .venv/bin/streamlit run tradecalc/app.py
"""
from __future__ import annotations

import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

# Streamlit Cloud runs this file directly; make the `tradecalc` package importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

st.set_page_config(page_title="Trade Calculator", page_icon="🏀", layout="wide")

# ESPN cookies for a private league can be supplied as Streamlit secrets (ESPN_S2, SWID).
try:
    for _k in ("ESPN_S2", "SWID"):
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:  # no secrets file locally
    pass

from tradecalc.session import load_state  # noqa: E402

# Hosted (shared) copy: injury corrections are read-only so one visitor can't change everyone's model.
HOSTED = bool(os.environ.get("TRADECALC_HOSTED")) or str(Path(__file__).resolve()).startswith("/mount/src")
REFRESH_COOLDOWN_S = 600


# --- state (shared by every visitor) --------------------------------------------------------

@st.cache_resource
def shared() -> dict:
    return {"nonce": 0, "last_refresh": 0.0}


@st.cache_resource(show_spinner="Loading league, stats and injuries…", max_entries=6)
def get_state(today: str, nonce: int):
    warnings = []
    s = load_state(force=False, today=None if today.startswith("auto:") else today, warn=warnings.append)
    return s, warnings


SH = shared()
with st.sidebar:
    st.header("🏀 Trade Calculator")
    use_today = st.checkbox("Use today's date", value=True)
    pretend = None if use_today else st.date_input("Pretend today is", value=date(2026, 11, 15)).isoformat()
    if st.button("Refresh data", help="Pull new games, injuries, rosters and this week's scores "
                                      "(also happens automatically every hour)", use_container_width=True):
        ago = time.time() - SH["last_refresh"]
        if ago < REFRESH_COOLDOWN_S:
            st.info(f"Data was refreshed {ago / 60:.0f} min ago; try again in "
                    f"{(REFRESH_COOLDOWN_S - ago) / 60:.0f} min.")
        else:
            with st.spinner("Refreshing from ESPN and NBA stats…"):
                load_state(force=True)
            SH["last_refresh"] = time.time()
            SH["nonce"] += 1

state, warnings = get_state(pretend or f"auto:{date.today().isoformat()}", SH["nonce"])
for w in warnings:
    st.sidebar.warning(w)

cfg = state.cfg
P = state.players
team_ids = sorted(state.rosters)
team_label = {t: f"{state.team_name(t)} ({t})" for t in team_ids}

try:
    qp_team = int(st.query_params.get("team", cfg["league"]["my_team_id"]))
except ValueError:
    qp_team = cfg["league"]["my_team_id"]
if st.session_state.get("my_team") not in team_ids:
    st.session_state.my_team = qp_team if qp_team in team_ids else team_ids[0]
_tc1, _tc2 = st.columns([2, 3])
ME = _tc1.selectbox("👤 Your team", team_ids, key="my_team", format_func=lambda t: state.team_name(t),
                    help="Matchup, trades and simulations are shown from this team's point of view. "
                         "The page link remembers your choice, so bookmark it.")
_tc2.caption("")
_tc2.caption("Pick your team — every tab switches to it. Bookmark the link to come back to it.")
if st.query_params.get("team") != str(ME):
    st.query_params["team"] = str(ME)
with st.sidebar:
    if st.session_state.get("me") != ME:  # clear results computed for another team
        st.session_state.me = ME
        st.session_state.pop("found", None)
        st.session_state.pop("sim", None)
    st.caption(f"League: {state.league.settings['name']} · {len(team_ids)} teams")
    st.caption(f"Model date: {state.today:%b %d, %Y}")
    if SH["last_refresh"]:
        st.caption(f"Last manual refresh: {time.strftime('%b %d %H:%M', time.localtime(SH['last_refresh']))}")


def owner(p) -> str:
    return state.team_name(p.on_team_id) if p.on_team_id else "FA"


def a_now(p) -> float:
    return state.a_next(p, max(state.today, date.today()))


# --- date banner ---------------------------------------------------------------------------
st.info(f"📅 {state.week_status()}"
        + (f"  ·  **Model date set to {state.today:%a, %b %d}** (sidebar)" if pretend else "")
        + ("" if pretend or (state.real_today or state.today) >= state.cal.first_day
           else f"  ·  Projections start from opening night ({state.cal.first_day:%a, %b %d})."))

# --- tabs --------------------------------------------------------------------------------

tab_match, tab_league, tab_sim, tab_build, tab_find, tab_players, tab_inj, tab_help = st.tabs(
    ["📅 Matchup", "🏆 League forecast", "🎲 Simulate", "🔁 Trade builder", "🔎 Trade finder", "📊 Players",
     "🩹 Injuries", "❓ Help & accuracy"])

from tradecalc import help_text as H  # noqa: E402


def how_it_works(md: str):
    with st.expander("ℹ️ How this works"):
        st.markdown(md)


@st.cache_resource(show_spinner="Building projection engine (rates, minutes, rotations, defense)…", max_entries=4)
def get_world(today: str | None, nonce: int):
    from tradecalc.session import context
    from tradecalc.sim.live import live_world

    _, conn = context()
    return live_world(conn, cfg, state)

# ---------------------------------------------------------------------------- Matchup
with tab_match:
    how_it_works(H.matchup(cfg))
    from tradecalc.trade.matchup import p_win, p_win_monte_carlo, season_outlook, week_dist

    cur = state.cal.week_of(state.today) or state.cal.weeks[0]
    remaining = state.cal.remaining_weeks(state.today)
    wk_num = st.selectbox("Week", [w.number for w in remaining], index=0,
                          format_func=lambda n: f"Week {n}: {state.cal.weeks[n - 1].start:%b %d} – "
                                                f"{state.cal.weeks[n - 1].end:%b %d}")
    wk = state.cal.weeks[wk_num - 1]
    opp = state.cal.opponents[ME].get(wk.number)
    d1, d2 = week_dist(state.rosters[ME], wk, state, state.today), week_dist(state.rosters[opp], wk, state, state.today)
    (b1, last1), (b2, _) = state.banked(ME, wk), state.banked(opp, wk)
    d1, d2 = type(d1)(d1.M + b1, d1.V, d1.lineup), type(d2)(d2.M + b2, d2.V, d2.lineup)
    left_from = max(wk.start, state.today)
    st.caption(f"Days remaining in this projection: {left_from:%a %b %d} – {wk.end:%a %b %d}"
               + (f" · already scored through {last1:%a %b %d}: you {b1:,.0f}, them {b2:,.0f}" if (b1 or b2) else ""))
    pn = p_win(d1.M, d1.V, d2.M, d2.V)
    c1, c2, c3 = st.columns(3)
    c1.metric(state.team_name(ME), f"{d1.M:,.0f} pts", f"± {d1.V ** 0.5:.0f}", delta_color="off")
    c2.metric(state.team_name(opp), f"{d2.M:,.0f} pts", f"± {d2.V ** 0.5:.0f}", delta_color="off")
    c3.metric("Win probability", f"{pn:.0%}")
    if st.toggle("Check with Monte Carlo (10,000 simulated weeks)"):
        st.caption(f"Monte Carlo: {p_win_monte_carlo(d1, d2, state, cfg['trade']['monte_carlo_sims']):.1%}")

    def lineup_df(d):
        rows = []
        for pid, g, sa in zip(d.lineup.ids, d.lineup.g, d.lineup.sum_a):
            p = P[pid]
            rows.append({"Player": p.name, "Team": p.pro_team, "Pos": "/".join(p.positions), "FPPG": p.mu,
                         "Games left": state.cal.team_games_in_week(p.pro_team, wk, from_day=state.today),
                         "Week games": state.cal.team_games_in_week(p.pro_team, wk), "Starts": int(g),
                         "Exp. pts": sa * p.mu, "Notes": "; ".join(p.flags)})
        return pd.DataFrame(rows).sort_values("Exp. pts", ascending=False)

    fmt = {"FPPG": "{:.1f}", "Exp. pts": "{:.0f}"}
    l, r = st.columns(2)
    l.subheader("Your lineup")
    l.dataframe(lineup_df(d1).style.format(fmt), hide_index=True, use_container_width=True)
    r.subheader("Opponent")
    r.dataframe(lineup_df(d2).style.format(fmt), hide_index=True, use_container_width=True)

    st.subheader("Rest of season")
    out = season_outlook(ME, state.rosters, state)
    st.metric("Expected wins over remaining weeks", f"{out['e_wins']:.1f} of {sum(1 for x in out['weeks'] if x['p'] is not None)}")
    sched = pd.DataFrame([{"Week": x["week"], "Opponent": state.team_name(x["opp"]) if x["opp"] else "-",
                           "You": x["M"], "Them": x.get("M_opp"), "Win %": x["p"]} for x in out["weeks"]])
    st.bar_chart(sched.set_index("Week")["Win %"], height=220)
    st.dataframe(sched.style.format({"You": "{:.0f}", "Them": "{:.0f}", "Win %": "{:.0%}"}),
                 hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------- League forecast
@st.cache_resource(show_spinner=False, max_entries=6, ttl=3600)
def league_forecast(today_key: str, nonce: int, method: str, n: int):
    from tradecalc.trade.standings import forecast

    if method == "detailed":
        world, coaches_now = get_world(today_key, nonce)
        return forecast(state, n=n, seed=None, method="detailed", world=world, coaches_now=coaches_now)
    return forecast(state, n=n, seed=None, method="quick")


with tab_league:
    import altair as alt

    from tradecalc.sim.params import PARAMS_PATH as _PP
    from tradecalc.trade.standings import last_place_line

    how_it_works(H.league_forecast(cfg))
    fkey = (pretend or date.today().isoformat(), SH["nonce"])
    saved = SH.get("forecast")  # detailed forecast shared by everyone for an hour
    have_detailed = bool(saved and saved[0] == fkey and time.time() - saved[1] < 3600)
    c1, c2 = st.columns([3, 2])
    if _PP.exists():
        if c2.button("Run detailed forecast (~30 s)" if not have_detailed else "Re-run detailed forecast",
                     help="Game-by-game engine, 2,000 seasons. The result is shared with everyone for an hour."):
            with st.spinner("Simulating every NBA game of the rest of the season, 2,000 times…"):
                world, coaches_now = get_world(fkey[0], fkey[1])
                from tradecalc.trade.standings import forecast as _fc

                SH["forecast"] = (fkey, time.time(), _fc(state, n=2000, seed=None, method="detailed",
                                                          world=world, coaches_now=coaches_now))
            have_detailed = True
    if have_detailed:
        fc = SH["forecast"][2]
        c1.success(f"Detailed game-by-game forecast · run "
                   f"{time.strftime('%H:%M', time.localtime(SH['forecast'][1]))}")
    else:
        fc = league_forecast(fkey[0], fkey[1], "quick", 10000)
        c1.info("Quick forecast (season model). Press **Run detailed forecast** for the game-by-game engine.")

    champ, last = fc.iloc[0], fc.iloc[-1]
    st.markdown(f"## 🏆 Projected champion: **{champ['team']}**")
    st.markdown(f"Wins the league in **{champ['p_first']:.0%}** of simulated seasons, averaging "
                f"**{champ['exp_wins']:.1f}** wins.")
    st.error(f"💩 **{last_place_line(fc)}**  ·  projected last with {last['exp_wins']:.1f} wins "
             f"({last['p_last']:.0%} chance of finishing dead last)")

    show = pd.DataFrame({
        "Team": [("👤 " if r.team_id == ME else "") + r.team for r in fc.itertuples()],
        "Record": fc["record"],
        "Projected wins": fc["exp_wins"],
        "Wins range (80%)": [f"{a:.0f}–{b:.0f}" for a, b in zip(fc.wins_p10, fc.wins_p90)],
        "Win league": fc["p_first"], "Top 3": fc["p_top3"], "Last place": fc["p_last"],
        "Avg finish": fc["avg_rank"],
    }, index=fc.index)
    show.index.name = "Rank"
    st.dataframe(show.style.format({"Projected wins": "{:.1f}", "Win league": "{:.0%}", "Top 3": "{:.0%}",
                                    "Last place": "{:.0%}", "Avg finish": "{:.1f}"}), use_container_width=True)
    st.caption(f"{fc.attrs['n']:,} simulated seasons · {fc.attrs['weeks_left']} weeks left · "
               f"{'game-by-game engine' if fc.attrs['method'] == 'detailed' else 'season model'} · "
               "no playoffs, so most wins takes the league (ties broken by total points). "
               "Finished weeks use real results; this week's points so far are included. Updates hourly.")

    st.markdown("**Where each team finishes** (share of simulated seasons)")
    rd = fc.attrs["rank_dist"]
    heat = pd.DataFrame([{"Team": state.team_name(t), "Finish": i + 1, "Share": v}
                         for t, dist in rd.items() for i, v in enumerate(dist)])
    order = list(fc["team"])
    st.altair_chart(
        alt.Chart(heat).mark_rect().encode(
            x=alt.X("Finish:O", title="Final rank"), y=alt.Y("Team:N", sort=order, title=None),
            color=alt.Color("Share:Q", scale=alt.Scale(scheme="blues"), legend=alt.Legend(format="%")),
            tooltip=["Team", "Finish", alt.Tooltip("Share:Q", format=".0%")]),
        use_container_width=True)

# ---------------------------------------------------------------------------- Simulate
with tab_sim:
    import altair as alt
    import numpy as np

    from tradecalc.sim.engine import run_matchup
    from tradecalc.sim.params import PARAMS_PATH

    from tradecalc.sim.params import load_params as _lp

    how_it_works(H.simulate(_lp() if PARAMS_PATH.exists() else None))
    st.markdown("Full projection engine: per-stat models, minutes from rotations and depth charts, "
                "teammate usage, opponent defense and pace, injuries, rest on back-to-backs, coach "
                "tendencies — simulated game by game, with daily lineups set from who is active.")
    if not PARAMS_PATH.exists():
        st.warning("Engine parameters not fitted yet. Run `tradecalc sim-fit` once, then reload.")
    else:
        rem = state.cal.remaining_weeks(state.today)
        weeks_left = [w.number for w in rem]

        def _opponent(team, week):
            o = state.cal.opponents.get(team, {}).get(week)
            return o if o is not None and o != team else next(x for x in team_ids if x != team)

        def _sync_b():  # Team A or week changed -> Team B becomes Team A's real opponent that week
            st.session_state.sim_b = _opponent(st.session_state.sim_a, st.session_state.sim_wk)

        if st.session_state.get("sim_wk") not in weeks_left:
            st.session_state.sim_wk = weeks_left[0]
        if st.session_state.get("sim_a") not in team_ids or st.session_state.get("sim_for") != ME:
            st.session_state.sim_a = ME          # follow "Your team" when it changes
            st.session_state.sim_for = ME
            _sync_b()
        if st.session_state.get("sim_b") not in team_ids or st.session_state.sim_b == st.session_state.sim_a:
            _sync_b()

        c1, c2, c3, c4 = st.columns([3, 3, 2, 2])
        ta = c1.selectbox("Team A", team_ids, format_func=team_label.get, key="sim_a", on_change=_sync_b,
                          help="Any team in the league. Team B switches to Team A's opponent that week.")
        wk_n = c3.selectbox("Week", weeks_left, key="sim_wk", on_change=_sync_b)
        tb = c2.selectbox("Team B", [x for x in team_ids if x != ta], format_func=team_label.get, key="sim_b",
                          help="Defaults to Team A's scheduled opponent; pick any other team for a what-if.")
        n_sims = c4.select_slider("Simulations", [1000, 5000, 10000, 25000], value=10000)
        if st.button("Run simulation", type="primary"):
            world, coaches_now = get_world(pretend or date.today().isoformat(), SH["nonce"])
            with st.spinner(f"Simulating {n_sims:,} weeks…"):
                res, sims = run_matchup(world, state, ta, tb, state.cal.weeks[wk_n - 1], n=n_sims, seed=None,
                                        coaches_now=coaches_now)
            st.session_state.sim = (ta, tb, wk_n, res, sims, world, state.today)
        sim_ = st.session_state.get("sim")
        if sim_ and (len(sim_) != 7 or sim_[6] != state.today or sim_[:3] != (ta, tb, wk_n)):
            st.info("The date, teams or week changed since the last run. Press **Run simulation** to update.")
            sim_ = None
        if sim_:
            ta_, tb_, wk_, res, sims, world, _ = sim_
            la, lb = state.team_name(ta_), state.team_name(tb_)
            A, B = res.totals[la], res.totals[lb]
            wkobj = state.cal.weeks[wk_ - 1]
            st.subheader(f"Week {wk_}: {la} vs {lb}")
            if res.days:
                st.caption(f"Simulated days: {res.days[0]:%a %b %d} – {res.days[-1]:%a %b %d} "
                           f"({len(res.days)} of {len(wkobj.days())})"
                           + (" · days before today use the actual points already scored: "
                              + ", ".join(f"{k} {v:,.0f}" for k, v in res.banked.items()) if any(res.banked.values()) else ""))
            else:
                st.caption("This week is over; totals are the actual points scored.")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"{la} wins", f"{res.p_win:.1%}")
            m2.metric(f"{lb} wins", f"{1 - res.p_win - res.p_tie:.1%}")
            m3.metric(f"{la} projected", f"{A.mean():,.0f}", f"80%: {np.percentile(A, 10):,.0f}–{np.percentile(A, 90):,.0f}",
                      delta_color="off")
            m4.metric(f"{lb} projected", f"{B.mean():,.0f}", f"80%: {np.percentile(B, 10):,.0f}–{np.percentile(B, 90):,.0f}",
                      delta_color="off")
            k = min(len(A), 5000)
            dist = pd.DataFrame({"Points": np.concatenate([A[:k], B[:k]]), "Team": [la] * k + [lb] * k})
            hist = alt.Chart(dist).mark_area(opacity=0.45, interpolate="step").encode(
                x=alt.X("Points:Q", bin=alt.Bin(maxbins=60)), y=alt.Y("count()", stack=None, title="Simulations"),
                color=alt.Color("Team:N", legend=alt.Legend(orient="top"),
                                scale=alt.Scale(domain=[la, lb], range=["#3b82f6", "#f97316"])))
            st.altair_chart(hist, use_container_width=True)
            diff = pd.DataFrame({"Margin": (A - B)[:k]})
            st.caption(f"Margin ({la} − {lb}): mean {np.mean(A - B):+.0f}, 80% range "
                       f"{np.percentile(A - B, 10):+.0f} to {np.percentile(A - B, 90):+.0f}")
            st.altair_chart(alt.Chart(diff).mark_bar().encode(x=alt.X("Margin:Q", bin=alt.Bin(maxbins=60)),
                                                             y=alt.Y("count()", title="Simulations")).properties(height=160),
                            use_container_width=True)
            box = {}
            for (key, d), bm in sims.box_mean.items():
                box.setdefault(key, []).append(bm)
            cols = st.columns(2)
            for col, lab in zip(cols, (la, lb)):
                sub = res.players[res.players.fantasy_team == lab].sort_values("exp_fp_to_team", ascending=False)
                rows = []
                for r in sub.itertuples():
                    b_ = pd.DataFrame(box.get(r.key, [])).mean() if r.key in box else pd.Series(dtype=float)
                    rows.append({"Player": r.player, "NBA": r.nba_team, "G": r.games, "Active": r.p_active,
                                 "Starts": r.exp_starts, "MIN": b_.get("min", 0), "PTS": b_.get("pts", 0),
                                 "REB": b_.get("reb", 0), "AST": b_.get("ast", 0), "STL": b_.get("stl", 0),
                                 "BLK": b_.get("blk", 0), "TO": b_.get("tov", 0), "FP/G": r.fp_per_game,
                                 "FP to team": r.exp_fp_to_team})
                col.markdown(f"**{lab}**")
                col.dataframe(pd.DataFrame(rows).style.format(
                    {"Active": "{:.0%}", "Starts": "{:.1f}", "MIN": "{:.1f}", "PTS": "{:.1f}", "REB": "{:.1f}",
                     "AST": "{:.1f}", "STL": "{:.1f}", "BLK": "{:.1f}", "TO": "{:.1f}", "FP/G": "{:.1f}",
                     "FP to team": "{:.0f}"}), hide_index=True, use_container_width=True)
            st.caption("Per-game stat lines are averages over simulations in which the player is active. "
                       "'FP to team' counts only games he is started. 'G' = games left in the simulated days.")

            st.markdown("**Schedule this week**  ·  ● game left to play · ○ game already played · blank = no game")
            sim_days = set(res.days)
            grid = []
            for lab in (la, lb):
                for r in res.players[res.players.fantasy_team == lab].sort_values("exp_fp_to_team", ascending=False).itertuples():
                    row = {"Team": lab, "Player": r.player}
                    for d in wkobj.days():
                        plays = state.cal.team_plays(r.nba_team, d)
                        row[f"{d:%a %d}"] = ("●" if d in sim_days else "○") if plays else ""
                    row["Left"] = r.games
                    grid.append(row)
            st.dataframe(pd.DataFrame(grid), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------- Trade builder
with tab_build:
    how_it_works(H.trade_builder(cfg))
    from tradecalc.trade.evaluate import Trade, apply_trade, evaluate
    from tradecalc.trade.matchup import season_outlook

    partner = st.selectbox("Trade partner", [t for t in team_ids if t != ME], format_func=team_label.get)

    def label(pid):
        p = P[pid]
        return f"{p.name} · {'/'.join(p.positions)} · {p.mu:.1f} FPPG" + (" · " + p.flags[0] if p.flags else "")

    a, b = st.columns(2)
    give = a.multiselect("You give", sorted(state.tradeable(ME), key=lambda i: -P[i].mu), format_func=label)
    get = b.multiselect("You get", sorted(state.tradeable(partner), key=lambda i: -P[i].mu), format_func=label)
    precise = st.checkbox("Monte Carlo win probabilities (slower)")

    if give and get:
        trade = Trade(ME, partner, tuple(give), tuple(get))
        with st.spinner("Simulating the rest of the season…"):
            r = evaluate(trade, state, precise=precise)
        from tradecalc.trade.explain import explain

        ex = explain(r, state)
        color = {"ACCEPT": "green", "DECLINE": "red"}.get(r.verdict, "orange")
        st.markdown(f"### Verdict: :{color}[{r.verdict}]")
        st.markdown(ex.paragraph)
        icon = {1: ":green[▲] ", -1: ":red[▼] ", 0: ""}
        st.markdown("\n".join(f"- {icon[s]}{b}" for s, b in zip(ex.signs, ex.bullets)))
        st.caption("▲ helps you · ▼ hurts you · no arrow = context. Reasons are ordered by how many fantasy points "
                   "they are worth over the rest of the season.")

        if ex.this_week is not None and len(ex.weeks):
            st.markdown("**Weekly win probability, before → after**")
            wcols = st.columns(min(2, len(ex.weeks)))
            for col, w in zip(wcols, ex.weeks.head(len(wcols)).itertuples()):
                tag = f" ({ex.this_week['label']})" if w.week == ex.this_week["week"] and ex.this_week["label"] != f"week {int(w.week)}" else ""
                col.metric(f"Week {int(w.week)}{tag} vs {w.opponent} · was {w.before:.0%}", f"{w.after:.0%}",
                           f"{round(w.delta * 100):+d}%", help=f"vs {w.opponent}: {w.before:.0%} → {w.after:.0%}")
            with st.expander("All remaining weeks"):
                wk_df = ex.weeks.rename(columns={"week": "Week", "opponent": "Opponent", "before": "Before",
                                                 "after": "After", "delta": "Change"})
                st.dataframe(wk_df.style.format({"Before": "{:.0%}", "After": "{:.0%}",
                                                 "Change": lambda v: f"{round(v * 100):+d}%"})
                             .map(lambda v: f"color: {'#16a34a' if v > 0 else '#dc2626' if v < 0 else 'inherit'}",
                                  subset=["Change"]),
                             hide_index=True, use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Your expected wins", f"{r.you.wins_after:.2f}", f"{r.you.d_wins:+.2f}")
        c2.metric("Your points / week", f"{r.you.pts_after:,.0f}", f"{r.you.d_pts:+.1f}")
        c3.metric(f"{state.team_name(partner)} wins", f"{r.partner.wins_after:.2f}", f"{r.partner.d_wins:+.2f}")
        c4.metric(f"{state.team_name(partner)} pts / week", f"{r.partner.pts_after:,.0f}", f"{r.partner.d_pts:+.1f}")
        with st.expander("Model details"):
            st.markdown("\n".join(f"- {d}" for d in r.drivers))

        new, _ = apply_trade(trade, state.rosters, state)
        before, after = season_outlook(ME, state.rosters, state), season_outlook(ME, new, state)
        chart = pd.DataFrame({"Week": [x["week"] for x in before["weeks"]],
                              "Before": [x["p"] for x in before["weeks"]],
                              "After": [x["p"] for x in after["weeks"]]}).set_index("Week")
        st.caption("Your weekly win probability, before vs after")
        st.line_chart(chart, height=260, color=["#94a3b8", "#3b82f6"])
    else:
        st.info("Pick at least one player on each side.")

# ---------------------------------------------------------------------------- Trade finder
with tab_find:
    how_it_works(H.trade_finder(cfg))
    from tradecalc.trade.evaluate import format_result
    from tradecalc.trade.finder import find_trades

    c1, c2, c3 = st.columns(3)
    who = c1.selectbox("Partner", ["All teams"] + [t for t in team_ids if t != ME],
                       format_func=lambda t: t if isinstance(t, str) else team_label[t])
    max_size = c2.radio("Max players per side", [1, 2], index=1, horizontal=True)
    top = c3.number_input("Trades per team", 1, 20, cfg["trade"]["top_n_per_opponent"])
    st.caption(f"Keeps trades that raise your expected wins while the partner loses at most "
               f"{cfg['trade']['acceptance_threshold']:.2f} wins (a stand-in for 'they might say yes').")

    if st.button("Find trades", type="primary"):
        partners = None if who == "All teams" else [who]
        n = len(partners or [t for t in team_ids if t != ME])
        bar = st.progress(0.0, "Scanning…")
        done = []

        def progress(p, evaluated, kept):
            done.append(p)
            bar.progress(len(done) / n, f"{state.team_name(p)}: {evaluated} evaluated, {kept} kept")

        st.session_state.found = find_trades(ME, state, max_size=max_size, partners=partners, top=int(top),
                                             progress=progress)
        bar.empty()

    found = st.session_state.get("found")
    if found:
        rows = []
        for partner, rs in found.items():
            for r in rs:
                rows.append({"Partner": state.team_name(partner),
                             "You give": " + ".join(P[i].name for i in r.trade.give),
                             "You get": " + ".join(P[i].name for i in r.trade.get),
                             "Your Δ wins": r.you.d_wins, "Their Δ wins": r.partner.d_wins,
                             "Your Δ pts/wk": r.you.d_pts, "_r": r})
        if not rows:
            st.warning("No trades met the criteria.")
        else:
            df = pd.DataFrame(rows).sort_values("Your Δ wins", ascending=False)
            st.dataframe(df.drop(columns="_r").style.format(
                {"Your Δ wins": "{:+.2f}", "Their Δ wins": "{:+.2f}", "Your Δ pts/wk": "{:+.1f}"}),
                hide_index=True, use_container_width=True)
            for _, row in df.head(15).iterrows():
                with st.expander(f"{row['You give']} → {row['You get']} ({row['Partner']}, {row['Your Δ wins']:+.2f} wins)"):
                    st.text(format_result(row["_r"], state))

# ---------------------------------------------------------------------------- Players
with tab_players:
    how_it_works(H.players(cfg))
    c1, c2, c3 = st.columns(3)
    own = c1.selectbox("Owner", ["All", "Free agents"] + team_ids,
                       format_func=lambda t: t if isinstance(t, str) else team_label[t])
    pos = c2.multiselect("Position", ["PG", "SG", "SF", "PF", "C"])
    q = c3.text_input("Search")
    rows = []
    for p in P.values():
        if own == "Free agents" and p.on_team_id:
            continue
        if isinstance(own, int) and p.on_team_id != own:
            continue
        if pos and not set(pos) & set(p.positions):
            continue
        if q and q.lower() not in p.name.lower():
            continue
        rows.append({"Player": p.name, "Team": p.pro_team, "Pos": "/".join(p.positions), "Owner": owner(p),
                     "FPPG": p.mu, "Prior": p.prior, "GP": p.n, "Avail.": a_now(p),
                     "Games left": p.games_remaining, "ROS value": p.ros_value, "Notes": "; ".join(p.flags)})
    df = pd.DataFrame(rows).sort_values("ROS value", ascending=False)
    st.dataframe(df.style.format({"FPPG": "{:.1f}", "Prior": "{:.1f}", "Avail.": "{:.0%}", "ROS value": "{:,.0f}"}),
                 hide_index=True, use_container_width=True, height=600)
    rep = ", ".join(f"{k}: {v.name} ({v.mu:.1f})" for k, v in state.replacement.items())
    st.caption(f"Replacement level (best free agent): {rep}")

# ---------------------------------------------------------------------------- Injuries
with tab_inj:
    how_it_works(H.injuries(cfg))
    st.markdown("How the model treats each injured or recently-returned player. "
                "Research behind the assumptions: `docs/injury_research.md`.")
    only_rostered = st.checkbox("Rostered players only", value=True)
    base_day = max(state.today, date.today())
    rows = []
    for p in P.values():
        v = p.injury
        if not v or not (v.current or v.history) or (only_rostered and not p.on_team_id):
            continue
        rows.append({"Player": p.name, "Owner": owner(p), "FPPG": p.mu,
                     "Now": state.a_next(p, base_day),
                     "+30 days": state.a_next(p, base_day + timedelta(days=30)),
                     "+90 days": state.a_next(p, base_day + timedelta(days=90)),
                     "ROS value": p.ros_value, "Assumptions": "; ".join(p.flags)})
    if rows:
        df = pd.DataFrame(rows).sort_values("ROS value", ascending=False)
        st.dataframe(df.style.format({"FPPG": "{:.1f}", "Now": "{:.0%}", "+30 days": "{:.0%}",
                                      "+90 days": "{:.0%}", "ROS value": "{:,.0f}"}),
                     hide_index=True, use_container_width=True, height=420)
    st.caption("'Now / +30 / +90 days': chance he plays his team's next game on or after that date.")

    st.subheader("Injury corrections")
    ov_path = cfg.path(cfg["data"].get("injury_overrides_csv", "data/injury_overrides.csv"))
    hi_path = cfg.path(cfg["data"].get("injury_history_csv", "data/injury_history.csv"))
    ov = pd.read_csv(ov_path, dtype=str) if ov_path.exists() else pd.DataFrame(
        columns=["name", "status", "games_out", "return_date"])
    hi = pd.read_csv(hi_path, dtype=str) if hi_path.exists() else pd.DataFrame(columns=["name", "injury", "returned"])
    if HOSTED:
        st.caption("Corrections the league's model is using. They're read-only here because this copy is shared: "
                   "ask the commissioner to change them (they edit the files in the project and redeploy).")
        c1, c2 = st.columns(2)
        c1.markdown("**Current injury overrides**")
        c1.dataframe(ov, hide_index=True, use_container_width=True)
        c2.markdown("**Injury history**")
        c2.dataframe(hi, hide_index=True, use_container_width=True)
    else:
        st.caption("Saved to tradecalc/data/. Overrides change a current injury; history adds a past "
                   "injury the model should know about (blank 'returned' = start of this season).")
        from tradecalc.model.injury import load_profiles

        prof_keys = list(load_profiles().profiles)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Current injury overrides**")
            ov_new = st.data_editor(ov, num_rows="dynamic", use_container_width=True, key="ov",
                                    column_config={"status": st.column_config.SelectboxColumn(
                                        options=["ACTIVE", "DAY_TO_DAY", "OUT", "OUT_FOR_SEASON"])})
        with c2:
            st.markdown("**Injury history**")
            hi_new = st.data_editor(hi, num_rows="dynamic", use_container_width=True, key="hi",
                                    column_config={"injury": st.column_config.SelectboxColumn(options=prof_keys)})
        if st.button("Save corrections and recalculate"):
            ov_new.dropna(how="all").to_csv(ov_path, index=False)
            hi_new.dropna(how="all").to_csv(hi_path, index=False)
            SH["nonce"] += 1
            st.rerun()

# ---------------------------------------------------------------------------- Help & accuracy
with tab_help:
    from tradecalc.sim.live import load_validation

    st.markdown(H.help_overview(cfg))
    with st.expander("How trades are evaluated and found", expanded=False):
        st.markdown(H.help_trades(cfg))
    with st.expander("How injuries affect the model", expanded=False):
        st.markdown(H.injuries(cfg))
    st.markdown(H.help_accuracy(cfg, load_validation()))
    st.markdown(H.help_glossary())
