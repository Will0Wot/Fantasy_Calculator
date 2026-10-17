"""League forecast: Monte Carlo of the rest of the regular season for every team.

Each simulated season:
  - weeks already finished use the real result from ESPN;
  - every remaining week draws each team's score (points already scored this week are added),
    then every scheduled matchup is decided (a tie is worth tie_value);
  - teams are ranked by wins, ties broken by total points (no playoffs: most wins takes the league).

Two ways to draw weekly scores:
  quick     the season model: score ~ Normal(expected points, spread) per team-week (seconds)
  detailed  the game-by-game engine (sim/engine.py): every NBA game, minutes, usage, rest...
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradecalc.trade.matchup import week_dist


def _finished_results(state) -> tuple[dict, dict, dict, dict]:
    """Record so far from weeks that are over: wins, losses, ties and points per team."""
    w, l, t, pts = ({tid: 0.0 for tid in state.rosters} for _ in range(4))
    done = {wk.number for wk in state.cal.weeks if wk.end < state.today}
    for m in getattr(state.league, "matchups", []):
        if m["week"] not in done or m["away"] is None:
            continue
        h, a = m["home"], m["away"]
        pts[h] += m.get("home_total", 0.0)
        pts[a] += m.get("away_total", 0.0)
        res = m.get("winner")
        if res == "HOME":
            w[h] += 1; l[a] += 1
        elif res == "AWAY":
            w[a] += 1; l[h] += 1
        elif res == "TIE":
            t[h] += 1; t[a] += 1
    return w, l, t, pts


def _weekly_scores_quick(state, week, rng, n, team_form: dict | None = None) -> dict[int, np.ndarray]:
    out = {}
    for tid, roster in state.rosters.items():
        d = week_dist(roster, week, state, state.today)
        bank = state.banked(tid, week)[0] if hasattr(state, "banked") else 0.0
        f = team_form[tid] if team_form else 1.0
        out[tid] = rng.normal(d.M, np.sqrt(max(d.V, 1e-9)), n) * f + bank
    return out


def _team_form(state, n, rng) -> dict:
    """Quick method's version of the season layer: one persistent form multiplier per team per
    simulated season (player form averaged over ~10 starters), shrinking as games are played."""
    from tradecalc.sim.engine import SEASON_LAYER

    played = [p.n for t in state.rosters for p in (state.players[i] for i in state.rosters[t])]
    gp = float(np.mean(played)) if played else 0.0
    sd = SEASON_LAYER["talent_sd"] * np.sqrt(SEASON_LAYER["talent_k"] / (SEASON_LAYER["talent_k"] + gp)) / np.sqrt(10)
    return {t: np.exp(rng.normal(0, sd, n) - sd ** 2 / 2) for t in state.rosters}


def _weekly_scores_engine(state, week, world, coaches_now, n, seed, layer=None, week_index: int = 0) -> dict[int, np.ndarray]:
    from tradecalc.model.names import norm_name
    from tradecalc.sim.engine import fantasy_week, live_avail, simulate_games
    from tradecalc.teams import to_nba
    from tradecalc.trade.lineup import slot_list

    cfg = state.cfg
    days = [d for d in week.days() if d >= state.today]
    games = [(g["date"], to_nba(g["home"]), to_nba(g["away"])) for g in state.league.pro_games if g["date"] in set(days)]
    rosters, elig = {}, {}
    for tid, ids in state.rosters.items():
        keys = []
        for i in ids:
            k = norm_name(state.players[i].name)
            if k in world.players.index:
                keys.append(k)
                elig[k] = tuple(state.players[i].positions) or ("SF", "PF")
                tm = world.players.loc[k, "team"]
                if k not in world.rotation.get(tm, []):
                    world.rotation.setdefault(tm, []).append(k)
        rosters[tid] = keys
    avail = live_avail(world, state, coaches_now)
    if layer is not None:  # outside long absences a player is available a bit more often
        base_avail = avail
        avail = lambda k, t, d: min(base_avail(k, t, d) * layer.short_scale, 0.99)  # noqa: E731
    sims = simulate_games(world, games, avail, n=n, seed=seed, interest=set(elig))
    if layer is not None:
        layer.apply(sims, week_index)
    res = fantasy_week(sims, rosters, days, slot_list(cfg), elig, cfg["roster"]["slot_eligibility"], n)
    return {tid: res.totals[tid] + (state.banked(tid, week)[0] if hasattr(state, "banked") else 0.0)
            for tid in state.rosters}


def forecast(state, n: int = 10000, seed: int | None = 0, method: str = "quick", world=None,
             coaches_now=None, progress=None) -> pd.DataFrame:
    """One row per team, best projected finish first."""
    rng = np.random.default_rng(seed)
    tie = state.cfg["league"]["tie_value"]
    teams = sorted(state.rosters)
    idx = {t: i for i, t in enumerate(teams)}
    w0, l0, t0, p0 = _finished_results(state)
    wins = np.tile(np.array([w0[t] + tie * t0[t] for t in teams]), (n, 1))
    points = np.tile(np.array([p0[t] for t in teams]), (n, 1))
    remaining = [wk for wk in state.cal.weeks if wk.end >= state.today]
    # Season layer: luck that persists all season (form, multi-week injuries); backtested in
    # sim/forecast_test.py. Without it, season ranges were far too narrow.
    layer, team_form = None, None
    if method == "detailed":
        from tradecalc.model.names import norm_name
        from tradecalc.sim.engine import SeasonLayer

        keys = sorted({norm_name(state.players[i].name) for t in state.rosters for i in state.rosters[t]})
        gp = {norm_name(p.name): p.n for p in state.players.values()}
        layer = SeasonLayer(keys, len(remaining), n, rng, gp)
    else:
        team_form = _team_form(state, n, rng)
    for j, wk in enumerate(remaining):
        if method == "detailed":
            scores = _weekly_scores_engine(state, wk, world, coaches_now, n, None if seed is None else seed + wk.number,
                                           layer, j)
        else:
            scores = _weekly_scores_quick(state, wk, rng, n, team_form)
        for t in teams:
            points[:, idx[t]] += scores[t]
        seen = set()
        for t in teams:
            o = state.cal.opponents.get(t, {}).get(wk.number)
            if o is None or (o, t) in seen:
                continue
            seen.add((t, o))
            a, b = scores[t], scores[o]
            wins[:, idx[t]] += (a > b) + tie * (a == b)
            wins[:, idx[o]] += (b > a) + tie * (a == b)
        if progress:
            progress(j + 1, len(remaining))
    # rank by wins, then total points (1 = first place)
    key = wins * 1e7 + points
    rank = (-key).argsort(axis=1).argsort(axis=1) + 1
    nt = len(teams)
    rows = []
    for t in teams:
        i = idx[t]
        rows.append({
            "team_id": t, "team": state.team_name(t),
            "record": f"{int(w0[t])}-{int(l0[t])}" + (f"-{int(t0[t])}" if t0[t] else ""),
            "exp_wins": float(wins[:, i].mean()),
            "wins_p10": float(np.percentile(wins[:, i], 10)), "wins_p90": float(np.percentile(wins[:, i], 90)),
            "p_first": float((rank[:, i] == 1).mean()), "p_top3": float((rank[:, i] <= 3).mean()),
            "p_last": float((rank[:, i] == nt).mean()), "avg_rank": float(rank[:, i].mean()),
            "exp_points": float(points[:, i].mean()),
        })
    df = pd.DataFrame(rows).sort_values(["avg_rank", "exp_wins"], ascending=[True, False]).reset_index(drop=True)
    df.index = df.index + 1
    df.attrs.update(n=n, method=method, weeks_left=len(remaining),
                    rank_dist={t: np.bincount(rank[:, idx[t]] - 1, minlength=nt) / n for t in teams})
    return df


def last_place_line(df: pd.DataFrame) -> str:
    return f"{df.iloc[-1]['team']} will be eating poop"
