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


def result_of(m: dict) -> str | None:
    """HOME, AWAY or TIE for a matchup whose week is over. ESPN can leave it UNDECIDED for hours after
    the week ends; the point totals decide it then (None if there are no totals yet)."""
    res = m.get("winner")
    if res in ("HOME", "AWAY", "TIE"):
        return res
    h, a = float(m.get("home_total") or 0.0), float(m.get("away_total") or 0.0)
    if h == a == 0:
        return None
    return "HOME" if h > a else "AWAY" if a > h else "TIE"


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
        res = result_of(m)
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


def luck(state, predictions: pd.DataFrame | None = None) -> pd.DataFrame:
    """The real standings so far, next to the record each team was expected to have. Ties count tie_value.

      exp_record    a win for every week the model favored the team before the week started (win chance
                    over 50%), a loss when it was the underdog, a tie for an exact toss-up: "supposed to be
                    2-0" (the season tracker's frozen predictions: tracking/<season>/matchups.csv, pre-week)
      vs_expected   wins minus expected wins by that record (+1: won a week it was supposed to lose)
      surprises     every week that went against the favorite: week, result (won / lost / tied), the
                    team's pre-week win chance, opponent
      exp_wins      the same predictions added up as probabilities (a 60% favorite counts 0.6 of a win)
      luck          wins minus exp_wins, over the weeks that have a prediction
      allplay_wins  each week, the share of the other teams the team outscored: its record if it had
                    played everyone (schedule luck: a high score that loses still counts)
      allplay_luck  wins minus allplay_wins
    One row per team, ranked like the league: wins, then points scored."""
    from collections import defaultdict

    tie = state.cfg["league"]["tie_value"]
    done = {wk.number for wk in state.cal.weeks if wk.end < state.today}
    pre = {}
    if predictions is not None and len(predictions):
        pw = predictions[predictions.kind == "pre-week"] if "kind" in predictions else predictions
        pre = {(int(w), int(t)): float(p) for w, t, p in zip(pw.week, pw.team_id, pw.win_prob)}
    rec = {t: {"W": 0, "L": 0, "T": 0, "pf": 0.0, "exp": 0.0, "won_pred": 0.0, "ap": 0.0, "EW": 0, "EL": 0, "ET": 0,
               "exp_won": 0.0, "surprises": []} for t in state.rosters}
    scores = defaultdict(dict)
    for m in getattr(state.league, "matchups", []):
        res = result_of(m) if m["week"] in done and m.get("away") is not None else None
        if res is None:
            continue
        for me, opp, side, pts in ((m["home"], m["away"], "HOME", m.get("home_total")),
                                   (m["away"], m["home"], "AWAY", m.get("away_total"))):
            if me not in rec:
                continue
            r = rec[me]
            won = 1.0 if res == side else tie if res == "TIE" else 0.0
            r["W"] += res == side
            r["T"] += res == "TIE"
            r["L"] += res not in (side, "TIE")
            r["pf"] += float(pts or 0.0)
            if (m["week"], me) in pre:
                p = pre[(m["week"], me)]
                r["exp"] += p
                r["won_pred"] += won
                expected = "won" if p > 0.5 else "lost" if p < 0.5 else None     # None: an exact toss-up
                r["EW"] += expected == "won"
                r["EL"] += expected == "lost"
                r["ET"] += expected is None
                r["exp_won"] += 1.0 if expected == "won" else tie if expected is None else 0.0
                outcome = "won" if res == side else "tied" if res == "TIE" else "lost"
                if expected is not None and outcome != expected:
                    r["surprises"].append({"week": int(m["week"]), "result": outcome, "p": p, "opp": opp})
            scores[m["week"]][me] = float(pts or 0.0)
    for week_scores in scores.values():
        for t, s in week_scores.items():
            others = [v for u, v in week_scores.items() if u != t]
            if others:
                rec[t]["ap"] += (sum(s > v for v in others) + tie * sum(s == v for v in others)) / len(others)
    rows = []
    for t, r in rec.items():
        wins = r["W"] + tie * r["T"]
        rows.append({"team_id": t, "team": state.team_name(t),
                     "record": f"{r['W']}-{r['L']}" + (f"-{r['T']}" if r["T"] else ""), "wins": wins,
                     "exp_record": f"{r['EW']}-{r['EL']}" + (f"-{r['ET']}" if r["ET"] else ""),
                     "vs_expected": r["won_pred"] - r["exp_won"],
                     "surprises": sorted(r["surprises"], key=lambda x: x["week"]),
                     "exp_wins": r["exp"], "luck": r["won_pred"] - r["exp"], "allplay_wins": r["ap"],
                     "allplay_luck": wins - r["ap"], "points_for": r["pf"]})
    df = pd.DataFrame(rows).sort_values(["wins", "points_for"], ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    return df
