"""Backtest of the league forecast on past NBA seasons with mock fantasy leagues.

Your ESPN league has no history, so this builds leagues the way ESPN leagues form:
  - 12 teams x 13 players, snake draft by ESPN's *preseason* projection for that season
    (with some randomness so leagues differ), league scoring, 24 weeks, round-robin schedule;
  - ground truth: what those exact rosters actually scored, from real box scores, with each
    manager starting his best available players every day (by his as-of-that-day FPPG);
  - forecasts at checkpoints (preseason and after weeks 6/12/18) using only information
    available on that date: finished weeks count as they happened, the rest is simulated
    by the projection engine (sim/engine.py).

Lineups here use a fast vectorised greedy fill (guards to PG/SG/G, forwards to SF/PF/F,
centers to C, everyone to UTIL), the same rule for forecasts and for ground truth.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from tradecalc.model.names import norm_name
from tradecalc.schedule.calendar import build_weeks
from tradecalc.sim.engine import World, _ewma, _rates_by_key, _team_strength, build_world_history, rest_factor, \
    simulate_games
from tradecalc.sim.defense import defender_impacts, estimate_defense
from tradecalc.sim.engine import STATS

N_TEAMS, ROSTER = 12, 13
GROUP_SLOTS = {"G": 3, "F": 3, "C": 1}     # PG+SG+G, SF+PF+F, C
UTIL = 3


# --- league setup ---------------------------------------------------------------------------

def season_weeks(H, season: str, n_weeks: int = 24):
    tl = H.team_logs[H.team_logs.season == season]
    days = sorted({date.fromisoformat(d) for d in tl.game_date})
    return build_weeks(days[0], days[-1], n_weeks, set(days))


def round_robin(n_teams: int, n_weeks: int) -> list[list[tuple[int, int]]]:
    teams = list(range(n_teams))
    rounds = []
    for r in range(n_teams - 1):
        rounds.append([(teams[i], teams[-1 - i]) for i in range(n_teams // 2)])
        teams = [teams[0]] + [teams[-1]] + teams[1:-1]
    return [rounds[w % len(rounds)] for w in range(n_weeks)]


PRESEASON_PROJ: dict[str, dict] = {}   # key -> ESPN preseason projection (per game), filled by preseason_values


def preseason_values(client, espn_season: int, scoring: dict) -> dict[str, float]:
    """ESPN's preseason projection (fantasy points per game x projected games) per player key.
    Also keeps the projections themselves (minutes, per-game stats) for the preseason world."""
    from tradecalc.ingest.league import _projection
    from tradecalc.model.scoring import fantasy_points

    pool = client.player_pool(limit=600, season=espn_season, league_scoped=False)
    out = {}
    PRESEASON_PROJ.clear()
    for e in pool.get("players", []):
        pr = _projection(e["player"], espn_season)
        if pr.get("GP"):
            k = norm_name(e["player"]["fullName"])
            out[k] = fantasy_points(pr, scoring) * min(pr["GP"], 82) / 82
            PRESEASON_PROJ[k] = pr
    return out


def mock_drafts(values: dict[str, float], playable: set[str], n_leagues: int, seed: int = 0) -> list[list[list[str]]]:
    """Snake drafts by preseason value; each pick takes one of the best 3 left (60/25/15%)."""
    rng = np.random.default_rng(seed)
    board = [k for k in sorted(values, key=lambda k: -values[k]) if k in playable]
    leagues = []
    for _ in range(n_leagues):
        left = list(board)
        teams = [[] for _ in range(N_TEAMS)]
        for rnd in range(ROSTER):
            order = range(N_TEAMS) if rnd % 2 == 0 else reversed(range(N_TEAMS))
            for t in order:
                j = min(rng.choice(3, p=[0.6, 0.25, 0.15]), len(left) - 1)
                teams[t].append(left.pop(j))
        leagues.append(teams)
    return leagues


# --- lineups (vectorised greedy, same rule for truth and forecast) -------------------------

def greedy_lineup(active: np.ndarray, prio: np.ndarray, groups: list[str]) -> np.ndarray:
    """active: n x m (bool), prio: m. Returns n x m bool of players started."""
    n, m = active.shape
    order = np.argsort(-prio)
    cap = {g: np.full(n, c) for g, c in GROUP_SLOTS.items()}
    util = np.full(n, UTIL)
    started = np.zeros((n, m), dtype=bool)
    for j in order:
        a = active[:, j]
        g = groups[j]
        ded = a & (cap[g] > 0)
        cap[g] = cap[g] - ded
        use_util = a & ~ded & (util > 0)
        util = util - use_util
        started[:, j] = ded | use_util
    return started


# --- ground truth ----------------------------------------------------------------------------

def actual_weekly(H, league: list[list[str]], weeks, season: str, pos_of) -> np.ndarray:
    """weeks x teams matrix of real fantasy points (best available lineup each day, ranked by
    each player's FPPG through the previous day; last season's FPPG before he has played)."""
    L = H.logs[H.logs.season == season].copy()
    L["key"] = L.player_name.map(norm_name)
    L["d"] = L.game_date.map(date.fromisoformat)
    prev = H.logs[H.logs.season == _prev(season)]
    prev_fp = prev.assign(key=prev.player_name.map(norm_name)).groupby("key")["fp"].mean()
    keys = {k for team in league for k in team}
    sub = L[L.key.isin(keys)].sort_values("d")
    sub["cum_fp"] = sub.groupby("key")["fp"].cumsum() - sub["fp"]
    sub["cum_n"] = sub.groupby("key").cumcount()
    sub["prio"] = np.where(sub.cum_n > 0, sub.cum_fp / sub.cum_n.clip(lower=1), sub.key.map(prev_fp).fillna(20.0))
    by_day = {d: g.set_index("key") for d, g in sub.groupby("d")}
    out = np.zeros((len(weeks), N_TEAMS))
    for wi, wk in enumerate(weeks):
        for d in wk.days():
            g = by_day.get(d)
            if g is None:
                continue
            for t, team in enumerate(league):
                ks = [k for k in team if k in g.index]
                if not ks:
                    continue
                act = np.ones((1, len(ks)), dtype=bool)
                st = greedy_lineup(act, g.loc[ks, "prio"].to_numpy(float), [pos_of(k) for k in ks])[0]
                out[wi, t] += g.loc[ks, "fp"].to_numpy()[st].sum()
    return out


def _prev(season: str) -> str:
    y = int(season[:4]) - 1
    return f"{y}-{str(y + 1)[2:]}"


def standings(wins: np.ndarray, points: np.ndarray) -> np.ndarray:
    """rank (1 = best) per row of wins / points arrays shaped (..., teams)."""
    key = wins * 1e7 + points
    return (-key).argsort(axis=-1).argsort(axis=-1) + 1


def play_weeks(scores: np.ndarray, sched, start_week: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """scores: (n, weeks, teams) or (weeks, teams). Returns wins, points summed from start_week."""
    s = scores if scores.ndim == 3 else scores[None]
    n, W, T = s.shape
    wins = np.zeros((n, T))
    for w in range(start_week, W):
        for a, b in sched[w]:
            sa, sb = s[:, w, a], s[:, w, b]
            wins[:, a] += (sa > sb) + 0.5 * (sa == sb)
            wins[:, b] += (sb > sa) + 0.5 * (sa == sb)
    return wins, s[:, start_week:, :].sum(axis=1)


# --- forecast at a checkpoint ----------------------------------------------------------------

def build_world_preseason(H, params: dict, season: str, prev: str, values: dict[str, float]) -> World:
    """World on opening night: rosters = who suited up for each team in its first 5 games
    (rosters are known before the season), minutes and rates from last season."""
    L = H.logs
    cur, prv = L[L.season == season], L[L.season == prev]
    first_days = cur.groupby("team")["game_date"].apply(lambda s: sorted(set(s))[:5])
    rows = []
    rates = _rates_by_key(H, cur.iloc[0:0], prv, params)
    lg_rates = rates.attrs["league"]
    prev_mpg = prv.assign(key=prv.player_name.map(norm_name)).groupby("key")["min"].mean()
    for team, days in first_days.items():
        g = cur[(cur.team == team) & (cur.game_date.isin(days))]
        from tradecalc.sim.engine import _rates_from_projection

        for k, name in {norm_name(n): n for n in g.player_name}.items():
            pos = H.pos_of(name) or "F"
            proj = PRESEASON_PROJ.get(k, {})
            views = [float(prev_mpg[k])] if k in prev_mpg.index else []
            if proj.get("MIN"):
                views.append(float(proj["MIN"]))      # like the live engine: last season + ESPN minutes
            exp_min = float(np.mean(views)) if views else 12.0
            row = {"key": k, "name": name, "team": team, "pos": pos, "age": np.nan, "exp_min": exp_min}
            if k in rates.index and k in prev_mpg.index:
                row.update(rates.loc[k, STATS].to_dict())
            else:
                row.update(_rates_from_projection(proj) or lg_rates.get(pos, lg_rates["ALL"]))
            rows.append(row)
    P = pd.DataFrame(rows).drop_duplicates("key").set_index("key")
    from tradecalc.sim.engine import trim_rotation

    for t, sub in P.groupby("team"):
        top = sub.sort_values("exp_min", ascending=False).index[:13]
        P.loc[top, "exp_min"] = trim_rotation(P.loc[top, "exp_min"].to_numpy(float), params.get("rotation_exp_min_sum", 282.0))
    rotation = {t: sub[sub.exp_min >= 6].sort_values("exp_min", ascending=False).index[:13].tolist()
                for t, sub in P.groupby("team")}
    tl = H.team_logs
    dm = estimate_defense(tl.iloc[0:0], params["def_k"], tl[tl.season == prev])
    imp = defender_impacts(H.onoff[H.onoff.season == prev]) if len(H.onoff) else pd.Series(dtype=float)
    w = World(date.fromisoformat(cur.game_date.min()), P, rotation, dm, _team_strength(tl[tl.season == prev]), {},
              params, H.scoring)
    from tradecalc.sim.engine import _b2b_days_from_logs

    w.b2b_days = _b2b_days_from_logs(tl[tl.season == season])
    return w


def _avail_fn(H, W: World, params, asof: str, season: str, prev: str):
    """P(plays): games-played rate (this season so far, else last season), x back-to-back rest."""
    L = H.logs
    cur = L[(L.season == season) & (L.game_date < asof)]
    prv = L[L.season == prev]
    team_games_prev = 82
    rate_prev = (prv.assign(key=prv.player_name.map(norm_name)).groupby("key").size() / team_games_prev).clip(upper=0.95)
    cur_k = cur.assign(key=cur.player_name.map(norm_name))
    apps = cur_k.groupby("key").size()
    tl = H.team_logs[(H.team_logs.season == season) & (H.team_logs.game_date < asof)]
    tg = tl.groupby("team").size()

    def avail(k, team, d):
        a = apps.get(k, 0)
        n_team = tg.get(team, 0)
        base_prev = rate_prev.get(k, 0.85)
        base = (10 * base_prev + a) / (10 + max(n_team, a)) if n_team else base_prev
        row = W.players.loc[k] if k in W.players.index else None
        f = rest_factor(params, 26 if row is None else (row["age"] if row["age"] == row["age"] else 26),
                        20 if row is None else row["exp_min"], d in W.b2b_days.get(team, set()))
        return float(np.clip(min(base, 0.97) * f, 0, 1))
    return avail


def forecast_checkpoint(H, params, leagues, weeks, sched, season, prev, checkpoint: int, actual: list[np.ndarray],
                        values: dict, n: int = 400, seed: int = 0, layer_cfg: dict | None = None,
                        use_layer: bool = True) -> list[dict]:
    """Forecast every league at `checkpoint` (number of weeks already played)."""
    rng_seed = seed + checkpoint
    asof = weeks[checkpoint].start.isoformat() if checkpoint < len(weeks) else weeks[-1].end.isoformat()
    W = build_world_preseason(H, params, season, prev, values) if checkpoint == 0 else \
        build_world_history(H, params, asof, season, prev)
    avail0 = _avail_fn(H, W, params, asof, season, prev)
    interest = {k for lg in leagues for team in lg for k in team}
    layer = None
    if use_layer:
        from tradecalc.sim.engine import SeasonLayer

        cur = H.logs[(H.logs.season == season) & (H.logs.game_date < asof)]
        gp = cur.assign(key=cur.player_name.map(norm_name)).groupby("key").size().to_dict()
        layer = SeasonLayer(sorted(interest), len(weeks), n, np.random.default_rng(seed + 991 + checkpoint), gp, layer_cfg)
        avail = lambda k, t, d: min(avail0(k, t, d) * layer.short_scale, 0.99)  # noqa: E731
    else:
        avail = avail0
    for k in interest:  # make sure every rostered player is simulated if he is in the world
        if k in W.players.index:
            t = W.players.loc[k, "team"]
            if k not in W.rotation.get(t, []):
                W.rotation.setdefault(t, []).append(k)
    tl = H.team_logs[(H.team_logs.season == season) & H.team_logs.home]
    sim_scores = [np.zeros((n, len(weeks), N_TEAMS)) for _ in leagues]
    groups_of = {k: (W.players.loc[k, "pos"] if k in W.players.index else (H.pos_of(k) or "F")) for k in interest}
    for wi in range(checkpoint, len(weeks)):
        wk = weeks[wi]
        g = tl[(tl.game_date >= wk.start.isoformat()) & (tl.game_date <= wk.end.isoformat())]
        games = [(date.fromisoformat(d), h, a) for d, h, a in zip(g.game_date, g.team, g.opp)]
        S = simulate_games(W, games, avail, n=n, seed=rng_seed + wi, interest=interest)
        if layer is not None:
            layer.apply(S, wi)
        for d in wk.days():
            for li, lg in enumerate(leagues):
                for t, team in enumerate(lg):
                    ks = [k for k in team if (k, d) in S.fp]
                    if not ks:
                        continue
                    act = np.stack([S.played[(k, d)] for k in ks], axis=1)
                    fp = np.stack([S.fp[(k, d)] for k in ks], axis=1)
                    prio = np.array([S.box_mean[(k, d)]["fp"] for k in ks])
                    st = greedy_lineup(act, prio, [groups_of[k] for k in ks])
                    sim_scores[li][:, wi, t] += (fp * st).sum(axis=1)
    out = []
    for li, lg in enumerate(leagues):
        s = sim_scores[li]
        s[:, :checkpoint, :] = actual[li][None, :checkpoint, :]          # finished weeks are real
        wins, pts = play_weeks(s, sched)
        rank = standings(wins, pts)
        a_wins, a_pts = play_weeks(actual[li], sched)
        a_rank = standings(a_wins, a_pts)[0]
        for t in range(N_TEAMS):
            out.append({"league": li, "checkpoint": checkpoint, "team": t,
                        "exp_wins": float(wins[:, t].mean()), "p10": float(np.percentile(wins[:, t], 10)),
                        "p90": float(np.percentile(wins[:, t], 90)), "p_first": float((rank[:, t] == 1).mean()),
                        "p_last": float((rank[:, t] == N_TEAMS).mean()), "avg_rank": float(rank[:, t].mean()),
                        "exp_points": float(pts[:, t].mean()),
                        "actual_wins": float(a_wins[0, t]), "actual_rank": int(a_rank[t]),
                        "actual_points": float(a_pts[0, t]),
                        "naive_value": float(sum(values.get(k, 0) for k in lg[t])),
                        "wins_so_far": float(play_weeks(actual[li][:checkpoint], sched[:checkpoint])[0][0, t])
                        if checkpoint else 0.0})
    return out


def run(H, params, client, season: str, espn_season: int, n_leagues: int = 12, checkpoints=(0, 6, 12, 18),
        n: int = 400, seed: int = 0, log=print, layer_cfg: dict | None = None, use_layer: bool = True) -> dict:
    prev = _prev(season)
    weeks = season_weeks(H, season)
    sched = round_robin(N_TEAMS, len(weeks))
    values = preseason_values(client, espn_season, H.scoring)
    playable = {norm_name(n) for n in H.logs[H.logs.season == season].player_name}
    leagues = mock_drafts(values, playable, n_leagues, seed)
    log(f"{season}: {n_leagues} mock leagues, {len(weeks)} weeks")
    actual = [actual_weekly(H, lg, weeks, season, lambda k: H.pos_of(k) or "F") for lg in leagues]
    rows = []
    for cp in checkpoints:
        log(f"  forecasting after week {cp} ...")
        rows += forecast_checkpoint(H, params, leagues, weeks, sched, season, prev, cp, actual, values, n, seed,
                                    layer_cfg, use_layer)
    res = pd.DataFrame(rows)
    names = {k: k for lg in leagues for team in lg for k in team}
    return {"season": season, "results": res, "leagues": leagues, "weeks": [(w.start.isoformat(), w.end.isoformat()) for w in weeks],
            "actual_weekly": [a.tolist() for a in actual], "schedule": sched, "n_sims": n}


# --- scoring the forecast --------------------------------------------------------------------

def metrics(res: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import spearmanr

    rows = []
    for cp, g in res.groupby("checkpoint"):
        rho = np.mean([spearmanr(-x.avg_rank, -x.actual_rank)[0] for _, x in g.groupby("league")])
        naive = np.mean([spearmanr(x.naive_value + x.wins_so_far * 1e3, -x.actual_rank)[0] for _, x in g.groupby("league")])
        champ = g[g.actual_rank == 1]
        last = g[g.actual_rank == N_TEAMS]
        brier_first = float(((g.p_first - (g.actual_rank == 1)) ** 2).mean())
        rows.append({
            "checkpoint": f"after week {cp}" if cp else "preseason",
            "rank_corr": rho, "rank_corr_naive": naive,
            "wins_MAE": float((g.exp_wins - g.actual_wins).abs().mean()),
            "wins_bias": float((g.exp_wins - g.actual_wins).mean()),
            "in_80pct_range": float(((g.actual_wins >= g.p10) & (g.actual_wins <= g.p90)).mean()),
            "champ_prob_given": float(champ.p_first.mean()),
            "champ_in_pred_top3": float(np.mean([
                (x.avg_rank.rank(method="first")[x.actual_rank == 1] <= 3).any() for _, x in g.groupby("league")])),
            "brier_champion": brier_first, "brier_coinflip": float(((1 / N_TEAMS - (g.actual_rank == 1)) ** 2).mean()),
            "last_prob_given": float(last.p_last.mean()),
        })
    return pd.DataFrame(rows)


# --- report for the app ----------------------------------------------------------------------

REPORT_PATH = None  # set lazily (config/forecast_test.json)


def report_path():
    from tradecalc.sim.params import PARAMS_PATH

    return PARAMS_PATH.parent / "forecast_test.json"


def build_report(H, before: dict, after: dict, tuning: list[dict], fit_seasons: list[str]) -> dict:
    """Everything the Forecast tests page shows, as plain JSON."""
    from datetime import date as _date

    names = H.logs.assign(key=H.logs.player_name.map(norm_name)).groupby("key")["player_name"].last().to_dict()
    lgs = [[[names.get(k, k) for k in team] for team in lg] for lg in after["leagues"]]
    cols = ["league", "checkpoint", "team", "exp_wins", "p10", "p90", "p_first", "p_last", "avg_rank",
            "actual_wins", "actual_rank", "naive_value", "wins_so_far"]
    return {
        "run_on": __import__("tradecalc.clock", fromlist=["league_today"]).league_today().isoformat(), "test_season": after["season"], "fit_seasons": fit_seasons,
        "n_leagues": len(after["leagues"]), "n_sims": after["n_sims"],
        "metrics_before": metrics(before["results"]).to_dict("records"),
        "metrics_after": metrics(after["results"]).to_dict("records"),
        "rows_before": before["results"][cols].round(4).to_dict("records"),
        "rows_after": after["results"][cols].round(4).to_dict("records"),
        "leagues": lgs, "tuning": tuning,
    }


def full_backtest(H, conn, client, coaches, test_season: str, espn_season: int, n_leagues: int = 12,
                  talent_grid=(0.2, 0.25, 0.3, 0.35), log=print) -> dict:
    """Fit on the two seasons before `test_season`, tune the form size on the season before,
    validate before/after on `test_season`. Returns the report (also saved to JSON)."""
    import json

    from tradecalc.sim.engine import SEASON_LAYER
    from tradecalc.sim.params import fit_noise, fit_params
    from tradecalc.sim.validate import fit_calibration

    tune_season = _prev(test_season)
    s0 = _prev(tune_season)
    log(f"Fitting the engine on {s0} + {tune_season} ...")
    p = fit_params(H, coaches, [s0, tune_season])
    p.update(fit_noise(H, p, tune_season, s0))
    mondays = [d for d in sorted(H.team_logs[H.team_logs.season == tune_season].game_date.unique())
               if pd.Timestamp(d).weekday() == 0][4::5]
    p.update(fit_calibration(H, p, tune_season, s0, mondays))
    p["form_sd"] = 0.18
    tuning = []
    for sd in talent_grid:
        log(f"Tuning on {tune_season}: season form {sd} ...")
        out = run(H, p, client, tune_season, espn_season - 1, n_leagues=max(6, n_leagues // 2), n=300,
                  log=lambda m: None, layer_cfg={"talent_sd": sd})
        m = metrics(out["results"])
        tuning.append({"talent_sd": sd, "coverage": list(m.in_80pct_range.round(3)),
                       "brier_champion": list(m.brier_champion.round(4)), "checkpoints": list(m.checkpoint)})
    best = min(tuning, key=lambda r: abs(np.mean(r["coverage"]) - 0.8))
    log(f"Chosen season form {best['talent_sd']} (current setting {SEASON_LAYER['talent_sd']}). Validating on {test_season} ...")
    before = run(H, p, client, test_season, espn_season, n_leagues=n_leagues, log=lambda m: None, use_layer=False)
    after = run(H, p, client, test_season, espn_season, n_leagues=n_leagues, log=lambda m: None,
                layer_cfg={"talent_sd": best["talent_sd"]})
    rep = build_report(H, before, after, tuning, [s0, tune_season])
    rep["chosen_talent_sd"] = best["talent_sd"]
    report_path().write_text(json.dumps(rep))
    return rep


def load_report() -> dict | None:
    import json

    try:
        return json.loads(report_path().read_text())
    except (FileNotFoundError, ValueError):
        return None
