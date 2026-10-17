"""Historical league states, real outcomes, and the trade / season-model tests.

A HistLeague is a mock 12-team league on a past season. At a checkpoint (start of fantasy
week c) it can build a LeagueState that the real trade calculator code (trade/evaluate.py,
trade/matchup.py) runs on unchanged, using only information known at that date:
  mu     the season model's blend: (k * prior + n * live) / (k + n), with prior = ESPN's
         preseason projection per game (last season's FPPG if none), live = FPPG so far
  sigma  shrunk per-game spread; availability = games-played rate so far, shrunk to last season's
Real outcomes come from real box scores: each team, each day, starts its best available
players (greedy by position group, ranked by FPPG through the previous day).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tradecalc.model.availability import Availability, calibrated_rate, espn_games_blend
from tradecalc.model.names import norm_name
from tradecalc.model.values import LeagueState, PlayerState
from tradecalc.model.variance import shrunk_sigma
from tradecalc.sim.forecast_test import (N_TEAMS, PRESEASON_PROJ, greedy_lineup, mock_drafts, preseason_values,
                                         round_robin, season_weeks)
from tradecalc.sim.history import prev_season

GROUP_POS = {"G": ("PG", "SG"), "F": ("SF", "PF"), "C": ("C",)}


class HistCal:
    def __init__(self, weeks, games_by_day, opponents):
        self.weeks, self.games_by_day, self.opponents = weeks, games_by_day, opponents
        self.first_day = weeks[0].start

    def team_plays(self, team, d):
        return team in self.games_by_day.get(d, ())

    def week_of(self, d):
        return next((w for w in self.weeks if w.start <= d <= w.end), None)

    def remaining_weeks(self, today):
        return [w for w in self.weeks if w.end >= today]

    def team_games_in_week(self, team, week, from_day=None):
        return sum(self.team_plays(team, d) for d in week.days() if from_day is None or d >= from_day)


@dataclass
class SeasonCtx:
    """Everything about one past season that the tests share."""
    season: str
    weeks: list
    sched: list
    values: dict                 # draft value (ESPN preseason FP/game x games/82)
    proj_fp: dict                # ESPN preseason FP per game
    proj_gp: dict                # ESPN preseason projected games played
    logs: pd.DataFrame           # this season's player-games (key, d, team, fp, ...)
    prev_fppg: pd.Series
    prev_rate: pd.Series         # last season games / 82
    games_by_day: dict
    pos_of: callable
    day_data: dict = field(default_factory=dict)   # date -> {key: (fp, prio)}
    prev_team: dict = field(default_factory=dict)  # last season's last team (team before any game)


def season_ctx(H, client, season: str) -> SeasonCtx:
    from tradecalc.model.scoring import fantasy_points

    espn = int(season[:4]) + 1
    values = preseason_values(client, espn, H.scoring)
    proj_fp = {k: fantasy_points(pr, H.scoring) for k, pr in PRESEASON_PROJ.items()}
    weeks = season_weeks(H, season)
    sched = round_robin(N_TEAMS, len(weeks))
    L = H.logs[H.logs.season == season].copy()
    L["key"] = L.player_name.map(norm_name)
    L["d"] = L.game_date.map(date.fromisoformat)
    P = H.logs[H.logs.season == prev_season(season)]
    P = P.assign(key=P.player_name.map(norm_name))
    prev_fppg = P.groupby("key")["fp"].mean()
    prev_rate = (P.groupby("key").size() / 82).clip(upper=0.95)
    tl = H.team_logs[H.team_logs.season == season]
    gbd = defaultdict(set)
    for d_, t in zip(tl.game_date, tl.team):
        gbd[date.fromisoformat(d_)].add(t)
    L = L.sort_values("d")
    L["cum_fp"] = L.groupby("key")["fp"].cumsum() - L["fp"]
    L["cum_n"] = L.groupby("key").cumcount()
    L["prio"] = np.where(L.cum_n > 0, L.cum_fp / L.cum_n.clip(lower=1), L.key.map(prev_fppg).fillna(20.0))
    day_data = {d_: dict(zip(g.key, zip(g.fp, g.prio))) for d_, g in L.groupby("d")}
    proj_gp = {k: pr.get("GP") for k, pr in PRESEASON_PROJ.items()}
    return SeasonCtx(season, weeks, sched, values, proj_fp, proj_gp, L, prev_fppg, prev_rate, dict(gbd),
                     lambda k: H.pos_of(k) or "F", day_data, P.sort_values("game_date").groupby("key")["team"].last().to_dict())


def team_weekly_actual(ctx: SeasonCtx, roster: list[str], week_idx) -> np.ndarray:
    """Real fantasy points per week for a roster (best available lineup each day)."""
    out = np.zeros(len(week_idx))
    groups = {k: ctx.pos_of(k) for k in roster}
    for j, wi in enumerate(week_idx):
        for d in ctx.weeks[wi].days():
            dd = ctx.day_data.get(d)
            if not dd:
                continue
            ks = [k for k in roster if k in dd]
            if not ks:
                continue
            fp = np.array([dd[k][0] for k in ks])
            pr = np.array([dd[k][1] for k in ks])
            st = greedy_lineup(np.ones((1, len(ks)), dtype=bool), pr, [groups[k] for k in ks])[0]
            out[j] += fp[st].sum()
    return out


def actual_matrix(ctx: SeasonCtx, league: list[list[str]]) -> np.ndarray:
    W = range(len(ctx.weeks))
    return np.stack([team_weekly_actual(ctx, team, W) for team in league], axis=1)   # weeks x teams


def wins_from(scores: np.ndarray, sched, start: int) -> np.ndarray:
    w = np.zeros(scores.shape[1])
    for wi in range(start, scores.shape[0]):
        for a, b in sched[wi]:
            w[a] += (scores[wi, a] > scores[wi, b]) + 0.5 * (scores[wi, a] == scores[wi, b])
            w[b] += (scores[wi, b] > scores[wi, a]) + 0.5 * (scores[wi, a] == scores[wi, b])
    return w


# --- LeagueState as of a past date -----------------------------------------------------------

def _team_dates(ctx: SeasonCtx) -> dict:
    out = defaultdict(list)
    for d_, teams in sorted(ctx.games_by_day.items()):
        for t in teams:
            out[t].append(d_)
    return out


def player_table(ctx: SeasonCtx, cfg, today: date, rostered: dict) -> pd.DataFrame:
    """Every player the model would know about on `today`: projection (mu), games so far (n),
    availability before streak adjustment (base) and games missed in a row (streak).
    Uses only information from before `today`: a player who hasn't played yet sits on last
    season's team (a rookie with no games yet on the team he will first play for)."""
    m = cfg["model"]
    before = ctx.logs[ctx.logs.d < today]
    stats = before.groupby("key").agg(n=("fp", "size"), live=("fp", "mean"), sd=("fp", "std"), team=("team", "last"),
                                      last_d=("d", "max"))
    first_team = ctx.logs.groupby("key")["team"].first()
    known = set(stats.index) | set(ctx.proj_fp) | set(rostered)
    team_dates = _team_dates(ctx)
    team_played = {t: sum(1 for x in ds if x < today) for t, ds in team_dates.items()}
    rows = []
    for k in known:
        n = int(stats.n.get(k, 0)) if k in stats.index else 0
        team = stats.team[k] if n else (ctx.prev_team.get(k) or first_team.get(k))
        if team is None or team not in team_dates:
            continue
        prior = ctx.proj_fp.get(k, ctx.prev_fppg.get(k, 15.0))
        live = float(stats.live[k]) if n else 0.0
        mu = (m["k"] * prior + n * live) / (m["k"] + n)
        rate_prev = float(ctx.prev_rate.get(k, 0.85))
        tp = team_played.get(team, 0)
        if m.get("availability_calibration"):
            base = min(calibrated_rate(rate_prev, n, tp, m["availability_calibration"]), m["availability_cap"])
        else:
            base = min((10 * rate_prev + n) / (10 + max(tp, n)), m["availability_cap"]) if tp else min(rate_prev, m["availability_cap"])
        base = espn_games_blend(base, ctx.proj_gp.get(k), tp, m["availability_cap"])   # as the live model does
        last = stats.last_d[k] if n else None
        streak = sum(1 for x in team_dates[team] if x < today and (last is None or x > last))
        rows.append((k, team, mu, n, float(stats.sd[k]) if n >= 2 else np.nan, base, rostered.get(k, 0), streak))
    return pd.DataFrame(rows, columns=["key", "team", "mu", "n", "sd", "base", "owner", "streak"])


def hist_state(ctx: SeasonCtx, cfg, league: list[list[str]], checkpoint: int, fa_pool: int = 250) -> tuple:
    """(LeagueState, key->id, id->key) as of the start of week `checkpoint`."""
    m = cfg["model"]
    today = ctx.weeks[checkpoint].start
    rostered = {k: t + 1 for t, team in enumerate(league) for k in team}
    team_dates = _team_dates(ctx)
    team_game_index = {t: {d_: j for j, d_ in enumerate([x for x in ds if x >= today])} for t, ds in team_dates.items()}
    df = player_table(ctx, cfg, today, rostered)
    # keep rostered players + the best free agents
    fa = df[df.owner == 0].nlargest(fa_pool, "mu")
    df = pd.concat([df[df.owner > 0], fa])
    sig = shrunk_sigma(df.mu.to_numpy(), df.sd.to_numpy(), df.n.to_numpy(), m["k"], m["cv_typical"])
    players, id_of = {}, {}
    for i, (r, s) in enumerate(zip(df.itertuples(), sig), start=1):
        js = sorted(team_game_index.get(r.team, {}).items(), key=lambda x: x[1])
        av = Availability(float(r.base))
        ros = sum(av.for_game(j, d_) for d_, j in js) * r.mu
        players[i] = PlayerState(i, r.key, r.team, GROUP_POS[ctx.pos_of(r.key)], int(r.owner), "ACTIVE",
                                 float(r.mu), float(s), av, n=int(r.n), ros_value=float(ros), games_remaining=len(js))
        id_of[r.key] = i
    rosters = {t + 1: [id_of[k] for k in team if k in id_of] for t, team in enumerate(league)}
    opponents = defaultdict(dict)
    for wi, pairs in enumerate(ctx.sched):
        for a, b in pairs:
            opponents[a + 1][ctx.weeks[wi].number] = b + 1
            opponents[b + 1][ctx.weeks[wi].number] = a + 1
    cal = HistCal(ctx.weeks, ctx.games_by_day, opponents)
    lg = SimpleNamespace(teams={t: SimpleNamespace(name=f"Team {t}", roster=r) for t, r in rosters.items()},
                         matchups=[], players={}, settings={"name": "mock"})
    st = LeagueState(cfg, lg, cal, today, players, rosters, {}, team_game_index, {}, real_today=today)
    return st, id_of, {v: k for k, v in id_of.items()}


# --- tests -----------------------------------------------------------------------------------

def season_model_matchups(ctx, cfg, league, actual, checkpoint) -> pd.DataFrame:
    """Season model (normal approximation): P(win) per remaining matchup vs real result, and
    expected remaining wins vs real remaining wins."""
    from tradecalc.trade.matchup import season_outlook

    from tradecalc.trade.matchup import week_dist

    st, _, _ = hist_state(ctx, cfg, league, checkpoint)
    rows = []
    real_wins = wins_from(actual, ctx.sched, checkpoint)
    for t in range(1, N_TEAMS + 1):
        out = season_outlook(t, st.rosters, st)
        for wk in out["weeks"]:
            if wk["opp"] is None or wk["opp"] < t:
                continue
            wi = wk["week"] - 1
            a, b = actual[wi, t - 1], actual[wi, wk["opp"] - 1]
            w_ = ctx.weeks[wi]
            d1, d2 = week_dist(st.rosters[t], w_, st, st.today), week_dist(st.rosters[wk["opp"]], w_, st, st.today)
            rows.append({"Vg": d1.V_games, "Vg_opp": d2.V_games, "U": d1.U, "U_opp": d2.U, "R": d1.R, "R_opp": d2.R,
                         "checkpoint": checkpoint, "week": wk["week"], "p": wk["p"], "M": wk["M"], "M_opp": wk["M_opp"],
                         "V": wk["V"], "V_opp": wk["V_opp"], "won": float(a > b) + 0.5 * float(a == b),
                         "actual": a, "actual_opp": b, "team": t, "kind": "matchup"})
        rows.append({"checkpoint": checkpoint, "team": t, "kind": "season", "exp_wins": out["e_wins"],
                     "actual_wins": real_wins[t - 1]})
    return pd.DataFrame(rows)


def trade_counterfactuals(ctx, cfg, league, actual, checkpoint, n_trades: int, rng) -> pd.DataFrame:
    """Random trades: the trade calculator's predicted change in expected wins vs what the
    post-trade rosters (incl. the same forced drops / free-agent signings) actually did."""
    from tradecalc.trade.evaluate import Trade, apply_trade, evaluate

    st, id_of, key_of = hist_state(ctx, cfg, league, checkpoint)
    W = list(range(checkpoint, len(ctx.weeks)))
    base_wins = wins_from(actual, ctx.sched, checkpoint)
    rows = []
    shapes = [(1, 1)] * 5 + [(2, 1)] * 2 + [(1, 2)] * 2 + [(2, 2)]
    for _ in range(n_trades):
        a, b = rng.choice(np.arange(1, N_TEAMS + 1), 2, replace=False)
        ga, gb = shapes[rng.integers(len(shapes))]
        if len(st.rosters[a]) < ga or len(st.rosters[b]) < gb:
            continue
        give = tuple(int(x) for x in rng.choice(st.rosters[a], ga, replace=False))
        get = tuple(int(x) for x in rng.choice(st.rosters[b], gb, replace=False))
        tr = Trade(int(a), int(b), give, get)
        r = evaluate(tr, st)
        new, moves = apply_trade(tr, st.rosters, st)
        scores = actual.copy()
        for t in (a, b):
            keys = [key_of[i] for i in new[t]]
            scores[checkpoint:, t - 1] = team_weekly_actual(ctx, keys, W)
        new_wins = wins_from(scores, ctx.sched, checkpoint)
        d_ros = sum(st.players[i].ros_value for i in (*get, *moves[a][1])) - \
            sum(st.players[i].ros_value for i in (*give, *moves[a][0]))
        rows.append({"checkpoint": checkpoint, "shape": f"{ga}-for-{gb}", "pred_you": r.you.d_wins,
                     "pred_partner": r.partner.d_wins, "real_you": new_wins[a - 1] - base_wins[a - 1],
                     "real_partner": new_wins[b - 1] - base_wins[b - 1], "verdict": r.verdict,
                     "d_ros": d_ros, "pred_pts": r.you.d_pts,
                     "real_pts": float((scores[checkpoint:, a - 1] - actual[checkpoint:, a - 1]).mean())})
    return pd.DataFrame(rows)


def run_season_model_tests(H, client, cfg, season: str, n_leagues: int = 4, checkpoints=(0, 6, 12),
                           trade_checkpoints=(6, 12), n_trades: int = 30, seed: int = 0, log=print) -> dict:
    ctx = season_ctx(H, client, season)
    playable = set(ctx.logs.key)
    leagues = mock_drafts(ctx.values, playable, n_leagues, seed)
    rng = np.random.default_rng(seed)
    mats, trades = [], []
    for li, lg in enumerate(leagues):
        actual = actual_matrix(ctx, lg)
        for cp in checkpoints:
            mats.append(season_model_matchups(ctx, cfg, lg, actual, cp).assign(league=li))
        for cp in trade_checkpoints:
            trades.append(trade_counterfactuals(ctx, cfg, lg, actual, cp, n_trades, rng).assign(league=li))
        log(f"    league {li + 1}/{n_leagues} done")
    return {"matchups": pd.concat(mats, ignore_index=True),
            "trades": pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()}


# Absence-streak diagnostic: games missed in a row (0, 1-2, 3-5, 6-10, 11-20, 21+) x how far ahead
# (next 5, 5-9, 10-19, 20-39, 40+ team games).
STREAK_EDGES = (0, 1, 3, 6, 11, 21)
HORIZON_EDGES = (0, 5, 10, 20, 40)


def bin_of(x: float, edges) -> int:
    return max(i for i, e in enumerate(edges) if x >= e)


def fit_absence_streak(ctx: SeasonCtx, cfg, prior_games: float = 20.0, top: int | None = 200) -> dict:
    """Fit the absence-streak table on one past season: for every week start, every player the
    model knew about, and each of his team's later games, compare games actually played with the
    availability the model predicted (before streak adjustment), by streak bin x horizon bin.
    Only the `top` players by projection count: deep-bench players sit for coach's decisions and
    would make a starter's two-game absence look far worse than it is.
    Each cell is shrunk toward 1 with `prior_games` predicted games."""
    team_dates = _team_dates(ctx)
    played = defaultdict(set)
    for k, d_ in zip(ctx.logs.key, ctx.logs.d):
        played[k].add(d_)
    num = np.zeros((len(STREAK_EDGES), len(HORIZON_EDGES)))
    den = np.zeros_like(num)
    for c in range(1, len(ctx.weeks) - 1):
        today = ctx.weeks[c].start
        tab = player_table(ctx, cfg, today, {})
        if top:                                   # fantasy-relevant players only (rostered + waiver wire)
            tab = tab.nlargest(top, "mu")
        for r in tab.itertuples():
            pl = played.get(r.key, ())
            sb = bin_of(r.streak, STREAK_EDGES)
            for j, d_ in enumerate(x for x in team_dates[r.team] if x >= today):
                h = bin_of(j, HORIZON_EDGES)
                num[sb, h] += d_ in pl
                den[sb, h] += r.base
    table = (num + prior_games) / (den + prior_games)
    return {"table": table.round(3).tolist(), "played": num.tolist(), "predicted": den.round(1).tolist()}
