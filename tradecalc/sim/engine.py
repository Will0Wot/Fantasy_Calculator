"""Monte Carlo projection engine.

World       as-of snapshot: per-player rates, expected minutes and team rotation; team
            defense, pace and strength; defender impacts. Players are keyed by normalized
            name so ESPN (rosters, injuries, depth charts) and stats.nba.com (box scores) join.
simulate    vectorized simulation of NBA games. For every simulation draw:
              1. who plays: each rotation player's P(plays) (injury model, injury-report
                 status, rest on back-to-backs) -> joint absences on both teams
              2. game shocks shared by everyone in the game: final margin (drives blowout
                 minutes) and pace
              3. minutes: expected minutes + redistribution of absent teammates' minutes
                 (same position counts double) + blowout effect + noise
              4. usage: counting rates scale with the share of team usage that is absent
              5. opponent: per-stat defensive factors, pace, absent opposing defenders
              6. box score: FGA ~ Poisson, 3PA/makes ~ Binomial, FTA ~ NegBin, ... ->
                 points are built from makes, so every stat line is internally consistent
week        fantasy matchup between two rosters: each simulated day, the manager starts
            the best eligible lineup among players who are active (known at lock time),
            then the starters' simulated fantasy points are summed.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from tradecalc.model.names import norm_name
from tradecalc.sim.context import EWMA_ALPHA
from tradecalc.sim.defense import defender_impacts, estimate_defense
from tradecalc.sim.project import COUNT_STATS, minutes_sd
from tradecalc.sim.rates import RATES, estimate_rates

STATS = list(RATES)
BOX = ["min", "pts", "reb", "ast", "stl", "blk", "tov", "fgm", "fga", "ftm", "fta", "fg3m", "fg3a"]


@dataclass
class World:
    asof: date
    players: pd.DataFrame                 # index key: name, team, pos, age, exp_min, + rate columns
    rotation: dict[str, list[str]]        # NBA team -> player keys expected to play
    defense: object                       # sim.defense.DefenseModel
    net: dict[str, float]                 # team net rating per 100 possessions
    impact: dict[str, float]              # key -> defensive on/off impact (DRtg points)
    params: dict
    scoring: dict
    b2b_days: dict[str, set] = field(default_factory=dict)  # team -> dates that are 2nd night of a B2B


# --- building a world ---------------------------------------------------------------------

def _ewma(series: pd.Series) -> float:
    v = None
    for m in series:
        v = m if v is None else (1 - EWMA_ALPHA) * v + EWMA_ALPHA * m
    return float(v)


def _rates_by_key(H, cur, prev, params, extra_prior: dict | None = None) -> pd.DataFrame:
    rm = estimate_rates(cur, prev, params["rate_k"], H.pos_of)
    names = pd.concat([cur, prev]).groupby("player_id")["player_name"].last()
    r = rm.rates.copy()
    r["key"] = names.reindex(r.index).map(norm_name)
    r = r.dropna(subset=["key"]).drop_duplicates("key").set_index("key")
    r.attrs["league"] = rm.league
    r.attrs["minutes"] = pd.concat([cur, prev]).assign(key=lambda d: d.player_name.map(norm_name)).groupby("key")["min"].sum()
    return r


def _team_strength(tl: pd.DataFrame) -> dict[str, float]:
    g = tl.groupby("team")
    return (100 * (g["pts"].sum() - g["opp_pts"].sum()) / g["poss"].sum()).to_dict()


def build_world_history(H, params: dict, asof: str, season: str, prev: str) -> World:
    """World from game logs only (backtests): rotation = recent appearances with the team."""
    L = H.logs
    cur = L[(L.season == season) & (L.game_date < asof)]
    prv = L[L.season == prev]
    tl = H.team_logs
    tl_cur, tl_prev = tl[(tl.season == season) & (tl.game_date < asof)], tl[tl.season == prev]
    rates = _rates_by_key(H, cur, prv, params)
    rows, rotation = [], defaultdict(list)
    for pid, g in cur.groupby("player_id"):
        team = g["team"].iat[-1]
        tg = tl_cur[tl_cur.team == team]["game_date"].sort_values().tail(10)
        recent = g[g.game_date.isin(set(tg))]
        if len(recent) < 3:
            continue
        key = norm_name(g["player_name"].iat[-1])
        exp_min = _ewma(g[g.team == team]["min"]) if (g.team == team).sum() >= 2 else _ewma(g["min"])
        rows.append({"key": key, "name": g["player_name"].iat[-1], "team": team, "nba_id": pid,
                     "pos": H.pos_of(g["player_name"].iat[-1]) or "F", "age": H.age.get((pid, season), np.nan),
                     "exp_min": exp_min})
    P = pd.DataFrame(rows).drop_duplicates("key").set_index("key")
    P = P.join(rates, how="left")
    for s in STATS:
        P[s] = P[s].fillna(P["pos"].map(lambda g_: rates.attrs["league"].get(g_, rates.attrs["league"]["ALL"])[s]))
    for team, sub in P.groupby("team"):
        rotation[team] = sub.sort_values("exp_min", ascending=False).index[:13].tolist()
    dm = estimate_defense(tl_cur, params["def_k"], tl_prev)
    imp = defender_impacts(H.onoff[H.onoff.season == prev]) if len(H.onoff) else pd.Series(dtype=float)
    id2key = P.reset_index().set_index("nba_id")["key"].to_dict()
    impact = {id2key[i]: v for i, v in imp.items() if i in id2key}
    w = World(date.fromisoformat(asof), P, dict(rotation), dm, _team_strength(tl_cur if len(tl_cur) >= 60 else tl_prev),
              impact, params, H.scoring)
    w.b2b_days = _b2b_days_from_logs(tl[tl.season == season])
    return w


def _b2b_days_from_logs(tl) -> dict[str, set]:
    out = defaultdict(set)
    for t, d in zip(tl.team, tl.game_date):
        out[t].add(d)
    res = {}
    for t, ds in out.items():
        days = sorted(date.fromisoformat(x) for x in ds)
        res[t] = {b for a, b in zip(days, days[1:]) if (b - a).days == 1}
    return res


# --- simulating NBA games ----------------------------------------------------------------

def _p_rest(params: dict, age: float, exp_min: float, b2b: bool, coach_rate: float | None = None) -> float:
    r = params["rest"]
    a28 = max((age if age == age else 26.0) - 28, 0)
    load = exp_min / 30
    z = (r["Intercept"] + r["b2b"] * b2b + r["age28"] * a28 + r["b2b:age28"] * b2b * a28
         + r["load"] * load + r["b2b:load"] * b2b * load)
    if b2b and coach_rate:
        lg = params["rest"]["league_b2b_rate"]
        z += np.log(coach_rate / (1 - coach_rate)) - np.log(lg / (1 - lg))
    return float(1 / (1 + np.exp(-z)))


def rest_factor(params, age, exp_min, b2b, coach_rate=None) -> float:
    """Multiplier on a base availability rate for this game: P(no rest | game) / P(no rest | average game).
    Base availability rates already include average rest, so only the B2B contrast is applied."""
    p_game = _p_rest(params, age, exp_min, b2b, coach_rate)
    share_b2b = 0.18
    p_avg = share_b2b * _p_rest(params, age, exp_min, True, coach_rate) + (1 - share_b2b) * _p_rest(params, age, exp_min, False)
    return (1 - p_game) / (1 - p_avg)


@dataclass
class GameSims:
    fp: dict          # (key, date) -> array[n] fantasy points (0 if did not play)
    played: dict      # (key, date) -> bool array[n]
    box_mean: dict    # (key, date) -> {stat: mean over sims where played}


def _nb(rng, mean, r, n):
    mean = np.maximum(mean, 1e-9)
    if r >= 500:
        return rng.poisson(mean)
    p = r / (r + mean)
    return rng.negative_binomial(r, p)


def simulate_games(world: World, games: list[tuple[date, str, str]], avail, n: int = 10000,
                   seed: int | None = 0, interest: set[str] | None = None) -> GameSims:
    """games: (date, home NBA team, away NBA team). avail(key, team, date) -> P(plays).
    interest: keys to return samples for (default: everyone simulated)."""
    rng = np.random.default_rng(seed)
    pr, P = world.params, world.players
    # Weekly form / rate-estimate error: one multiplier per player per simulation, shared by all
    # of his games in this call, so a player's games within a week are correlated.
    form_sd = pr.get("form_sd", 0.0)
    form: dict[str, np.ndarray] = {}

    def form_of(k):
        if k not in form:
            form[k] = np.maximum(rng.normal(1.0, form_sd, n), 0.2) if form_sd else np.ones(n)
        return form[k]

    mp, ub, disp = pr["minutes"], pr["usage_beta"], pr["dispersion"]
    dm = world.defense
    fp_out, played_out, box_out = {}, {}, {}
    w = world.scoring
    for d, home, away in games:
        teams = (home, away)
        exp_pace = {t: dm.pace.get(t, dm.league_pace) for t in teams}
        game_pace = exp_pace[home] + exp_pace[away] - dm.league_pace
        pace_shock = rng.normal(1.0, pr["pace_sd"], n)
        exp_margin = ((world.net.get(home, 0) - world.net.get(away, 0)) / 100 * game_pace + pr.get("home_edge", 1.5))
        margin = rng.normal(exp_margin, pr["margin_sd"], n)                    # home perspective
        blow = np.maximum(np.abs(margin) - mp.get("blowout_threshold", 12), 0)
        plays, info = {}, {}
        for t in teams:
            keys = [k for k in world.rotation.get(t, []) if k in P.index]
            sub = P.loc[keys]
            p = np.array([avail(k, t, d) for k in keys])
            plays[t] = rng.random((n, len(keys))) < p[None, :]
            info[t] = (keys, sub)
        for t in teams:
            o = away if t == home else home
            keys, sub = info[t]
            if not keys:
                continue
            pl = plays[t]
            em = sub["exp_min"].to_numpy()
            absent = ~pl
            # usage: share of team expected usage that is absent, per sim and stat
            miss = {}
            for s in ["fga", "fta", "tov", "ast", "reb", "stl", "blk", "fg3a"]:
                base = (sub["fg3a_share"] * sub["fga"]).to_numpy() if s == "fg3a" else sub[s].to_numpy()
                wts = base * em
                tot = wts.sum()
                miss[s] = (absent * wts[None, :]).sum(axis=1) / tot if tot > 0 else np.zeros(n)
            pos = sub["pos"].to_numpy()
            abs_by_pos = {g: (absent[:, pos == g] * em[pos == g][None, :]).sum(axis=1) for g in ("G", "F", "C")}
            abs_total = sum(abs_by_pos.values())
            # opponent's absent defenders -> efficiency multiplier
            okeys, osub = info[o]
            if okeys:
                oimp = np.array([world.impact.get(k, 0.0) for k in okeys]) * osub["exp_min"].to_numpy() / 48
                eff = 1 + ((~plays[o]) * oimp[None, :]).sum(axis=1) / dm.league_drtg
            else:
                eff = np.ones(n)
            def_f = dm.factors.loc[o].to_dict() if o in dm.factors.index else {}
            pace_ratio = game_pace / exp_pace[t] * pace_shock
            b2b = d in world.b2b_days.get(t, set())
            # minutes for every rotation player, then scale so the team plays ~240 minutes
            M = np.zeros((n, len(keys)))
            for j in range(len(keys)):
                g_ = pos[j]
                same, other = abs_by_pos[g_], abs_total - abs_by_pos[g_]
                mhat = (em[j] + mp["const"] + mp["abs_min_same"] * same + mp["abs_min_other"] * other + mp["b2b"] * b2b)
                if em[j] >= 30:
                    mhat = mhat + mp["blow_star"] * blow
                elif em[j] < 20:
                    mhat = mhat + mp["blow_bench"] * blow
                M[:, j] = np.clip(rng.normal(mhat, minutes_sd(em[j], mp)), 0, 48) * pl[:, j]
            target = pr.get("team_minutes", 241.0) * pr.get("rotation_minutes_share", 0.97)
            tot = M.sum(axis=1, keepdims=True)
            M = np.clip(M * np.where(tot > 0, target / np.maximum(tot, 1e-9), 0), 0, 48)
            rate_cal = pr.get("rate_calibration", 1.0)
            for j, k in enumerate(keys):
                if interest is not None and k not in interest:
                    continue
                row = sub.iloc[j]
                box = _sample_box(rng, row, M[:, j], miss, ub, def_f, pace_ratio * rate_cal * form_of(k), eff, disp, n)
                fp = sum(wt * box[s.lower()] for s, wt in w.items())
                m = pl[:, j]
                fp_out[(k, d)] = fp
                played_out[(k, d)] = m
                box_out[(k, d)] = {s: float(box[s][m].mean()) if m.any() else 0.0 for s in BOX}
                box_out[(k, d)]["fp"] = float(fp[m].mean()) if m.any() else 0.0
    return GameSims(fp_out, played_out, box_out)


def trim_rotation(exp_min: np.ndarray, total: float) -> np.ndarray:
    """If a team's expected minutes add up to more than `total`, remove the excess from the
    bottom of the rotation first (the way a coach shortens the bench)."""
    order = np.argsort(-exp_min)
    m = exp_min[order]
    excess = max(m.sum() - total, 0.0)
    below = np.cumsum(m[::-1])[::-1] - m
    m = m - np.clip(excess - below, 0, m)
    out = np.empty_like(exp_min)
    out[order] = m
    return out


def _sample_box(rng, row, mins, miss, ub, def_f, pace_ratio, eff, disp, n):
    lam = {}
    for s in COUNT_STATS:
        u = 1 + ub.get(s, 0.0) * miss[s]
        lam[s] = row[s] * mins * u * def_f.get(s, 1.0) * pace_ratio
    share = np.clip(row["fg3a_share"] * (1 + ub.get("fg3a", 0) * miss["fg3a"]) / (1 + ub.get("fga", 0) * miss["fga"])
                    * def_f.get("fg3a_share", 1.0), 0, 0.95)
    fga = _nb(rng, lam["fga"], disp.get("fga", 1000), n)
    fg3a = rng.binomial(fga, share)
    fg2a = fga - fg3a
    p2 = np.clip(row["fg2"] * def_f.get("fg2", 1.0) * eff, 0, 0.99)
    p3 = np.clip(row["fg3"] * def_f.get("fg3", 1.0) * eff, 0, 0.99)
    fg2m, fg3m = rng.binomial(fg2a, p2), rng.binomial(fg3a, p3)
    fta = _nb(rng, lam["fta"], disp.get("fta", 1000), n)
    ftm = rng.binomial(fta, np.clip(row["ft"] * def_f.get("ft", 1.0), 0, 0.99))
    out = {"min": mins, "fga": fga, "fgm": fg2m + fg3m, "fg3a": fg3a, "fg3m": fg3m, "fta": fta, "ftm": ftm,
           "pts": 2 * fg2m + 3 * fg3m + ftm}
    for s in ["reb", "ast", "stl", "blk", "tov"]:
        out[s] = _nb(rng, lam[s], disp.get(s, 1000), n)
    return out


# --- fantasy week --------------------------------------------------------------------------

@dataclass
class MatchupResult:
    p_win: float
    p_tie: float
    totals: dict           # team label -> array[n] weekly fantasy points (incl. banked points)
    players: pd.DataFrame  # per player: team, games, P(active), E[starts], E[FP/game], E[FP to team]
    banked: dict = field(default_factory=dict)   # team label -> points already scored this week
    days: list = field(default_factory=list)     # days simulated


def fantasy_week(sims: GameSims, rosters: dict[str, list[str]], days: list[date], slots: list[str],
                 elig: dict[str, list[str]], slot_elig: dict, n: int, tie_value: float = 0.5,
                 names: dict | None = None, teams_of: dict | None = None) -> MatchupResult:
    """rosters: label -> player keys. elig: key -> ESPN positions. Daily lineup per simulation:
    best eligible starters among players active that day, ranked by expected FP (manager knows
    who is active at lock time, not how they will score)."""
    from tradecalc.trade.lineup import best_lineup, eligibility_matrix

    totals = {}
    rows = []
    for label, keys in rosters.items():
        tot = np.zeros(n)
        starts = defaultdict(lambda: np.zeros(n))
        contrib = defaultdict(lambda: np.zeros(n))
        E = eligibility_matrix([tuple(elig.get(k, ("PG", "SG", "SF", "PF", "C"))) for k in keys], slots, slot_elig)
        for d in days:
            idx = [i for i, k in enumerate(keys) if (k, d) in sims.fp]
            if not idx:
                continue
            act = np.stack([sims.played[(keys[i], d)] for i in idx], axis=1)          # n x m
            fpm = np.stack([sims.fp[(keys[i], d)] for i in idx], axis=1)
            prio = np.array([sims.box_mean[(keys[i], d)]["fp"] for i in idx])
            sub_e = E[idx]
            # one lineup decision per distinct availability pattern
            pat = np.packbits(act, axis=1)
            uniq, inv = np.unique(pat, axis=0, return_inverse=True)
            start_mask = np.zeros_like(act)
            for u in range(len(uniq)):
                rows_u = np.flatnonzero(inv.ravel() == u)
                a0 = act[rows_u[0]]
                st = best_lineup(np.where(a0, prio, 0.0), sub_e)
                start_mask[rows_u] = st[None, :]
            gained = (fpm * start_mask).sum(axis=1)
            tot += gained
            for c, i in enumerate(idx):
                starts[keys[i]] += start_mask[:, c]
                contrib[keys[i]] += fpm[:, c] * start_mask[:, c]
        totals[label] = tot
        for k in keys:
            gs = [d for d in days if (k, d) in sims.fp]
            act_rate = np.mean([sims.played[(k, d)].mean() for d in gs]) if gs else 0.0
            fpg = np.mean([sims.box_mean[(k, d)]["fp"] for d in gs]) if gs else 0.0
            rows.append({"fantasy_team": label, "key": k, "player": (names or {}).get(k, k),
                         "nba_team": (teams_of or {}).get(k, ""), "games": len(gs), "p_active": act_rate,
                         "exp_starts": float(starts[k].mean()) if gs else 0.0, "fp_per_game": fpg,
                         "exp_fp_to_team": float(contrib[k].mean()) if gs else 0.0})
    labels = list(rosters)
    if len(labels) == 2:  # head-to-head: P(first team wins), P(tie)
        ta, tb = totals[labels[0]], totals[labels[1]]
        return MatchupResult(float((ta > tb).mean()), float((ta == tb).mean()), totals, pd.DataFrame(rows))
    return MatchupResult(float("nan"), float("nan"), totals, pd.DataFrame(rows))


# --- live world (current season, ESPN rosters / depth charts / injuries) -------------------

DEPTH_MINUTES = {1: 31.6, 2: 22.0, 3: 13.5}   # by position depth; replaced by params["depth_minutes"] ranks
PROJ_KEYS = {"fga": "FGA", "fta": "FTA", "reb": "REB", "ast": "AST", "stl": "STL", "blk": "BLK", "tov": "TOV"}


def _depth_prior(params: dict) -> dict[int, float]:
    dm = {int(k): v for k, v in params.get("depth_minutes", {}).items()}
    if not dm:
        return DEPTH_MINUTES
    avg = lambda rs: float(np.mean([dm[r] for r in rs if r in dm]))  # noqa: E731
    return {1: avg(range(1, 6)), 2: avg(range(6, 11)), 3: avg(range(11, 16)), 4: 4.0}


def _rates_from_projection(proj: dict) -> dict | None:
    m = proj.get("MIN") or 0
    if not m or m <= 0:
        return None
    fga, fg3a = proj.get("FGA", 0), proj.get("FG3A", 0)
    fgm, fg3m = proj.get("FGM", 0), proj.get("FG3M", 0)
    fta, ftm = proj.get("FTA", 0), proj.get("FTM", 0)
    r = {s: proj.get(c, 0) / m for s, c in PROJ_KEYS.items()}
    r["fg3a_share"] = fg3a / fga if fga else 0.35
    r["fg2"] = (fgm - fg3m) / (fga - fg3a) if fga > fg3a else 0.5
    r["fg3"] = fg3m / fg3a if fg3a else 0.35
    r["ft"] = ftm / fta if fta else 0.75
    return r


def build_world_live(H, params: dict, state, depth: pd.DataFrame, coaches: pd.DataFrame | None,
                     season: str, prev: str) -> World:
    """World for the live league as of state.today.

    Expected minutes blend three views, then this season's games take over (weight n/(n+3)):
      - last season's minutes (last 20 games), if he is on the same team
      - his ESPN depth-chart slot (typical minutes for that depth)
      - ESPN's projected minutes
    A new head coach rescales starters' minutes by the coaches' rotation concentration.
    Rates come from box scores; players with under 100 NBA minutes use ESPN's projection.
    """
    from tradecalc.teams import to_nba

    today = state.today.isoformat()
    L = H.logs
    cur = L[(L.season == season) & (L.game_date < today)]
    prv = L[L.season == prev]
    rates = _rates_by_key(H, cur, prv, params)
    lg_rates = rates.attrs["league"]
    nba_minutes = rates.attrs["minutes"]
    depth_min = _depth_prior(params)

    last20 = prv.sort_values("game_date").groupby("player_id").tail(20)
    last_team = prv.sort_values("game_date").groupby("player_id")["team"].last()
    prev_mpg = last20.groupby("player_id")["min"].mean()
    pid_by_key = pd.concat([prv, cur]).assign(key=lambda d: d.player_name.map(norm_name)).groupby("key")["player_id"].last()

    coach_now, coach_before = {}, {}
    if coaches is not None and len(coaches):
        cmap = coaches.set_index(["team", "season"])["coach"].to_dict()
        coach_now = {t: cmap.get((t, season)) for t in coaches.team.unique()}
        coach_before = {t: cmap.get((t, prev)) for t in coaches.team.unique()}
    cpar, clg = params.get("coach", {}), params.get("coach_league", {})

    # depth chart: key -> (team, best depth)
    dpt = {}
    for r in depth.itertuples():
        k = norm_name(r.name)
        t = to_nba(r.team)
        if k not in dpt or r.depth < dpt[k][1]:
            dpt[k] = (t, int(r.depth))

    espn = {norm_name(p.name): p for p in state.players.values()}
    keys = set(dpt) | {k for k, p in espn.items() if p.pro_team and p.pro_team != "FA"}
    rows = []
    for k in keys:
        p = espn.get(k)
        team = to_nba(p.pro_team) if p is not None and p.pro_team not in ("", "FA") else dpt.get(k, (None,))[0]
        if not team:
            continue
        pid = pid_by_key.get(k)
        views = []
        if pid is not None and pid in prev_mpg.index and last_team.get(pid) == team:
            views.append(float(prev_mpg[pid]))
        if k in dpt:
            views.append(depth_min.get(min(dpt[k][1], 4), 4.0))
        proj = (state.league.players.get(p.espn_id).projection if p is not None and p.espn_id in state.league.players else {}) or {}
        if proj.get("MIN"):
            views.append(float(proj["MIN"]))
        if pid is not None and pid in prev_mpg.index and not views:
            views.append(float(prev_mpg[pid]) * 0.8)
        if not views:
            continue
        prior = float(np.mean(views))
        cn, cb = coach_now.get(team), coach_before.get(team)
        if cn and cb and cn != cb and prior >= 28 and cn in cpar and cb in cpar:
            prior *= cpar[cn].get("top5_share", clg.get("top5_share", 1)) / cpar[cb].get("top5_share", clg.get("top5_share", 1))
        g = cur[(cur.player_name.map(norm_name) == k) & (cur.team == team)] if len(cur) else cur
        n = len(g)
        exp_min = prior if n == 0 else (n * _ewma(g["min"]) + 3 * prior) / (n + 3)
        pos = None
        if p is not None and p.positions:
            ps = [x for x in p.positions if x in ("PG", "SG", "SF", "PF", "C")]
            pos = "C" if "C" in ps else ("G" if ps and ps[0] in ("PG", "SG") else "F")
        pos = pos or H.pos_of(k) or "F"
        row = {"key": k, "name": p.name if p is not None else k, "team": team, "nba_id": pid, "pos": pos,
               "age": H.age.get((pid, prev), np.nan) + 1 if pid is not None else np.nan, "exp_min": exp_min,
               "espn_id": p.espn_id if p is not None else None}
        if k in rates.index and nba_minutes.get(k, 0) >= 100:
            row.update(rates.loc[k, STATS].to_dict())
        else:
            pr = _rates_from_projection(proj) if proj else None
            row.update(pr or lg_rates.get(pos, lg_rates["ALL"]))
        rows.append(row)
    P = pd.DataFrame(rows).drop_duplicates("key").set_index("key")
    # Preseason priors (depth chart, projections) over-allocate bench minutes; trim each team's
    # top 13 to the total that real rotations carry (fitted from past seasons).
    total = params.get("rotation_exp_min_sum", 282.0)
    for t, sub in P.groupby("team"):
        top = sub.sort_values("exp_min", ascending=False).index[:13]
        P.loc[top, "exp_min"] = trim_rotation(P.loc[top, "exp_min"].to_numpy(dtype=float), total)
    rotation = {t: sub[sub.exp_min >= 6].sort_values("exp_min", ascending=False).index[:13].tolist()
                for t, sub in P.groupby("team")}
    tl = H.team_logs
    tl_cur, tl_prev = tl[(tl.season == season) & (tl.game_date < today)], tl[tl.season == prev]
    dm = estimate_defense(tl_cur, params["def_k"], tl_prev)
    imp = defender_impacts(H.onoff[H.onoff.season == prev]) if len(H.onoff) else pd.Series(dtype=float)
    id2key = P.dropna(subset=["nba_id"]).reset_index().set_index("nba_id")["key"].to_dict()
    impact = {id2key[i]: v for i, v in imp.items() if i in id2key}
    net = _team_strength(tl_cur if len(tl_cur) >= 300 else tl_prev)
    w = World(state.today, P, rotation, dm, net, impact, params, H.scoring)
    # back-to-backs from the ESPN schedule
    by_team = defaultdict(set)
    for g_ in state.league.pro_games:
        for t in (g_["home"], g_["away"]):
            by_team[to_nba(t)].add(g_["date"])
    w.b2b_days = {t: {b for a, b in zip(sorted(ds), sorted(ds)[1:]) if (b - a).days == 1} for t, ds in by_team.items()}
    return w


def live_avail(world: World, state, coaches_now: dict[str, str] | None = None):
    """P(plays): injury model + injury report (LeagueState availability) x back-to-back rest."""
    by_espn = {p.espn_id: p for p in state.players.values()}
    cpar = world.params.get("coach", {})

    def avail(k, team, d):
        row = world.players.loc[k] if k in world.players.index else None
        base = 0.85
        if row is not None and row.get("espn_id") in by_espn:
            ps = by_espn[row["espn_id"]]
            base = state.a_for(ps, d) if state.cal.team_plays(ps.pro_team, d) else ps.avail.for_game(0, d)
        b2b = d in world.b2b_days.get(team, set())
        coach = (coaches_now or {}).get(team)
        crate = cpar.get(coach, {}).get("b2b_rest") if coach else None
        age = row["age"] if row is not None else 26.0
        em = row["exp_min"] if row is not None else 15.0
        return float(np.clip(base * rest_factor(world.params, age, em, b2b, crate), 0, 1))
    return avail


def run_matchup(world: World, state, team_a: int, team_b: int, week, n: int = 10000, seed: int | None = 0,
                coaches_now: dict | None = None) -> tuple[MatchupResult, GameSims]:
    """Monte Carlo of fantasy teams team_a vs team_b for one fantasy week (remaining days only
    if the week is in progress)."""
    from tradecalc.teams import to_nba
    from tradecalc.trade.lineup import slot_list

    cfg = state.cfg
    days = [d for d in week.days() if d >= state.today]
    games = [(g["date"], to_nba(g["home"]), to_nba(g["away"])) for g in state.league.pro_games if g["date"] in set(days)]
    rosters, elig, names, teams_of = {}, {}, {}, {}
    key_of = {i: norm_name(state.players[i].name) for t in (team_a, team_b) for i in state.rosters[t]}
    for t in (team_a, team_b):
        keys = []
        for i in state.rosters[t]:
            k = key_of[i]
            if k not in world.players.index:
                continue
            ps = state.players[i]
            keys.append(k)
            elig[k] = tuple(ps.positions) or ("SF", "PF")
            names[k] = ps.name
            teams_of[k] = ps.pro_team
        rosters[state.team_name(t)] = keys
    # make sure every rostered player is simulated even if outside his team's top-13 rotation
    for k in elig:
        t = world.players.loc[k, "team"]
        if k not in world.rotation.get(t, []):
            world.rotation.setdefault(t, []).append(k)
    sims = simulate_games(world, games, live_avail(world, state, coaches_now), n=n, seed=seed, interest=set(elig))
    res = fantasy_week(sims, rosters, days, slot_list(cfg), elig, cfg["roster"]["slot_eligibility"], n,
                       cfg["league"]["tie_value"], names, teams_of)
    # Days already played this week: add each team's actual points, then re-score the matchup.
    for t in (team_a, team_b):
        lab = state.team_name(t)
        res.banked[lab] = state.banked(t, week)[0]
        res.totals[lab] = res.totals[lab] + res.banked[lab]
    la, lb = state.team_name(team_a), state.team_name(team_b)
    res.p_win = float((res.totals[la] > res.totals[lb]).mean())
    res.p_tie = float((res.totals[la] == res.totals[lb]).mean())
    res.days = days
    return res, sims
