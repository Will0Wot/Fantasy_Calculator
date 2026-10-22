"""Possession-by-possession game simulator (RotoBot-style), turned into fantasy predictions.

Every simulated game is played one possession at a time. For each possession:
  1. Who is on the floor: 5 players per team, drawn so each plays his simulated minutes
     (systematic sampling with inclusion probability minutes / 48).
  2. Turnover? Probability = the on-court players' turnover rates x the defense's forced-turnover
     factor. Who turned it over, and was it a steal (and by whom).
  3. Otherwise a shot: who shoots (usage among the five), fouled before the shot (2-3 free throws),
     or where he shoots from (his zone mix, shifted by how the defense funnels shots), who is
     guarding him (matchup history; at the rim, who protects the rim), whether it goes in (his make
     rate in that zone + the defender's effect), blocked?, and-one?, assisted? and by whom (who he
     passes to), free throws.
  4. Missed shot or last free throw: offensive or defensive rebound (on-court rebound rates);
     an offensive rebound continues the possession (sometimes a putback by the rebounder).
Possessions per team follow the game's pace; tied games go to overtime. Each simulation's box score
gives every player's fantasy points, so the result is a full distribution per player per game.

The player minutes (absences, back-to-backs, blowouts) come from the box engine's fitted minutes
model (sim/engine.py: draw_minutes), so both engines agree on playing time.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from tradecalc.ingest.pbp import ZONES
from tradecalc.sim.engine import BOX, GameSims, draw_minutes

ZI = {z: i for i, z in enumerate(ZONES)}
THREE = np.array([z in ("C3", "AB3") for z in ZONES])
STATS = ["pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov"]
SI = {s: i for i, s in enumerate(STATS)}
# share of FGA blocked, by zone (league, play-by-play 2024-25)
BLOCKED = np.array([0.110, 0.086, 0.022, 0.011, 0.008])
# and-ones per made shot by zone, relative to the player's overall and-one rate
AND1_W = np.array([1.9, 1.3, 0.5, 0.15, 0.15])
POS_SIM = {("G", "G"): 1.0, ("F", "F"): 1.0, ("C", "C"): 1.0, ("G", "F"): 0.35, ("F", "G"): 0.35,
           ("F", "C"): 0.35, ("C", "F"): 0.35, ("G", "C"): 0.08, ("C", "G"): 0.08}

# Fitted on the 2022-23 season (tuning season; never on the test seasons 2023-24 to 2025-26):
# league totals of each stat matched to box scores, usage concentration chosen by player-game error,
# weekly form (a player's share of his team's chances that week) by weekly 50% and 80%-range coverage.
DEFAULT_CAL = {"poss_scale": 0.965, "home_logit": 0.03, "tov_scale": 0.956, "trip_scale": 1.08, "oreb_scale": 1.0,
               "ft_miss_oreb": 0.45, "putback": 0.30, "assist_scale": 1.0, "zone_logit": [0.0] * 5,
               "pace_sd": 0.035, "ot_poss": 10.5, "three_foul": 0.09, "ast_lineup": 1.0, "pair_shrink": 10.0,
               "stl_scale": 0.93, "blk_scale": 0.87, "three_scale": 1.04, "usage_power": 1.5, "form_sd": 0.30,
               "stint": 4}


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _pick(w: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Row-wise categorical draw: w (R, p) >= 0, u (R,) uniform -> index (R,)."""
    cw = np.cumsum(w, axis=1)
    t = cw[:, -1:] * u[:, None]
    return np.minimum((cw < t).sum(axis=1), w.shape[1] - 1)


@dataclass
class TeamGame:
    """One team in one game: player parameters (p players) and simulated minutes (n x p)."""
    keys: list
    M: np.ndarray            # n x p minutes
    opp: np.ndarray          # shot opportunities per possession on the floor
    trip: np.ndarray
    zp: np.ndarray           # p x 5
    fgl: np.ndarray          # p x 5 logit make rate
    as_: np.ndarray          # p x 5 assisted share
    and1: np.ndarray
    ft: np.ndarray
    tov: np.ndarray
    ast: np.ndarray
    stl: np.ndarray
    blk: np.ndarray
    oreb: np.ndarray
    dreb: np.ndarray
    eff: np.ndarray          # p x 5 defender effect (logit) when guarding a shot in zone z
    rim: np.ndarray          # rim contests per game
    pos: np.ndarray
    pairs: np.ndarray        # p x p: assists passer -> shooter (pseudo-counts)
    ast_ref: np.ndarray      # p: passing (sum of teammates' assist rates) he usually has on the floor
    tov_f: float             # this team's forced-turnover factor (as the defense)
    zone_f: np.ndarray       # 5: how this team's defense shifts opponents' shot zones


class PossessionModel:
    """Profiles + defense model + calibration, ready to build TeamGame objects for a World."""

    def __init__(self, prof, dmodel, cal: dict | None = None):
        from tradecalc.model.names import match_names

        self.prof, self.dm = prof, dmodel
        self.cal = {**DEFAULT_CAL, **(cal or {})}
        self._match = match_names
        self._map = {}
        P = prof.players
        self.blk_ref = float((P.blk * P["min"]).sum() / max(P["min"].sum(), 1) * 5)

    def key(self, k: str) -> str | None:
        if k not in self._map:
            if k in self.prof.players.index:
                self._map[k] = k
            else:
                hit = self._match([k], list(self.prof.players.index), cutoff=90)
                self._map[k] = hit.get(k)
        return self._map[k]

    def team_game(self, keys: list, pos: np.ndarray, M: np.ndarray, team: str) -> TeamGame:
        P, pri = self.prof.players, self.prof.league["pos"]
        rows = []
        for k, g in zip(keys, pos):
            pk = self.key(k)
            if pk is not None and pk in P.index:
                rows.append(P.loc[pk])
            else:                                      # no play-by-play yet (rookie): position average
                pr = pri.get(g, pri.get("F"))
                rows.append(pd.Series({**pr, "pos": g}))
        R = pd.DataFrame(rows)
        zp = R[[f"zp_{z}" for z in ZONES]].to_numpy(float)
        fgl = _logit(R[[f"fg_{z}" for z in ZONES]].to_numpy(float))
        as_ = R[[f"as_{z}" for z in ZONES]].to_numpy(float)
        eff = np.array([[self.dm.effect.loc[k, z] if k in self.dm.effect.index else 0.0 for z in ZONES] for k in keys])
        rim = np.array([self.dm.rim.get(k, 0.5) for k in keys])
        pr_keys = [self.key(k) for k in keys]
        pa = self.prof.pairs
        sub = pa[pa.passer.isin(pr_keys) & pa.shooter.isin(pr_keys)]
        idx = {k: i for i, k in enumerate(pr_keys) if k is not None}
        pairs = np.zeros((len(keys), len(keys)))
        for a, s, n_ in zip(sub.passer, sub.shooter, sub.n):
            if a in idx and s in idx:
                pairs[idx[a], idx[s]] += n_
        ast = R["ast"].to_numpy(float)
        np.fill_diagonal(pairs, 0.0)
        # who he passes to: each passer's share follows his own assist rate, tilted by the pair's
        # history ("chemistry") relative to what that rate alone would predict, shrunk to 1
        n_i = pairs.sum(axis=0)                                   # assisted makes per shooter (in pairs)
        share = np.where(np.eye(len(keys), dtype=bool), 0.0, ast[:, None])
        share = share / np.maximum(share.sum(axis=0, keepdims=True), 1e-12)
        expected = share * n_i[None, :]
        c = self.cal.get("pair_shrink", 10.0)
        pairs = ast[:, None] * (pairs + c) / (expected + c)
        np.fill_diagonal(pairs, 0.0)
        pres = np.clip(M.mean(axis=0) / 48.0, 0, 1)                # how often each player is on the floor
        tot_p, tot_ap = pres.sum(), (ast * pres).sum()
        ast_ref = (tot_ap - ast * pres) * 4.0 / np.maximum(tot_p - pres, 1e-9)
        T = self.prof.teams
        tov_f = float(T.loc[team, "tov_f"]) if team in T.index else 1.0
        zone_f = (T.loc[team, [f"zone_{z}" for z in ZONES]].to_numpy(float) if team in T.index else np.ones(5))
        return TeamGame(keys, M, R["opp"].to_numpy(float), R["trip"].to_numpy(float), zp, fgl, as_,
                        R["and1"].to_numpy(float), R["ft"].to_numpy(float), R["tov"].to_numpy(float), ast,
                        R["stl"].to_numpy(float), R["blk"].to_numpy(float), R["oreb"].to_numpy(float),
                        R["dreb"].to_numpy(float), eff, rim, np.asarray(pos), pairs, ast_ref, tov_f, zone_f)

    def affinity(self, off: TeamGame, de: TeamGame) -> np.ndarray:
        """p_off x p_def weights for who guards whom: past matchups + position similarity x minutes."""
        A = np.zeros((len(off.keys), len(de.keys)))
        mins = de.M.mean(axis=0)
        for i, ko in enumerate(off.keys):
            for j, kd in enumerate(de.keys):
                A[i, j] = self.dm.affinity.get((ko, kd), 0.0) + 20.0 * POS_SIM.get((off.pos[i], de.pos[j]), 0.2) * mins[j] / 48
        return A


# --- the possession loop (a batch of games x simulations, vectorized) -------------------------

class _Side:
    """One side (home or away) of every game in a batch. Rows = game x simulation; players are
    padded to P slots (padding never takes the floor)."""

    def __init__(self, tgs: list, forms: list, aff: list, n: int, P: int, rng, usage_power: float):
        G = len(tgs)
        R = G * n
        f32 = np.float32
        self.n, self.P = n, P

        def per_row(attr, fill=0.0):
            a = np.full((G, P), fill, dtype=f32)
            for g, tg in enumerate(tgs):
                v = getattr(tg, attr)
                a[g, :len(v)] = v
            return np.repeat(a, n, axis=0)

        def per_game(attr, shape):
            a = np.zeros((G, P) + shape, dtype=f32)
            for g, tg in enumerate(tgs):
                v = getattr(tg, attr)
                a[g, :v.shape[0], ...] = v if not shape or len(shape) == 1 else v[:, :shape[0]]
            return a

        self.opp_a = per_row("opp") ** usage_power
        for attr in ("trip", "and1", "ft", "tov", "ast", "stl", "blk", "oreb", "dreb", "rim", "ast_ref"):
            setattr(self, attr, per_row(attr))
        self.zp, self.fgl, self.as_, self.eff = (per_game(a, (5,)) for a in ("zp", "fgl", "as_", "eff"))
        self.pairs = np.zeros((G, P, P), dtype=f32)
        self.aff = np.zeros((G, P, P), dtype=f32)              # this side's shooters x the other side's defenders
        for g, tg in enumerate(tgs):
            k = len(tg.keys)
            self.pairs[g, :k, :k] = tg.pairs
            self.aff[g, :k, :aff[g].shape[1]] = aff[g]
        self.tov_f = np.array([tg.tov_f for tg in tgs], dtype=f32)
        self.zone_f = np.stack([tg.zone_f for tg in tgs]).astype(f32)
        pi = np.zeros((R, P))
        self.form = np.ones((R, P), dtype=f32)
        for g, tg in enumerate(tgs):
            k = len(tg.keys)
            pi[g * n:(g + 1) * n, :k] = np.clip(tg.M / 48.0, 0, 1)
            if forms[g] is not None and k:
                self.form[g * n:(g + 1) * n, :k] = forms[g]
        for _ in range(3):                                  # 5 players on the floor on average
            s = pi.sum(axis=1, keepdims=True)
            pi = np.where(s > 0, np.clip(pi * 5.0 / np.maximum(s, 1e-9), 0, 1), 0)
        self.order = np.argsort(rng.random((R, P)), axis=1)          # random order per simulation
        self.inv = np.argsort(self.order, axis=1)
        po = np.take_along_axis(pi, self.order, axis=1)
        self.C = np.cumsum(po, axis=1)
        self.Cp = self.C - po
        self.stats = np.zeros(R * P * len(STATS))
        self.score = np.zeros(R)
        self._idx, self._val = [], []

    def new_stint(self, u: np.ndarray) -> None:
        """Draw the five on the floor for every simulation (kept for the next few possessions)."""
        sel = (np.floor(self.C - u[:, None]) - np.floor(self.Cp - u[:, None])) > 0
        self.on = np.take_along_axis(sel, self.inv, axis=1).astype(np.float32)

    def on_court(self, rows: np.ndarray) -> np.ndarray:
        return self.on[rows]

    def add(self, rows, who, stat, val=1.0):
        self._idx.append((rows * self.P + who) * len(STATS) + SI[stat])
        self._val.append(np.broadcast_to(np.asarray(val, dtype=float), rows.shape))

    def flush(self):
        if self._idx:
            np.add.at(self.stats, np.concatenate(self._idx), np.concatenate(self._val))
            self._idx, self._val = [], []


def simulate_batch(model: PossessionModel, items: list, n: int, rng) -> list:
    """items: [(home TeamGame, away TeamGame, possessions per team, form_h, form_a)]. Returns, per
    game, (home stats n x p x S, away stats)."""
    cal = model.cal
    G = len(items)
    P = max(max(len(h.keys), len(a.keys)) for h, a, *_ in items) or 1
    R = G * n
    aff_h = [model.affinity(h, a) for h, a, *_ in items]
    aff_a = [model.affinity(a, h) for h, a, *_ in items]
    H = _Side([it[0] for it in items], [it[3] for it in items], aff_h, n, P, rng, cal["usage_power"])
    A = _Side([it[1] for it in items], [it[4] for it in items], aff_a, n, P, rng, cal["usage_power"])
    g_of = np.repeat(np.arange(G), n)
    n_poss = np.repeat(np.array([it[2] for it in items]), n)
    pace = np.maximum(rng.normal(1.0, cal["pace_sd"], R), 0.8) * n_poss * cal["poss_scale"]
    total = (2 * np.floor(pace) + (rng.random(R) < (pace - np.floor(pace)))).astype(int)
    zl = np.asarray(cal["zone_logit"], dtype=np.float32)
    three_w = np.where(THREE, cal["three_scale"], 1.0).astype(np.float32)
    rows_all = np.arange(R)
    u = rng.random

    def possession(O: _Side, D: _Side, rows: np.ndarray, home_side: bool):
        m = len(rows)
        if m == 0:
            return
        on_o = O.on_court(rows)
        on_d = D.on_court(rows)
        g = g_of[rows]
        # 1. turnover
        w_tov = on_o * O.tov[rows]
        p_tov = np.clip(w_tov.sum(axis=1) * D.tov_f[g] * cal["tov_scale"], 0.02, 0.35)
        is_tov = u(m) < p_tov
        if is_tov.any():
            r = rows[is_tov]
            O.add(r, _pick(w_tov[is_tov], u(r.size)), "tov")
            w_stl = on_d[is_tov] * D.stl[r]
            st = u(r.size) < np.clip(w_stl.sum(axis=1) * cal["stl_scale"] / p_tov[is_tov], 0, 0.95)
            if st.any():
                D.add(r[st], _pick(w_stl[st], u(int(st.sum()))), "stl")
        live = ~is_tov
        rr, oo, dd = rows[live], on_o[live], on_d[live]
        fo, fd = O.form[rr], D.form[rr]
        putback_by = np.full(rr.size, -1)
        for chain in range(4):
            k = rr.size
            if k == 0:
                break
            gg = g_of[rr]
            ar = np.arange(k)
            # 2. who shoots (sometimes a putback by the offensive rebounder)
            shooter = _pick(oo * fo * O.opp_a[rr], u(k))
            pb = (putback_by >= 0) & (u(k) < cal["putback"])
            shooter = np.where(pb, putback_by, shooter)
            # 3. fouled before the shot (2-3 free throws, no FGA), or where he shoots from
            trip = (u(k) < np.clip(O.trip[rr, shooter] * cal["trip_scale"], 0, 0.6)) & ~pb
            zone = _pick(O.zp[gg, shooter] * D.zone_f[gg] * three_w[None, :], u(k))
            zone = np.where(pb, ZI["RA"], zone)
            miss_reb = np.zeros(k, dtype=bool)
            ft_miss_reb = np.zeros(k, dtype=bool)
            if trip.any():
                ti = np.flatnonzero(trip)
                nft = np.where(THREE[zone[ti]] & (u(ti.size) < cal["three_foul"]), 3, 2)
                pft = O.ft[rr[ti], shooter[ti]]
                makes = np.zeros(ti.size)
                last = np.zeros(ti.size, dtype=bool)
                for j in range(3):
                    mk = (u(ti.size) < pft) & (nft > j)
                    makes += mk
                    last = np.where(nft == j + 1, mk, last)
                O.add(rr[ti], shooter[ti], "fta", nft)
                O.add(rr[ti], shooter[ti], "ftm", makes)
                O.add(rr[ti], shooter[ti], "pts", makes)
                O.score[rr[ti]] += makes
                ft_miss_reb[ti] = ~last
            fi = np.flatnonzero(~trip)
            if fi.size:
                s, z, gf, r_ = shooter[fi], zone[fi], gg[fi], rr[fi]
                ddf, oof, fof = dd[fi], oo[fi], fo[fi]
                # who's guarding him: matchup history; at the rim, partly whoever protects the rim
                w_def = ddf * O.aff[gf, s]
                w_def = w_def / np.maximum(w_def.sum(axis=1, keepdims=True), 1e-9)
                inside = z <= ZI["PAINT"]
                if inside.any():
                    rimw = ddf[inside] * D.rim[r_[inside]]
                    rimw = rimw / np.maximum(rimw.sum(axis=1, keepdims=True), 1e-9)
                    w_def[inside] = 0.5 * w_def[inside] + 0.5 * rimw
                guard = _pick(w_def, u(fi.size))
                logit = O.fgl[gf, s, z] + D.eff[gf, guard, z] + zl[z] + (cal["home_logit"] if home_side else 0.0)
                p_make = 1 / (1 + np.exp(-logit))
                made = u(fi.size) < p_make
                three = THREE[z]
                pts = np.where(three, 3, 2) * made
                O.add(r_, s, "fga")
                O.add(r_, s, "fgm", made)
                O.add(r_, s, "fg3a", three)
                O.add(r_, s, "fg3m", three & made)
                O.add(r_, s, "pts", pts)
                O.score[r_] += pts
                # block (on misses), credited by block rate
                w_blk = ddf * D.blk[r_]
                p_blk = np.clip(BLOCKED[z] * w_blk.sum(axis=1) / max(model.blk_ref, 1e-9) * cal["blk_scale"]
                                / np.maximum(1 - p_make, 0.05), 0, 0.7)
                bl = ~made & (u(fi.size) < p_blk)
                if bl.any():
                    D.add(r_[bl], _pick(w_blk[bl], u(int(bl.sum()))), "blk")
                # assisted? the shooter's tendency in that zone, raised or lowered by how much the
                # teammates on the floor pass compared with his usual teammates; then who passed
                w_ast = oof * O.ast[r_]
                passing = w_ast.sum(axis=1) - w_ast[np.arange(fi.size), s]
                lift = (passing / np.maximum(O.ast_ref[r_, s], 1e-9)) ** cal["ast_lineup"]
                ast = made & (u(fi.size) < np.clip(O.as_[gf, s, z] * lift * cal["assist_scale"], 0, 0.99))
                if ast.any():
                    ai = np.flatnonzero(ast)
                    wp = oof[ai] * fof[ai] * O.pairs[gf[ai], :, s[ai]]
                    ok = wp.sum(axis=1) > 0
                    if ok.any():
                        O.add(r_[ai][ok], _pick(wp[ok], u(int(ok.sum()))), "ast")
                # and-one
                a1 = made & (u(fi.size) < np.clip(O.and1[r_, s] * AND1_W[z] * cal["trip_scale"], 0, 0.5))
                if a1.any():
                    i1 = np.flatnonzero(a1)
                    mk = u(i1.size) < O.ft[r_[i1], s[i1]]
                    O.add(r_[i1], s[i1], "fta")
                    O.add(r_[i1], s[i1], "ftm", mk)
                    O.add(r_[i1], s[i1], "pts", mk)
                    O.score[r_[i1]] += mk
                    ft_miss_reb[fi[i1]] = ~mk
                miss_reb[fi] = ~made
            # 4. rebounds
            need = miss_reb | ft_miss_reb
            if not need.any():
                break
            ni = np.flatnonzero(need)
            r_ = rr[ni]
            w_o = oo[ni] * fo[ni] * O.oreb[r_]
            w_d = dd[ni] * fd[ni] * D.dreb[r_]
            so = w_o.sum(axis=1) * cal["oreb_scale"]
            so = np.where(ft_miss_reb[ni] & ~miss_reb[ni], so * cal["ft_miss_oreb"], so)
            sd = w_d.sum(axis=1)
            off_gets = u(ni.size) < so / np.maximum(so + sd, 1e-9)
            credit = u(ni.size) < np.clip(so + sd, 0, 1)
            oc = off_gets & credit
            reb = None
            if oc.any():
                reb = _pick(w_o[oc], u(int(oc.sum())))
                O.add(r_[oc], reb, "oreb")
            dc = ~off_gets & credit
            if dc.any():
                D.add(r_[dc], _pick(w_d[dc], u(int(dc.sum()))), "dreb")
            keep = ni[off_gets]                               # offensive rebounds keep the ball
            nb = np.full(keep.size, -1)
            if reb is not None:
                nb[np.flatnonzero(oc[off_gets])] = reb
            rr, oo, dd, fo, fd, putback_by = rr[keep], oo[keep], dd[keep], fo[keep], fd[keep], nb

    kmax = int(total.max()) if R else 0
    stint = int(cal.get("stint", 4))
    for k in range(kmax):
        if k % stint == 0:
            H.new_stint(u(R))
            A.new_stint(u(R))
        rows = rows_all[k < total]
        if k % 2 == 0:
            possession(H, A, rows, True)
        else:
            possession(A, H, rows, False)
        if k % 16 == 15:
            H.flush()
            A.flush()
    for _ in range(3):                                      # overtime while tied
        H.flush()
        A.flush()
        tied = rows_all[H.score == A.score]
        if tied.size == 0:
            break
        for k in range(int(2 * cal["ot_poss"])):
            if k % stint == 0:
                H.new_stint(u(R))
                A.new_stint(u(R))
            possession(H, A, tied, True) if k % 2 == 0 else possession(A, H, tied, False)
    H.flush()
    A.flush()
    S = len(STATS)
    hs, as_ = H.stats.reshape(R, P, S), A.stats.reshape(R, P, S)
    return [(hs[g * n:(g + 1) * n, :len(items[g][0].keys)], as_[g * n:(g + 1) * n, :len(items[g][1].keys)])
            for g in range(G)]


def fantasy(stats: np.ndarray, scoring: dict) -> np.ndarray:
    """stats n x p x S -> fantasy points n x p (league weights; REB = OREB + DREB)."""
    s = {k: stats[..., SI[k]] for k in STATS}
    s["reb"] = s["oreb"] + s["dreb"]
    return sum(w * s[k.lower()] for k, w in scoring.items() if k.lower() in s)


def _prepare_game(world, d, home: str, away: str, avail, model: PossessionModel, rng, n: int, played_draw=None):
    """Minutes for both rotations in every simulation (absences, blowouts, back-to-backs), turned
    into TeamGame objects. Returns ({team: (TeamGame, played, minutes)}, possessions per team)."""
    pr, P, dm = world.params, world.players, world.defense
    teams = (home, away)
    exp_pace = {t: dm.pace.get(t, dm.league_pace) for t in teams}
    game_pace = exp_pace[home] + exp_pace[away] - dm.league_pace
    margin = rng.normal(((world.net.get(home, 0) - world.net.get(away, 0)) / 100 * game_pace
                         + pr.get("home_edge", 1.5)), pr["margin_sd"], n)
    blow = np.maximum(np.abs(margin) - pr["minutes"].get("blowout_threshold", 12), 0)
    tgs = {}
    for t in teams:
        keys = [k for k in world.rotation.get(t, []) if k in P.index]
        sub = P.loc[keys]
        p = np.array([avail(k, t, d) for k in keys])
        pl = played_draw(keys, p) if played_draw is not None else rng.random((n, len(keys))) < p[None, :]
        pos = sub["pos"].to_numpy()
        M = draw_minutes(sub["exp_min"].to_numpy(), pos, pl, d in world.b2b_days.get(t, set()), blow, pr, rng, n)
        tgs[t] = (model.team_game(keys, pos, M, t), pl, M)
    return tgs, game_pace


@dataclass
class GameDetail:
    """Every simulation of one NBA game: box scores (n x players x stats) for both teams."""
    date: date
    home: str
    away: str
    keys: dict          # team -> player keys
    stats: dict         # team -> n x p x S
    played: dict        # team -> n x p bool
    minutes: dict       # team -> n x p
    fp: dict            # team -> n x p fantasy points
    score: dict         # team -> n points

    @property
    def n(self) -> int:
        return len(self.score[self.home])

    def typical(self) -> int:
        """The most typical simulation: closest to the median score for both teams and, more
        loosely, to every player's median fantasy points (standardized distance)."""
        z = 0.0
        for t in (self.home, self.away):
            s = self.score[t]
            z = z + 3.0 * ((s - np.median(s)) / max(s.std(), 1e-9)) ** 2
            fp, pl = self.fp[t], self.played[t]
            med = np.array([np.median(fp[pl[:, j], j]) if pl[:, j].any() else 0.0 for j in range(fp.shape[1])])
            sd = np.maximum(fp.std(axis=0), 1.0)
            z = z + (((fp - med) / sd) ** 2 * pl).sum(axis=1) / max(fp.shape[1], 1)
        return int(np.argmin(z))

    def box(self, team: str, sim: int, names: dict | None = None) -> pd.DataFrame:
        st = self.stats[team][sim]
        rows = []
        for j, k in enumerate(self.keys[team]):
            if not self.played[team][sim, j] or self.minutes[team][sim, j] <= 0:
                continue
            r = {s: st[j, SI[s]] for s in STATS}
            rows.append({"Player": (names or {}).get(k, k.title()), "MIN": self.minutes[team][sim, j],
                         "PTS": r["pts"], "REB": r["oreb"] + r["dreb"], "AST": r["ast"], "STL": r["stl"],
                         "BLK": r["blk"], "TOV": r["tov"], "3PM": r["fg3m"], "FG": f"{int(r['fgm'])}-{int(r['fga'])}",
                         "FT": f"{int(r['ftm'])}-{int(r['fta'])}", "FP": self.fp[team][sim, j]})
        return pd.DataFrame(rows).sort_values("MIN", ascending=False)

    def players(self, team: str, names: dict | None = None) -> pd.DataFrame:
        """Each player's fantasy distribution over the simulations he played."""
        rows = []
        for j, k in enumerate(self.keys[team]):
            m = self.played[team][:, j]
            if not m.any():
                continue
            fp = self.fp[team][m, j]
            st = self.stats[team][m, j]
            rows.append({"Player": (names or {}).get(k, k.title()), "Plays": m.mean(),
                         "MIN": self.minutes[team][m, j].mean(), "PTS": st[:, SI["pts"]].mean(),
                         "REB": (st[:, SI["oreb"]] + st[:, SI["dreb"]]).mean(), "AST": st[:, SI["ast"]].mean(),
                         "FP": fp.mean(), "FP 10%": np.quantile(fp, .1), "FP 90%": np.quantile(fp, .9)})
        return pd.DataFrame(rows).sort_values("FP", ascending=False)


def game_detail(world, d, home: str, away: str, avail, model: PossessionModel, n: int = 5000,
                seed: int | None = None) -> GameDetail:
    """Simulate one NBA game n times and keep every box score."""
    rng = np.random.default_rng(seed)
    tgs, pace = _prepare_game(world, d, home, away, avail, model, rng, n)
    th, ta = tgs[home][0], tgs[away][0]
    sd = model.cal.get("form_sd", 0.0)
    fh, fa = ((np.maximum(rng.lognormal(-sd ** 2 / 2, sd, (n, len(tg.keys))), 0.2) if sd else None) for tg in (th, ta))
    (sh, sa), = simulate_batch(model, [(th, ta, pace, fh, fa)], n, rng)
    out = {"keys": {}, "stats": {}, "played": {}, "minutes": {}, "fp": {}, "score": {}}
    for t, stats in ((home, sh), (away, sa)):
        tg, pl, M = tgs[t]
        fp = fantasy(stats, world.scoring) * pl
        out["keys"][t], out["stats"][t], out["played"][t], out["minutes"][t] = tg.keys, stats, pl, M * pl
        out["fp"][t], out["score"][t] = fp, stats[..., SI["pts"]].sum(axis=1)
    return GameDetail(d, home, away, **out)


def simulate_games(world, games: list[tuple[date, str, str]], avail, model: PossessionModel, n: int = 5000,
                   seed: int | None = 0, interest: set[str] | None = None, max_rows: int = 40000,
                   rho: float = 0.0) -> GameSims:
    """Same interface and output as sim/engine.py simulate_games, with possession-level games."""
    from tradecalc.sim.engine import absence_weeks

    rng = np.random.default_rng(seed)
    played_draw = absence_weeks(rng, n, rho)
    pr, P = world.params, world.players
    dm = world.defense
    fp_out, played_out, box_out = {}, {}, {}
    form_sd = model.cal.get("form_sd", 0.0)
    form: dict[str, np.ndarray] = {}

    def form_of(keys):
        for k in keys:
            if k not in form:
                form[k] = np.maximum(rng.lognormal(-form_sd ** 2 / 2, form_sd, n), 0.2) if form_sd else np.ones(n)
        return np.stack([form[k] for k in keys], axis=1) if keys else np.ones((n, 0))
    prepared = [(d, home, away, *_prepare_game(world, d, home, away, avail, model, rng, n, played_draw))
                for d, home, away in games]
    per_batch = max(1, max_rows // max(n, 1))
    for b0 in range(0, len(prepared), per_batch):
        chunk = prepared[b0:b0 + per_batch]
        items = [(tgs[h][0], tgs[a][0], pace, form_of(tgs[h][0].keys), form_of(tgs[a][0].keys))
                 for _, h, a, tgs, pace in chunk]
        for (d, home, away, tgs, _), (sh, sa) in zip(chunk, simulate_batch(model, items, n, rng)):
            for t, stats in ((home, sh), (away, sa)):
                tg, pl, M = tgs[t]
                fp = fantasy(stats, world.scoring)
                for j, k in enumerate(tg.keys):
                    if interest is not None and k not in interest:
                        continue
                    m = pl[:, j]
                    fp_out[(k, d)] = fp[:, j] * m
                    played_out[(k, d)] = m
                    box = {s: stats[:, j, SI[s]] for s in STATS}
                    box["reb"] = box["oreb"] + box["dreb"]
                    box["min"] = M[:, j]
                    box_out[(k, d)] = {s: float(box[s][m].mean()) if m.any() else 0.0 for s in BOX}
                    box_out[(k, d)]["fp"] = float(fp[:, j][m].mean()) if m.any() else 0.0
    return GameSims(fp_out, played_out, box_out)
