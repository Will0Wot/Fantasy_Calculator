"""Player and team possession profiles from play-by-play (ingest/pbp.py), for the possession
simulator. Everything is "as of" a date: only events before it are used, so the same code serves
backtests and the live season.

Per player (offense):
  opp        shot opportunities per team possession he is on the floor (FGA + fouled-shot trips)
  trip       share of his shot opportunities that end in a shooting foul with no FGA (2-3 FTs)
  zp_<z>     where he shoots from: share of FGA by zone (RA, PAINT, MID, C3, AB3)
  fg_<z>     how often he makes it, by zone
  as_<z>     share of his makes in that zone that are assisted
  and1       and-one free throws per made shot
  ft         free-throw percentage
  tov        turnovers per team possession on the floor
  ast        assists per team possession on the floor (passing volume)
Per player (defense / boards):
  stl        steals per opponent possession on the floor
  blk        blocks per opponent shot attempt on the floor
  oreb, dreb share of his team's / the opponent's missed shots he rebounds while on the floor
Pairs: assists from passer to shooter ("who he passes to").
Team defense: forced-turnover factor and shot-zone allowance factors (how often opponents get to
each zone vs league average).

Exposure (possessions on the floor) is the team's possessions in each game times his share of the
team's minutes. Every rate is shrunk toward his position group's average (G / F / C) by an
empirical-Bayes weight n / (n + K), and last season counts at half weight as the prior.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tradecalc.ingest.pbp import ZONES
from tradecalc.model.names import norm_name

PREV_WEIGHTS = (0.5,)       # weight of last season, the season before, ... (this season: 1)
# prior strength (in exposure units: possessions, attempts or rebound chances), fitted by split-half
# prediction (odd games predict even games) on 2021-22 and 2022-23
K = {"opp": 50, "trip": 80, "zp": 15, "fg_RA": 120, "fg_PAINT": 80, "fg_MID": 120, "fg_C3": 200, "fg_AB3": 300,
     "as": 10, "and1": 80, "ft": 30, "tov": 200, "ast": 80, "stl": 800, "blk": 300, "oreb": 120, "dreb": 120}
TEAM_K = {"tov": 2500, "zone": 1500}


@dataclass
class Profiles:
    players: pd.DataFrame                      # index: player key (normalized name)
    pairs: pd.DataFrame                        # passer, shooter, n
    teams: pd.DataFrame                        # index: NBA team; defensive factors
    league: dict
    names: dict = field(default_factory=dict)  # key -> display name


def _possessions(E: pd.DataFrame) -> pd.DataFrame:
    """Per game and team: possessions, missed shots (FG + last free throw), FGA."""
    g = ["game_id", "team"]
    sh = E[E.etype == "shot"]
    ft = E[E.etype == "ft"]
    c = pd.DataFrame({
        "fga": sh.groupby(g).size(),
        "fg_miss": sh[~sh.made.astype(bool)].groupby(g).size(),
        "fta": ft.groupby(g).size(),
        "ft_last_miss": ft[(ft.ft_n == ft.ft_of) & ~ft.made.astype(bool)].groupby(g).size(),
        "oreb": E[E.etype == "oreb"].groupby(g).size(),
        "tov": E[E.etype == "tov"].groupby(g).size(),
    }).fillna(0)
    c["poss"] = c.fga + 0.44 * c.fta - c.oreb + c.tov
    c["miss"] = c.fg_miss + c.ft_last_miss
    return c


def _counts(E: pd.DataFrame, PL: pd.DataFrame, key_of: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per player counts and exposures, and assist pairs, from one window of events."""
    E = E.copy()
    E["k1"] = E.p1.map(key_of)
    E["k2"] = E.p2.map(key_of)
    tg = _possessions(E)
    opp_of = E.dropna(subset=["team", "opp"]).groupby(["game_id", "team"]).opp.first()
    tg["opp"] = opp_of.reindex(tg.index)
    o = tg.reset_index().set_index(["game_id", "opp"])[["poss", "miss", "fga"]]
    o.index.names = ["game_id", "team"]
    tg = tg.join(o.rename(columns={"poss": "opp_poss", "miss": "opp_miss", "fga": "opp_fga"}))
    # minutes share per player-game -> exposures
    pl = PL[PL["min"] > 0].copy()
    pl["key"] = pl.espn_id.map(key_of)
    team_min = pl.groupby(["game_id", "team"])["min"].transform("sum")
    pl["share"] = 5 * pl["min"] / team_min.where(team_min > 0, 240)
    pl = pl.join(tg[["poss", "miss", "opp_poss", "opp_miss", "opp_fga"]], on=["game_id", "team"])
    for c in ("poss", "miss", "opp_poss", "opp_miss", "opp_fga"):
        pl[f"{c}_on"] = pl[c] * pl["share"]
    expo = pl.groupby("key")[["min", "poss_on", "miss_on", "opp_poss_on", "opp_miss_on", "opp_fga_on"]].sum()
    expo["games"] = pl.groupby("key").size()

    sh = E[E.etype == "shot"]
    made = sh[sh.made.astype(bool)]
    ft = E[E.etype == "ft"]
    cnt = pd.DataFrame(index=expo.index)
    for z in ZONES:
        cnt[f"fga_{z}"] = sh[sh.zone == z].groupby("k1").size()
        cnt[f"fgm_{z}"] = made[made.zone == z].groupby("k1").size()
        cnt[f"ast_{z}"] = made[(made.zone == z) & made.k2.notna()].groupby("k1").size()
    cnt["trips"] = ft[(ft.ft_n == 1) & ft.ft_of.isin([2, 3])].groupby("k1").size()
    cnt["and1"] = ft[ft.ft_of == 1].groupby("k1").size()
    cnt["fta"] = ft.groupby("k1").size()
    cnt["ftm"] = ft[ft.made.astype(bool)].groupby("k1").size()
    tov = E[E.etype == "tov"]
    cnt["tov"] = tov.groupby("k1").size()
    cnt["stl"] = tov[tov.k2.notna()].groupby("k2").size()
    cnt["blk"] = sh[sh.blocked.astype(bool) & sh.k2.notna()].groupby("k2").size()
    cnt["oreb"] = E[E.etype == "oreb"].groupby("k1").size()
    cnt["dreb"] = E[E.etype == "dreb"].groupby("k1").size()
    cnt["ast"] = made[made.k2.notna()].groupby("k2").size()
    out = expo.join(cnt.fillna(0), how="left").fillna(0)
    pairs = made[made.k2.notna() & made.k1.notna()].groupby(["k2", "k1"]).size().rename("n").reset_index()
    pairs.columns = ["passer", "shooter", "n"]
    return out, pairs


def _team_defense(E: pd.DataFrame) -> pd.DataFrame:
    """Per defensive team: opponent turnovers per possession and opponent zone shares."""
    tg = _possessions(E).reset_index()
    opp = E.dropna(subset=["team", "opp"]).groupby(["game_id", "team"]).opp.first().rename("defense")
    tg = tg.join(opp, on=["game_id", "team"])
    d = tg.groupby("defense")[["poss", "tov"]].sum()
    sh = E[E.etype == "shot"]
    z = sh.groupby(["opp", "zone"]).size().unstack(fill_value=0)
    return d.join(z, how="left").fillna(0)


@dataclass
class Counts:
    """One window of play-by-play reduced to counts (what profiles are built from). Small enough to
    commit for finished seasons, so the app never needs raw play-by-play."""
    players: pd.DataFrame       # index player key: exposures and event counts
    pairs: pd.DataFrame         # passer, shooter, n
    teams: pd.DataFrame         # index defensive team: possessions, turnovers, opponent shots by zone
    names: dict

    def save(self, base: Path) -> None:
        base.parent.mkdir(parents=True, exist_ok=True)
        self.players.assign(name=[self.names.get(k, k) for k in self.players.index]).to_parquet(f"{base}_players.parquet")
        self.pairs.to_parquet(f"{base}_pairs.parquet", index=False)
        self.teams.to_parquet(f"{base}_teams.parquet")

    @classmethod
    def load(cls, base: Path) -> "Counts | None":
        if not Path(f"{base}_players.parquet").exists():
            return None
        pl = pd.read_parquet(f"{base}_players.parquet")
        names = dict(zip(pl.index, pl.pop("name")))
        return cls(pl, pd.read_parquet(f"{base}_pairs.parquet"), pd.read_parquet(f"{base}_teams.parquet"), names)


def counts(E: pd.DataFrame, PL: pd.DataFrame) -> Counts:
    key_of = {i: norm_name(n) for i, n in zip(PL.espn_id, PL.name)}
    names = {norm_name(n): n for n in PL.name}
    c, pr = _counts(E, PL, key_of)
    return Counts(c, pr, _team_defense(E), names)


def build(events: list[tuple[pd.DataFrame, pd.DataFrame, float]], pos_of) -> Profiles:
    """events: [(events, players, weight)], e.g. [(this season before the date, ..., 1.0),
    (last season, ..., 0.5)]. pos_of(key) -> 'G' | 'F' | 'C'."""
    return from_counts([(counts(E, PL), w) for E, PL, w in events if E is not None and len(E)], pos_of)


def from_counts(windows: list[tuple[Counts, float]], pos_of) -> Profiles:
    """Profiles from weighted count windows (see build)."""
    names, tot, pairs, tdef = {}, None, [], None
    for cn, w in windows:
        names.update(cn.names)
        c = cn.players * w
        tot = c if tot is None else tot.add(c, fill_value=0)
        pairs.append(cn.pairs.assign(n=cn.pairs.n * w))
        td = cn.teams * w
        tdef = td if tdef is None else tdef.add(td, fill_value=0)
    pairs = (pd.concat(pairs).groupby(["passer", "shooter"]).n.sum().reset_index() if pairs
             else pd.DataFrame(columns=["passer", "shooter", "n"]))
    tot = tot.fillna(0)
    tot["pos"] = [pos_of(k) or "F" for k in tot.index]
    fga = tot[[f"fga_{z}" for z in ZONES]].sum(axis=1)
    fgm = tot[[f"fgm_{z}" for z in ZONES]].sum(axis=1)
    tot["fga"], tot["fgm"] = fga, fgm
    tot["shot_opp"] = fga + tot.trips

    # position-group priors (league averages within the window)
    def rate(num, den):
        return lambda g: g[num].sum() / max(g[den].sum(), 1e-9)

    pri = {}
    for grp, g in tot.groupby("pos"):
        p = {"opp": rate("shot_opp", "poss_on")(g), "trip": rate("trips", "shot_opp")(g),
             "and1": rate("and1", "fgm")(g), "ft": rate("ftm", "fta")(g), "tov": rate("tov", "poss_on")(g),
             "ast": rate("ast", "poss_on")(g), "stl": rate("stl", "opp_poss_on")(g), "blk": rate("blk", "opp_fga_on")(g),
             "oreb": rate("oreb", "miss_on")(g), "dreb": rate("dreb", "opp_miss_on")(g)}
        zf = g[[f"fga_{z}" for z in ZONES]].sum()
        for z in ZONES:
            p[f"zp_{z}"] = zf[f"fga_{z}"] / max(zf.sum(), 1e-9)
            p[f"fg_{z}"] = g[f"fgm_{z}"].sum() / max(g[f"fga_{z}"].sum(), 1e-9)
            p[f"as_{z}"] = g[f"ast_{z}"].sum() / max(g[f"fgm_{z}"].sum(), 1e-9)
        pri[grp] = p

    P = pd.DataFrame(index=tot.index)
    P["pos"] = tot.pos
    P["min"], P["games"], P["poss_on"] = tot["min"], tot.games, tot.poss_on

    def shrink(name, num, den, kk):
        prior = tot.pos.map(lambda g: pri.get(g, pri.get("F"))[name])
        P[name] = (tot[num] + kk * prior) / (tot[den] + kk)

    shrink("opp", "shot_opp", "poss_on", K["opp"])
    shrink("trip", "trips", "shot_opp", K["trip"])
    shrink("and1", "and1", "fgm", K["and1"])
    shrink("ft", "ftm", "fta", K["ft"])
    shrink("tov", "tov", "poss_on", K["tov"])
    shrink("ast", "ast", "poss_on", K["ast"])
    shrink("stl", "stl", "opp_poss_on", K["stl"])
    shrink("blk", "blk", "opp_fga_on", K["blk"])
    shrink("oreb", "oreb", "miss_on", K["oreb"])
    shrink("dreb", "dreb", "opp_miss_on", K["dreb"])
    for z in ZONES:
        prior_zp = tot.pos.map(lambda g: pri.get(g, pri["F"])[f"zp_{z}"])
        P[f"zp_{z}"] = (tot[f"fga_{z}"] + K["zp"] * prior_zp) / (fga + K["zp"])
        shrink(f"fg_{z}", f"fgm_{z}", f"fga_{z}", K[f"fg_{z}"])
        shrink(f"as_{z}", f"ast_{z}", f"fgm_{z}", K["as"])
    zcols = [f"zp_{z}" for z in ZONES]
    P[zcols] = P[zcols].div(P[zcols].sum(axis=1), axis=0)

    # league numbers and team defense
    L = {g: pri[g] for g in pri}
    zf_all = tot[[f"fga_{z}" for z in ZONES]].sum()
    lg = {"zone_freq": {z: zf_all[f"fga_{z}"] / zf_all.sum() for z in ZONES},
          "fg": {z: tot[f"fgm_{z}"].sum() / max(tot[f"fga_{z}"].sum(), 1) for z in ZONES},
          "tov": tot.tov.sum() / max(tot.poss_on.sum() / 5, 1), "pos": L}
    T = pd.DataFrame(index=tdef.index)
    lg_tov = tdef.tov.sum() / tdef.poss.sum()
    T["tov_f"] = ((tdef.tov + TEAM_K["tov"] * lg_tov) / (tdef.poss + TEAM_K["tov"])) / lg_tov
    zt = tdef[[z for z in ZONES if z in tdef]].sum(axis=1)
    for z in ZONES:
        share = lg["zone_freq"][z]
        T[f"zone_{z}"] = ((tdef.get(z, 0) + TEAM_K["zone"] * share) / (zt + TEAM_K["zone"])) / share
    return Profiles(P, pairs, T, lg, names)


def load_events(pbp_dir: Path, season: str, before: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    from tradecalc.ingest.pbp import load

    E, PL = load(pbp_dir, season)
    if before is not None and len(E):
        E = E[E.date < before]
        PL = PL[PL.date < before]
    return E, PL
