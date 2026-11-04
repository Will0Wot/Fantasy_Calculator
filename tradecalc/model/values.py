"""Build per-player model state (mu, sigma, availability, ROS value) for the current league.

Joins ESPN players to stats.nba.com game logs by normalized name, then applies the
section 4 model: ADP/projection prior, shrinkage blend, variance, availability,
replacement level and ROS value.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from tradecalc.ingest import adp as adp_mod
from tradecalc.ingest import nba
from tradecalc.ingest.league import LeagueData, Player
from tradecalc.model.availability import (OUT_STATUSES, Availability, availability, calibrated_rate,
                                          games_played_rate)
from tradecalc.model.blend import blend_minutes
from tradecalc.model.injury import (InjuryView, injury_view, load_profiles, log_current_injuries,
                                    logged_history, long_absences, manual_history)
from tradecalc.model.names import norm_name
from tradecalc.model.prior import fit_adp_curve, prior_table
from tradecalc.model.scoring import add_fp, bonus_rates, fantasy_points, live_stats
from tradecalc.model.variance import shrunk_sigma
from tradecalc.schedule.calendar import Calendar


@dataclass
class PlayerState:
    espn_id: int
    name: str
    pro_team: str
    positions: tuple[str, ...]
    on_team_id: int
    injury_status: str
    mu: float
    sigma: float
    avail: Availability
    n: int = 0
    prior: float = float("nan")
    live: float = float("nan")
    role_change: bool = False
    ros_value: float = 0.0
    games_remaining: int = 0
    flags: list[str] = field(default_factory=list)
    injury: InjuryView | None = None
    minutes: float = float("nan")    # projected minutes per game (depth chart, last season, this season)
    role: str = ""                   # Starter / Rotation / Bench
    mu_by_date: dict = field(default_factory=dict)   # per-game projection where game simulations exist
    last_fppg: float = float("nan")  # track record: last season's FPPG and games (the season before
    last_games: int = 0              # if he barely played last season)
    last_season: str = ""

    def mu_on(self, d) -> float:
        """Projected fantasy points if he plays on date d (that game's projection, else his average)."""
        return self.mu_by_date.get(d, self.mu)


@dataclass
class LeagueState:
    cfg: dict
    league: LeagueData
    cal: Calendar
    today: date
    players: dict[int, PlayerState]
    rosters: dict[int, list[int]]
    replacement: dict[str, PlayerState]   # best free agent per position
    # pro team -> {date: index of that game among the team's games on/after today}
    team_game_index: dict[str, dict[date, int]]
    ir: dict[int, set[int]] = field(default_factory=dict)      # team -> players in IR slots
    fills: dict[int, list[int]] = field(default_factory=dict)  # team -> FAs filling open active spots
    real_today: date | None = None   # the actual calendar date (today is moved to opening night preseason)

    def banked(self, team_id: int, week) -> tuple[float, date | None]:
        """Fantasy points a team has already scored this week on days before `today`, and the
        last such day. Those days are not simulated; their points are added as-is."""
        pts, last = 0.0, None
        for m in getattr(self.league, "matchups", []):
            if m["week"] != week.number:
                continue
            side = "home_points" if m["home"] == team_id else "away_points" if m["away"] == team_id else None
            if side is None:
                continue
            for sp, v in m[side].items():
                d = self.cal.first_day + timedelta(days=sp - 1)
                if week.start <= d < self.today:
                    pts += v
                    last = d if last is None or d > last else last
        return pts, last

    def week_status(self) -> str:
        """Human-readable date line: today's date, season phase, current week and days left."""
        real = self.real_today or self.today
        head = f"{real:%A, %b %d, %Y}"
        if real < self.cal.first_day:
            return f"{head} · Preseason: opening night is {self.cal.first_day:%a, %b %d}"
        w = self.cal.week_of(real)
        if w is None:
            return f"{head} · Regular season is over"
        total = len(w.days())
        left = (w.end - real).days + 1
        return (f"{head} · Week {w.number} ({w.start:%b %d}–{w.end:%b %d}) · day {total - left + 1} of {total}, "
                f"{left} day{'s' if left != 1 else ''} left including today")

    @property
    def active_size(self) -> int:
        r = self.cfg["roster"]
        return sum(r["slots"].values()) + r["bench"]

    def ir_eligible(self, pid: int) -> bool:
        from tradecalc.model.availability import OUT_STATUSES

        return self.players[pid].injury_status in OUT_STATUSES

    def tradeable(self, team_id: int) -> list[int]:
        fill = set(self.fills.get(team_id, ()))
        return [p for p in self.rosters[team_id] if p not in fill]

    def team_name(self, team_id: int) -> str:
        return self.league.teams[team_id].name

    def a_for(self, p: PlayerState, d: date) -> float:
        j = self.team_game_index.get(p.pro_team, {}).get(d)
        return 0.0 if j is None else p.avail.for_game(j, d)

    def a_next(self, p: PlayerState, day: date) -> float:
        """P(plays) in his team's first game on or after `day` (0 if none left)."""
        games = self.team_game_index.get(p.pro_team, {})
        nxt = min((d for d in games if d >= day), default=None)
        return 0.0 if nxt is None else p.avail.for_game(games[nxt], nxt)

    def free_agents(self) -> list[PlayerState]:
        return [p for p in self.players.values() if p.on_team_id == 0]


LONG_ABSENCE_DAYS = 14   # out at least this much longer: valued on his record before the injury


def _season_before(season: str) -> str:
    y = int(season[:4]) - 1
    return f"{y}-{str(y + 1)[2:]}"


def _status(p: Player, rep, view: InjuryView, override: dict | None) -> tuple[str, int | None, date | None]:
    """(status, games out, return date): the injury report's status (else the fantasy platform's),
    with any manual correction from injury_overrides.csv."""
    status = rep.status if rep is not None else p.injury_status
    games_out, return_date = None, view.return_date
    if override is not None:
        status = str(override.get("status") or status)
        go = override.get("games_out")
        games_out = int(go) if go == go and go is not None else None
        rd = override.get("return_date")
        if isinstance(rd, str) and rd:
            return_date = date.fromisoformat(rd)
    return status, games_out, return_date


def _long_absence(s: tuple, view: InjuryView, today: date) -> bool:
    """Out at least LONG_ABSENCE_DAYS more (by return date, games out, else the injury's usual length)."""
    status, games_out, return_date = s
    if status == "OUT_FOR_SEASON":
        return True
    if return_date is not None:
        return (return_date - today).days >= LONG_ABSENCE_DAYS
    if games_out is not None:
        return games_out >= 7
    return (view.typical_days or 0) >= LONG_ABSENCE_DAYS


def _record_fppg(prev_logs: pd.DataFrame, prev2_logs: pd.DataFrame, scoring, min_games: int = 20) -> pd.Series:
    """FPPG in a player's last season with at least `min_games` games (last season, else the one
    before), by normalized name."""
    out = pd.Series(dtype=float)
    for logs in (prev2_logs, prev_logs):               # last season wins where both qualify
        if not len(logs):
            continue
        logs = logs if "fp" in logs else add_fp(logs, scoring)
        g = logs[logs["min"] > 0].assign(key=lambda d: d["player_name"].map(norm_name)).groupby("key")["fp"]
        out = g.mean()[g.size() >= min_games].combine_first(out)
    return out


def _track_record(prev_logs: pd.DataFrame, prev2_logs: pd.DataFrame, scoring, prev_season: str,
                  min_games: int = 20) -> dict[str, dict]:
    """{normalized name: last_fppg, last_games, last_season}: last season, or the season before when
    he played fewer than `min_games` last season (and more then)."""
    out: dict[str, dict] = {}
    for logs, season in ((prev2_logs, _season_before(prev_season)), (prev_logs, prev_season)):
        if not len(logs):
            continue
        logs = logs if "fp" in logs else add_fp(logs, scoring)
        g = logs[logs["min"] > 0].assign(key=lambda d: d["player_name"].map(norm_name)).groupby("key")["fp"]
        for k, (fppg, n) in pd.DataFrame({"f": g.mean(), "n": g.size()}).iterrows():
            if k not in out or n >= min_games or n > out[k]["last_games"]:
                out[k] = {"last_fppg": float(fppg), "last_games": int(n), "last_season": season}
    return out


def _prior_minutes(league: LeagueData, ptab: pd.DataFrame, minutes_proj, out_keys: set[str],
                   prev_logs: pd.DataFrame) -> pd.Series:
    """Minutes per game behind the prior, by normalized name, so that prior / minutes is a real
    points-per-minute rate: the average of the minutes each piece of the prior assumes (ESPN's projected
    minutes for its projection, the engine's minutes for its piece; healthy minutes for injured players).
    A prior made only of the ADP piece takes last season's minutes (10+ games)."""
    has_espn = set(ptab.index[ptab["proj_fppg"].notna()]) if "proj_fppg" in ptab else set()
    espn = {}
    for p in league.players.values():
        k = norm_name(p.name)
        if k in has_espn and (p.projection or {}).get("MIN"):
            espn.setdefault(k, float(p.projection["MIN"]))
    parts = [pd.Series(espn, dtype=float)]
    if minutes_proj is not None and len(minutes_proj):
        eng = minutes_proj["minutes"].groupby(level=0).first()
        if "healthy_min" in minutes_proj:
            healthy = minutes_proj["healthy_min"].groupby(level=0).first().reindex(eng.index)
            eng = eng.where(~eng.index.isin(list(out_keys)), healthy)
        parts.append(eng)
    out = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
    if len(prev_logs):
        g = prev_logs[prev_logs["min"] > 0].assign(key=lambda d: d["player_name"].map(norm_name)).groupby("key")["min"]
        out = out.combine_first(g.mean()[g.size() >= 10])
    return out


def load_injury_overrides(path) -> dict[str, dict]:
    """Optional CSV: name, status, games_out, return_date. Overrides ESPN injury info per player."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return {}
    df.columns = [c.strip().lower() for c in df.columns]
    return {norm_name(r["name"]): r for r in df.to_dict("records")}


def build_state(cfg, conn, league: LeagueData, today: date | None = None, injuries=None,
                minutes_proj: pd.DataFrame | None = None) -> LeagueState:
    """injuries: list of ingest.injuries.Injury (ESPN injury report), or None to use only
    the fantasy API's injury status.
    minutes_proj: sim.engine.minutes_projection() output. When given, each player's prior is
    the average of three views: ADP curve, ESPN projection, and projected minutes x his
    per-minute production, so role (starter vs bench) and minutes changes move the projection."""
    lg, m, scoring = cfg["league"], cfg["model"], cfg["scoring"]
    cal = Calendar(league, cfg)
    from tradecalc.clock import league_today

    real_today = league_today(cfg)
    today = today or real_today
    if today < cal.first_day:
        today = cal.first_day
    ic = load_profiles()
    injuries = injuries or []
    if injuries and today == max(real_today, cal.first_day):
        log_current_injuries(conn, injuries, ic, real_today)

    cur_logs = add_fp(nba.game_logs(conn, lg["nba_season"]), scoring)
    prev_logs = add_fp(nba.game_logs(conn, lg["prev_nba_season"]), scoring)
    prev2_logs = nba.game_logs(conn, _season_before(lg["prev_nba_season"]))

    # --- prior: ADP curve (fit on last season) averaged with ESPN's projection.
    curve = fit_adp_curve(adp_mod.load_adp(cfg.path(cfg["data"]["prev_adp_csv"])), live_stats(prev_logs, scoring))
    # Bonuses (double-doubles etc.) aren't in a projection of averages: add each player's own rate.
    bonus = bonus_rates(pd.concat([prev_logs, cur_logs]), scoring, norm_name)
    proj = {p.name: fantasy_points(p.projection, scoring) + float(bonus.get(norm_name(p.name), 0.0))
            for p in league.players.values() if p.projection.get("GP")}
    ptab = prior_table(adp_mod.load_adp(cfg.path(cfg["data"]["adp_csv"])), curve, proj)

    # --- injuries: current report + history -> per-player view (model/injury.py).
    prev_logs_raw = nba.game_logs(conn, lg["prev_nba_season"])
    report = {i.espn_id: i for i in injuries if i.espn_id is not None}
    report_by_name = {norm_name(i.name): i for i in injuries}
    logged = logged_history(conn, ic, real_today, set(report))
    manual = manual_history(cfg.path(cfg["data"].get("injury_history_csv", "data/injury_history.csv")))
    absent = {norm_name(n): g for g, n in long_absences(prev_logs_raw, ic.long_absence_games).values()}
    from tradecalc.model.injury import HistoryEntry
    views: dict[int, InjuryView] = {}
    for p in league.players.values():
        key = norm_name(p.name)
        hist = list(logged.get(p.espn_id, [])) + list(manual.get(key, []))
        if not hist and key in absent:
            hist.append(HistoryEntry("long_absence", None, f"game logs: missed {absent[key]} straight"))
        views[p.espn_id] = injury_view(p, report.get(p.espn_id) or report_by_name.get(key), hist, ic,
                                       today, cal.first_day)
    overrides = load_injury_overrides(cfg.path(cfg["data"].get("injury_overrides_csv", "data/injury_overrides.csv")))
    statuses = {p.espn_id: _status(p, report.get(p.espn_id) or report_by_name.get(norm_name(p.name)),
                                   views[p.espn_id], overrides.get(norm_name(p.name)))
                for p in league.players.values()}
    # Research multiplier applies to the ADP-curve and engine pieces of the prior; ESPN's
    # projection already reflects their analysts' view of the injury.
    mult = pd.Series({norm_name(league.players[i].name): v.fppg_mult for i, v in views.items()})
    mult = mult.groupby(level=0).min().reindex(ptab.index).fillna(1.0)
    curve_adj = ptab["curve_fppg"] * mult
    # A player who is out is valued per game once he's back. A long absence also sinks his draft
    # position (drafters price in the games he'll miss, which availability already counts), so his
    # record before the injury stands in for the ADP piece. Players come back at about 92% of their
    # earlier level after long absences (2021-22..2025-26, docs/model_audit.md §10); the injury
    # multiplier covers the bigger drops after major injuries.
    out_keys = {norm_name(league.players[i].name) for i, s in statuses.items() if s[0] in OUT_STATUSES}
    long_out = {norm_name(league.players[i].name) for i, s in statuses.items()
                if s[0] in OUT_STATUSES and _long_absence(s, views[i], today)}
    record = _record_fppg(prev_logs, prev2_logs, scoring)
    track = _track_record(prev_logs, prev2_logs, scoring, lg["prev_nba_season"])
    swap = [k for k in long_out if k in curve_adj.index and k in record.index]
    curve_adj.loc[swap] = record.loc[swap].to_numpy() * mult.loc[swap].to_numpy()
    comps = [curve_adj, ptab["proj_fppg"]]
    if minutes_proj is not None and len(minutes_proj):
        mp = minutes_proj["fppg_min"].groupby(level=0).first()
        if "fppg_healthy" in minutes_proj:
            healthy = minutes_proj["fppg_healthy"].groupby(level=0).first()
            hurt = mp.index.isin(list(out_keys))
            mp = mp.where(~hurt, healthy.reindex(mp.index))
        if len(bonus):
            mp = mp + bonus.reindex(mp.index).fillna(0.0)
        ptab = ptab.reindex(ptab.index.union(mp.index))
        curve_adj = curve_adj.reindex(ptab.index)
        mult = mult.reindex(ptab.index).fillna(1.0)
        comps = [curve_adj, ptab["proj_fppg"], mp.reindex(ptab.index) * mult]
    prior = pd.concat(comps, axis=1).mean(axis=1, skipna=True).dropna()

    # --- live stats this season, keyed by normalized name.
    live = live_stats(cur_logs, scoring, recent_games=m["role_change_recent_games"]) if len(cur_logs) else None
    if live is not None:
        live = live.assign(key=live["player_name"].map(norm_name)).drop_duplicates("key").set_index("key")
        b = blend_minutes(prior, _prior_minutes(league, ptab, minutes_proj, out_keys, prev_logs), live,
                          minutes_k=m["minutes_k"], rate_k=m["rate_k"], recent_weight=m["recent_minutes_weight"],
                          role_threshold=m["role_change_threshold"], role_min_games=m["role_change_min_games"],
                          replacement=curve.tail_value, k_fallback=m["k"])
    # --- games-played rate over the last two seasons, keyed by normalized name. Each season's
    # longest absence (if >= long_absence_games) is dropped from the denominator, giving a
    # healthy baseline: the injury itself is modeled by its profile (model/injury.py), so it
    # is not counted twice. Repeated shorter absences still lower the rate.
    names = pd.concat([prev_logs, prev2_logs])[["player_id", "player_name"]].drop_duplicates("player_id")
    excluded: dict[int, int] = {}
    for logs_s in (prev_logs, prev2_logs):
        if len(logs_s):
            for pid, (g, _) in long_absences(logs_s, ic.long_absence_games).items():
                excluded[pid] = excluded.get(pid, 0) + g
    gp = games_played_rate([l for l in (prev2_logs, prev_logs) if len(l)], excluded=excluded)
    gp_by_name = {norm_name(n): gp.get(pid) for pid, n in zip(names["player_id"], names["player_name"])}

    team_games: dict[str, list[date]] = {}
    for d, teams in sorted(cal.games_by_day.items()):
        if d >= today:
            for t in teams:
                team_games.setdefault(t, []).append(d)
    team_game_index = {t: {d: j for j, d in enumerate(ds)} for t, ds in team_games.items()}
    # this season so far: team games played and each player's appearances (for availability)
    team_played = {}
    for d, teams in cal.games_by_day.items():
        if d < today:
            for t_ in teams:
                team_played[t_] = team_played.get(t_, 0) + 1
    _played = (cur_logs[cur_logs["min"] > 0].assign(key=lambda d_: d_.player_name.map(norm_name))
               if len(cur_logs) else pd.DataFrame(columns=["key", "game_date"]))
    cur_apps = _played.groupby("key").size().to_dict()

    players: dict[int, PlayerState] = {}
    for p in league.players.values():
        key = norm_name(p.name)
        view = views[p.espn_id]
        status, games_out, return_date = statuses[p.espn_id]
        acal = m.get("availability_calibration")
        if acal:
            ng = int(cur_apps.get(key, 0))
            tg = team_played.get(p.pro_team, 0)
            mins = None                                  # expected minutes: engine projection, else ESPN's
            if minutes_proj is not None and key in minutes_proj.index:
                mins = float(minutes_proj.loc[[key], "minutes"].iloc[0])
            elif (p.projection or {}).get("MIN"):
                mins = float(p.projection["MIN"])
            gp_rate = calibrated_rate(gp_by_name.get(key), ng, tg, acal, mins)
        else:
            gp_rate = gp_by_name.get(key)
        av = availability(status, gp_rate, cap=m["availability_cap"], dtd_prob=m["dtd_availability"],
                          out_default_games=m["out_default_games"], games_until_return=games_out,
                          default_rate=m["default_gp_rate"], return_date=return_date, today=today,
                          window=ic.window, rtp=view.rtp, typical_days=view.typical_days,
                          modifiers=view.modifiers, healthy_now=True)
        flags = list(view.notes)
        if live is not None and key in b.index:
            r = b.loc[key]
            mu, n, sd, lv, role = float(r["blended_fppg"]), int(r["n"]), r["sd_fp"], float(r["live_fppg"]), bool(r["role_change"])
            if role:
                flags.append(f"role change: {r['mpg_season']:.0f} -> {r['mpg_last5']:.0f} mpg")
        else:
            mu = float(prior.get(key, curve.tail_value))
            n, sd, lv, role = 0, np.nan, np.nan, False
        sigma = float(shrunk_sigma([mu], [sd], [n], m["k"], m["cv_typical"])[0]) * view.sigma_mult
        if status != "ACTIVE":
            flags.insert(0, status.replace("_", "-").lower())
        js = team_games.get(p.pro_team, [])
        # Second opinion on games played: blend the model's expected games with ESPN's projected
        # games (scaled to the games left). ESPN's weight starts at 50% and fades to 0 by the
        # middle of the season, as this season's real availability takes over.
        model_games = sum(av.for_game(j, d) for j, d in enumerate(js))
        espn_gp = (p.projection or {}).get("GP")
        played_so_far = max(82 - len(js), 0)
        w = 0.5 * max(0.0, 1 - played_so_far / 41)
        if espn_gp and model_games > 0 and w > 0 and js:
            espn_games = min(float(espn_gp), 82) * len(js) / 82
            target = (1 - w) * model_games + w * espn_games
            scale = float(np.clip(target / model_games, 0.7, 1.3))
            if abs(scale - 1) > 0.01:
                from dataclasses import replace as _replace

                av = _replace(av, base=min(av.base * scale, m["availability_cap"]))
                new_games = sum(av.for_game(j, d) for j, d in enumerate(js))
                if abs(new_games - model_games) >= 2:
                    flags.append(f"games: model {model_games:.0f}, ESPN {espn_games:.0f} → {new_games:.0f}")
        ros = sum(av.for_game(j, d) for j, d in enumerate(js)) * mu
        players[p.espn_id] = PlayerState(
            p.espn_id, p.name, p.pro_team, tuple(p.positions), p.on_team_id, status, mu, sigma, av,
            n=n, prior=float(prior.get(key, np.nan)), live=lv, role_change=role, ros_value=ros,
            games_remaining=len(js), flags=flags, injury=view, **track.get(key, {}),
        )
        if minutes_proj is not None and key in minutes_proj.index:
            mrow = minutes_proj.loc[key]
            mrow = mrow.iloc[0] if isinstance(mrow, pd.DataFrame) else mrow
            players[p.espn_id].minutes = float(mrow["minutes"])
            players[p.espn_id].role = str(mrow["role"])

    # Per-game projections from the possession simulator (sim/possession/forecast.py, refreshed each
    # morning): each game = average of the season-long estimate and that game's simulation.
    eng = cfg.get("engine", {})
    if eng.get("mode") == "possession":
        from tradecalc.sim.possession import forecast

        if forecast.apply(players, lg["nba_season"], today, float(eng.get("game_weight", 0.5)), norm_name,
                          scoring=scoring):
            for p in players.values():
                if p.mu_by_date:
                    p.ros_value = sum(p.avail.for_game(j, d) * p.mu_on(d)
                                      for j, d in enumerate(team_games.get(p.pro_team, [])))

    rosters = {tid: list(t.roster) for tid, t in league.teams.items()}
    ir = {tid: {i for i in t.roster if league.players[i].lineup_slot == "IR"} for tid, t in league.teams.items()}
    replacement = {}
    fas = [p for p in players.values() if p.on_team_id == 0 and p.avail.base > 0]
    for pos in ("PG", "SG", "SF", "PF", "C"):
        cands = [p for p in fas if pos in p.positions]
        if cands:
            replacement[pos] = max(cands, key=lambda p: p.mu)
    st = LeagueState(cfg, league, cal, today, players, rosters, replacement, team_game_index, ir,
                     real_today=real_today)
    # Section 4.6: value open active roster spots at replacement level by filling them with
    # the best remaining free agents (by ROS value). Fills are not tradeable.
    pool = sorted((p for p in fas if p.espn_id not in {i for r in rosters.values() for i in r}),
                  key=lambda p: -p.ros_value)
    for tid in sorted(rosters, key=lambda t: len(rosters[t]) - len(ir[t])):
        while len(rosters[tid]) - len(ir[tid]) < st.active_size and pool:
            fa = pool.pop(0)
            rosters[tid].append(fa.espn_id)
            st.fills.setdefault(tid, []).append(fa.espn_id)
    return st
