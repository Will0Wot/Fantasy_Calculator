"""Season tracker: log the model's predictions before games are played, fill in what really
happened afterwards, so the season can be graded at the end.

Three tables per season in tracking/<season>/ (CSV, committed to the repo):
  matchups.csv   every team, every week: projected points, spread, win chance, and the opponent;
                 kind = "pre-week" (made before the week's first game: the honest prediction) or
                 "in-week" (made during the week, with points already scored added in)
  players.csv    every rostered player, every week (pre-week): projected points per game, expected
                 games, projected points; then real games, points and points per game
  standings.csv  before each week: every team's expected final wins, 80% range, chance to win the
                 league or finish last; final wins and rank are filled in when the season ends
Each row records the model version (git commit) that made it. `snapshot` is safe to run any number
of times a day: a newer snapshot of the same prediction replaces the older one, and predictions for
weeks that have started are never overwritten. `settle` fills the actual columns once a week ends.
Run both with `tradecalc track`; `excel()` builds the workbook (also downloadable in the app).
"""
from __future__ import annotations

import io
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from tradecalc.model.names import norm_name

ROOT = Path(__file__).resolve().parent.parent / "tracking"
KEYS = ("matchups", "players", "standings")


def season_dir(season: str) -> Path:
    return ROOT / season


def load(season: str) -> dict[str, pd.DataFrame]:
    d = season_dir(season)
    return {name: pd.read_csv(d / f"{name}.csv") if (d / f"{name}.csv").exists() else pd.DataFrame()
            for name in KEYS}


def save(season: str, tables: dict[str, pd.DataFrame]) -> None:
    d = season_dir(season)
    d.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        if len(df):
            df.to_csv(d / f"{name}.csv", index=False, float_format="%.4f")


def model_version() -> str:
    sha = os.environ.get("GITHUB_SHA", "")[:7]
    if sha:
        return sha
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parent, timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# --- predictions -----------------------------------------------------------------------------

def _target_weeks(state) -> tuple:
    """(next week not yet started -> pre-week, week in progress -> in-week or None)."""
    real = state.real_today or state.today
    upcoming = next((w for w in state.cal.weeks if w.start >= real), None)
    current = next((w for w in state.cal.weeks if w.start < real <= w.end), None)
    return upcoming, current


def snapshot(state, n_forecast: int = 10000) -> dict[str, pd.DataFrame]:
    """The model's predictions right now (see module docstring)."""
    from tradecalc.trade.lineup import simulate_week
    from tradecalc.trade.matchup import season_outlook

    real = state.real_today or state.today
    version = model_version()
    upcoming, current = _target_weeks(state)
    targets = [(w, "pre-week") for w in (upcoming,) if w] + [(w, "in-week") for w in (current,) if w]
    rows = []
    for t in sorted(state.rosters):
        out = {x["week"]: x for x in season_outlook(t, state.rosters, state)["weeks"]}
        for w, kind in targets:
            x = out.get(w.number)
            if x is None or x.get("opp") is None:
                continue
            banked = state.banked(t, w)[0] if kind == "in-week" else 0.0
            rows.append({"week": w.number, "week_start": w.start.isoformat(), "week_end": w.end.isoformat(),
                         "kind": kind, "as_of": real.isoformat(), "team_id": t, "team": state.team_name(t),
                         "opp_id": x["opp"], "opp": state.team_name(x["opp"]), "proj_points": x["M"],
                         "proj_sd": float(np.sqrt(x["V"])), "banked_points": banked, "win_prob": x["p"],
                         "model": version})
    matchups = pd.DataFrame(rows)

    players = pd.DataFrame()
    if upcoming is not None:
        prow = []
        for t, roster in sorted(state.rosters.items()):
            res = simulate_week(roster, upcoming, state)
            counted = dict(zip(res.ids, res.sum_a))
            for i in roster:
                p = state.players[i]
                days = [d for d in upcoming.days() if state.cal.team_plays(p.pro_team, d)]
                exp_games = float(sum(state.a_for(p, d) for d in days))
                prow.append({"week": upcoming.number, "week_start": upcoming.start.isoformat(),
                             "week_end": upcoming.end.isoformat(), "as_of": real.isoformat(), "team_id": t,
                             "team": state.team_name(t), "espn_id": i, "player": p.name, "pro_team": p.pro_team,
                             "status": p.injury_status, "team_games": len(days), "proj_fppg": p.mu,
                             "proj_games": exp_games, "proj_points": p.mu * exp_games,
                             "proj_points_counted": p.mu * counted.get(i, 0.0), "model": version})
        players = pd.DataFrame(prow)

    standings = pd.DataFrame()
    if upcoming is not None:
        from tradecalc.trade.standings import forecast

        fc = forecast(state, n=n_forecast, seed=0).reset_index(drop=True)
        name_to_id = {state.team_name(t): t for t in state.rosters}
        standings = pd.DataFrame({
            "before_week": upcoming.number, "as_of": real.isoformat(),
            "team_id": fc["team"].map(name_to_id), "team": fc["team"], "record": fc.get("record"),
            "exp_wins": fc["exp_wins"], "wins_p10": fc["wins_p10"], "wins_p90": fc["wins_p90"],
            "p_first": fc["p_first"], "p_top3": fc["p_top3"], "p_last": fc["p_last"],
            "proj_rank": np.arange(1, len(fc) + 1), "model": version})
    return {"matchups": matchups, "players": players, "standings": standings}


def _replace(old: pd.DataFrame, new: pd.DataFrame, drop) -> pd.DataFrame:
    """old rows minus those drop(old) marks, plus new rows."""
    if old is None or old.empty:
        return new.copy()
    if new.empty:
        return old
    return pd.concat([old[~drop(old)], new], ignore_index=True)


def record(state, season: str, log=print) -> dict[str, pd.DataFrame]:
    """Take a snapshot and merge it into the season's tables.

    A snapshot only makes pre-week predictions for a week that has not started, so a newer one
    simply replaces that week's earlier pre-week rows; once the week starts nothing replaces them
    (frozen). In-week predictions keep one snapshot per day."""
    real = (state.real_today or state.today).isoformat()
    old, new = load(season), snapshot(state)
    nm, npl, ns = new["matchups"], new["players"], new["standings"]
    pre_weeks = set(nm[nm.kind == "pre-week"].week) if len(nm) else set()
    in_weeks = set(nm[nm.kind == "in-week"].week) if len(nm) else set()
    m = _replace(old["matchups"], nm, lambda d: ((d.kind == "pre-week") & d.week.isin(pre_weeks))
                 | ((d.kind == "in-week") & d.week.isin(in_weeks) & (d.as_of.astype(str) == real)))
    p = _replace(old["players"], npl, lambda d: d.week.isin(set(npl.week) if len(npl) else set()))
    s = _replace(old["standings"], ns, lambda d: d.before_week.isin(set(ns.before_week) if len(ns) else set()))
    log(f"snapshot {real}: {len(nm)} matchup rows, {len(npl)} player rows, {len(ns)} standings rows")
    return {"matchups": m, "players": p, "standings": s}


# --- actual results --------------------------------------------------------------------------

def settle(state, tables: dict[str, pd.DataFrame], logs: pd.DataFrame, log=print) -> dict[str, pd.DataFrame]:
    """Fill actual columns for weeks that have finished. logs: this season's box scores with fp."""
    real = state.real_today or state.today
    done = {w.number for w in state.cal.weeks if w.end < real}
    tie = state.cfg["league"]["tie_value"]
    score = {}
    for mt in state.league.matchups:
        if mt["week"] in done and mt.get("away") is not None and mt.get("winner", "UNDECIDED") != "UNDECIDED":
            h, a = mt["home_total"], mt["away_total"]
            score[(mt["week"], mt["home"])] = (h, a, 1.0 if h > a else 0.0 if h < a else tie)
            score[(mt["week"], mt["away"])] = (a, h, 1.0 if a > h else 0.0 if a < h else tie)
    m = tables["matchups"].copy()
    if len(m):
        for col in ("actual_points", "opp_actual_points", "won"):
            if col not in m:
                m[col] = np.nan
        for i, r in m.iterrows():
            s = score.get((int(r["week"]), int(r["team_id"])))
            if s is not None:
                m.loc[i, ["actual_points", "opp_actual_points", "won"]] = s
        log(f"settled matchups: {int(m.actual_points.notna().sum())} of {len(m)} rows have results")

    p = tables["players"].copy()
    if len(p) and len(logs):
        L = logs[logs["min"] > 0].assign(key=logs.player_name.map(norm_name), d=logs.game_date.astype(str).str[:10])
        for col in ("games_played", "actual_points", "actual_fppg"):
            if col not in p:
                p[col] = np.nan
        for wk, g in p.groupby("week"):
            if int(wk) not in done:
                continue
            ws, we = str(g.week_start.iloc[0]), str(g.week_end.iloc[0])
            box = L[(L.d >= ws) & (L.d <= we)].groupby("key")["fp"].agg(["size", "sum"])
            keys = g.player.map(norm_name)
            p.loc[g.index, "games_played"] = keys.map(box["size"]).fillna(0).to_numpy()
            p.loc[g.index, "actual_points"] = keys.map(box["sum"]).fillna(0).to_numpy()
        p["actual_fppg"] = np.where(p.games_played > 0, p.actual_points / p.games_played.replace(0, np.nan), np.nan)

    s = tables["standings"].copy()
    if len(s) and all(w.number in done for w in state.cal.weeks):
        wins = {}
        for (wk, t), (_, _, won) in score.items():
            wins[t] = wins.get(t, 0.0) + won
        s["final_wins"] = s.team_id.map(wins)
        final_rank = {t: i + 1 for i, t in enumerate(sorted(wins, key=lambda t: -wins[t]))}
        s["final_rank"] = s.team_id.map(final_rank)
    return {"matchups": m, "players": p, "standings": s}


# --- grading ---------------------------------------------------------------------------------

def summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """How close the model has been so far (pre-week predictions only)."""
    rows = []
    m = tables.get("matchups", pd.DataFrame())
    if len(m) and "won" in m:
        g = m[(m.kind == "pre-week") & m.won.notna()]
        if len(g):
            fav = g[g.win_prob != 0.5]
            rows += [
                ("Matchups", "weeks graded", int(g.week.nunique()), ""),
                ("Matchups", "predicted favourite won", float(((fav.win_prob > 0.5) == (fav.won > 0.5)).mean()),
                 "share of team-weeks"),
                ("Matchups", "Brier score", float(((g.win_prob - g.won) ** 2).mean()), "0.25 = always saying 50%"),
                ("Matchups", "average win chance given to the winner", float(g[g.won == 1].win_prob.mean()), ""),
                ("Team points", "average miss (points / week)", float((g.proj_points - g.actual_points).abs().mean()), ""),
                ("Team points", "bias (projected − actual)", float((g.proj_points - g.actual_points).mean()), ""),
                ("Team points", "inside projected ±1 s.d.",
                 float(((g.actual_points - g.proj_points).abs() <= g.proj_sd).mean()), "68% = honest"),
            ]
    p = tables.get("players", pd.DataFrame())
    if len(p) and "actual_points" in p:
        g = p[p.actual_points.notna()]
        if len(g):
            played = g[g.games_played > 0]
            rows += [
                ("Players", "player-weeks graded", int(len(g)), ""),
                ("Players", "average miss (points / week)", float((g.proj_points - g.actual_points).abs().mean()), ""),
                ("Players", "bias (projected − actual, points / week)", float((g.proj_points - g.actual_points).mean()), ""),
                ("Players", "average miss (points per game, when he played)",
                 float((played.proj_fppg - played.actual_fppg).abs().mean()), ""),
                ("Players", "expected vs real games (bias)", float((g.proj_games - g.games_played).mean()), "games / week"),
            ]
    s = tables.get("standings", pd.DataFrame())
    if len(s) and "final_wins" in s and s.final_wins.notna().any():
        from scipy.stats import spearmanr

        for wk, g in s.groupby("before_week"):
            rows.append(("Standings", f"before week {int(wk)}: rank correlation with final",
                         float(spearmanr(g.exp_wins, g.final_wins)[0]), "1 = perfect order"))
    return pd.DataFrame(rows, columns=["area", "measure", "value", "note"], dtype=object)   # counts stay whole


def excel(tables: dict[str, pd.DataFrame]) -> bytes:
    """Workbook: Summary, Matchups, Players, Standings."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        summary(tables).to_excel(xw, sheet_name="Summary", index=False)
        for name in ("matchups", "players", "standings"):
            df = tables.get(name, pd.DataFrame())
            (df if len(df) else pd.DataFrame({"note": ["no rows yet"]})).to_excel(xw, sheet_name=name.title(), index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 45)
            ws.freeze_panes = "A2"
    return buf.getvalue()


def run(state, conn, season: str, log=print) -> dict[str, pd.DataFrame]:
    """Snapshot + settle + save. Returns the tables."""
    from tradecalc.ingest import nba
    from tradecalc.model.scoring import add_fp

    tables = record(state, season, log)
    logs = add_fp(nba.game_logs(conn, season), state.cfg["scoring"])
    tables = settle(state, tables, logs, log)
    save(season, tables)
    return tables
