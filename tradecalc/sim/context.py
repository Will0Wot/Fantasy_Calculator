"""As-of context for every team-game: rotation, absences, baselines.

For each team-game, in date order, using only earlier games:
  rotation   players with >= ROT_MIN_APPS appearances in the team's last ROT_WINDOW games
             and >= ROT_MIN_MPG minutes per appearance
  baseline   each player's expected minutes (EWMA of recent appearances) and per-minute
             rates (season to date, falling back to last season)
  absent     rotation players who did not play this game

One output row per (team-game, player) for rotation players plus anyone who played.
This table is what the usage, minutes and rest models are fitted on.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

ROT_WINDOW, ROT_MIN_APPS, ROT_MIN_MPG = 10, 4, 12.0
EWMA_ALPHA = 0.25
USAGE_STATS = ["fga", "fta", "tov", "ast", "reb", "stl", "blk", "fg3a"]


def build_context(logs: pd.DataFrame, team_logs: pd.DataFrame, pos_of, ages: dict) -> pd.DataFrame:
    logs = logs.sort_values(["game_date", "game_id"])
    margin = team_logs.set_index(["game_id", "team"])["plus_minus"].to_dict()
    b2b = team_logs.set_index(["game_id", "team"])["b2b"].to_dict()

    ewma: dict[int, float] = {}
    apps: dict[int, int] = defaultdict(int)          # appearances this season
    cum = defaultdict(lambda: defaultdict(float))    # player -> stat -> season total (incl. min)
    prev_cum: dict[int, dict] = {}
    team_recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=ROT_WINDOW))
    names: dict[int, str] = {}
    season = None
    rows = []

    for (gid, team), g in logs.groupby(["game_id", "team"], sort=False):
        s = g["season"].iat[0]
        if s != season:  # new season: carry totals to prev_cum, reset
            prev_cum = {p: dict(v) for p, v in cum.items()}
            cum = defaultdict(lambda: defaultdict(float))
            apps = defaultdict(int)
            team_recent = defaultdict(lambda: deque(maxlen=ROT_WINDOW))
            season = s
        date = g["game_date"].iat[0]
        recent = team_recent[team]
        counts, mins = defaultdict(int), defaultdict(float)
        for d in recent:
            for p, m in d.items():
                counts[p] += 1
                mins[p] += m
        rotation = {p for p in counts if counts[p] >= ROT_MIN_APPS and mins[p] / counts[p] >= ROT_MIN_MPG}
        present = dict(zip(g["player_id"], g.index))

        def rate(p, stat):
            c = cum.get(p)
            if c and c["min"] >= 100:
                return c[stat] / c["min"]
            pc = prev_cum.get(p)
            if pc and pc.get("min", 0) >= 100:
                return pc[stat] / pc["min"]
            return np.nan

        exp_min = {p: ewma.get(p, mins[p] / max(counts[p], 1)) for p in rotation}
        absent = rotation - set(present)
        team_tot = {s_: np.nansum([rate(p, s_) * exp_min[p] for p in rotation]) for s_ in USAGE_STATS}
        miss = {s_: (np.nansum([rate(p, s_) * exp_min[p] for p in absent]) / team_tot[s_] if team_tot[s_] else 0.0)
                for s_ in USAGE_STATS}
        abs_by_pos = defaultdict(float)
        for p in absent:
            abs_by_pos[pos_of(names.get(p, "")) or "F"] += exp_min[p]
        abs_total = sum(abs_by_pos.values())

        common = dict(game_id=gid, team=team, game_date=date, season=s, b2b=bool(b2b.get((gid, team), False)),
                      margin=float(margin.get((gid, team), 0.0)), team_games=len(recent),
                      abs_min_total=abs_total, n_absent=len(absent),
                      **{f"miss_{k}": v for k, v in miss.items()})
        for p in rotation | set(present):
            pg = pos_of(names.get(p, g.loc[present[p], "player_name"] if p in present else "")) or "F"
            r = dict(common, player_id=p, in_rotation=p in rotation, played=p in present,
                     exp_min=exp_min.get(p, ewma.get(p, np.nan)), apps=apps[p],
                     age=ages.get((p, s), np.nan), pos=pg,
                     abs_min_same=abs_by_pos.get(pg, 0.0), abs_min_other=abs_total - abs_by_pos.get(pg, 0.0),
                     **{f"base_{k}": rate(p, k) for k in USAGE_STATS})
            if p in present:
                row = g.loc[present[p]]
                r.update(min=row["min"], **{k: row[k] for k in USAGE_STATS})
            rows.append(r)

        # update state with this game
        d = {}
        for p, i in present.items():
            row = g.loc[i]
            names[p] = row["player_name"]
            m = float(row["min"])
            ewma[p] = m if p not in ewma else (1 - EWMA_ALPHA) * ewma[p] + EWMA_ALPHA * m
            apps[p] += 1
            c = cum[p]
            c["min"] += m
            for k in USAGE_STATS:
                c[k] += float(row[k])
            d[p] = m
        recent.append(d)
    return pd.DataFrame(rows)
