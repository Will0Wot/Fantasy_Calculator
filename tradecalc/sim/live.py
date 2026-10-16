"""Glue for the live league: fit/save engine parameters, build the current World."""
from __future__ import annotations

import pandas as pd

from tradecalc.ingest import nba_extra
from tradecalc.sim.history import load_history, prev_season
from tradecalc.sim.params import PARAMS_PATH, fit_noise, fit_params, load_params, save_params
from tradecalc.teams import to_nba

_H_CACHE: dict = {}


def coaches_table(conn, espn_pro_teams: dict[int, str], espn_seasons: list[int]) -> pd.DataFrame:
    hc = nba_extra.head_coaches(conn, espn_pro_teams, espn_seasons)
    if hc.empty:
        return hc
    hc["team"] = hc.team.map(to_nba)
    hc["season"] = hc.season.map(lambda y: f"{y - 1}-{str(y)[2:]}")
    return hc


def history(conn, cfg, league):
    season = cfg["league"]["nba_season"]
    key = season  # one history per process; it is read-only and shared by every visitor
    if key not in _H_CACHE:
        seasons = [prev_season(prev_season(prev_season(season))), prev_season(prev_season(season)),
                   prev_season(season), season]
        _H_CACHE[key] = load_history(conn, cfg["scoring"], list(league.players.values()), seasons, current_season=season)
    return _H_CACHE[key]


def fit_and_save(conn, cfg, league, log=print) -> dict:
    """Refit every engine parameter on the two latest complete seasons and save it."""
    from tradecalc.sim.validate import coverage, fit_calibration

    season = cfg["league"]["nba_season"]
    s1, s0 = prev_season(season), prev_season(prev_season(season))
    H = history(conn, cfg, league)
    y = int(season[:4]) + 1
    coaches = coaches_table(conn, league.pro_teams, [y - 3, y - 2, y - 1, y])
    log(f"Fitting rates, usage, minutes, rest and coach models on {s0} and {s1}...")
    p = fit_params(H, coaches, [s0, s1])
    p.update(fit_noise(H, p, s1, s0))
    mondays = sorted(H.team_logs[(H.team_logs.season == s1)].game_date.unique())
    mondays = [d for d in mondays if pd.Timestamp(d).weekday() == 0][4::5]
    log(f"Calibrating minutes share and rate scale on {len(mondays)} weeks of {s1}...")
    p.update(fit_calibration(H, p, s1, s0, mondays))
    best = None
    for sd in (0.12, 0.18, 0.24):
        c = coverage(H, dict(p, form_sd=sd), s1, s0, mondays[:4], n=1500)
        score = abs(c["week_in80"] - 0.8) + abs(c["week_in50"] - 0.5)
        log(f"  form_sd {sd}: weekly 80% coverage {c['week_in80']:.3f}, 50% coverage {c['week_in50']:.3f}")
        if best is None or score < best[0]:
            best = (score, sd)
    p["form_sd"] = best[1]
    save_params(p)
    log(f"Saved {PARAMS_PATH}")
    return p


def live_world(conn, cfg, state):
    from tradecalc.sim.engine import build_world_live

    season = cfg["league"]["nba_season"]
    if not PARAMS_PATH.exists():
        raise FileNotFoundError("No engine parameters yet: run `tradecalc sim-fit` first.")
    params = load_params()
    H = history(conn, cfg, state.league)
    depth = nba_extra.depth_charts(conn, state.league.pro_teams)
    y = int(season[:4]) + 1
    coaches = coaches_table(conn, state.league.pro_teams, [y - 1, y])
    world = build_world_live(H, params, state, depth, coaches, season, prev_season(season))
    coaches_now = coaches[coaches.season == season].set_index("team")["coach"].to_dict() if len(coaches) else {}
    return world, coaches_now


def validate_last_season(conn, cfg, league, log=print) -> dict:
    """Out-of-sample check: fit on the two seasons before last season, test on last season."""
    from tradecalc.config import load_config
    from tradecalc.sim.backtest import run as run_bt, summarize
    from tradecalc.sim.validate import coverage, fit_calibration, matchup_calibration
    from tradecalc.trade.lineup import slot_list

    season = cfg["league"]["nba_season"]
    test = prev_season(season)
    f1, f0 = prev_season(test), prev_season(prev_season(test))
    H = history(conn, cfg, league)
    y = int(test[:4]) + 1
    coaches = coaches_table(conn, league.pro_teams, [y - 2, y - 1, y])
    log(f"Fitting on {f0} + {f1}; testing on {test}.")
    p = fit_params(H, coaches, [f0, f1])
    p.update(fit_noise(H, p, f1, f0))
    fit_mondays = [d for d in sorted(H.team_logs[H.team_logs.season == f1].game_date.unique())
                   if pd.Timestamp(d).weekday() == 0][4::5]
    p.update(fit_calibration(H, p, f1, f0, fit_mondays))
    p["form_sd"] = load_params().get("form_sd", 0.18) if PARAMS_PATH.exists() else 0.18
    test_mondays = [d for d in sorted(H.team_logs[H.team_logs.season == test].game_date.unique())
                    if pd.Timestamp(d).weekday() == 0][4::4]
    log("Point projections vs the trade calculator's FPPG...")
    pts = summarize(run_bt(H, p, test, f1, every=6))
    log("Distribution coverage...")
    cov = coverage(H, p, test, f1, test_mondays)
    log("Fantasy matchup win-probability calibration...")
    mc = matchup_calibration(H, p, test, f1, test_mondays, matchups_per_week=40,
                             slots_cfg=(slot_list(cfg), cfg["roster"]["slot_eligibility"]))
    return {"points": pts, "coverage": cov, "matchups": mc, "test_season": test, "fit_seasons": [f0, f1]}


VALIDATION_PATH = PARAMS_PATH.parent / "validation.json"


def save_validation(r: dict, path=VALIDATION_PATH):
    """Store the headline numbers from validate_last_season for the app's Help tab."""
    import json
    from datetime import date

    from sklearn.metrics import brier_score_loss

    pts, cov, m = r["points"], r["coverage"], r["matchups"].copy()
    m["bucket"] = pd.cut(m.p_win, [0, .2, .4, .6, .8, 1])
    buckets = m.groupby("bucket", observed=True).agg(n=("won", "size"), predicted=("p_win", "mean"),
                                                     actual=("won", "mean")).reset_index()
    out = {
        "run_on": date.today().isoformat(),
        "test_season": r.get("test_season"),
        "fit_seasons": r.get("fit_seasons"),
        "points": {k: {c: (None if pd.isna(v) else round(float(v), 3)) for c, v in row.items()}
                   for k, row in pts.to_dict("index").items()},
        "n_player_games": int(pts.attrs.get("n", 0)),
        "coverage": {k: round(float(v), 4) for k, v in cov.items()},
        "matchups": {"n": int(len(m)), "brier": round(float(brier_score_loss(m.won, m.p_win)), 4),
                     "buckets": [{"range": str(b.bucket), "n": int(b.n), "predicted": round(float(b.predicted), 3),
                                  "actual": round(float(b.actual), 3)} for b in buckets.itertuples()]},
    }
    path.write_text(json.dumps(out, indent=2))
    return path


def load_validation(path=VALIDATION_PATH) -> dict | None:
    import json

    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return None
