"""tradecalc command-line interface."""
from __future__ import annotations

from collections import Counter
from datetime import date

import typer

app = typer.Typer(add_completion=False, help="Fantasy basketball trade calculator.")


@app.callback()
def main():
    """Fantasy basketball trade calculator."""


def _ctx(cfg_path=None):
    from tradecalc.session import context

    return context(cfg_path)


def _client(cfg, conn):
    from tradecalc.session import espn_client

    return espn_client(cfg, conn)


@app.command()
def check(force: bool = typer.Option(False, help="Bypass caches.")):
    """Phase 0: verify every data source and print a summary of each."""
    from tradecalc.ingest import adp, nba
    from tradecalc.ingest.league import check_settings_against_config, load_league
    from tradecalc.schedule.calendar import Calendar

    cfg, conn = _ctx()
    lg = cfg["league"]
    client = _client(cfg, conn)

    typer.secho("\n== League (ESPN) ==", bold=True)
    league = load_league(client, force=force)
    s = league.settings
    typer.echo(f"{s['name']}: {len(league.teams)} teams, {s['scoringSettings']['scoringType']}, "
               f"{s['scheduleSettings']['matchupPeriodCount']} weeks, "
               f"lock={s['rosterSettings'].get('lineupLocktimeType')}, "
               f"tie rule={s['scoringSettings'].get('matchupTieRule')}")
    problems = check_settings_against_config(s, cfg)
    for p in problems:
        typer.secho(f"  CONFIG MISMATCH: {p}", fg="red")
    if not problems:
        typer.secho("  league.yaml matches ESPN scoring, roster slots, weeks and tie rule.", fg="green")
    for t in league.teams.values():
        mark = "  <- you" if t.team_id == lg["my_team_id"] else ""
        typer.echo(f"  [{t.team_id:>2}] {t.name:<30} {len(t.roster)} players{mark}")
    me = league.teams[lg["my_team_id"]]
    typer.echo(f"\n  Your roster ({me.name}):")
    for pid in me.roster:
        p = league.players[pid]
        proj = p.projection
        fp = sum(cfg["scoring"].get(k, 0) * v for k, v in proj.items())
        typer.echo(f"    {p.lineup_slot or '':<5} {p.name:<26} {p.pro_team:<4} {'/'.join(p.positions):<12} "
                   f"{p.injury_status:<12} ADP {p.adp or 0:>6.1f}  proj {fp:5.1f} FPPG")
    fa = [p for p in league.players.values() if p.on_team_id == 0]
    inj = Counter(p.injury_status for p in league.players.values() if p.on_team_id)
    typer.echo(f"  Free agents in pool: {len(fa)}; rostered injury statuses: {dict(inj)}")

    typer.secho("\n== NBA schedule (ESPN pro schedule) ==", bold=True)
    cal = Calendar(league, cfg)
    typer.echo(f"  {len(league.pro_games)} games, {league.pro_games[0]['date']} to {league.pro_games[-1]['date']}; "
               f"scoring period 1 = {league.first_day}")
    per_team = Counter()
    for g in league.pro_games:
        per_team[g["home"]] += 1
        per_team[g["away"]] += 1
    typer.echo(f"  games per team: min {min(per_team.values())}, max {max(per_team.values())} "
               f"(82 expected; unscheduled NBA Cup knockout games are added later by ESPN)")
    typer.echo(f"  {len(cal.weeks)} fantasy weeks:")
    for w in cal.weeks:
        opp = cal.opponents[lg["my_team_id"]].get(w.number)
        n_games = sum(len(cal.games_by_day.get(d, ())) // 2 for d in w.days())
        typer.echo(f"    wk {w.number:>2}: {w.start} - {w.end} ({len(w.days())}d, {n_games:>3} NBA games) "
                   f"vs {league.teams[opp].name if opp else '-'}")

    typer.secho("\n== Box scores (stats.nba.com) ==", bold=True)
    interval = cfg["data"]["nba_min_request_interval_s"]
    for season, complete in ((lg["prev_nba_season"], True), (lg["nba_season"], False)):
        r = nba.refresh_season(conn, season, complete=complete, min_interval=interval, force=force)
        n = conn.execute("SELECT COUNT(*) c, COUNT(DISTINCT player_id) p FROM game_logs WHERE season=?",
                         (season,)).fetchone()
        typer.echo(f"  {season}: {r} -> cached {n['c']} player-games, {n['p']} players")
    star = league.players[me.roster[0]].name
    hit = nba.find_player_id(conn, star)
    if hit:
        pid, nm = hit
        gl = nba.player_game_log(conn, lg["prev_nba_season"], pid)
        typer.echo(f"  Game log for {nm} ({lg['prev_nba_season']}): {len(gl)} games; last 3:")
        typer.echo(gl.tail(3)[["game_date", "matchup", "min", "pts", "reb", "ast", "stl", "blk", "tov"]]
                   .to_string(index=False))
    typer.echo(f"  stats.nba.com requests this run: {nba.API_CALLS}")

    typer.secho("\n== ADP ==", bold=True)
    cur_path, prev_path = cfg.path(cfg["data"]["adp_csv"]), cfg.path(cfg["data"]["prev_adp_csv"])
    if force or not cur_path.exists():
        n = adp.export_espn_adp(client.player_pool(limit=600, force=force), cur_path)
        typer.echo(f"  wrote {n} rows of ESPN {lg['season']} ADP -> {cur_path.name}")
    if force or not prev_path.exists():
        pool_prev = client.player_pool(limit=600, force=force, season=lg["prev_season"], league_scoped=False)
        n = adp.export_espn_adp(pool_prev, prev_path, use_rank=True)
        typer.echo(f"  wrote {n} rows of ESPN {lg['prev_season']} preseason draft rank -> {prev_path.name}")
    for label, path in (("current", cur_path), ("previous season", prev_path)):
        df = adp.load_adp(path)
        top = ", ".join(f"{r.name} ({r.adp:g})" for r in df.head(5).itertuples())
        typer.echo(f"  {label}: {len(df)} players; top 5: {top}")

    typer.echo(f"\nESPN requests this run: {client.api_calls}")


def _state(force: bool = False, today: str | None = None):
    from tradecalc.session import load_state

    st = load_state(force, today, warn=lambda m: typer.secho(m, fg="yellow", err=True))
    note = f"  (model date set to {st.today:%a %b %d})" if today else ""
    typer.secho(f"📅 {st.week_status()}{note}", fg="cyan", err=True)
    return st


@app.command()
def refresh():
    """Pull new games, injuries and rosters."""
    from tradecalc.ingest import nba

    st = _state(force=True)
    typer.echo(f"Refreshed. {len(st.players)} players, {nba.API_CALLS} stats.nba.com request(s); "
               f"{sum(1 for p in st.players.values() if p.n)} players with games this season.")


@app.command()
def lineup(
    week: int = typer.Option(None, help="Fantasy week (default: current)."),
    team: int = typer.Option(None, help="ESPN team id (default: yours)."),
    today: str = typer.Option(None, help="Pretend today is YYYY-MM-DD."),
):
    """Phase 3: simulated starts (g_i) per player for one week."""
    from tradecalc.trade.lineup import simulate_week

    st = _state(today=today)
    team = team or st.cfg["league"]["my_team_id"]
    wk = st.cal.weeks[week - 1] if week else st.cal.week_of(st.today) or st.cal.weeks[0]
    res = simulate_week(st.rosters[team], wk, st, from_day=st.today if wk.start <= st.today <= wk.end else None)
    typer.secho(f"{st.team_name(team)} - week {wk.number} ({wk.start} to {wk.end})", bold=True)
    typer.echo(f"{'player':<26}{'team':<5}{'pos':<12}{'mu':>6}{'a':>6}{'games':>7}{'starts':>8}{'E[FP]':>8}")
    rows = sorted(zip(res.ids, res.g, res.sum_a), key=lambda r: -st.players[r[0]].mu)
    for pid, g, sa in rows:
        p = st.players[pid]
        games = st.cal.team_games_in_week(p.pro_team, wk)
        typer.echo(f"{p.name:<26}{p.pro_team:<5}{'/'.join(p.positions):<12}{p.mu:6.1f}{p.avail.base:6.2f}"
                   f"{games:7d}{int(g):8d}{sa * p.mu:8.1f}  {' '.join(p.flags)}")
    typer.echo(f"Total starts {int(res.g.sum())}, expected FP {sum(sa * st.players[i].mu for i, sa in zip(res.ids, res.sum_a)):.0f}")


def _match_player(name: str, st, pool_ids) -> int:
    from rapidfuzz import fuzz, process

    from tradecalc.model.names import norm_name

    choices = {pid: norm_name(st.players[pid].name) for pid in pool_ids}
    hit = process.extractOne(norm_name(name), choices, scorer=fuzz.WRatio, score_cutoff=75)
    if not hit:
        raise typer.BadParameter(f"no player matching '{name}'")
    return hit[2]


def parse_trade(text: str, st, partner: int | None):
    from tradecalc.trade.evaluate import Trade

    if " for " not in text:
        raise typer.BadParameter('format: "Player A + Player B for Player C"')
    give_s, get_s = text.split(" for ", 1)
    me = st.cfg["league"]["my_team_id"]
    give = tuple(_match_player(n.strip(), st, st.tradeable(me)) for n in give_s.split("+"))
    others = partner and st.tradeable(partner) or [p for t in st.rosters if t != me for p in st.tradeable(t)]
    get = tuple(_match_player(n.strip(), st, others) for n in get_s.split("+"))
    teams = {st.players[i].on_team_id for i in get}
    if len(teams) != 1:
        raise typer.BadParameter("players you receive must all come from one team")
    return Trade(me, teams.pop(), give, get)


@app.command()
def evaluate(
    trade: str = typer.Argument(..., help='"Player A + Player B for Player C"'),
    with_: int = typer.Option(None, "--with", help="Partner ESPN team id (default: inferred)."),
    precise: bool = typer.Option(False, help="Monte Carlo win probabilities."),
    today: str = typer.Option(None, help="Pretend today is YYYY-MM-DD."),
):
    """Evaluate one trade for both teams."""
    from tradecalc.trade.evaluate import evaluate as run_eval, format_result

    st = _state(today=today)
    from tradecalc.trade.explain import explain, format_explanation

    t = parse_trade(trade, st, with_)
    r = run_eval(t, st, precise=precise)
    typer.echo(format_result(r, st))
    typer.secho("\nWhy", bold=True)
    typer.echo(format_explanation(explain(r, st)))


@app.command()
def find(
    with_: int = typer.Option(None, "--with", help="Only this partner ESPN team id."),
    max_size: int = typer.Option(2, help="Max players per side."),
    top: int = typer.Option(None, help="Trades per opponent (default from config)."),
    today: str = typer.Option(None, help="Pretend today is YYYY-MM-DD."),
):
    """Find trades that help you and that the partner might accept."""
    import time

    from tradecalc.trade.evaluate import format_result
    from tradecalc.trade.finder import find_trades

    st = _state(today=today)
    me = st.cfg["league"]["my_team_id"]
    t0 = time.time()
    res = find_trades(me, st, max_size=max_size, partners=[with_] if with_ else None, top=top,
                      progress=lambda p, n, k: typer.echo(f"  {st.team_name(p):<30} {n:>4} evaluated, {k:>3} kept",
                                                          err=True))
    typer.echo(f"Scan took {time.time() - t0:.1f}s\n", err=True)
    for partner, rs in res.items():
        if not rs:
            continue
        typer.secho(f"== {st.team_name(partner)} ==", bold=True)
        for r in rs:
            typer.echo(format_result(r, st) + "\n")


@app.command()
def week(
    week: int = typer.Option(None, help="Fantasy week (default: current)."),
    precise: bool = typer.Option(False, help="Monte Carlo win probability."),
    today: str = typer.Option(None, help="Pretend today is YYYY-MM-DD."),
):
    """Projected matchup and win probability for one week."""
    from tradecalc.trade.matchup import p_win, p_win_monte_carlo, week_dist

    st = _state(today=today)
    me = st.cfg["league"]["my_team_id"]
    wk = st.cal.weeks[week - 1] if week else st.cal.week_of(st.today) or st.cal.weeks[0]
    opp = st.cal.opponents[me].get(wk.number)
    d1, d2 = week_dist(st.rosters[me], wk, st, st.today), week_dist(st.rosters[opp], wk, st, st.today)
    p = (p_win_monte_carlo(d1, d2, st, st.cfg["trade"]["monte_carlo_sims"], tie_value=st.cfg["league"]["tie_value"])
         if precise else p_win(d1.M, d1.V, d2.M, d2.V))
    typer.secho(f"Week {wk.number} ({wk.start:%a %b %d} to {wk.end:%a %b %d}): {st.team_name(me)} vs {st.team_name(opp)}", bold=True)
    bm, bo = st.banked(me, wk)[0], st.banked(opp, wk)[0]
    if bm or bo:
        typer.echo(f"  Already scored: you {bm:.0f}, opponent {bo:.0f} (days before today)")
    d1 = type(d1)(d1.M + bm, d1.V, d1.lineup)
    d2 = type(d2)(d2.M + bo, d2.V, d2.lineup)
    for label, tid, d in (("You", me, d1), ("Opponent", opp, d2)):
        typer.echo(f"  {label:<9} {d.M:7.0f} ± {d.V ** 0.5:4.0f} pts, {int(d.lineup.g.sum())} starts left")
    typer.secho(f"  Win probability: {p:.1%}", bold=True)


@app.command()
def values(
    team: int = typer.Option(None, help="Only this ESPN team id (0 = free agents)."),
    top: int = typer.Option(50, help="Rows to show."),
    today: str = typer.Option(None, help="Pretend today is YYYY-MM-DD."),
):
    """Rest-of-season values and flags."""
    st = _state(today=today)
    ps = [p for p in st.players.values() if team is None or p.on_team_id == team]
    ps.sort(key=lambda p: -p.ros_value)
    typer.echo(f"{'#':>3} {'player':<26}{'team':<5}{'pos':<12}{'owner':<16}{'FPPG':>6}{'prior':>7}{'GP':>4}"
               f"{'a':>6}{'G left':>7}{'ROS':>7}  flags")
    for i, p in enumerate(ps[:top], 1):
        owner = st.team_name(p.on_team_id)[:15] if p.on_team_id else "FA"
        typer.echo(f"{i:>3} {p.name:<26}{p.pro_team:<5}{'/'.join(p.positions):<12}{owner:<16}{p.mu:6.1f}"
                   f"{p.prior:7.1f}{p.n:4d}{p.avail.base:6.2f}{p.games_remaining:7d}{p.ros_value:7.0f}  "
                   f"{'; '.join(p.flags)}")
    if team is None:
        rep = ", ".join(f"{pos} {p.name} {p.mu:.1f}" for pos, p in st.replacement.items())
        typer.echo(f"\nReplacement level (best FA): {rep}")


@app.command()
def injuries(
    all_: bool = typer.Option(False, "--all", help="Include free agents."),
    today: str = typer.Option(None, help="Pretend today is YYYY-MM-DD."),
):
    """Injured and recently-returned players, with what the model assumes for each."""
    st = _state(today=today)
    rows = [p for p in st.players.values()
            if p.injury and (p.injury.current or p.injury.history) and (all_ or p.on_team_id)]
    rows.sort(key=lambda p: -p.ros_value)
    typer.echo(f"{'player':<26}{'owner':<16}{'FPPG':>6}{'a now':>7}{'a +30d':>8}{'a +90d':>8}  assumptions")
    from datetime import timedelta
    for p in rows:
        owner = st.team_name(p.on_team_id)[:15] if p.on_team_id else "FA"
        a = [st.a_next(p, st.today + timedelta(days=x)) for x in (0, 30, 90)]
        typer.echo(f"{p.name:<26}{owner:<16}{p.mu:6.1f}{a[0]:7.2f}{a[1]:8.2f}{a[2]:8.2f}  {'; '.join(p.flags)}")


@app.command()
def standings(
    detailed: bool = typer.Option(False, help="Use the game-by-game engine (slower)."),
    sims: int = typer.Option(None, help="Simulated seasons (default 10,000 quick / 2,000 detailed)."),
    today: str = typer.Option(None, help="Pretend today is YYYY-MM-DD."),
):
    """League forecast: Monte Carlo of the rest of the season for every team."""
    import time

    from tradecalc.trade.standings import forecast, last_place_line

    st = _state(today=today)
    t0 = time.time()
    if detailed:
        from tradecalc.sim.live import live_world

        cfg, conn = _ctx()
        world, coaches_now = live_world(conn, cfg, st)
        fc = forecast(st, n=sims or 2000, seed=0, method="detailed", world=world, coaches_now=coaches_now)
    else:
        fc = forecast(st, n=sims or 10000, seed=0)
    typer.secho(f"\nLeague forecast ({fc.attrs['n']:,} seasons, {fc.attrs['method']}, {time.time() - t0:.1f}s)", bold=True)
    typer.echo(f"{'#':>3} {'team':<30}{'record':>8}{'proj W':>8}{'80% range':>11}{'win lg':>8}{'top 3':>7}{'last':>7}")
    for rank, r in fc.iterrows():
        typer.echo(f"{rank:>3} {r.team:<30}{r.record:>8}{r.exp_wins:>8.1f}{f'{r.wins_p10:.0f}-{r.wins_p90:.0f}':>11}"
                   f"{r.p_first:>8.0%}{r.p_top3:>7.0%}{r.p_last:>7.0%}")
    typer.secho(f"\n🏆 {fc.iloc[0]['team']} wins the league in {fc.iloc[0]['p_first']:.0%} of seasons.", bold=True)
    typer.secho(f"💩 {last_place_line(fc)}", bold=True)


@app.command("sim-fit")
def sim_fit():
    """Fit the projection engine (rates, usage, minutes, rest, coaches, noise) on recent seasons."""
    from tradecalc.sim.live import fit_and_save

    st = _state()
    cfg, conn = _ctx()
    fit_and_save(conn, cfg, st.league, log=typer.echo)


@app.command("sim-validate")
def sim_validate():
    """Out-of-sample validation of the projection engine on last season."""
    import pandas as pd
    from sklearn.metrics import brier_score_loss

    from tradecalc.sim.live import validate_last_season

    st = _state()
    cfg, conn = _ctx()
    r = validate_last_season(conn, cfg, st.league, log=typer.echo)
    typer.secho("\nPer-game fantasy points (lower is better)", bold=True)
    typer.echo(r["points"].round(2).to_string())
    c = r["coverage"]
    typer.secho("\nCalibration of simulated ranges (target 80% / 50%)", bold=True)
    typer.echo(f"  single games:  80% range {c['game_in80']:.1%}, 50% range {c['game_in50']:.1%}, bias {c['game_bias']:+.2f}")
    typer.echo(f"  player-weeks:  80% range {c['week_in80']:.1%}, 50% range {c['week_in50']:.1%}, bias {c['week_bias']:+.2f}")
    m = r["matchups"]
    m["bucket"] = pd.cut(m.p_win, [0, .2, .4, .6, .8, 1])
    typer.secho(f"\nFantasy matchups ({len(m)}): predicted vs actual win rate", bold=True)
    typer.echo(m.groupby("bucket", observed=True).agg(n=("won", "size"), predicted=("p_win", "mean"),
                                                      actual=("won", "mean")).round(3).to_string())
    typer.echo(f"  Brier score {brier_score_loss(m.won, m.p_win):.3f} (coin flip 0.250)")

    from tradecalc.sim.live import save_validation

    path = save_validation(r)
    typer.echo(f"\nSaved results to {path} (shown in the app's Help & accuracy tab).")


@app.command()
def simulate(
    a: int = typer.Option(None, "--a", help="ESPN team id (default: yours)."),
    b: int = typer.Option(None, "--b", help="ESPN team id (default: your opponent that week)."),
    week: int = typer.Option(None, help="Fantasy week (default: current)."),
    sims: int = typer.Option(10000, help="Number of simulated weeks."),
    seed: int = typer.Option(0, help="Random seed."),
    today: str = typer.Option(None, help="Pretend today is YYYY-MM-DD."),
):
    """Monte Carlo of a fantasy matchup with the full projection engine."""
    import time

    import numpy as np

    from tradecalc.sim.engine import run_matchup
    from tradecalc.sim.live import live_world

    st = _state(today=today)
    cfg, conn = _ctx()
    a = a or cfg["league"]["my_team_id"]
    wk = st.cal.weeks[week - 1] if week else st.cal.week_of(st.today) or st.cal.weeks[0]
    b = b or st.cal.opponents[a].get(wk.number)
    t0 = time.time()
    world, coaches_now = live_world(conn, cfg, st)
    res, _ = run_matchup(world, st, a, b, wk, n=sims, seed=seed, coaches_now=coaches_now)
    la, lb = st.team_name(a), st.team_name(b)
    ta, tb = res.totals[la], res.totals[lb]
    typer.secho(f"\nWeek {wk.number} ({wk.start:%a %b %d} to {wk.end:%a %b %d}): {la} vs {lb}  "
                f"[{sims:,} simulations, {time.time() - t0:.1f}s]", bold=True)
    if res.days:
        typer.echo(f"  Simulating {res.days[0]:%a %b %d} to {res.days[-1]:%a %b %d} "
                   f"({len(res.days)} of {len(wk.days())} days).")
    else:
        typer.echo("  This week is over: nothing left to simulate.")
    if any(res.banked.values()):
        typer.echo("  Already scored this week (added to totals): "
                   + ", ".join(f"{k} {v:.0f}" for k, v in res.banked.items()))
    for lab, tot in ((la, ta), (lb, tb)):
        q = np.percentile(tot, [10, 50, 90])
        typer.echo(f"  {lab:<28} mean {tot.mean():7.0f}   10th {q[0]:6.0f}   median {q[1]:6.0f}   90th {q[2]:6.0f}")
    typer.secho(f"  P({la} wins) = {res.p_win:.1%}   P(tie) = {res.p_tie:.2%}", bold=True)
    margin = ta - tb
    typer.echo(f"  Margin: mean {margin.mean():+.0f}, 80% range {np.percentile(margin, 10):+.0f} to {np.percentile(margin, 90):+.0f}")
    df = res.players.sort_values(["fantasy_team", "exp_fp_to_team"], ascending=[True, False])
    for lab, sub in df.groupby("fantasy_team", sort=False):
        typer.secho(f"\n  {lab}", bold=True)
        typer.echo(f"  {'player':<26}{'NBA':<5}{'games left':>11}{'P(active)':>10}{'starts':>8}{'FP/game':>9}{'FP to team':>11}")
        for r in sub.itertuples():
            typer.echo(f"  {r.player:<26}{r.nba_team:<5}{r.games:>11}{r.p_active:>10.0%}{r.exp_starts:>8.1f}"
                       f"{r.fp_per_game:>9.1f}{r.exp_fp_to_team:>11.0f}")


@app.command()
def backtest(save: bool = typer.Option(True, help="Save the chosen k and cv_typical to league.yaml.")):
    """Section 8: MAE of prior / live / blended FPPG on last season; picks k."""
    from tradecalc.backtest.run import format_table, run
    from tradecalc.config import save_model_value

    cfg, conn = _ctx()
    bt = cfg["backtest"]
    table, best_k, cv, curve, proj, blend_prior = run(cfg, conn, _client(cfg, conn))
    typer.secho(f"\nBacktest: curve fit on {bt['fit_season']}, tested on {bt['test_season']} "
                f"(MAE in FPPG, lower is better)", bold=True)
    typer.echo(format_table(table))
    typer.echo(f"\nBlend rows use prior = {blend_prior}.")
    if blend_prior == "proj":
        typer.secho("ADP CSVs are ESPN draft ranks, which ESPN revises in-season (they leak results); "
                    "k is chosen with the preseason projection prior. Supply genuine historical ADP "
                    "CSVs to test the ADP curve.", fg="yellow")
    pts = ", ".join(f"#{r}: {curve(r)[0]:.1f}" for r in (1, 10, 25, 50, 100, 150, curve.tail_rank))
    typer.echo(f"\nADP curve ({bt['fit_season']}): {pts}")
    typer.echo(f"ESPN {bt['test_espn_season']} projections used for {len(proj)} players")
    typer.echo(f"cv_typical (median per-game FP CV): {cv:.3f}")
    typer.secho(f"Best k = {best_k}", fg="green", bold=True)
    if save:
        save_model_value("k", best_k)
        save_model_value("cv_typical", round(cv, 4))
        typer.echo("Saved k and cv_typical to config/league.yaml")


if __name__ == "__main__":
    app()
