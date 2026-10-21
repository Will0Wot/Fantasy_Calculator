"""Streaming guide: which player to drop and which free agent to add (and on which day) to raise a
team's chance of winning this week's matchup.

For a team and week, the planner tries moves "on day D, drop X and add free agent Y" (or just add
Y if the roster has an open spot), and scores each with the same weekly model the app uses
everywhere: daily lineups with fill-ins, the player's games left that week and his availability.
The week is split at the move: days before D are played by the old roster, days from D on by the
new one. Moves are chosen greedily (best first) up to the league's acquisition limit.

Only bench-level players are considered for dropping (the lowest rest-of-season values on the
roster), only players ESPN lists as free agents (not on waivers) are added, and every suggested move
also reports what keeping it would do to the rest of the season, so a one-week rental is labelled
as one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from tradecalc.trade.matchup import p_win, season_outlook, week_dist


@dataclass
class Move:
    day: date
    kind: str                 # "drop" | "ir" (move an injured player to an IR slot) | "open" (empty spot)
    out: int | None           # player dropped or moved to IR (None for an open spot)
    add: int
    gain_pts: float           # expected points this week vs the plan without this move
    p_after: float | None     # this week's win chance after this move (and the ones before it)
    add_games: int            # games the added player's team has from `day` to the end of the week
    out_games: int            # games the dropped player's team has from `day` on
    d_week: float = 0.0       # change in this week's win chance
    d_future: float = 0.0     # change in expected wins in later weeks if the move is kept


@dataclass
class StreamPlan:
    team: int
    week: object
    opp: int | None
    start: date
    M_before: float
    M_after: float
    p_before: float | None
    p_after: float | None
    moves: list[Move] = field(default_factory=list)
    open_spots: int = 0
    free_ir: int = 0
    future_d_wins: float = float("nan")   # keeping every move: change in expected wins after this week
    options: list = field(default_factory=list)   # first move: the best alternatives (in case a player is taken)
    protected: list = field(default_factory=list)  # low-value players kept because a rival would claim them


def real_roster(state, team: int) -> list[int]:
    """The team's actual roster: without the free agents the model pencils into open spots."""
    fills = set(state.fills.get(team, ()))
    return [i for i in state.rosters[team] if i not in fills]


def _days(week, start: date) -> list[date]:
    return [d for d in week.days() if d >= start]


def _games(state, pid: int, week, start: date) -> int:
    return sum(state.cal.team_plays(state.players[pid].pro_team, d) for d in _days(week, start))


def _seg(state, roster, week, start: date) -> tuple[float, float]:
    """Expected counted points and variance for days >= start of the week."""
    if start > week.end:
        return 0.0, 0.0
    d = week_dist(roster, week, state, start if start > week.start else None)
    return d.M, d.V


def timeline_dist(state, week, timeline: list[tuple[date, list[int]]]) -> tuple[float, float]:
    """timeline: [(first day, roster from that day)], sorted by day. Points and variance for the week."""
    M = V = 0.0
    for k, (d, roster) in enumerate(timeline):
        nxt = timeline[k + 1][0] if k + 1 < len(timeline) else week.end + timedelta(1)
        m1, v1 = _seg(state, roster, week, d)
        m2, v2 = _seg(state, roster, week, nxt)
        M += m1 - m2
        V += max(v1 - v2, 0.0)
    return M, V


def week_points(state, pid: int, week, start: date) -> float:
    """A player's expected fantasy points over the rest of the week (all his games)."""
    p = state.players[pid]
    return p.mu * sum(state.a_for(p, d) for d in _days(week, start) if state.cal.team_plays(p.pro_team, d))


def available_free_agents(state, exclude=()) -> list[int]:
    """Free agents that can be added today (ESPN status FREEAGENT; waiver players excluded)."""
    taken = {i for t in state.rosters for i in real_roster(state, t)} | set(exclude)
    out = []
    for p in state.players.values():
        if p.on_team_id != 0 or p.espn_id in taken or p.mu <= 0:
            continue
        lp = getattr(state.league, "players", {}).get(p.espn_id)
        if lp is not None and getattr(lp, "pool_status", "") == "WAIVERS":
            continue
        out.append(p.espn_id)
    return out


def streamer_table(state, week, start: date | None = None, top: int = 25) -> pd.DataFrame:
    """Best free agents for the rest of this week, by expected fantasy points."""
    start = max(start or state.today, week.start)
    rows = []
    for i in available_free_agents(state):
        p = state.players[i]
        g = _games(state, i, week, start)
        if not g:
            continue
        rows.append({"id": i, "Player": p.name, "NBA": p.pro_team, "Pos": "/".join(p.positions),
                     "FPPG": p.mu, "Games left": g, "Exp. pts this week": week_points(state, i, week, start),
                     "Status": p.injury_status.replace("_", "-").lower() if p.injury_status != "ACTIVE" else "",
                     "Rest-of-season value": p.ros_value})
    df = pd.DataFrame(rows)
    return df.sort_values("Exp. pts this week", ascending=False).head(top).reset_index(drop=True) if len(df) else df


def plan(state, team: int, week, max_moves: int | None = None, n_drop: int | None = None,
         n_add: int | None = None, min_gain: float = 3.0, n_eval: int = 12) -> StreamPlan:
    """Streaming plan for `team` in `week` (see module docstring).

    Each step tries every (drop / IR / open spot, free agent, day) and keeps the n_eval moves that
    add the most points this week; those are then scored by the total change in expected wins:
    this week's win chance plus every later week if the move is kept. The best is taken if it
    helps, up to max_moves moves."""
    cfg = state.cfg.get("streaming", {})
    max_moves = cfg.get("moves_per_week", 1) if max_moves is None else max_moves
    n_drop = n_drop or cfg.get("drop_candidates", 5)
    n_add = n_add or cfg.get("add_candidates", 30)
    shrink = state.cfg["model"].get("matchup_shrink", 1.0)
    start = max(state.today, week.start)
    opp = state.cal.opponents.get(team, {}).get(week.number)

    roster = real_roster(state, team)
    ir = [i for i in roster if i in set(state.ir.get(team, ()))]
    open_spots = max(state.active_size - (len(roster) - len(ir)), 0)
    free_ir = max(state.cfg["roster"].get("ir", 0) - len(ir), 0)
    mid_week = start > week.start
    bank_me = state.banked(team, week)[0] if mid_week else 0.0
    if opp is not None:
        d_opp = week_dist(state.rosters[opp], week, state, start if mid_week else None)
        bank_opp = state.banked(opp, week)[0] if mid_week else 0.0
        M_opp, V_opp = d_opp.M + bank_opp, d_opp.V

    def win(M, V):
        return None if opp is None else p_win(M + bank_me, V, M_opp, V_opp, shrink)

    future = _future_fn(state, team, week, roster)
    timeline = [(start, [i for i in roster if i not in ir])]          # players in IR slots can't play
    M0, V0 = timeline_dist(state, week, timeline)
    out = StreamPlan(team, week, opp, start, M0 + bank_me, M0 + bank_me, win(M0, V0), win(M0, V0),
                     open_spots=open_spots, free_ir=free_ir, future_d_wins=0.0)
    pool = available_free_agents(state)
    by_week = sorted(pool, key=lambda i: -week_points(state, i, week, start))[:n_add]
    by_ros = sorted(pool, key=lambda i: -state.players[i].ros_value)[:10]
    adds_all = list(dict.fromkeys(by_week + by_ros))
    # A dropped player goes to waivers. If he would be the best player available there (better per
    # game, or more rest-of-season value, than every free agent left after this pickup), a rival
    # claims him: you lose him for good and strengthen an opponent. Such players are kept. A player
    # no better than what is already free costs nothing to drop: rivals could already add his equal.
    P = state.players
    top_mu = sorted(pool, key=lambda i: -P[i].mu)[:2]
    top_ros = sorted(pool, key=lambda i: -P[i].ros_value)[:2]

    def claimable(x, add=None) -> bool:
        mu = next((P[i].mu for i in top_mu if i != add), float("-inf"))
        ros = next((P[i].ros_value for i in top_ros if i != add), float("-inf"))
        return P[x].mu > mu or P[x].ros_value > ros

    by_value = sorted(timeline[0][1], key=lambda i: P[i].ros_value)
    out.protected = [i for i in by_value[:n_drop] if claimable(i)]
    M_cur, V_cur, fut_cur = M0, V0, 0.0
    gone, spots, ir_left, to_ir = set(), open_spots, free_ir, []
    for _ in range(max_moves):
        last_day, cur = timeline[-1]
        drops = [i for i in sorted(cur, key=lambda i: P[i].ros_value) if not claimable(i)][:n_drop]
        options = ([("open", None)] if spots > 0 else []) \
            + ([("ir", i) for i in cur if state.ir_eligible(i)] if ir_left > 0 else []) \
            + [("drop", i) for i in drops]
        cands = {}
        for add in adds_all:
            if add in cur or add in gone:
                continue
            p_add = state.players[add]
            for d in _days(week, last_day):
                if not state.cal.team_plays(p_add.pro_team, d):
                    continue
                for kind, x in options:
                    if kind == "drop" and claimable(x, add):      # he'd be the best free agent once you add `add`
                        continue
                    new = [i for i in cur if i != x] + [add]
                    tl = timeline + [(d, new)] if d > last_day else timeline[:-1] + [(d, new)]
                    M, V = timeline_dist(state, week, tl)
                    key = (kind, x, add)
                    if key not in cands or M > cands[key][2]:
                        cands[key] = (d, tl, M, V)
        best = None
        for (kind, x, add), (d, tl, M, V) in sorted(cands.items(), key=lambda kv: -kv[1][2])[:n_eval]:
            if M - M_cur < min_gain:
                continue
            keeps = tl[-1][1] + ir + to_ir + ([x] if kind == "ir" else [])
            fut = future(keeps)
            d_week = 0.0 if opp is None else win(M, V) - win(M_cur, V_cur)
            total = d_week + (fut - fut_cur)
            if not out.moves and total > 0.005:
                out.options.append(Move(d, kind, x, add, M - M_cur, win(M, V), _games(state, add, week, d),
                                        _games(state, x, week, d) if x is not None else 0, d_week, fut - fut_cur))
            if best is None or total > best[0]:
                best = (total, d_week, fut, kind, x, add, d, tl, M, V)
        if best is None or best[0] <= 0.005:
            break
        total, d_week, fut, kind, x, add, d, tl, M, V = best
        out.moves.append(Move(d, kind, x, add, M - M_cur, win(M, V), _games(state, add, week, d),
                              _games(state, x, week, d) if x is not None else 0, d_week, fut - fut_cur))
        timeline, M_cur, V_cur, fut_cur = tl, M, V, fut
        if kind == "open":
            spots -= 1
        elif kind == "ir":
            ir_left -= 1
            to_ir.append(x)
        else:
            gone.add(x)
    out.M_after, out.p_after, out.future_d_wins = M_cur + bank_me, win(M_cur, V_cur), fut_cur
    out.options.sort(key=lambda m: -(m.d_week + m.d_future))
    return out


def _future_fn(state, team: int, week, roster: list[int]):
    """keeps -> change in expected wins after this week vs keeping the current real roster.
    Other teams' rosters are as the model sees them."""
    nxt = week.end + timedelta(1)
    if nxt > state.cal.weeks[-1].end:
        return lambda keeps: 0.0
    rosters = dict(state.rosters)

    def e_wins(r):
        rosters[team] = list(dict.fromkeys(r))
        return season_outlook(team, rosters, state, nxt)["e_wins"]

    base = e_wins(roster)
    memo = {}

    def f(keeps):
        k = tuple(sorted(set(keeps)))
        if k not in memo:
            memo[k] = e_wins(list(k)) - base
        return memo[k]
    return f


def describe(state, m: Move) -> str:
    P = state.players
    if m.kind == "open":
        return f"add {P[m.add].name} (open roster spot)"
    if m.kind == "ir":
        return f"move {P[m.out].name} to IR, add {P[m.add].name}"
    return f"drop {P[m.out].name}, add {P[m.add].name}"


def league_summary(state, week, max_moves: int | None = None) -> pd.DataFrame:
    """Every team's best plan this week."""
    rows = []
    for t in sorted(state.rosters):
        pl = plan(state, t, week, max_moves=max_moves)
        rows.append({"Team": state.team_name(t),
                     "Plan": "; ".join(f"{m.day:%a}: {describe(state, m)}" for m in pl.moves) or "no move helps",
                     "Extra pts": pl.M_after - pl.M_before, "Win chance now": pl.p_before, "With moves": pl.p_after,
                     "Long-term (wins)": pl.future_d_wins})
    return pd.DataFrame(rows)
