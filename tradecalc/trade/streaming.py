"""Streaming guide: which player to drop and which free agent to add (and on which day) to raise a
team's chance of winning this week's matchup, without hurting the rest of its season.

For a team and week, the planner tries moves "on day D, drop X and add free agent Y" (or move an
injured X to a free IR slot, or fill an open roster spot), and scores each with the same weekly model
the app uses everywhere: daily lineups with fill-ins, games left that week and availability. The
week is split at the move: days before D are played by the old roster, days from D on by the new one.

Every move is judged on its total effect on the team's expected wins, three parts:
  this week       the change in this week's win chance
  later weeks     keeping the move for the rest of the season (losing X, having Y)
  rival pickup    X goes on waivers. If he would be the best player available there, a rival claims
                  him: the one with the best waiver priority who would want him (he beats their
                  weakest player). That rival is then stronger in every week you still play them.
                  If free agents just as good as X are already available, nobody gains by claiming
                  him and this part is zero.
Moves are taken best first while the total is positive, up to the league's acquisition limit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from tradecalc.trade.matchup import p_win, season_outlook, week_dist

CLAIM_BAND = 0.10     # within +-5% of the best free agent's value, a claim is a coin flip


@dataclass
class Claim:
    p: float                      # chance a rival claims the dropped player
    team: int | None = None       # the rival who would get him
    rank: int | None = None       # that rival's waiver priority (1 = first)
    cost: float = 0.0             # p x change in your expected wins after this week
    weeks: list = field(default_factory=list)   # remaining weeks (after this one) you play that rival


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
    claim: Claim = field(default_factory=lambda: Claim(0.0))

    @property
    def total(self) -> float:
        return self.d_week + self.d_future + self.claim.cost


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
    options: list = field(default_factory=list)      # first move: alternatives that also help (best first)
    drop_costs: list = field(default_factory=list)   # first move: the best move dropping each bench player

    @property
    def d_future(self) -> float:
        return sum(m.d_future for m in self.moves)

    @property
    def d_rival(self) -> float:
        return sum(m.claim.cost for m in self.moves)


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
    on = getattr(p, "mu_on", lambda d: p.mu)
    return sum(on(d) * state.a_for(p, d) for d in _days(week, start) if state.cal.team_plays(p.pro_team, d))


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

    Each step tries every (drop / IR / open spot, free agent, day), keeps the moves that add the most
    points this week (plus the best one for each bench player), scores them by total expected wins
    and takes the best if it helps, up to max_moves moves."""
    cfg = state.cfg.get("streaming", {})
    max_moves = cfg.get("moves_per_week", 1) if max_moves is None else max_moves
    n_drop = n_drop or cfg.get("drop_candidates", 5)
    n_add = n_add or cfg.get("add_candidates", 30)
    shrink = state.cfg["model"].get("matchup_shrink", 1.0)
    start = max(state.today, week.start)
    opp = state.cal.opponents.get(team, {}).get(week.number)
    P = state.players

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

    pool = available_free_agents(state)
    later = _Later(state, team, week, roster)
    claim = _claim_fn(state, team, week, pool, later)
    timeline = [(start, [i for i in roster if i not in ir])]          # players in IR slots can't play
    M0, V0 = timeline_dist(state, week, timeline)
    out = StreamPlan(team, week, opp, start, M0 + bank_me, M0 + bank_me, win(M0, V0), win(M0, V0),
                     open_spots=open_spots, free_ir=free_ir)
    by_week = sorted(pool, key=lambda i: -week_points(state, i, week, start))[:n_add]
    by_ros = sorted(pool, key=lambda i: -P[i].ros_value)[:10]
    adds_all = list(dict.fromkeys(by_week + by_ros))
    M_cur, V_cur, fut_cur = M0, V0, 0.0
    gone, spots, ir_left, to_ir = set(), open_spots, free_ir, []
    for step in range(max_moves):
        last_day, cur = timeline[-1]
        drops = sorted(cur, key=lambda i: P[i].ros_value)[:n_drop]
        options = ([("open", None)] if spots > 0 else []) \
            + ([("ir", i) for i in cur if state.ir_eligible(i)] if ir_left > 0 else []) \
            + [("drop", i) for i in drops]
        cands = {}
        for add in adds_all:
            if add in cur or add in gone:
                continue
            for d in _days(week, last_day):
                if not state.cal.team_plays(P[add].pro_team, d):
                    continue
                for kind, x in options:
                    new = [i for i in cur if i != x] + [add]
                    tl = timeline + [(d, new)] if d > last_day else timeline[:-1] + [(d, new)]
                    M, V = timeline_dist(state, week, tl)
                    key = (kind, x, add)
                    if key not in cands or M > cands[key][2]:
                        cands[key] = (d, tl, M, V)
        # score the moves that add the most points, and each bench player's best drop
        ranked = sorted(cands.items(), key=lambda kv: -kv[1][2])
        pick = dict(ranked[:n_eval])
        for x in drops:
            mine = [kv for kv in ranked if kv[0][0] == "drop" and kv[0][1] == x]
            if mine:
                pick.setdefault(*mine[0])
        scored = []
        for (kind, x, add), (d, tl, M, V) in pick.items():
            keeps = tl[-1][1] + ir + to_ir + ([x] if kind == "ir" else [])
            fut = later.delta(keeps)
            m = Move(d, kind, x, add, M - M_cur, win(M, V), _games(state, add, week, d),
                     _games(state, x, week, d) if x is not None else 0,
                     0.0 if opp is None else win(M, V) - win(M_cur, V_cur), fut - fut_cur,
                     claim(x, add, keeps) if kind == "drop" else Claim(0.0))
            scored.append((m, tl, M, V, fut))
        if step == 0:
            out.options = sorted([m for m, *_ in scored if m.total > 0.005 and m.gain_pts >= min_gain],
                                 key=lambda m: -m.total)
            best_drop = {}
            for m, *_ in scored:
                if m.kind == "drop" and (m.out not in best_drop or m.total > best_drop[m.out].total):
                    best_drop[m.out] = m
            out.drop_costs = sorted(best_drop.values(), key=lambda m: -m.total)
        good = [s for s in scored if s[0].gain_pts >= min_gain and s[0].total > 0.005]
        if not good:
            break
        m, tl, M, V, fut = max(good, key=lambda s: s[0].total)
        out.moves.append(m)
        timeline, M_cur, V_cur, fut_cur = tl, M, V, fut
        if m.kind == "open":
            spots -= 1
        elif m.kind == "ir":
            ir_left -= 1
            to_ir.append(m.out)
        else:
            gone.add(m.out)
    out.M_after, out.p_after = M_cur + bank_me, win(M_cur, V_cur)
    return out


class _Later:
    """Your expected wins after this week for a given roster (and optionally a changed rival roster),
    relative to keeping your current roster. Other teams as the model sees them."""

    def __init__(self, state, team: int, week, roster: list[int]):
        self.state, self.team = state, team
        self.nxt = week.end + timedelta(1)
        self.done = self.nxt > state.cal.weeks[-1].end
        self.memo = {}
        self.base = 0.0 if self.done else self.e_wins(roster)

    def e_wins(self, keeps, rival: int | None = None, rival_roster=None) -> float:
        if self.done:
            return 0.0
        key = (tuple(sorted(set(keeps))), rival, tuple(sorted(rival_roster)) if rival_roster else None)
        if key not in self.memo:
            rosters = dict(self.state.rosters)
            rosters[self.team] = list(key[0])
            if rival is not None:
                rosters[rival] = list(rival_roster)
            self.memo[key] = season_outlook(self.team, rosters, self.state, self.nxt)["e_wins"]
        return self.memo[key]

    def delta(self, keeps) -> float:
        return self.e_wins(keeps) - self.base


def _claim_fn(state, team: int, week, pool: list[int], later: _Later):
    """(dropped player, player added, roster you keep) -> Claim (see module docstring)."""
    P = state.players
    top = sorted(pool, key=lambda i: -P[i].ros_value)[:2]
    teams = getattr(state.league, "teams", {})
    rank = {t: getattr(teams.get(t), "waiver_rank", None) for t in state.rosters}
    rivals = sorted((t for t in state.rosters if t != team), key=lambda t: (rank.get(t) or 99, t))
    remaining = [w for w in state.cal.weeks if w.start > week.end]

    def claim(x: int, add: int, keeps) -> Claim:
        best = next((P[i].ros_value for i in top if i != add), 0.0)       # best free agent left
        gap = (P[x].ros_value - best) / max(best, 1.0)
        p = min(max(0.5 + gap / CLAIM_BAND, 0.0), 1.0)
        if p <= 0:
            return Claim(0.0)
        for r in rivals:                                  # waiver order: first team that wants him
            ir_r = set(state.ir.get(r, ()))
            active = [i for i in state.rosters[r] if i not in ir_r]
            weakest = min(active, key=lambda i: P[i].ros_value) if active else None
            if weakest is None or P[x].ros_value > P[weakest].ros_value:
                new_r = [i for i in state.rosters[r] if i != weakest] + [x]
                cost = min(later.e_wins(keeps, r, new_r) - later.e_wins(keeps), 0.0)   # a stronger rival never helps
                weeks = [w.number for w in remaining if state.cal.opponents.get(team, {}).get(w.number) == r]
                return Claim(p, r, rank.get(r), p * cost, weeks)
        return Claim(0.0)
    return claim


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
                     "Later weeks": pl.d_future, "Rival pickups": pl.d_rival})
    return pd.DataFrame(rows)
