"""Evaluate one trade for both teams (section 7.1)."""
from __future__ import annotations

from dataclasses import dataclass, field

from tradecalc.trade.matchup import season_outlook, season_outlook_mc


@dataclass(frozen=True)
class Trade:
    you: int                    # user's team id
    partner: int
    give: tuple[int, ...]       # user's players sent
    get: tuple[int, ...]        # partner's players received


@dataclass
class TeamDelta:
    team_id: int
    pts_before: float
    pts_after: float
    wins_before: float
    wins_after: float
    starts_before: int = 0
    starts_after: int = 0
    dropped: list[int] = field(default_factory=list)
    added: list[int] = field(default_factory=list)
    weeks_before: list[dict] = field(default_factory=list)   # season_outlook()["weeks"] before the trade
    weeks_after: list[dict] = field(default_factory=list)

    @property
    def d_pts(self) -> float:
        return self.pts_after - self.pts_before

    @property
    def d_wins(self) -> float:
        return self.wins_after - self.wins_before


@dataclass
class TradeResult:
    trade: Trade
    you: TeamDelta
    partner: TeamDelta
    verdict: str
    drivers: list[str]


def apply_trade(trade: Trade, rosters: dict[int, list[int]], state) -> tuple[dict[int, list[int]], dict]:
    """Post-trade rosters. Each team keeps its active roster (non-IR) at full size: the side
    with too many active players drops its lowest-ROS ones; the side with too few signs the
    best free agents. Incoming OUT players go to a free IR slot if the team has one."""
    new = {t: list(r) for t, r in rosters.items()}
    new[trade.you] = [p for p in new[trade.you] if p not in trade.give] + list(trade.get)
    new[trade.partner] = [p for p in new[trade.partner] if p not in trade.get] + list(trade.give)
    moves = {trade.you: ([], []), trade.partner: ([], [])}  # team -> (dropped, added)
    taken = {p for r in new.values() for p in r}
    fas = sorted((p for p in state.free_agents() if p.espn_id not in taken), key=lambda p: -p.ros_value)
    n_ir = state.cfg["roster"]["ir"]
    for team, incoming in ((trade.you, trade.get), (trade.partner, trade.give)):
        ir = {p for p in state.ir.get(team, ()) if p in new[team]}
        for p in incoming:
            if len(ir) < n_ir and state.ir_eligible(p):
                ir.add(p)
        active = [p for p in new[team] if p not in ir]
        target = state.active_size
        if len(active) > target:
            drop = sorted(active, key=lambda i: state.players[i].ros_value)[: len(active) - target]
            new[team] = [p for p in new[team] if p not in drop]
            moves[team][0].extend(drop)
        while len([p for p in new[team] if p not in ir]) < target and fas:
            fa = fas.pop(0)
            new[team].append(fa.espn_id)
            moves[team][1].append(fa.espn_id)
    return new, moves


def _verdict(d: float, cfg) -> str:
    t = cfg["trade"]
    return "ACCEPT" if d >= t["verdict_accept"] else "DECLINE" if d <= t["verdict_decline"] else "FAIR-ISH"


def _drivers(trade: Trade, moves: dict, you: "TeamDelta", state) -> list[str]:
    P = state.players
    out = []
    ds = you.starts_after - you.starts_before
    if ds:
        out.append(f"{ds:+d} starts over the remaining schedule")
    g_in = sum(P[i].games_remaining for i in trade.get)
    g_out = sum(P[i].games_remaining for i in trade.give)
    if len(trade.get) == len(trade.give) and g_in != g_out:
        out.append(f"{g_in - g_out:+d} games remaining for traded players")
    dropped, added = moves[trade.you]
    mu_in = sum(P[i].mu for i in (*trade.get, *added))
    mu_out = sum(P[i].mu for i in (*trade.give, *dropped))
    out.append(f"{mu_in - mu_out:+.1f} roster FPPG incl. FA moves ({mu_out:.1f} out, {mu_in:.1f} in)")
    for i in (*trade.give, *trade.get):
        for f in P[i].flags:
            out.append(f"{f} on {P[i].name}")
    for team, (dropped, added) in moves.items():
        who = "you" if team == trade.you else "partner"
        for i in dropped:
            out.append(f"{who} drop {P[i].name} ({P[i].mu:.1f})")
        for i in added:
            out.append(f"{who} add FA {P[i].name} ({P[i].mu:.1f})")
    return out


def evaluate(trade: Trade, state, precise: bool = False, mc_sims: int | None = None) -> TradeResult:
    outlook = season_outlook_mc if precise else season_outlook
    kw = {"n": mc_sims or state.cfg["trade"]["monte_carlo_sims"]} if precise else {}
    before = state.rosters
    after, moves = apply_trade(trade, before, state)
    deltas = {}
    for team in (trade.you, trade.partner):
        b = outlook(team, before, state, **kw)
        a = outlook(team, after, state, **kw)
        deltas[team] = TeamDelta(team, b["pts_per_week"], a["pts_per_week"], b["e_wins"], a["e_wins"],
                                 b["starts"], a["starts"], dropped=moves[team][0], added=moves[team][1],
                                 weeks_before=b["weeks"], weeks_after=a["weeks"])
    you = deltas[trade.you]
    return TradeResult(trade, you, deltas[trade.partner], _verdict(you.d_wins, state.cfg),
                       _drivers(trade, moves, you, state))


def format_result(r: TradeResult, state) -> str:
    P = state.players
    give = " + ".join(P[i].name for i in r.trade.give)
    get = " + ".join(P[i].name for i in r.trade.get)
    lines = [
        f"You give {give}  |  you get {get}  (with {state.team_name(r.trade.partner)})",
        f"You:     {r.you.d_pts:+6.1f} pts/wk   {r.you.d_wins:+.2f} expected wins "
        f"({r.you.wins_before:.2f} -> {r.you.wins_after:.2f})",
        f"Partner: {r.partner.d_pts:+6.1f} pts/wk   {r.partner.d_wins:+.2f} expected wins "
        f"({r.partner.wins_before:.2f} -> {r.partner.wins_after:.2f})",
        f"Verdict: {r.verdict}",
        "Key drivers: " + "; ".join(r.drivers),
    ]
    return "\n".join(lines)
