"""Trade finder (section 7.2): enumerate small trades, prune cheaply, evaluate survivors."""
from __future__ import annotations

from itertools import combinations

from tradecalc.trade.evaluate import Trade, TradeResult, evaluate


def _ros_delta(roster: list[int], give: tuple, get: tuple, state, fa_ros: float, ir: set) -> float:
    """Cheap ROS change for one team, approximating apply_trade's roster-size rules."""
    P = state.players
    d = sum(P[i].ros_value for i in get) - sum(P[i].ros_value for i in give)
    extra = sum(i not in ir for i in get) - sum(i not in ir for i in give)
    if extra > 0:
        after = [i for i in roster if i not in give] + list(get)
        d -= sum(sorted(P[i].ros_value for i in after)[:extra])
    elif extra < 0:
        d += -extra * fa_ros
    return d


def candidate_trades(user: int, partner: int, state, max_size: int = 2):
    """Yield (trade, cheap ROS delta for user, for partner) for every 1..max_size x 1..max_size trade."""
    mine, theirs = state.rosters[user], state.rosters[partner]
    ir_me, ir_them = state.ir.get(user, set()), state.ir.get(partner, set())
    fas = sorted((p.ros_value for p in state.free_agents()), reverse=True)
    fa_ros = fas[0] if fas else 0.0
    for a in range(1, max_size + 1):
        for b in range(1, max_size + 1):
            for give in combinations(state.tradeable(user), a):
                for get in combinations(state.tradeable(partner), b):
                    yield (Trade(user, partner, give, get),
                           _ros_delta(mine, give, get, state, fa_ros, ir_me),
                           _ros_delta(theirs, get, give, state, fa_ros, ir_them))


def find_trades(user: int, state, max_size: int = 2, partners: list[int] | None = None,
                top: int | None = None, progress=None) -> dict[int, list[TradeResult]]:
    cfg = state.cfg["trade"]
    top = top or cfg["top_n_per_opponent"]
    partners = partners or [t for t in state.rosters if t != user]
    out = {}
    for partner in partners:
        survivors = []
        for trade, d_you, d_them in candidate_trades(user, partner, state, max_size):
            give_ros = sum(state.players[i].ros_value for i in trade.give)
            get_ros = sum(state.players[i].ros_value for i in trade.get)
            if d_you < -cfg["user_loss_prune_margin"] * give_ros:
                continue
            if d_them < -cfg["partner_loss_prune_margin"] * get_ros:
                continue
            survivors.append((d_you, trade))
        # Bound the full evaluations per partner, keeping the best by the cheap metric.
        survivors.sort(key=lambda x: -x[0])
        survivors = survivors[: cfg["max_full_evals_per_opponent"]]
        results = []
        for _, trade in survivors:
            r = evaluate(trade, state)
            # only trades the builder would call ACCEPT (not close calls)
            if r.you.d_wins >= cfg.get("finder_min_gain", cfg["verdict_accept"]) \
                    and r.partner.d_wins >= -cfg["acceptance_threshold"]:
                results.append(r)
        results.sort(key=lambda r: -r.you.d_wins)
        out[partner] = results[:top]
        if progress:
            progress(partner, len(survivors), len(results))
    return out
