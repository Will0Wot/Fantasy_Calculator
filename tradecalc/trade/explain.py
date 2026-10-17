"""Plain-English explanation of a trade verdict: a summary paragraph, ranked reasons, and
weekly win probability before vs after.

Everything here is derived from the evaluation itself (no new modelling):
  production    fantasy points per game in vs out, counting forced drops and free-agent adds
  games         expected games actually played in vs out (schedule x availability)
  lineup        how much of the roster change reaches your starting lineup (starts, points/week)
  health/role   injury and role-change flags on the players involved
  partner       how the trade changes the other team's expected wins (would they say yes?)
  weeks         win probability per remaining week, before and after
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Explanation:
    verdict: str
    headline: str
    paragraph: str
    bullets: list[str]          # most important first
    this_week: dict | None      # {"week", "opp", "before", "after", "delta"} (probabilities 0-1)
    weeks: pd.DataFrame         # week, opponent, before, after, delta


def _names(ids, P) -> str:
    names = [P[i].name for i in ids]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _dpct(x: float) -> str:
    v = round(x * 100)
    return f"{'+' if v > 0 else '−' if v < 0 else '±'}{abs(v)}%"


def _exp_games(p) -> float:
    """Expected games he actually plays the rest of the season (schedule x availability)."""
    return p.ros_value / p.mu if p.mu > 0 else 0.0


def _signed(x: float, fmt: str = ".0f") -> str:
    s = format(abs(x), fmt)
    return ("+" if x > 0 else "−" if x < 0 else "±") + s


def _sign(x: float) -> int:
    return 1 if x > 0 else -1 if x < 0 else 0


def _games_lost_share(p) -> float:
    """Share of his remaining games he is expected to miss beyond his normal rate (injury effect)."""
    full = p.games_remaining * p.avail.base
    return float(min(max(1 - _exp_games(p) / full, 0.0), 1.0)) if full > 0 else 0.0


def _health_notes(p) -> list[str]:
    notes = []
    for f in p.flags:
        low = f.lower()
        if any(w in low for w in ("out", "day-to-day", "back from", "tear", "rupture", "surgery", "fracture",
                                  "strain", "absence", "injur")):
            notes.append(f)
    return notes


@dataclass
class Reason:
    weight: float      # how much it matters (for ordering)
    sign: int          # +1 helps you, -1 hurts you, 0 neutral/context
    text: str          # bullet
    phrase: str        # short clause for the summary paragraph


def explain(r, state) -> Explanation:
    P = state.players
    t = r.trade
    you, partner = r.you, r.partner
    pname = state.team_name(t.partner)
    give, get = list(t.give), list(t.get)
    out_ids, in_ids = give + you.dropped, get + you.added

    fppg_out = sum(P[i].mu for i in out_ids)
    fppg_in = sum(P[i].mu for i in in_ids)
    games_out = sum(_exp_games(P[i]) for i in out_ids)
    games_in = sum(_exp_games(P[i]) for i in in_ids)
    n_weeks = max(len(you.weeks_before), 1)
    raw_per_week = (sum(P[i].ros_value for i in in_ids) - sum(P[i].ros_value for i in out_ids)) / n_weeks
    lineup_per_week = you.d_pts
    R: list[Reason] = []

    d = fppg_in - fppg_out
    avg_out = fppg_out / max(len(out_ids), 1)
    R.append(Reason(abs(d) * games_in / max(len(in_ids), 1), _sign(d),
                    f"Production: you send out {fppg_out:.1f} fantasy points per game ({_names(out_ids, P)}) and "
                    f"take back {fppg_in:.1f} ({_names(in_ids, P)}), {_signed(d, '.1f')} per game.",
                    f"you give up {fppg_out:.1f} points per game and get {fppg_in:.1f} back"))
    dg = games_in - games_out
    if abs(dg) >= 1:
        R.append(Reason(abs(dg) * avg_out, _sign(dg),
                        f"Games: the players you get are expected to play {games_in:.0f} games the rest of the way "
                        f"versus {games_out:.0f} for the players you lose ({dg:+.0f}), counting the schedule, "
                        f"injuries and rest.",
                        f"the incoming players should play about {abs(dg):.0f} {'more' if dg > 0 else 'fewer'} games"))
    def _role(i):
        q = P[i]
        m = getattr(q, "minutes", float("nan"))
        r_ = getattr(q, "role", "") or "?"
        return m, r_

    mins_out = [(_role(i)[0], i) for i in give if _role(i)[0] == _role(i)[0]]
    mins_in = [(_role(i)[0], i) for i in get if _role(i)[0] == _role(i)[0]]
    if mins_out and mins_in:
        m_out, m_in = sum(m for m, _ in mins_out), sum(m for m, _ in mins_in)
        desc = lambda ids: "; ".join(f"{P[i].name} ~{_role(i)[0]:.0f} min ({_role(i)[1].lower()})" for i in ids)  # noqa: E731
        roles_out = {_role(i)[1] for i in give}
        roles_in = {_role(i)[1] for i in get}
        dm = m_in - m_out
        if abs(dm) >= 3 or roles_out != roles_in:
            R.append(Reason(abs(dm) * 0.9 * 60, _sign(dm),
                            f"Role and minutes: you give {desc(give)} and get {desc(get)} "
                            f"({dm:+.0f} projected minutes per game).",
                            f"you'd go from {m_out:.0f} to {m_in:.0f} projected minutes "
                            f"({'/'.join(sorted(roles_out)).lower()} → {'/'.join(sorted(roles_in)).lower()})"))

    gap = lineup_per_week - raw_per_week
    if abs(gap) >= 8:
        if gap < 0:
            R.append(Reason(abs(gap) * n_weeks, -1,
                            f"Lineup crunch: on paper the roster changes by {raw_per_week:+.0f} points a week, but "
                            f"only {lineup_per_week:+.0f} reach your starting lineup — many of the extra games fall on "
                            f"nights your slots are already full.",
                            "much of the extra production would sit on your bench"))
        else:
            R.append(Reason(abs(gap) * n_weeks, 1,
                            f"Lineup fit: the trade opens starting spots, so your lineup gains {lineup_per_week:+.0f} "
                            f"points a week even though the raw production change is {raw_per_week:+.0f}.",
                            "it fits your lineup better than the raw numbers suggest"))
    ds = you.starts_after - you.starts_before
    if abs(ds) >= 3:
        R.append(Reason(abs(ds) * avg_out * 0.5, _sign(ds),
                        f"Starts: you would fill {abs(ds)} {'more' if ds > 0 else 'fewer'} lineup spots over the rest "
                        f"of the season.",
                        f"you'd fill {abs(ds)} {'more' if ds > 0 else 'fewer'} lineup spots"))
    for i in get:
        notes = _health_notes(P[i])
        if notes:
            lost = _games_lost_share(P[i])
            extra = f" — expected to miss about {lost:.0%} of his remaining games" if lost >= 0.03 else ""
            R.append(Reason(P[i].mu * P[i].games_remaining * P[i].avail.base * lost + 1, -1, f"Risk on {P[i].name} (incoming): {'; '.join(notes)}{extra}.",
                            f"{P[i].name} is expected to miss about {lost:.0%} of his games" if lost >= 0.03
                            else f"{P[i].name} has a minor injury tag ({notes[0]})"))
        if P[i].role_change:
            R.append(Reason(P[i].mu * 5, 0, f"{P[i].name}'s role is changing — recent minutes differ a lot from his "
                                              f"season average.", f"{P[i].name}'s role is in flux"))
    for i in give:
        lost = _games_lost_share(P[i])
        if _health_notes(P[i]) and lost >= 0.05:
            R.append(Reason(P[i].mu * P[i].games_remaining * P[i].avail.base * lost, 1, f"{P[i].name} (outgoing) is expected to miss about {lost:.0%} "
                                                   f"of his remaining games, risk you would be shedding.",
                            f"you'd be shedding {P[i].name}'s injury risk"))
    for i in you.dropped:
        R.append(Reason(P[i].ros_value, -1, f"Roster spot: to fit the extra player you would have to drop "
                                           f"{P[i].name} ({P[i].mu:.1f} per game).", f"you'd have to drop {P[i].name}"))
    for i in you.added:
        R.append(Reason(P[i].ros_value * 0.3, 1, f"Open spot: the trade frees a roster spot, filled here with the best "
                                          f"free agent, {P[i].name} ({P[i].mu:.1f} per game).",
                        f"the open roster spot goes to free agent {P[i].name}"))

    pw = partner.d_wins
    if pw > 0.1:
        ptxt = (f"Partner: {pname} gains {pw:+.2f} expected wins, so they would probably say yes — "
                f"a sign the deal leans their way." if you.d_wins < 0 else
                f"Partner: {pname} also gains ({pw:+.2f} wins), so this could be a genuine win-win.")
    elif pw < -t_threshold(state):
        ptxt = f"Partner: {pname} loses {abs(pw):.2f} expected wins, so they are unlikely to accept as offered."
    else:
        ptxt = f"Partner: {pname} comes out about even ({pw:+.2f} wins), so they might accept."

    # order: for a decline lead with what hurts you, for an accept with what helps
    pref = {"DECLINE": -1, "ACCEPT": 1}.get(r.verdict, 0)
    R.sort(key=lambda x: (-(x.sign == pref) if pref else 0, -x.weight))
    bullets = [x.text for x in R] + [ptxt]
    signs = [x.sign for x in R] + [0]

    # --- weekly win probability ------------------------------------------------------------
    rows = []
    for b, a in zip(you.weeks_before, you.weeks_after):
        if b["p"] is None:
            continue
        rows.append({"week": b["week"], "opponent": state.team_name(b["opp"]) if b["opp"] else "-",
                     "before": b["p"], "after": a["p"],
                     "delta": (round(a["p"] * 100) - round(b["p"] * 100)) / 100})
    weeks = pd.DataFrame(rows)
    this_week = rows[0] if rows else None
    if len(weeks):
        top = weeks.reindex(weeks["delta"].abs().sort_values(ascending=False).index).head(2)
        swing = "; ".join(f"week {int(w.week)} vs {w.opponent}: {_pct(w.before)} → {_pct(w.after)} ({_dpct(w.delta)})"
                          for w in top.itertuples() if abs(w.delta) >= 0.02)
        if swing:
            bullets.append(f"Biggest weekly swings — {swing}.")
            signs.append(0)

    # --- summary paragraph -----------------------------------------------------------------
    dw, dp = you.d_wins, you.d_pts
    lead = [x.phrase for x in R if (x.sign == pref if pref else True)][:2]
    because = (" because " + lead[0] + (f", and {lead[1]}" if len(lead) > 1 else "")) if lead else ""
    if this_week:
        this_week["label"] = week_label(state, this_week["week"])
    tw = (f" {this_week['label'].capitalize()} (week {this_week['week']} vs {this_week['opponent']}) your chance to win goes from "
          f"{_pct(this_week['before'])} to {_pct(this_week['after'])} ({_dpct(this_week['delta'])}).") if this_week else ""
    if r.verdict == "DECLINE":
        para = (f"Decline. Trading {_names(give, P)} for {_names(get, P)} costs you about {abs(dw):.2f} expected wins "
                f"over the remaining {n_weeks} weeks ({_signed(dp)} points per week in your starting lineup){because}.{tw}")
    elif r.verdict == "ACCEPT":
        para = (f"Accept. Trading {_names(give, P)} for {_names(get, P)} adds about {dw:.2f} expected wins over the "
                f"remaining {n_weeks} weeks ({_signed(dp)} points per week in your starting lineup){because}.{tw}")
    else:
        para = (f"Close call. Trading {_names(give, P)} for {_names(get, P)} moves your expected wins by only "
                f"{dw:+.2f} over {n_weeks} weeks ({_signed(dp)} points per week), so it comes down to preference — "
                f"for example, which players you trust to stay healthy.{tw}")
    para += (f" {pname} would gain {pw:.2f} wins." if pw >= 0 else f" {pname} would lose {abs(pw):.2f} wins.")
    headline = {"ACCEPT": "Accept", "DECLINE": "Decline", "FAIR-ISH": "Close call"}[r.verdict]
    ex = Explanation(r.verdict, f"{headline}: {dw:+.2f} expected wins for you", para, bullets, this_week, weeks)
    ex.signs = signs
    return ex


def week_label(state, week_no: int) -> str:
    real = getattr(state, "real_today", None) or state.today
    if real < state.cal.first_day:
        return "opening week"
    w = state.cal.week_of(state.today)
    return "this week" if w is not None and w.number == week_no else f"week {week_no}"


def t_threshold(state) -> float:
    return state.cfg["trade"]["acceptance_threshold"]


def format_explanation(e: Explanation) -> str:
    lines = [e.paragraph, ""]
    lines += [f"  • {b}" for b in e.bullets]
    if e.this_week:
        w = e.this_week
        lines += ["", f"{w.get('label', 'this week').capitalize()} (week {w['week']} vs {w['opponent']}): win chance "
                      f"{_pct(w['before'])} → {_pct(w['after'])} ({_dpct(w['delta'])})"]
    if len(e.weeks) > 1:
        lines.append("Week by week:  " + "  ".join(
            f"W{int(r.week)} {_pct(r.before)}→{_pct(r.after)} ({_dpct(r.delta)})" for r in e.weeks.itertuples()))
    return "\n".join(lines)
