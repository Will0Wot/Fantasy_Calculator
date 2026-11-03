"""Discord bot: the data it answers from, and the weekly post.

Every morning the GitHub job runs `tradecalc discord --post`, which writes discord-bot/data/snapshot.json
(committed, so the bot can read it from GitHub):
  - the standings so far, next to the record each team was supposed to have (a win for every week the
    model favored it) and the upsets,
  - the league forecast (projected final standings),
  - each team's projected score for every remaining week (points already scored this week included),
  - the chance that any team beats any other team in any remaining week, by the same formula the app
    uses (trade/matchup.py p_win with the fitted matchup shrink),
  - the standings and matchup tables, already formatted for Discord.
The Cloudflare Worker in discord-bot/ answers /odds, /standings, /forecast and /matchups from it, so its
answers always match the app, and formatting changes here need no redeploy of the Worker.

At the start of each fantasy week the job also posts the week's preview (last week's results, the standings
with expected wins, the projected final standings, this week's matchups) to the channel webhook in
DISCORD_WEBHOOK_URL, once per week.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "discord-bot" / "data"
SNAPSHOT_PATH = DATA_DIR / "snapshot.json"
POSTED_PATH = DATA_DIR / "posted.json"
NAME_WIDTH = 14          # team names in the standings table (fits a phone screen: 40 characters a line)
COLOR = 0xF26522         # basketball orange
USER_AGENT = "DiscordBot (https://github.com/Will0Wot/Fantasy_Calculator, 1.0)"


# --- formatting ----------------------------------------------------------------------------

def pct(p: float | None) -> str:
    if p is None:
        return "–"
    if p < 0.005:
        return "<1%"
    if p > 0.995:
        return ">99%"
    return f"{p * 100:.0f}%"


def md(text) -> str:
    """A team name for Discord text: markdown characters escaped."""
    return re.sub(r"([\\*_~`|>])", r"\\\1", str(text))


def _plain(text, width: int) -> str:
    """A team name for a code block: no backticks, cut to `width` characters."""
    s = str(text).replace("`", "'")
    return s if len(s) <= width else s[: width - 1] + "…"


def week_label(w) -> str:
    if w.start.month == w.end.month:
        days = f"{w.start:%b} {w.start.day}–{w.end.day}"
    else:
        days = f"{w.start:%b} {w.start.day}–{w.end:%b} {w.end.day}"
    return f"Week {w.number} ({days})"


def _week_status(w, today) -> str:
    return "final" if w.end < today else "in_progress" if w.start <= today else "upcoming"


def current_week(state):
    """The week being played (or starting today); before the season or between weeks, the next one."""
    weeks, today = state.cal.weeks, state.today
    return (next((w for w in weeks if w.start <= today <= w.end), None)
            or next((w for w in weeks if w.start > today), None))


def _signed(x: float) -> str:
    return "0" if abs(x) < 1e-9 else f"{x:+.0f}" if float(x).is_integer() else f"{x:+.1f}"


def standings_text(snap: dict) -> str:
    """The real standings next to the record each team was supposed to have, and last week's upsets."""
    names = {t["id"]: t["name"] for t in snap["teams"]}
    rows = snap["standings"]
    if not rows or not any(r["record"] != "0-0" for r in rows):
        return "No games finished yet."
    w = NAME_WIDTH
    lines = [f"    {'Team':<{w}} {'Rec':>5} {'Exp':>5} {'+/-':>3} {'Pts':>5}"]
    for r in rows:
        lines.append(f"{r['rank']:>2}. {_plain(names[r['team_id']], w):<{w}} {r['record']:>5} {r['exp_record']:>5} "
                     f"{_signed(r['vs_expected']):>3} {r['points_for'] / 1000:>4.1f}k")
    out = "```\n" + "\n".join(lines) + "\n```\n"
    out += ("Exp = the record you were supposed to have: a win for every week the model favored you before it "
            "started · +/- = wins above or below it")
    finished = [int(k) for k in snap.get("results", {})]
    if finished:
        wk = max(finished)
        flops = [(r, s_) for r in rows for s_ in r["surprises"] if s_["week"] == wk and s_["result"] != "won"]
        if flops:
            out += f"\n\n**Week {wk}: supposed to win, didn't**"
            for r, s_ in flops:
                verb = "lost to" if s_["result"] == "lost" else "only tied"
                out += (f"\n😬 **{md(names[r['team_id']])}** ({pct(s_['p'])} to win) {verb} "
                        f"**{md(names[s_['opp']])}**")
        elif any(g.get("p_home") is not None for g in snap["results"][str(wk)]):
            out += f"\n\n✅ Every favorite won in week {wk}."
    return out


def forecast_text(snap: dict) -> str:
    """The projected final standings."""
    names = {t["id"]: t["name"] for t in snap["teams"]}
    w = NAME_WIDTH
    lines = [f"    {'Team':<{w}} {'Rec':>5} {'Proj':>5} {'1st':>4} {'Last':>4}"]
    for r in snap["forecast"]:
        lines.append(f"{r['rank']:>2}. {_plain(names[r['team_id']], w):<{w}} {r['record']:>5} {r['exp_wins']:>5.1f} "
                     f"{pct(r['p_first']):>4} {pct(r['p_last']):>4}")
    out = "```\n" + "\n".join(lines) + "\n```\n"
    out += "Proj = projected final wins · 1st = chance to win the league · Last = chance to finish last"
    if snap.get("last_place"):
        out += f"\n💩 {md(snap['last_place'])}"
    return out


def matchups_text(snap: dict, week: int) -> str:
    wk = str(week)
    names = {t["id"]: t["name"] for t in snap["teams"]}
    idx = {t: i for i, t in enumerate(snap["team_ids"])}
    odds, proj = snap["odds"].get(wk), snap["proj"].get(wk)
    if not odds or not snap["schedule"].get(wk):
        return "No matchups this week."
    lines = []
    for a, b in snap["schedule"][wk]:
        pa = odds[idx[a]][idx[b]]
        ma, mb = proj[str(a)], proj[str(b)]
        line = (f"**{md(names[a])}** {pct(pa)} – {pct(1 - pa)} **{md(names[b])}**\n"
                f"projected {ma['mean']:,.0f} – {mb['mean']:,.0f}")
        if ma["banked"] or mb["banked"]:
            line += f" · so far {ma['banked']:,.0f} – {mb['banked']:,.0f}"
        lines.append(line)
    return "\n\n".join(lines)


def results_text(snap: dict, week: int) -> str:
    """Final scores, each with the winner's pre-week win chance; upsets marked."""
    names = {t["id"]: t["name"] for t in snap["teams"]}
    lines = []
    for m in snap["results"].get(str(week), []):
        h, a = md(names[m["home"]]), md(names[m["away"]])
        hs, as_ = f"{m['home_total']:,.0f}", f"{m['away_total']:,.0f}"
        if m["winner"] == "TIE":
            lines.append(f"🤝 {h} {hs} – {as_} {a} (tie)")
            continue
        home_won = m["winner"] == "HOME"
        win, lose, ws, ls = (h, a, hs, as_) if home_won else (a, h, as_, hs)
        p = m.get("p_home") if home_won else m.get("p_away")
        if p is None:
            lines.append(f"✅ **{win}** {ws} – {ls} {lose}")
        elif p < 0.5:
            lines.append(f"😮 **{win}** {ws} – {ls} {lose} (upset: had {pct(p)} to win)")
        else:
            lines.append(f"✅ **{win}** {ws} – {ls} {lose} (had {pct(p)} to win)")
    return "\n".join(lines) or "No results."


# --- the snapshot --------------------------------------------------------------------------

def _pairs(state, week, teams) -> list[list[int]]:
    """This week's matchups as [home, away], in ESPN's order when it has them."""
    out = [[m["home"], m["away"]] for m in getattr(state.league, "matchups", [])
           if m["week"] == week.number and m.get("away") is not None]
    if out:
        return out
    seen = set()
    for t in teams:
        o = state.cal.opponents.get(t, {}).get(week.number)
        if o is not None and (o, t) not in seen:
            seen.add((t, o))
            out.append([t, o])
    return out


def snapshot(state, n_forecast: int = 10000, now: datetime | None = None, predictions=None) -> dict:
    """Everything the bot answers from (see module docstring). predictions: the season tracker's
    matchups table (default: tracking/<season>/matchups.csv) for the expected wins so far."""
    from tradecalc.clock import league_tz
    from tradecalc.trade.matchup import p_win, week_dist
    from tradecalc import tracking
    from tradecalc.trade.standings import forecast, last_place_line, luck, result_of

    cfg = state.cfg
    shrink = float(cfg["model"].get("matchup_shrink", 1.0))
    teams = sorted(state.rosters)
    today = state.today
    now = now or datetime.now(league_tz(cfg))
    proj, odds, schedule = {}, {}, {}
    for w in state.cal.remaining_weeks(today):
        dist = {}
        for t in teams:
            d = week_dist(state.rosters[t], w, state, today)
            bank = float(state.banked(t, w)[0]) if hasattr(state, "banked") else 0.0
            dist[t] = (d.M + bank, max(float(d.V), 0.0), bank)
        wk = str(w.number)
        proj[wk] = {str(t): {"mean": round(M, 1), "sd": round(math.sqrt(V), 1), "banked": round(b, 1)}
                    for t, (M, V, b) in dist.items()}
        odds[wk] = [[None if a == b else round(p_win(dist[a][0], dist[a][1], dist[b][0], dist[b][1], shrink), 4)
                     for b in teams] for a in teams]
        schedule[wk] = _pairs(state, w, teams)
    done = {w.number for w in state.cal.weeks if w.end < today}
    results: dict[str, list] = {}
    for m in getattr(state.league, "matchups", []):
        if m["week"] in done and m.get("away") is not None and result_of(m):
            results.setdefault(str(m["week"]), []).append(
                {"home": m["home"], "away": m["away"], "home_total": round(float(m.get("home_total", 0.0)), 1),
                 "away_total": round(float(m.get("away_total", 0.0)), 1), "winner": result_of(m)})
    fc = forecast(state, n=n_forecast, seed=0)
    projected = [{"rank": int(i), "team_id": int(r.team_id), "record": r.record, "exp_wins": round(float(r.exp_wins), 1),
                  "wins_p10": float(r.wins_p10), "wins_p90": float(r.wins_p90), "p_first": round(float(r.p_first), 4),
                  "p_top3": round(float(r.p_top3), 4), "p_last": round(float(r.p_last), 4)} for i, r in fc.iterrows()]
    predictions = tracking.load(cfg["league"].get("nba_season", ""))["matchups"] if predictions is None else predictions
    pre = predictions[predictions.kind == "pre-week"] if len(predictions) and "kind" in predictions else predictions
    pre_p = {(int(w), int(t)): float(x) for w, t, x in zip(pre.week, pre.team_id, pre.win_prob)} if len(pre) else {}
    for wk, games in results.items():
        for g in games:
            g["p_home"], g["p_away"] = pre_p.get((int(wk), g["home"])), pre_p.get((int(wk), g["away"]))
    lk = luck(state, predictions)
    standings = [{"rank": int(i), "team_id": int(r.team_id), "record": r.record, "wins": float(r.wins),
                  "exp_record": r.exp_record, "vs_expected": float(r.vs_expected),
                  "surprises": [{"week": x["week"], "result": x["result"], "p": round(x["p"], 4), "opp": int(x["opp"])}
                                for x in r.surprises],
                  "exp_wins": round(float(r.exp_wins), 2), "luck": round(float(r.luck), 2),
                  "allplay_wins": round(float(r.allplay_wins), 2), "allplay_luck": round(float(r.allplay_luck), 2),
                  "points_for": round(float(r.points_for), 1)} for i, r in lk.iterrows()]
    cur = current_week(state)
    settings = getattr(state.league, "settings", {}) or {}
    clock = f"{now:%I:%M %p}".lstrip("0")
    snap = {
        "version": 1,
        "league": settings.get("name") or "Fantasy league",
        "season": cfg["league"].get("nba_season"),
        "as_of": now.isoformat(timespec="minutes"),
        "as_of_label": f"{now:%a %b} {now.day}, {clock} {now:%Z}",
        "app_url": (cfg.get("discord") or {}).get("app_url", ""),
        "current_week": cur.number if cur else None,
        "weeks": [{"number": w.number, "start": w.start.isoformat(), "end": w.end.isoformat(), "label": week_label(w),
                   "status": _week_status(w, today)} for w in state.cal.weeks],
        "teams": [{"id": t, "name": state.team_name(t)} for t in teams],
        "team_ids": teams,
        "standings": standings,
        "forecast": projected,
        "last_place": last_place_line(fc) if len(fc) else "",
        "proj": proj, "odds": odds, "schedule": schedule, "results": results,
    }
    snap["text"] = {"standings": standings_text(snap), "forecast": forecast_text(snap),
                    "matchups": {wk: matchups_text(snap, int(wk)) for wk in schedule},
                    "results": {wk: results_text(snap, int(wk)) for wk in results}}
    return snap


def save(snap: dict, path: Path = SNAPSHOT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snap, ensure_ascii=False, separators=(",", ":")))
    return path


# --- the weekly post -----------------------------------------------------------------------

def _week(snap: dict, number: int | None) -> dict | None:
    return next((w for w in snap["weeks"] if w["number"] == number), None)


def weekly_post(snap: dict) -> dict:
    """Webhook payload: last week's results, projected standings, this week's matchups."""
    cur = _week(snap, snap["current_week"])
    last = max((int(k) for k in snap["results"]), default=None)
    embeds = []
    if last is not None:
        embeds.append({"title": f"{_week(snap, last)['label']}: results", "color": COLOR,
                       "description": snap["text"]["results"][str(last)]})
    if last is not None:
        embeds.append({"title": f"Standings after week {last}: actual vs expected record", "color": COLOR,
                       "description": snap["text"]["standings"]})
    embeds.append({"title": "Projected final standings" + (f" before week {cur['number']}" if cur else ""), "color": COLOR,
                   "description": snap["text"]["forecast"]})
    if cur and str(cur["number"]) in snap["text"]["matchups"]:
        embeds.append({"title": f"{cur['label']}: matchups", "color": COLOR,
                       "description": snap["text"]["matchups"][str(cur["number"])]})
    embeds[-1]["footer"] = {"text": f"Projections as of {snap['as_of_label']} · ask me: /odds, /standings, /forecast, /matchups"}
    head = f"**{md(snap['league'])}: {cur['label'] if cur else 'season'} preview**"
    if snap.get("app_url"):
        head += f"\nFull projections: <{snap['app_url']}>"
    return {"content": head, "embeds": embeds, "allowed_mentions": {"parse": []}}


def post_due(snap: dict, posted_path: Path = POSTED_PATH) -> bool:
    """Once per fantasy week: from the day before it starts (opening week) to its second day, so a
    failed run is caught up the next morning."""
    cur = _week(snap, snap["current_week"])
    if cur is None:
        return False
    try:
        if json.loads(posted_path.read_text()).get("week") == cur["number"]:
            return False
    except (FileNotFoundError, ValueError):
        pass
    today = datetime.fromisoformat(snap["as_of"]).date()
    return -1 <= (today - datetime.fromisoformat(cur["start"]).date()).days <= 1


def mark_posted(snap: dict, posted_path: Path = POSTED_PATH) -> None:
    posted_path.parent.mkdir(parents=True, exist_ok=True)
    posted_path.write_text(json.dumps({"week": snap["current_week"], "at": snap["as_of"]}))


def send(payload: dict, webhook_url: str) -> None:
    import requests

    r = requests.post(webhook_url, params={"wait": "true"}, json=payload, timeout=30,
                      headers={"User-Agent": USER_AGENT})
    if r.status_code >= 400:
        raise RuntimeError(f"Discord webhook returned {r.status_code}: {r.text[:300]}")
