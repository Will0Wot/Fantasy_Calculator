"""Injury-aware adjustments (profiles in config/injury_profiles.yaml).

For each player this module works out:
  - the current injury (ESPN injury report, else fantasy status), classified to a profile;
  - past major injuries ("history"): from the persisted injury log, ESPN's season outlook,
    long absences in last season's game logs, and data/injury_history.csv;
and turns them into model inputs:
  - when he is back (return window, P(returns this season));
  - date-windowed availability multipliers (ramp-up right after return, then year-1 and
    year-2 reinjury/rest effects);
  - multipliers on the ADP-curve half of his prior FPPG and on his volatility.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

from tradecalc.model.names import norm_name

PROFILES_PATH = Path(__file__).resolve().parent.parent / "config" / "injury_profiles.yaml"
YEAR = timedelta(days=365)


@dataclass(frozen=True)
class Profile:
    key: str
    label: str
    match: tuple
    match_type: tuple
    typical_days: int
    rtp: float
    year1: dict
    year2: dict
    ramp_days: int
    ramp: float
    confidence: str


@dataclass
class InjuryConfig:
    profiles: dict[str, Profile]
    major: set[str]
    window: tuple[float, int, int]
    long_absence_games: int


@lru_cache(maxsize=None)
def load_profiles(path: str = str(PROFILES_PATH)) -> InjuryConfig:
    d = yaml.safe_load(Path(path).read_text())
    profs = {}
    for k, v in d["profiles"].items():
        profs[k] = Profile(
            k, v["label"],
            tuple(re.compile(x, re.I) for x in v.get("match") or []),
            tuple(re.compile(x, re.I) for x in v.get("match_type") or []),
            int(v["typical_days"]), float(v["rtp"]), dict(v["year1"]), dict(v["year2"]),
            int(v["ramp_days"]), float(v["ramp"]), v["confidence"],
        )
    w = d["return_window"]
    return InjuryConfig(profs, set(d["major"]), (w["frac"], w["min_days"], w["max_days"]), int(d["long_absence_games"]))


def classify(type_text: str, text: str, ic: InjuryConfig) -> str:
    """Profile key for an injury. type_text: ESPN 'side type detail'; text: news/outlook prose."""
    for k, p in ic.profiles.items():
        if any(r.search(text or "") or r.search(type_text or "") for r in p.match):
            return k
        if any(r.search(type_text or "") for r in p.match_type):
            return k
    return "other"


@dataclass
class HistoryEntry:
    profile: str
    returned: date          # first day back (or expected first game of this season)
    source: str             # injury log | outlook | game logs | manual


@dataclass
class InjuryView:
    """Everything the model needs about one player's injuries."""
    current: str | None = None            # profile key of current injury
    current_text: str = ""
    return_date: date | None = None
    history: list[HistoryEntry] = field(default_factory=list)
    fppg_mult: float = 1.0                # on the ADP-curve half of the prior
    sigma_mult: float = 1.0
    modifiers: tuple = ()                 # (from, to, mult) availability windows
    rtp: float = 1.0
    typical_days: int | None = None
    notes: list[str] = field(default_factory=list)


# --- history sources -------------------------------------------------------------------

def log_current_injuries(conn: sqlite3.Connection, injuries, ic: InjuryConfig, today: date) -> None:
    """Record today's injury report so history accumulates over the season. Each player's
    status is also snapshotted per day (status_snapshots), so the chance a day-to-day player
    actually plays can later be measured against box scores."""
    for inj in injuries:
        if inj.espn_id is None:
            continue
        conn.execute("INSERT OR REPLACE INTO status_snapshots VALUES (?,?,?)", (inj.espn_id, str(today), inj.status))
        prof = classify(inj.injury_type, inj.comment, ic)
        row = conn.execute(
            "SELECT first_seen FROM injury_log WHERE espn_id=? AND profile=? AND last_seen>=? "
            "ORDER BY last_seen DESC LIMIT 1",
            (inj.espn_id, prof, str(today - timedelta(days=7))),
        ).fetchone()
        rd = str(inj.return_date) if inj.return_date else None
        if row:
            conn.execute("UPDATE injury_log SET last_seen=?, status=?, return_date=?, text=? "
                         "WHERE espn_id=? AND profile=? AND first_seen=?",
                         (str(today), inj.status, rd, inj.comment, inj.espn_id, prof, row["first_seen"]))
        else:
            conn.execute("INSERT OR REPLACE INTO injury_log VALUES (?,?,?,?,?,?,?,?)",
                         (inj.espn_id, inj.name, prof, inj.status, str(today), str(today), rd, inj.comment))
    conn.commit()


def logged_history(conn: sqlite3.Connection, ic: InjuryConfig, real_today: date,
                   currently_injured: set[int]) -> dict[int, list[HistoryEntry]]:
    """Major injuries from the log that have ended: not seen on the report since before
    yesterday (real date), and the player is not on today's report."""
    out: dict[int, list[HistoryEntry]] = {}
    for r in conn.execute("SELECT espn_id, profile, last_seen FROM injury_log WHERE last_seen < ?",
                          (str(real_today - timedelta(days=1)),)):
        if r["profile"] in ic.major and r["espn_id"] not in currently_injured:
            out.setdefault(r["espn_id"], []).append(
                HistoryEntry(r["profile"], date.fromisoformat(r["last_seen"]) + timedelta(days=1), "injury log"))
    return out


def long_absences(logs: pd.DataFrame, min_games: int) -> dict[int, tuple[int, str]]:
    """Longest run of team games each player missed in one season, if >= min_games.

    Returns {player_id: (games missed, player name)}. Gaps are counted date by date against
    the schedule of the team he was on when he came back (so trades are not mistaken for
    absences), and include games missed at the start and end of the season.
    """
    if logs.empty:
        return {}
    team_dates = {t: sorted(set(s)) for t, s in logs.groupby("team")["game_date"]}
    first_day, last_day = logs["game_date"].min(), logs["game_date"].max()

    def missed(team, lo, hi):  # team games strictly between lo and hi
        return sum(1 for d in team_dates.get(team, ()) if lo < d < hi)

    out = {}
    for pid, g in logs[logs["min"] > 0].sort_values("game_date").groupby("player_id"):
        dates, teams = g["game_date"].tolist(), g["team"].tolist()
        gaps = [missed(teams[0], "0000", dates[0]), missed(teams[-1], dates[-1], "9999")]
        gaps += [missed(teams[i + 1], dates[i], dates[i + 1]) for i in range(len(dates) - 1)]
        # A start-of-season gap can also be a late signing / two-way call-up; those are
        # fringe players, so the error is harmless next to missing an injured starter.
        worst = max(gaps)
        if worst >= min_games:
            out[int(pid)] = (worst, g["player_name"].iloc[-1])
    return out


def manual_history(path) -> dict[str, list[HistoryEntry]]:
    """data/injury_history.csv: name, injury, returned (YYYY-MM-DD; blank = start of this season)."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return {}
    df.columns = [c.strip().lower() for c in df.columns]
    out = {}
    ic = load_profiles()
    for r in df.to_dict("records"):
        prof = r.get("injury") if r.get("injury") in ic.profiles else classify(str(r.get("injury")), str(r.get("injury")), ic)
        ret = r.get("returned")
        ret = date.fromisoformat(str(ret)) if isinstance(ret, str) and ret else None
        out.setdefault(norm_name(r["name"]), []).append(HistoryEntry(prof, ret, "manual"))
    return out


# --- effects ---------------------------------------------------------------------------

def _windows(entry: HistoryEntry, p: Profile) -> list[tuple]:
    r = entry.returned
    w = []
    if p.ramp_days and p.ramp < 1:
        w.append((r, r + timedelta(days=p.ramp_days), p.ramp))
    w.append((r, r + YEAR, p.year1["avail"]))
    w.append((r + YEAR, r + 2 * YEAR, p.year2["avail"]))
    return [x for x in w if x[2] != 1.0]


def _phase(entry: HistoryEntry, p: Profile, on: date) -> dict | None:
    if entry.returned <= on < entry.returned + YEAR:
        return p.year1
    if entry.returned + YEAR <= on < entry.returned + 2 * YEAR:
        return p.year2
    if on < entry.returned:
        return p.year1
    return None


def injury_view(
    player,                      # ingest.league.Player
    report,                      # ingest.injuries.Injury | None
    history: list[HistoryEntry],
    ic: InjuryConfig,
    today: date,
    season_start: date,
) -> InjuryView:
    v = InjuryView()
    if report is not None:
        v.current = classify(report.injury_type, report.comment, ic)
        v.current_text = report.injury_type or report.comment[:60]
        v.return_date = report.return_date
    elif (player.injury_status or "ACTIVE") != "ACTIVE":
        v.current = "other"

    # ESPN season outlook: a major injury mentioned there is treated as history returning
    # at the start of this season, unless the player is still out with it.
    entries = list(history)
    if player.outlook:
        k = classify("", player.outlook, ic)
        if k in ic.major and k != v.current and not any(e.profile == k for e in entries):
            entries.append(HistoryEntry(k, season_start, "outlook"))
    # A generic "long absence" is only kept when nothing more specific explains it: not when
    # the player is still out (it is the same injury), nor when a named injury is on record.
    still_out = report is not None and report.status in ("OUT", "OUT_FOR_SEASON")
    if still_out or any(e.profile != "long_absence" for e in entries):
        entries = [e for e in entries if e.profile != "long_absence"]
    for e in entries:
        if e.returned is None:
            e.returned = season_start
    v.history = entries

    mods: list[tuple] = []
    fppg, sigma = 1.0, 1.0
    if v.current is not None:
        p = ic.profiles[v.current]
        v.rtp = p.rtp
        v.typical_days = p.typical_days
        back = v.return_date or (today + timedelta(days=p.typical_days))
        mods += _windows(HistoryEntry(v.current, max(back, today), "current"), p)
        if v.current in ic.major:
            fppg *= p.year1["fppg"]
            sigma *= p.year1["sigma"]
        if v.current != "other":
            v.notes.append(f"{p.label}" + (f", back ~{v.return_date:%b %d}" if v.return_date else "")
                           + (f", P(returns this season) {p.rtp:.0%}" if p.rtp < 0.95 else ""))
    for e in entries:
        p = ic.profiles[e.profile]
        ph = _phase(e, p, today)
        if ph is None:
            continue
        mods += _windows(e, p)
        fppg *= ph["fppg"]
        sigma *= ph["sigma"]
        yr = 1 if ph is p.year1 else 2
        v.notes.append(f"year {yr} back from {p.label.lower()} ({e.source})")
    v.fppg_mult, v.sigma_mult, v.modifiers = fppg, sigma, tuple(mods)
    return v
