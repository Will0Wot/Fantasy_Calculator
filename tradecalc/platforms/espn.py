"""ESPN fantasy basketball leagues: settings read from ESPN, so any ESPN league works, not just the home one.

The data itself (teams, rosters, matchups, the player pool) comes from ingest/league.py; this module
turns ESPN's league settings into platforms.base.LeagueSettings.
"""
from __future__ import annotations

from tradecalc.platforms.base import SLOT_ELIGIBILITY, LeagueSettings, merge_scoring

# ESPN stat ids -> box-score stats (missed shots = attempts - makes). Ids as in ESPN's API
# (and the espn-api package's basketball STATS_MAP).
ESPN_SCORING = {
    0: "PTS", 1: "BLK", 2: "STL", 3: "AST", 4: "OREB", 5: "DREB", 6: "REB", 9: "PF", 11: "TOV",
    13: "FGM", 14: "FGA", 15: "FTM", 16: "FTA", 17: "FG3M", 18: "FG3A",
    23: {"FGA": 1, "FGM": -1}, 24: {"FTA": 1, "FTM": -1}, 25: {"FG3A": 1, "FG3M": -1},
    37: "DD", 38: "TD", 39: "QD", 40: "MIN",
}
ESPN_UNSCORED = {7: "ejections", 8: "flagrant fouls", 10: "technical fouls", 12: "disqualifications",
                 41: "games started", 43: "team wins"}
ESPN_SLOT = {0: "PG", 1: "SG", 2: "SF", 3: "PF", 4: "C", 5: "G", 6: "F", 7: "SG/SF", 8: "G/F",
             9: "PF/C", 10: "F/C", 11: "UTIL", 12: "BE", 13: "IR"}


def settings_from_espn(raw: dict, league_id, season: int) -> LeagueSettings:
    """raw: ESPN's league response with the mSettings view (settings + status)."""
    s = raw["settings"]
    sc, ro, sched = s.get("scoringSettings", {}), s.get("rosterSettings", {}), s.get("scheduleSettings", {})
    acq = s.get("acquisitionSettings", {})
    stype = str(sc.get("scoringType") or "H2H_POINTS").upper()
    kind = "points" if "POINTS" in stype else ("roto" if "ROTO" in stype else "categories")
    notes = []
    parts, ignored = [], {}
    for item in sc.get("scoringItems", []):
        sid, pts = int(item["statId"]), float(item.get("points") or 0)
        if sid in ESPN_SCORING:
            parts.append((ESPN_SCORING[sid], pts))
        elif pts:
            ignored[ESPN_UNSCORED.get(sid, f"ESPN stat {sid}")] = pts
    if ignored:
        notes.append("Not modeled (rare, so worth little): " + ", ".join(f"{k} ({v:+g})" for k, v in ignored.items()))
    counts = {ESPN_SLOT.get(int(k), str(k)): int(v) for k, v in (ro.get("lineupSlotCounts") or {}).items() if v}
    bench, ir = counts.pop("BE", 0), counts.pop("IR", 0)
    weeks = int(sched.get("matchupPeriodCount") or 0)
    length = int(sched.get("matchupPeriodLength") or 1)
    if length > 1:
        notes.append(f"Matchups last {length} weeks; the app models them week by week.")
    tie = sc.get("matchupTieRule") or "NONE"
    if tie != "NONE":
        notes.append(f"ESPN tie rule {tie}: the app counts a tie as half a win.")
    lock = "weekly" if "WEEK" in str(ro.get("lineupLocktimeType", "")).upper() else "daily"
    if lock == "weekly":
        notes.append("Lineups lock weekly; the app assumes daily lineup moves, so bench depth is worth a bit less "
                     "than it shows.")
    limit = acq.get("matchupAcquisitionLimit")
    moves = None if limit is None or float(limit) < 0 else int(limit)
    if kind != "points":
        notes.append(f"Scoring type {stype}: only points leagues are supported so far.")
    return LeagueSettings(
        platform="espn", league_id=str(league_id), season=int(season), name=s.get("name") or f"ESPN league {league_id}",
        num_teams=int(s.get("size") or 0), scoring_type=kind, scoring=merge_scoring(parts), ignored_scoring=ignored,
        slots=counts, slot_eligibility={k: SLOT_ELIGIBILITY.get(k, SLOT_ELIGIBILITY["UTIL"]) for k in counts},
        bench=bench, ir=ir, regular_season_weeks=weeks, tie_value=0.5, moves_per_week=moves, lineup_lock=lock,
        playoffs=int(sched.get("playoffTeamCount") or 0) > 0, notes=notes)
