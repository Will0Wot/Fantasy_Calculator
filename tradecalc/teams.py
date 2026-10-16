"""Team abbreviation mapping between ESPN (schedule, rosters) and stats.nba.com (box scores)."""

ESPN_TO_NBA = {
    "GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS",
    "PHO": "PHX", "BKN": "BKN", "CHA": "CHA",
}
NBA_TO_ESPN = {v: k for k, v in ESPN_TO_NBA.items()}


def to_nba(abbrev: str) -> str:
    a = (abbrev or "").upper()
    return ESPN_TO_NBA.get(a, a)


def to_espn(abbrev: str) -> str:
    a = (abbrev or "").upper()
    return NBA_TO_ESPN.get(a, a)
