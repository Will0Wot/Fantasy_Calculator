"""What every fantasy platform's league is turned into, so the model never needs to know where it came from.

An adapter (platforms/espn.py, sleeper.py, yahoo.py) reads one league and returns ingest.league.LeagueData:
  meta      LeagueSettings below: scoring, lineup slots, weeks, ties, pickup limit, ...
  teams     fantasy teams and their rosters
  players   ESPN's NBA player pool (the one player ID every feature uses), with on_team_id, lineup slot
            and positions as this league has them
  matchups  every week's pairings and results
plus the NBA schedule. Players on other platforms are matched to ESPN's pool by name and team
(platforms/matching.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

POSITIONS = ("PG", "SG", "SF", "PF", "C")

# Which player positions each lineup slot takes. Platforms name slots the same way for basketball.
SLOT_ELIGIBILITY = {
    "PG": ["PG"], "SG": ["SG"], "SF": ["SF"], "PF": ["PF"], "C": ["C"],
    "G": ["PG", "SG"], "F": ["SF", "PF"], "SG/SF": ["SG", "SF"], "G/F": ["PG", "SG", "SF", "PF"],
    "PF/C": ["PF", "C"], "F/C": ["SF", "PF", "C"], "UTIL": list(POSITIONS),
}

# Box-score stats the model scores directly (stats.nba.com column names, upper case), and the
# per-game bonuses it scores from each game's box score (double-, triple-, quadruple-doubles).
LINEAR_STATS = ("PTS", "REB", "AST", "STL", "BLK", "TOV", "FGM", "FGA", "FTM", "FTA", "FG3M", "FG3A",
                "OREB", "DREB", "PF", "MIN")
BONUS_STATS = ("DD", "TD", "QD")


class UnsupportedLeague(Exception):
    """The league can be read, but the app can't model it (e.g. a category league). The message is
    shown to the user as is."""


class LeagueNotFound(Exception):
    """No such league, or it's private and the credentials given (if any) don't open it."""


@dataclass
class LeagueSettings:
    platform: str                         # espn | sleeper | yahoo
    league_id: str
    season: int                           # the platform's season id (ESPN 2027 = the 2026-27 season)
    name: str
    num_teams: int
    scoring_type: str = "points"          # points | categories | roto
    scoring: dict = field(default_factory=dict)          # stat (LINEAR_STATS / BONUS_STATS) -> points
    ignored_scoring: dict = field(default_factory=dict)  # the platform's rules the model can't score -> points
    slots: dict = field(default_factory=dict)            # starting slot -> count
    slot_eligibility: dict = field(default_factory=lambda: dict(SLOT_ELIGIBILITY))
    bench: int = 3
    ir: int = 0
    regular_season_weeks: int = 0
    weeks: list = field(default_factory=list)            # [(number, first day, last day)] if the platform says
    tie_value: float = 0.5
    moves_per_week: int | None = None     # pickups per matchup; None = no limit
    lineup_lock: str = "daily"            # daily | weekly
    playoffs: bool = False
    notes: list = field(default_factory=list)            # caveats for the user, e.g. ignored scoring rules

    @property
    def key(self) -> str:
        """Stable identifier: espn-1331882803-2027."""
        return f"{self.platform}-{self.league_id}-{self.season}"


def merge_scoring(parts: list[tuple[dict | str, float]]) -> dict[str, float]:
    """Sum (stat or {stat: sign}, points) pairs into {stat: points}: missed shots become attempts minus
    makes, e.g. FG missed -1 -> FGA -1, FGM +1."""
    out: dict[str, float] = {}
    for stat, pts in parts:
        for s, sign in ({stat: 1.0} if isinstance(stat, str) else stat).items():
            out[s] = out.get(s, 0.0) + sign * pts
    return {s: round(v, 6) for s, v in out.items() if abs(v) > 1e-9}


def weeks_from_dates(spans: list[tuple[date, date]]) -> list[tuple[int, date, date]]:
    return [(i + 1, s, e) for i, (s, e) in enumerate(sorted(spans))]
