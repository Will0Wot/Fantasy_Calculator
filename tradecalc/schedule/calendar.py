"""Fantasy weeks and NBA games per day/week."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class Week:
    number: int
    start: date
    end: date  # inclusive

    def days(self) -> list[date]:
        return [self.start + timedelta(d) for d in range((self.end - self.start).days + 1)]


def build_weeks(first_day: date, last_day: date, n_weeks: int, game_dates: set[date], overrides=None) -> list[Week]:
    """Week 1 = opening day through the first Sunday, then Monday-Sunday weeks.

    If that gives more weeks than the league has, merge the week where the All-Star break
    starts into the following week. The All-Star break is the longest run of days without
    games after January 1; December is skipped because NBA Cup knockout games are left
    off the schedule until the group stage ends, which makes early December look idle.
    `overrides` ({week_number: [start, end]}) replace computed boundaries.
    """
    spans: list[list[date]] = []
    start = first_day
    while start <= last_day:
        end = min(start + timedelta(days=(6 - start.weekday())), last_day)
        spans.append([start, end])
        start = end + timedelta(days=1)

    while len(spans) > n_weeks:
        brk = _longest_idle_run(game_dates, date(first_day.year + 1, 1, 1), last_day)
        i = next((j for j, s in enumerate(spans[:-1]) if brk and s[0] <= brk <= s[1]), len(spans) - 2)
        spans[i] = [spans[i][0], spans[i + 1][1]]
        del spans[i + 1]

    weeks = [Week(i + 1, s, e) for i, (s, e) in enumerate(spans)]
    for num, (s, e) in (overrides or {}).items():
        weeks[int(num) - 1] = Week(int(num), date.fromisoformat(str(s)), date.fromisoformat(str(e)))
    return weeks


def _longest_idle_run(game_dates: set[date], start: date, end: date) -> date | None:
    """First day of the longest stretch with no games in [start, end]."""
    best, best_len, run_start, run_len = None, 0, None, 0
    d = start
    while d <= end:
        if d in game_dates:
            run_len = 0
        else:
            if run_len == 0:
                run_start = d
            run_len += 1
            if run_len > best_len:
                best, best_len = run_start, run_len
        d += timedelta(days=1)
    return best


class Calendar:
    def __init__(self, league, cfg):
        self.first_day = league.first_day
        final_sp = league.settings.get("scheduleSettings", {}).get("finalScoringPeriod") or max(
            g["scoring_period"] for g in league.pro_games
        )
        last_day = max(league.first_day + timedelta(final_sp - 1), max(g["date"] for g in league.pro_games))
        self.games_by_day: dict[date, set[str]] = defaultdict(set)
        for g in league.pro_games:
            self.games_by_day[g["date"]].update((g["home"], g["away"]))
        self.weeks = build_weeks(
            self.first_day,
            last_day,
            cfg["league"]["regular_season_weeks"],
            set(self.games_by_day),
            cfg["weeks"].get("overrides"),
        )
        self.opponents: dict[int, dict[int, int]] = defaultdict(dict)  # team -> week -> opponent
        for m in league.matchups:
            if m["away"] is not None:
                self.opponents[m["home"]][m["week"]] = m["away"]
                self.opponents[m["away"]][m["week"]] = m["home"]

    def week_of(self, d: date) -> Week | None:
        return next((w for w in self.weeks if w.start <= d <= w.end), None)

    def remaining_weeks(self, today: date) -> list[Week]:
        """Weeks not yet finished. The current week counts only its remaining days (see team_games)."""
        return [w for w in self.weeks if w.end >= today]

    def team_plays(self, team: str, d: date) -> bool:
        return team in self.games_by_day.get(d, ())

    def team_games_in_week(self, team: str, week: Week, from_day: date | None = None) -> int:
        days = [d for d in week.days() if from_day is None or d >= from_day]
        return sum(self.team_plays(team, d) for d in days)
