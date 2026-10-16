"""Small synthetic league for lineup / matchup / trade tests."""
import copy
from collections import defaultdict
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from tradecalc.config import load_config
from tradecalc.model.availability import Availability
from tradecalc.model.values import LeagueState, PlayerState
from tradecalc.schedule.calendar import Week


class FakeCalendar:
    def __init__(self, games_by_day, weeks, opponents):
        self.games_by_day = games_by_day
        self.weeks = weeks
        self.opponents = opponents
        self.first_day = weeks[0].start

    def team_plays(self, team, d):
        return team in self.games_by_day.get(d, ())

    def week_of(self, d):
        return next((w for w in self.weeks if w.start <= d <= w.end), None)

    def remaining_weeks(self, today):
        return [w for w in self.weeks if w.end >= today]

    def team_games_in_week(self, team, week, from_day=None):
        return sum(self.team_plays(team, d) for d in week.days() if from_day is None or d >= from_day)


def make_player(pid, team_id, pro, pos, mu, sigma=None, a=1.0, out_games=0):
    return PlayerState(pid, f"P{pid}", pro, tuple(pos), team_id, "ACTIVE", mu,
                       sigma if sigma is not None else 0.4 * mu, Availability(a, out_games=out_games))


def make_state(players, games_by_day, n_weeks=1, opponents=None, cfg_overrides=None):
    cfg = copy.deepcopy(dict(load_config()))
    for path, v in (cfg_overrides or {}).items():
        d = cfg
        *head, last = path.split(".")
        for h in head:
            d = d[h]
        d[last] = v
    start = date(2026, 10, 26)  # Monday
    weeks = [Week(i + 1, start + timedelta(7 * i), start + timedelta(7 * i + 6)) for i in range(n_weeks)]
    cal = FakeCalendar(games_by_day, weeks, opponents or defaultdict(dict))
    team_game_index = defaultdict(dict)
    for d in sorted(games_by_day):
        for t in games_by_day[d]:
            team_game_index[t][d] = len(team_game_index[t])
    rosters = defaultdict(list)
    for p in players:
        if p.on_team_id:
            rosters[p.on_team_id].append(p.espn_id)
    league = SimpleNamespace(teams={t: SimpleNamespace(name=f"Team{t}", roster=r) for t, r in rosters.items()})
    st = LeagueState(cfg, league, cal, start, {p.espn_id: p for p in players}, dict(rosters), {}, team_game_index)
    for p in players:
        p.games_remaining = len(team_game_index[p.pro_team])
        p.ros_value = p.mu * p.avail.base * p.games_remaining
    return st


@pytest.fixture
def week_dates():
    start = date(2026, 10, 26)
    return [start + timedelta(i) for i in range(7)]
