"""Probability a player suits up for a given upcoming game (section 4.5).

Injuries use ESPN's expected return date R when there is one. Return dates slip far more
often than they come early, so instead of switching from 0 to the healthy rate on R, the
probability the player is back ramps linearly from 0 on R to 1 on R + w, where w grows
with the length of the absence (config: model.injury).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

OUT_STATUSES = {"OUT", "INJURY_RESERVE", "SUSPENSION", "FIFTEEN_DAY_DL", "SIXTY_DAY_DL",
                "SEVEN_DAY_DL", "TEN_DAY_DL", "OUT_FOR_SEASON"}
DTD_STATUSES = {"DAY_TO_DAY", "QUESTIONABLE", "DOUBTFUL", "GTD"}


@dataclass(frozen=True)
class Availability:
    base: float                     # healthy games-played rate
    out_games: int = 0              # upcoming team games missed for sure
    dtd_next: float | None = None   # P(plays) while day-to-day
    dtd_until: date | None = None   # day-to-day games before this date (else just the next game)
    back_from: date | None = None   # P(back) ramps from 0 on back_from ...
    back_by: date | None = None     # ... to rtp on back_by
    out_for_season: bool = False
    rtp: float = 1.0                # P(returns this season at all): ceiling on P(back)
    # (from, to, multiplier) date windows, e.g. post-injury ramp-up or reinjury risk
    modifiers: tuple = ()

    def p_back(self, d: date | None) -> float:
        if self.back_by is None or d is None:
            return 1.0
        if d < self.back_from:
            return 0.0
        if d >= self.back_by:
            return self.rtp
        return self.rtp * ((d - self.back_from).days + 1) / ((self.back_by - self.back_from).days + 1)

    def modifier(self, d: date | None) -> float:
        m = 1.0
        if d is not None:
            for frm, to, mult in self.modifiers:
                if frm <= d < to:
                    m *= mult
        return m

    def for_game(self, j: int, d: date | None = None) -> float:
        """Probability of playing the j-th upcoming team game (0 = next game), on date d."""
        if self.out_for_season or j < self.out_games:
            return 0.0
        p = self.base * self.p_back(d) * self.modifier(d)
        if self.dtd_next is not None:
            in_dtd = (d < self.dtd_until) if (self.dtd_until and d) else j == self.out_games
            if in_dtd:
                p = min(p, self.dtd_next)
        return min(p, 1.0)

    @property
    def is_injured(self) -> bool:
        return self.out_for_season or self.out_games > 0 or self.back_by is not None or self.dtd_next is not None


def games_played_rate(logs_by_season: list[pd.DataFrame], team_games: int = 82,
                      excluded: dict[int, int] | None = None) -> pd.Series:
    """Games played / team games over the given seasons, by player_id.

    Only seasons in which the player appeared count, so rookies are not penalized for
    seasons before they entered the league. `excluded` ({player_id: games}) removes a
    known injury absence from the denominator, giving a healthy baseline rate; the injury
    itself is then modeled by its profile (model/injury.py) instead of counted twice.
    """
    played = pd.concat(
        [l[l["min"] > 0].groupby("player_id").size().rename(i) for i, l in enumerate(logs_by_season)], axis=1
    )
    denom = played.notna().sum(axis=1) * team_games
    if excluded:
        denom = denom - pd.Series(excluded).reindex(denom.index).fillna(0)
    return (played.sum(axis=1) / denom.clip(lower=1)).clip(upper=1.0)


def return_window(today: date, return_date: date, frac: float, min_days: int, max_days: int) -> tuple[date, date]:
    """(R, R + w) with w = frac * days out, clamped to [min_days, max_days]."""
    from datetime import timedelta

    days_out = max((return_date - today).days, 0)
    w = int(round(min(max(frac * days_out, min_days), max_days)))
    return return_date, return_date + timedelta(days=w)


def availability(
    status: str,
    gp_rate: float | None,
    *,
    cap: float,
    dtd_prob: float,
    out_default_games: int,
    games_until_return: int | None = None,
    default_rate: float = 0.85,
    return_date: date | None = None,
    today: date | None = None,
    window: tuple[float, int, int] = (0.3, 4, 30),
    rtp: float = 1.0,
    typical_days: int | None = None,
    modifiers: tuple = (),
) -> Availability:
    """Availability for one player.

    Precedence for OUT players: explicit games_until_return (manual override) >
    return_date (ESPN injury report) > typical_days for the injury type > out_default_games.
    rtp and modifiers come from the injury profile (model/injury.py).
    """
    from datetime import timedelta

    base = min(cap, gp_rate if gp_rate is not None and gp_rate == gp_rate else default_rate)
    status = (status or "ACTIVE").upper()
    if status == "OUT_FOR_SEASON":
        return Availability(base, out_for_season=True)
    if status in OUT_STATUSES:
        if games_until_return is not None:
            return Availability(base, out_games=games_until_return, modifiers=modifiers)
        if return_date is None and typical_days and today is not None:
            return_date = today + timedelta(days=typical_days)
        if return_date is not None and today is not None:
            if return_date <= today:
                # Past the expected date but still listed OUT: assume a short slip.
                return_date = today
            frm, by = return_window(today, return_date, *window)
            return Availability(base, back_from=frm, back_by=by, rtp=rtp, modifiers=modifiers)
        return Availability(base, out_games=out_default_games, modifiers=modifiers)
    if status in DTD_STATUSES:
        until = return_date if return_date and today and return_date > today else None
        return Availability(base, dtd_next=dtd_prob, dtd_until=until, modifiers=modifiers)
    return Availability(base, modifiers=modifiers)


def in_season_calibration(rate: float, cal: dict, minutes: float | None = None) -> float:
    """Probability of playing, from this season's (shrunk) games-played rate and expected minutes.

    logit(a) = c0 + c1 * logit(rate) + c2 * (minutes / 30 - 1), fitted on 2022-23 rotation
    player-games (model audit). Raw rates are far too extreme (c1 ~ 0.36), and at the same rate a
    starter plays more often than a bench player (c2 > 0: rest days and DNPs hit bench roles).
    Unknown minutes use the typical rotation value (`default_minutes`)."""
    if "in_logit" not in cal:                      # older linear form
        return cal["in_intercept"] + cal["in_slope"] * rate
    c0, c1, c2 = cal["in_logit"]
    m = minutes if minutes is not None and minutes == minutes and minutes > 0 else cal.get("default_minutes", 25.0)
    r = min(max(float(rate), 1e-3), 0.97)
    return float(1.0 / (1.0 + np.exp(-(c0 + c1 * np.log(r / (1 - r)) + c2 * (m / 30.0 - 1.0)))))


def calibrated_rate(hist_rate: float | None, apps: int, team_games: int, cal: dict,
                    minutes: float | None = None) -> float:
    """Availability base rate, calibrated by the model audit (fitted on 2021-22 -> 2022-23).

    Raw games-played rates are far too extreme: for fantasy-relevant players, last season's rate
    (longest absence excluded) barely predicts next season (preseason line ~ flat), and in-season
    only part of a player's deviation from average carries forward. So:
      preseason  a0 = pre_intercept + pre_slope * hist_rate
      in-season  a1 = in_season_calibration((prior_games * hist_rate + apps) / (prior_games + team_games), minutes)
    blended by w = min(team_games / prior_games, 1). Known injuries are applied on top of this.
    """
    h = 0.85 if hist_rate is None or hist_rate != hist_rate else float(hist_rate)
    a0 = cal["pre_intercept"] + cal["pre_slope"] * h
    if team_games <= 0:
        return a0
    k = cal.get("prior_games", 10)
    rate = (k * h + apps) / (k + max(team_games, apps))
    a1 = in_season_calibration(rate, cal, minutes)
    w = min(team_games / k, 1.0)
    return (1 - w) * a0 + w * a1


def espn_games_blend(a: float, espn_gp: float | None, team_played: int, cap: float = 0.95,
                     weight: float = 0.5) -> float:
    """Second opinion on games played: blend the base rate with ESPN's projected games share.
    ESPN's weight starts at `weight` before the season and fades to 0 at mid-season (41 team games).
    (The live season model does the equivalent on expected remaining games; see values.py.)"""
    if not espn_gp:
        return a
    lam = weight * max(0.0, 1 - team_played / 41)
    return float(min((1 - lam) * a + lam * min(float(espn_gp), 82) / 82, cap))

