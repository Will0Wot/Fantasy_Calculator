"""Fantasy platforms: load any ESPN, Sleeper or Yahoo league into the same LeagueData (see base.py).

    ref = parse_league_input("https://fantasy.espn.com/basketball/league?leagueId=1331882803")
    league = load(ref, conn, cfg, creds)        # LeagueData with .meta (LeagueSettings)
    cfg_league = league_config(cfg, league.meta)
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlparse

from tradecalc.platforms.base import LeagueNotFound, LeagueSettings, UnsupportedLeague

PLATFORMS = {"espn": "ESPN", "sleeper": "Sleeper", "yahoo": "Yahoo"}
DEMO = None   # set below: the demo league (platforms/demo.py)


@dataclass(frozen=True)
class LeagueRef:
    platform: str                 # espn | sleeper | yahoo
    league_id: str
    season: int | None = None     # the platform's season id; None = the current season

    @property
    def key(self) -> str:
        return f"{self.platform}-{self.league_id}" + (f"-{self.season}" if self.season else "")

    def query(self) -> dict:
        """URL parameters for a bookmarkable link."""
        q = {"platform": self.platform, "league": self.league_id}
        if self.season:
            q["season"] = str(self.season)
        return q


@dataclass
class Credentials:
    """A visitor's own login for a private league, held only in their session (never stored)."""
    espn_s2: str = ""
    swid: str = ""
    yahoo_token: dict = field(default_factory=dict)

    def espn_cookies(self) -> dict | None:
        if self.espn_s2 and self.swid:
            swid = self.swid.strip()
            if not swid.startswith("{"):
                swid = "{" + swid.strip("{}") + "}"
            return {"espn_s2": self.espn_s2.strip(), "SWID": swid}
        return None


def parse_league_input(text: str, platform: str | None = None) -> LeagueRef:
    """A league ID or a league page link from ESPN, Sleeper or Yahoo -> LeagueRef.

    ESPN    https://fantasy.espn.com/basketball/league?leagueId=1331882803 (also team/scoreboard pages)
    Sleeper https://sleeper.com/leagues/1180313591236747264/... (or sleeper.app)
    Yahoo   https://basketball.fantasysports.yahoo.com/nba/12345[/...]
    A bare ID needs `platform`."""
    t = (text or "").strip()
    if not t:
        raise ValueError("Enter a league ID or paste your league's link.")
    if "://" not in t and ("." in t.split("/")[0] and "/" in t):
        t = "https://" + t
    if "://" in t:
        u = urlparse(t)
        host, path, q = u.netloc.lower(), u.path, parse_qs(u.query)
        if "espn" in host:
            lid = (q.get("leagueId") or [None])[0]
            season = (q.get("seasonId") or [None])[0]
            if not lid or not lid.isdigit():
                raise ValueError("That ESPN link has no leagueId. Open your league's page on ESPN and copy its link.")
            if "basketball" not in path and "fba" not in path and "basketball" not in host:
                raise ValueError("That's not an ESPN fantasy basketball link.")
            return LeagueRef("espn", lid, int(season) if season and season.isdigit() else None)
        if "sleeper" in host:
            m = re.search(r"/leagues?/(\d{6,})", path)
            if not m:
                raise ValueError("That Sleeper link has no league ID (it looks like sleeper.com/leagues/<number>).")
            return LeagueRef("sleeper", m.group(1))
        if "yahoo" in host:
            m = re.search(r"/nba/(\d+)", path)
            if not m:
                raise ValueError("That Yahoo link has no league ID (it looks like basketball.fantasysports.yahoo.com/nba/<number>).")
            return LeagueRef("yahoo", m.group(1))
        raise ValueError("Paste a league link from ESPN, Sleeper or Yahoo, or pick the site and enter the league ID.")
    if not platform:
        raise ValueError("Pick which site the league is on.")
    if platform not in PLATFORMS:
        raise ValueError(f"Unknown platform {platform!r}.")
    lid = re.sub(r"\s+", "", t)
    if platform == "yahoo":
        m = re.fullmatch(r"(?:\d+\.l\.)?(\d+)", lid)        # 466.l.12345 or 12345
        if not m:
            raise ValueError("A Yahoo league ID is a number, like 12345.")
        lid = m.group(1)
    elif not lid.isdigit():
        raise ValueError(f"A {PLATFORMS[platform]} league ID is a number.")
    return LeagueRef(platform, lid)


def load(ref: LeagueRef, conn, cfg, creds: Credentials | None = None, force: bool = False, nba=None):
    """Read the league from its platform -> ingest.league.LeagueData (with .meta). nba: the NBA layer
    (ingest.league.nba_layer) for platforms other than ESPN; built here if not given."""
    creds = creds or Credentials()
    if ref.platform == "espn":
        from tradecalc.ingest.league import EspnClient, load_league

        home = _is_home(ref, cfg)
        season = ref.season or int(cfg["league"]["season"])
        client = EspnClient(conn, int(ref.league_id), season, cfg["data"]["espn_cache_ttl_hours"],
                            cookies=creds.espn_cookies(), use_env=home)
        return load_league(client, force=force)
    if nba is None:
        nba = nba_context(conn, cfg, force)
    if ref.platform == "demo":
        from tradecalc.platforms.demo import load_demo

        return load_demo(ref, conn, cfg, nba)
    if ref.platform == "sleeper":
        from tradecalc.platforms.sleeper import load_sleeper

        return load_sleeper(ref, conn, cfg, nba, force=force)
    if ref.platform == "yahoo":
        from tradecalc.platforms.yahoo import load_yahoo

        return load_yahoo(ref, conn, cfg, nba, creds, force=force)
    raise UnsupportedLeague(f"{ref.platform} leagues aren't supported.")


def nba_context(conn, cfg, force: bool = False):
    from tradecalc.ingest.league import EspnClient, nba_layer

    return nba_layer(EspnClient(conn, 0, int(cfg["league"]["season"]), cfg["data"]["espn_cache_ttl_hours"],
                                use_env=False), force=force)


def home_ref(cfg) -> LeagueRef:
    lg = cfg["league"]
    return LeagueRef(lg.get("platform", "espn"), str(lg["league_id"]))


def _is_home(ref: LeagueRef, cfg) -> bool:
    h = home_ref(cfg)
    return (ref.platform, ref.league_id) == (h.platform, h.league_id)


LEAGUE_SECTIONS = ("league", "scoring", "roster", "lineup", "weeks", "streaming")
UNLIMITED_MOVES = 4      # what the streaming planner plans when the league has no weekly pickup limit


def league_config(base, meta: LeagueSettings):
    """The model settings from league.yaml with this league's own settings (from its platform) in the
    league-specific sections. `league.season` stays ESPN's season id: it selects the NBA data."""
    from tradecalc.config import Config

    cfg = Config(copy.deepcopy(dict(base)))
    lg = cfg["league"]
    home = _is_home(LeagueRef(meta.platform, meta.league_id), base)
    lg.update({"platform": meta.platform, "league_id": meta.league_id, "platform_season": meta.season,
               "name": meta.name, "num_teams": meta.num_teams, "tie_value": meta.tie_value,
               "playoffs": meta.playoffs})
    if meta.regular_season_weeks:
        lg["regular_season_weeks"] = meta.regular_season_weeks
    if not home:
        lg["my_team_id"] = None
    cfg["scoring"] = dict(meta.scoring)
    cfg["roster"] = {"slots": dict(meta.slots), "bench": meta.bench, "ir": meta.ir,
                     "slot_eligibility": {k: list(v) for k, v in meta.slot_eligibility.items()}}
    cfg["lineup"] = {**dict(cfg.get("lineup", {})), "lock": meta.lineup_lock}
    if not home:
        cfg["weeks"] = {"rule": "platform" if meta.weeks else "mon_sun_merge_allstar", "overrides": {}}
    cfg["streaming"] = {**dict(cfg.get("streaming", {})),
                        "moves_per_week": meta.moves_per_week if meta.moves_per_week is not None else UNLIMITED_MOVES,
                        "moves_unlimited": meta.moves_per_week is None}
    return cfg


DEMO = LeagueRef("demo", "demo")

__all__ = ["DEMO", "LeagueRef", "Credentials", "LeagueSettings", "LeagueNotFound", "UnsupportedLeague", "PLATFORMS",
           "parse_league_input", "load", "league_config", "home_ref", "nba_context"]
