"""Yahoo Fantasy basketball leagues. Every visitor signs in with Yahoo (OAuth 2.0); the app then reads the
Fantasy API (XML, which is far more regular than Yahoo's JSON):

  league/<key>/settings           scoring type, stat categories and point values, lineup slots, playoffs
  league/<key>/teams/roster       every team and its players (selected position IL = injured reserve)
  league/<key>/scoreboard;week=N  pairings, points, winner, and the week's dates

Setup (once, by whoever runs the app): create an app at https://developer.yahoo.com/apps/create/
(Web Application, Fantasy Sports: Read, redirect URI = the app's address) and set YAHOO_CLIENT_ID,
YAHOO_CLIENT_SECRET and YAHOO_REDIRECT_URI as secrets. A visitor's token stays in their session only.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import xml.etree.ElementTree as ET
from datetime import date

import requests

from tradecalc.platforms.base import (SLOT_ELIGIBILITY, LeagueNotFound, LeagueSettings, UnsupportedLeague,
                                      merge_scoring)
from tradecalc.platforms.matching import PlatformPlayer, match_players, stand_in_id

AUTH_URL = "https://api.login.yahoo.com/oauth2/request_auth"
TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"
API = "https://fantasysports.yahooapis.com/fantasy/v2"
NS = "{http://fantasysports.yahooapis.com/fantasy/v2/base.rng}"

# Yahoo stat display names -> box-score stats
YAHOO_STAT = {"PTS": "PTS", "REB": "REB", "OREB": "OREB", "DREB": "DREB", "AST": "AST", "ST": "STL", "STL": "STL",
              "BLK": "BLK", "TO": "TOV", "FGM": "FGM", "FGA": "FGA", "FTM": "FTM", "FTA": "FTA", "3PTM": "FG3M",
              "3PTA": "FG3A", "PF": "PF", "DD": "DD", "TD": "TD", "MIN": "MIN"}
YAHOO_SLOT = {"PG": "PG", "SG": "SG", "SF": "SF", "PF": "PF", "C": "C", "G": "G", "F": "F", "UTIL": "UTIL"}
YAHOO_STATUS = {"O": "OUT", "INJ": "OUT", "IL": "OUT", "SUSP": "SUSPENSION", "NA": "OUT",
                "GTD": "DAY_TO_DAY", "DTD": "DAY_TO_DAY"}


# --- sign-in -----------------------------------------------------------------------------------

def oauth_config(secrets=None) -> dict | None:
    """{client_id, client_secret, redirect_uri} from Streamlit secrets or the environment; None if unset."""
    def get(k):
        try:
            if secrets is not None and k in secrets:
                return str(secrets[k])
        except Exception:  # noqa: BLE001 - no secrets file
            pass
        return os.environ.get(k)
    c = {"client_id": get("YAHOO_CLIENT_ID"), "client_secret": get("YAHOO_CLIENT_SECRET"),
         "redirect_uri": get("YAHOO_REDIRECT_URI")}
    return c if all(c.values()) else None


def sign_state(oauth: dict, payload: dict) -> str:
    """The OAuth `state`: what to open after sign-in, signed and timestamped. Streamlit starts a new session
    when Yahoo sends the visitor back, so this can't live in session state; the signature stops a forged
    sign-in link."""
    body = base64.urlsafe_b64encode(json.dumps({**payload, "t": int(time.time())}).encode()).decode().rstrip("=")
    sig = hmac.new(oauth["client_secret"].encode(), body.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{body}.{sig}"


def verify_state(oauth: dict, state: str, max_age_s: int = 900) -> dict | None:
    try:
        body, sig = state.rsplit(".", 1)
        good = hmac.new(oauth["client_secret"].encode(), body.encode(), hashlib.sha256).hexdigest()[:32]
        if not hmac.compare_digest(sig, good):
            return None
        data = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        return data if time.time() - data.get("t", 0) <= max_age_s else None
    except Exception:  # noqa: BLE001 - anything malformed is simply not valid
        return None


def auth_url(oauth: dict, state: str) -> str:
    from urllib.parse import urlencode

    return AUTH_URL + "?" + urlencode({"client_id": oauth["client_id"], "redirect_uri": oauth["redirect_uri"],
                                       "response_type": "code", "language": "en-us", "state": state})


def _token_request(oauth: dict, data: dict, session=None) -> dict:
    auth = base64.b64encode(f"{oauth['client_id']}:{oauth['client_secret']}".encode()).decode()
    r = (session or requests).post(TOKEN_URL, data=data, timeout=30,
                                   headers={"Authorization": f"Basic {auth}",
                                            "Content-Type": "application/x-www-form-urlencoded"})
    if r.status_code != 200:
        raise LeagueNotFound("Yahoo sign-in failed. Try signing in again.")
    t = r.json()
    t["expires_at"] = time.time() + float(t.get("expires_in", 3600)) - 60
    return t


def exchange_code(oauth: dict, code: str, session=None) -> dict:
    return _token_request(oauth, {"grant_type": "authorization_code", "code": code,
                                  "redirect_uri": oauth["redirect_uri"]}, session)


def refresh_token(oauth: dict, token: dict, session=None) -> dict:
    return _token_request(oauth, {"grant_type": "refresh_token", "refresh_token": token["refresh_token"],
                                  "redirect_uri": oauth["redirect_uri"]}, session)


# --- API -----------------------------------------------------------------------------------------

def _t(el, path: str, default=None):
    """Text at a slash path of child tags (Yahoo's namespace added)."""
    if el is None:
        return default
    node = el.find("/".join(NS + p for p in path.split("/")))
    return node.text if node is not None and node.text is not None else default


def _all(el, path: str) -> list:
    return [] if el is None else el.findall("/".join(NS + p for p in path.split("/")))


class YahooClient:
    """Fantasy API calls with the visitor's token, refreshed when it expires. `token` is updated in place,
    so the caller's session keeps the fresh one."""

    def __init__(self, token: dict, oauth: dict | None, session=None):
        self.token, self.oauth = token, oauth
        self.session = session or requests.Session()

    def get(self, path: str) -> ET.Element:
        if self.oauth and self.token.get("expires_at", 0) < time.time() and self.token.get("refresh_token"):
            self.token.update(refresh_token(self.oauth, self.token, self.session))
        r = self.session.get(f"{API}/{path}", timeout=60,
                             headers={"Authorization": f"Bearer {self.token.get('access_token', '')}"})
        if r.status_code == 401 and self.oauth and self.token.get("refresh_token"):
            self.token.update(refresh_token(self.oauth, self.token, self.session))
            r = self.session.get(f"{API}/{path}", timeout=60,
                                 headers={"Authorization": f"Bearer {self.token['access_token']}"})
        if r.status_code in (400, 404):
            raise LeagueNotFound("Yahoo has no such league, or it isn't one of yours.")
        if r.status_code in (401, 403):
            raise LeagueNotFound("Yahoo didn't let this account see that league. Sign in with an account in the league.")
        r.raise_for_status()
        return ET.fromstring(r.content)


def user_leagues(client: YahooClient) -> list[dict]:
    """The signed-in user's NBA leagues this season: [{league_id, league_key, name, season, scoring_type}]."""
    root = client.get("users;use_login=1/games;game_codes=nba/leagues")
    out = []
    for lg in root.iter(NS + "league"):
        out.append({"league_id": _t(lg, "league_id"), "league_key": _t(lg, "league_key"), "name": _t(lg, "name"),
                    "season": _t(lg, "season"), "scoring_type": _t(lg, "scoring_type")})
    return out


def settings_from_yahoo(league: ET.Element) -> LeagueSettings:
    s = league.find(NS + "settings")
    stype = (_t(s, "scoring_type") or _t(league, "scoring_type") or "").lower()
    kind = "points" if stype in ("headpoint", "point") else ("roto" if stype == "roto" else "categories")
    names = {_t(st, "stat_id"): (_t(st, "display_name") or _t(st, "abbr") or _t(st, "name") or "").upper()
             for st in _all(s, "stat_categories/stats/stat")}
    parts, ignored, notes = [], {}, []
    for st in _all(s, "stat_modifiers/stats/stat"):
        name, val = names.get(_t(st, "stat_id"), f"stat {_t(st, 'stat_id')}"), float(_t(st, "value", 0) or 0)
        if not val:
            continue
        if name in YAHOO_STAT:
            parts.append((YAHOO_STAT[name], val))
        else:
            ignored[name] = val
    if ignored:
        notes.append("Not modeled: " + ", ".join(f"{k} ({v:+g})" for k, v in ignored.items()))
    if stype == "point":
        notes.append("A season-points league: the app's weekly matchup odds don't apply, everything else does.")
    if kind != "points":
        notes.append(f"Scoring type {stype}: only points leagues are supported so far.")
    slots, bench, ir = {}, 0, 0
    for rp in _all(s, "roster_positions/roster_position"):
        pos, n = (_t(rp, "position") or "").upper(), int(_t(rp, "count", 0) or 0)
        if pos == "BN":
            bench += n
        elif pos.startswith("IL"):
            ir += n
        elif pos in YAHOO_SLOT:
            slots[YAHOO_SLOT[pos]] = slots.get(YAHOO_SLOT[pos], 0) + n
        elif n:
            notes.append(f"Lineup slot {pos} isn't modeled; treated as a utility slot.")
            slots["UTIL"] = slots.get("UTIL", 0) + n
    start, end = int(_t(league, "start_week", 1) or 1), int(_t(league, "end_week", 0) or 0)
    playoff = int(_t(s, "playoff_start_week", 0) or 0) if _t(s, "uses_playoff") == "1" else 0
    last = (playoff - 1) if playoff > 1 else end
    adds = _t(s, "max_weekly_adds")
    return LeagueSettings(
        platform="yahoo", league_id=_t(league, "league_id", ""), season=int(_t(league, "season", 0) or 0),
        name=_t(league, "name", "Yahoo league"), num_teams=int(_t(league, "num_teams", 0) or 0), scoring_type=kind,
        scoring=merge_scoring(parts), ignored_scoring=ignored, slots=slots,
        slot_eligibility={k: SLOT_ELIGIBILITY.get(k, SLOT_ELIGIBILITY["UTIL"]) for k in slots},
        bench=bench, ir=ir, regular_season_weeks=max(last - start + 1, 0), moves_per_week=int(adds) if adds else None,
        playoffs=playoff > 1, notes=notes)


def load_yahoo(ref, conn, cfg, nba, creds, force: bool = False, client: YahooClient | None = None):
    """Yahoo league -> ingest.league.LeagueData, built on the NBA layer `nba` (ESPN's pool and schedule)."""
    import copy

    from tradecalc.clock import league_today
    from tradecalc.ingest.league import LeagueData, Player, Team

    if client is None:
        if not (creds and creds.yahoo_token):
            raise LeagueNotFound("Sign in with Yahoo to open a Yahoo league.")
        client = YahooClient(creds.yahoo_token, oauth_config())
    key = f"nba.l.{ref.league_id}"
    league = client.get(f"league/{key}/settings").find(NS + "league")
    if league is None:
        raise LeagueNotFound(f"Yahoo has no NBA league {ref.league_id} this season.")
    meta = settings_from_yahoo(league)
    if meta.scoring_type != "points":
        raise UnsupportedLeague(f"{meta.name} is a {meta.scoring_type} league. The app models points leagues only so "
                                "far; category and roto leagues are on the roadmap.")
    teams_el = client.get(f"league/{key}/teams/roster").find(NS + "league")
    raw = []
    for t in _all(teams_el, "teams/team"):
        tid = int(_t(t, "team_id"))
        plist = []
        for p in _all(t, "roster/players/player"):
            elig = [x.text for x in _all(p, "eligible_positions/position") if x.text in ("PG", "SG", "SF", "PF", "C")]
            plist.append((PlatformPlayer(_t(p, "player_key"), _t(p, "name/full", ""), _t(p, "editorial_team_abbr", ""),
                                         elig, _t(p, "status", "")),
                          (_t(p, "selected_position/position") or "").upper()))
        raw.append((tid, _t(t, "name", f"Team {tid}"), _t(t, "waiver_priority"), plist))
    to_espn, missing = match_players([pp for *_, pl in raw for pp, _ in pl], nba.players)
    players = {eid: copy.copy(p) for eid, p in nba.players.items()}
    for p in players.values():
        p.on_team_id, p.lineup_slot, p.pool_status = 0, None, "FREEAGENT"
    for pl in missing:
        eid = stand_in_id(pl.pid)
        to_espn[pl.pid] = eid
        players[eid] = Player(eid, pl.name, (pl.team or "FA").upper(), list(pl.positions),
                              YAHOO_STATUS.get(str(pl.status).upper(), "ACTIVE"))
    if missing:
        meta.notes.append(f"{len(missing)} rostered player(s) aren't in ESPN's player pool, so they have no "
                          "preseason projection: " + ", ".join(p.name for p in missing[:5]) + ("…" if len(missing) > 5 else ""))
    teams = {}
    for tid, name, wp, plist in raw:
        ids = []
        for pp, slot in plist:
            eid = to_espn[pp.pid]
            p = players[eid]
            p.on_team_id, p.pool_status = tid, "ONTEAM"
            p.lineup_slot = "IR" if slot.startswith("IL") else None
            if pp.positions:
                p.positions = list(pp.positions)
            ids.append(eid)
        teams[tid] = Team(tid, name, ids, int(wp) if wp and str(wp).isdigit() else None)

    today = league_today(cfg)
    matchups, weeks = [], []
    start_week = int(_t(league, "start_week", 1) or 1)
    for n in range(start_week, start_week + meta.regular_season_weeks):
        sb = client.get(f"league/{key}/scoreboard;week={n}").find(NS + "league")
        for m in _all(sb, "scoreboard/matchups/matchup"):
            ws, we = _t(m, "week_start"), _t(m, "week_end")
            if ws and we and (not weeks or weeks[-1][0] != n - start_week + 1):
                weeks.append((n - start_week + 1, date.fromisoformat(ws), date.fromisoformat(we)))
            side = _all(m, "teams/team")
            if len(side) != 2:
                continue
            (h, a) = sorted(side, key=lambda t: int(_t(t, "team_id")))
            ht, at = float(_t(h, "team_points/total", 0) or 0), float(_t(a, "team_points/total", 0) or 0)
            status = _t(m, "status", "")
            done = status == "postevent"
            win = _t(m, "winner_team_key")
            winner = ("TIE" if _t(m, "is_tied") == "1" else "HOME" if win == _t(h, "team_key") else "AWAY") if done \
                else "UNDECIDED"
            period = None
            if ws and status == "midevent":
                yday = min(date.fromisoformat(we), today) - date.resolution
                period = (yday - nba.first_day).days + 1
            matchups.append({"week": n - start_week + 1, "home": int(_t(h, "team_id")), "away": int(_t(a, "team_id")),
                             "home_points": {period: ht} if period and ht else {},
                             "away_points": {period: at} if period and at else {},
                             "winner": winner, "home_total": ht, "away_total": at})
    meta.weeks = weeks if len(weeks) == meta.regular_season_weeks else []
    return LeagueData({"name": meta.name}, teams, players, matchups, nba.pro_games, nba.pro_teams, nba.first_day, meta)
