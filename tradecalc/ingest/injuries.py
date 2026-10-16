"""NBA injury report with expected return dates, from ESPN's public site API.

The fantasy API only gives a status (OUT, DAY_TO_DAY, ...). The site injuries feed adds an
expected return date, injury type and a news blurb per player. Athlete ids there are the
same ids the fantasy API uses.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import date

import requests

from tradecalc.db import cache_get, cache_put

URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
_ID = re.compile(r"/id/(\d+)/")


@dataclass
class Injury:
    espn_id: int | None
    name: str
    status: str               # OUT | DAY_TO_DAY | OUT_FOR_SEASON
    return_date: date | None
    injury_type: str
    comment: str
    reported: str             # date of the latest note


def _status(raw: dict) -> str:
    fantasy = ((raw.get("details") or {}).get("fantasyStatus") or {}).get("abbreviation", "")
    if fantasy == "OFS":
        return "OUT_FOR_SEASON"
    s = (raw.get("status") or "").lower()
    return "OUT" if s == "out" else "DAY_TO_DAY" if "day" in s else s.upper() or "ACTIVE"


def parse(data: dict) -> list[Injury]:
    out = []
    for team in data.get("injuries", []):
        for i in team.get("injuries", []):
            a = i.get("athlete") or {}
            pid = None
            for link in a.get("links", []):
                m = _ID.search(link.get("href", ""))
                if m:
                    pid = int(m.group(1))
                    break
            det = i.get("details") or {}
            rd = det.get("returnDate")
            kind = " ".join(x for x in (det.get("side") if det.get("side") not in (None, "Not Specified") else "",
                                        det.get("type", ""), det.get("detail", "")) if x)
            out.append(Injury(pid, a.get("displayName", ""), _status(i),
                              date.fromisoformat(rd[:10]) if rd else None, kind,
                              i.get("shortComment", ""), (i.get("date") or "")[:10]))
    return out


def fetch_injuries(conn: sqlite3.Connection, ttl_hours: float = 3, force: bool = False) -> list[Injury]:
    data = None if force else cache_get(conn, URL, ttl_hours * 3600)
    if data is None:
        r = requests.get(URL, timeout=30, headers={"User-Agent": "tradecalc/0.1"})
        r.raise_for_status()
        data = r.json()
        cache_put(conn, URL, data)
    return parse(data)
