"""Historical NBA injury reports: the league's official pre-game report, one per game day.

The NBA publishes its injury report as a PDF several times a day at
  https://ak-static.cms.nba.com/referee/injury/Injury-Report_<YYYY-MM-DD>_<time>.pdf
and past reports stay online (2019-20 onward). We take the 5 PM ET report (about two hours
before most tip-offs; the time stamp moved from "05PM" to "05_00PM" during 2025-26) and parse it
into rows: game date and time, matchup, team, player, status (Out / Doubtful / Questionable /
Probable / Available) and reason.

Raw PDFs are cached in data/injury_reports/raw/ (not committed); the parsed rows go to
data/seed/injury_reports_<season>.csv.gz (committed, small). Collect with
`tradecalc injury-reports`. Parsing needs pdfplumber (a dev-only dependency).
"""
from __future__ import annotations

import re
import time
import unicodedata
from datetime import date
from pathlib import Path

import pandas as pd

URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{stamp}.pdf"
STAMPS = ("05PM", "05_00PM", "05_30PM", "04PM", "04_30PM", "06PM", "06_00PM")
STATUSES = ("Out", "Doubtful", "Questionable", "Probable", "Available")
COLUMNS = ["game_date", "game_time", "matchup", "team", "player", "status", "reason"]

TEAMS = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets", "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers", "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons", "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies", "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves", "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns", "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs", "TOR": "Toronto Raptors", "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}


def squash(s: str) -> str:
    """Letters and digits only, lower case, accents removed: the newer PDFs drop the spaces."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", s)


TEAM_BY_SQUASHED = {squash(n): a for a, n in TEAMS.items()} | {"losangelesclippers": "LAC"}


def player_key(printed: str) -> str:
    """'Carey Jr., Vernon' or 'CareyJr.,Vernon' -> 'verncarey'-style key: first + last name squashed,
    suffixes (Jr., Sr., II, III, IV) dropped. Match to box scores with the same function."""
    last, _, first = str(printed).partition(",")
    last = re.sub(r"\s*(Jr\.?|Sr\.?|III|II|IV)\s*$", "", last.strip())
    return squash(first + last)


def box_score_key(name: str) -> str:
    """The same key for a box-score name ('Vernon Carey Jr.')."""
    parts = re.sub(r"\s+(Jr\.?|Sr\.?|III|II|IV)\s*$", "", str(name).strip()).split()
    return squash("".join(parts))


# --- download --------------------------------------------------------------------------------

def fetch_report(day: date, raw_dir: Path, session=None, pause: float = 0.25) -> Path | None:
    """The 5 PM ET report for `day` (cached). None if no report exists for that day."""
    import requests

    raw_dir.mkdir(parents=True, exist_ok=True)
    path, missing = raw_dir / f"{day.isoformat()}.pdf", raw_dir / f"{day.isoformat()}.missing"
    if path.exists():
        return path
    if missing.exists():
        return None
    s = session or requests.Session()
    for stamp in STAMPS:
        r = s.get(URL.format(stamp=f"{day.isoformat()}_{stamp}"), timeout=30)
        time.sleep(pause)
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            path.write_bytes(r.content)
            return path
    missing.touch()
    return None


# --- parse -----------------------------------------------------------------------------------

def _column_starts(words) -> list[float] | None:
    """x positions where the 7 columns start, from the header line (with or without spaces)."""
    text = [(w["text"], w["x0"]) for w in words]
    keys = [("Game", "GameDate"), ("Game", "GameTime"), ("Matchup",), ("Team",), ("Player", "PlayerName"),
            ("Current", "CurrentStatus"), ("Reason",)]
    starts, i = [], 0
    for alts in keys:
        while i < len(text) and text[i][0] not in alts:
            i += 1
        if i == len(text):
            return None
        starts.append(text[i][1])
        i += 1
    return starts


def _lines(words, tol: float = 2.0) -> list[list[dict]]:
    out = []
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        if out and abs(out[-1][0]["top"] - w["top"]) <= tol:
            out[-1].append(w)
        else:
            out.append([w])
    return [sorted(line, key=lambda w: w["x0"]) for line in out]


def parse_report(path: Path) -> pd.DataFrame:
    """Rows of one report PDF. Wrapped cells (the reason often spans the lines just above and
    below its row) are attached to the nearest player row on the page."""
    import pdfplumber

    rows, ctx, starts = [], {}, None
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(keep_blank_chars=False, use_text_flow=False)
            page_rows, fragments = [], []
            for line in _lines(words):
                text = " ".join(w["text"] for w in line)
                if text.startswith("Injury Report:") or text.startswith("Page") or re.match(r"Page\s*\d+\s*of", text):
                    continue
                hdr = _column_starts(line)
                if hdr:
                    starts = hdr
                    continue
                if starts is None:
                    continue
                cells = [[] for _ in range(7)]
                for w in line:
                    col = max(j for j in range(7) if w["x0"] >= starts[j] - 4) if w["x0"] >= starts[0] - 4 else 0
                    cells[col].append(w["text"])
                cells = [" ".join(c) for c in cells]
                top = sum(w["top"] for w in line) / len(line)
                status = cells[5].strip()
                for j, name in enumerate(COLUMNS[:4]):           # carried down the table
                    if cells[j]:
                        ctx[name] = cells[j]
                if status in STATUSES:
                    page_rows.append({**ctx, "player": cells[4], "status": status, "_top": top,
                                      "_reason": [(top, cells[6])] if cells[6] else []})
                elif cells[6] and not any(cells[:6]):
                    fragments.append((top, cells[6]))
                # other lines (e.g. a team marked NOT YET SUBMITTED) only update the carried context
            for top, txt in fragments:
                if page_rows:
                    near = min(page_rows, key=lambda r: abs(r["_top"] - top))
                    near["_reason"].append((top, txt))
            for r in page_rows:
                r["reason"] = " ".join(t for _, t in sorted(r.pop("_reason")))
                r.pop("_top")
                rows.append(r)
    df = pd.DataFrame(rows, columns=COLUMNS)
    if len(df):
        df["game_date"] = pd.to_datetime(df.game_date, format="%m/%d/%Y", errors="coerce").dt.date.astype(str)
        df["game_time"] = df.game_time.str.replace(r"\s*\(ET\)", "", regex=True)
        df["team_abbr"] = df.team.map(lambda t: TEAM_BY_SQUASHED.get(squash(t)))
        df["matchup"] = df.matchup.str.replace(" ", "")
        df["player_key"] = df.player.map(player_key)
    return df


# --- collect ---------------------------------------------------------------------------------

def collect(game_days: dict[str, list[date]], raw_dir: Path, out_dir: Path, log=print) -> dict[str, pd.DataFrame]:
    """game_days: season -> dates with games. Downloads (cached), parses and writes
    out_dir/injury_reports_<season>.csv.gz. Returns the frames."""
    import requests

    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 (tradecalc injury-report collector)"
    out = {}
    for season, days in game_days.items():
        frames, n_missing = [], 0
        for i, d in enumerate(sorted(days)):
            p = fetch_report(d, raw_dir, s)
            if p is None:
                n_missing += 1
                continue
            try:
                df = parse_report(p)
            except Exception as e:                                # a damaged file: skip, keep going
                log(f"  {d}: could not parse ({type(e).__name__}: {e})")
                continue
            frames.append(df.assign(report_date=d.isoformat(), season=season))
            if (i + 1) % 25 == 0:
                log(f"  {season}: {i + 1}/{len(days)} days")
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / f"injury_reports_{season}.csv.gz", index=False,
                  compression={"method": "gzip", "mtime": 0})
        log(f"{season}: {len(days) - n_missing} reports, {len(df):,} rows ({n_missing} days without a report)")
        out[season] = df
    return out


def load(season: str, seed_dir: Path) -> pd.DataFrame:
    p = seed_dir / f"injury_reports_{season}.csv.gz"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def game_days_from_logs(logs: pd.DataFrame) -> list[date]:
    return sorted({date.fromisoformat(str(d)[:10]) for d in logs["game_date"].unique()})


ALIASES = {"carltoncarrington": "bubcarrington", "yongxicui": "cuiyongxi"}   # report name -> box-score name


def attach_played(reports: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    """Add box_key (the matching box-score player, or NaN) and played (he logged minutes in that
    game) to report rows. Names match exactly on box_score_key; otherwise through ALIASES, then by a
    unique last name on the same team that season."""
    L = logs[logs["min"] > 0].assign(box_key=logs.player_name.map(box_score_key), d=logs.game_date.astype(str).str[:10])
    names = logs.assign(box_key=logs.player_name.map(box_score_key))
    known = set(names.box_key)
    by_last = {}
    for name, team in zip(names.player_name, names.team):
        last = squash(re.sub(r"\s+(Jr\.?|Sr\.?|III|II|IV)\s*$", "", name.strip()).split()[-1])
        by_last.setdefault((last, team), set()).add(box_score_key(name))

    def match(printed, key, team):
        if key in known:
            return key
        if ALIASES.get(key) in known:
            return ALIASES[key]
        last = squash(re.sub(r"\s*(Jr\.?|Sr\.?|III|II|IV)\s*$", "", str(printed).partition(",")[0].strip()))
        hits = by_last.get((last, team), set())
        return next(iter(hits)) if len(hits) == 1 else None

    out = reports.copy()
    out["box_key"] = [match(p, k, t) for p, k, t in zip(out.player, out.player_key, out.team_abbr)]
    played = set(zip(L.d, L.box_key))
    out["played"] = [(str(d), k) in played for d, k in zip(out.game_date, out.box_key)]
    return out
