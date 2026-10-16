"""ADP import from CSV.

CSV columns: name, adp  (optional: team, espn_id, rank_type).
`export_espn_adp` writes these files from ESPN's own data (the platform's draft ADP),
so a FantasyPros or other CSV can replace them without code changes.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


def load_adp(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"name", "adp"} <= set(df.columns):
        raise ValueError(f"{path}: ADP CSV needs 'name' and 'adp' columns, got {list(df.columns)}")
    df = df.dropna(subset=["adp"]).sort_values("adp").reset_index(drop=True)
    df["adp_rank"] = df["adp"].rank(method="first").astype(int)
    return df


def export_espn_adp(pool: dict, path: str | Path, *, use_rank: bool = False) -> int:
    """Write an ADP CSV from an ESPN kona_player_info response.

    use_rank=True writes ESPN's STANDARD draft rank instead of ADP. ESPN resets last season's
    ADP after the season, but keeps the preseason draft ranks, so that is the best free stand-in.
    """
    rows = []
    for e in pool.get("players", []):
        p = e["player"]
        if use_rank:
            v = ((p.get("draftRanksByRankType") or {}).get("STANDARD") or {}).get("rank")
        else:
            v = (p.get("ownership") or {}).get("averageDraftPosition")
        if v and v > 0:
            rows.append({"name": p["fullName"], "adp": v, "espn_id": p["id"],
                         "rank_type": "espn_draft_rank" if use_rank else "espn_adp"})
    rows.sort(key=lambda r: r["adp"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "adp", "espn_id", "rank_type"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)
