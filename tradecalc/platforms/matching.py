"""Match another platform's players to ESPN's NBA pool (the one player ID the model uses).

By normalized name ('Nikola Jokić' == 'Nikola Jokic', suffixes dropped), the NBA team settling
duplicates, then a strict fuzzy match for spelling differences (e.g. 'Nic Claxton' / 'Nicolas Claxton').
A player still unmatched (a two-way call-up ESPN's pool doesn't list) keeps a stand-in ID below zero:
he's on the roster and his box scores still join by name, but he has no ESPN projection.
"""
from __future__ import annotations

from dataclasses import dataclass

from tradecalc.model.names import norm_name
from tradecalc.teams import to_nba


@dataclass
class PlatformPlayer:
    pid: str                      # the platform's player ID
    name: str
    team: str                     # NBA team abbreviation as the platform writes it ('' = free agent)
    positions: list
    status: str = ""              # the platform's injury status


def match_players(players: list[PlatformPlayer], pool: dict, fuzzy_cutoff: int = 92) -> tuple[dict[str, int], list]:
    """-> ({platform id: ESPN id}, [unmatched PlatformPlayer]). pool: ESPN id -> ingest.league.Player."""
    by_name: dict[str, list[int]] = {}
    for eid, p in pool.items():
        by_name.setdefault(norm_name(p.name), []).append(eid)
    out, missing = {}, []
    for pl in players:
        cands = by_name.get(norm_name(pl.name), [])
        if len(cands) > 1 and pl.team:
            same = [e for e in cands if to_nba(pool[e].pro_team) == to_nba(pl.team)]
            cands = same or cands
        if cands:
            out[pl.pid] = cands[0]
        else:
            missing.append(pl)
    if missing:
        from rapidfuzz import fuzz, process

        names = list(by_name)
        still = []
        for pl in missing:
            hit = process.extractOne(norm_name(pl.name), names, scorer=fuzz.token_sort_ratio, score_cutoff=fuzzy_cutoff)
            if hit:
                e = by_name[hit[0]][0]
                if not pl.team or not pool[e].pro_team or to_nba(pool[e].pro_team) == to_nba(pl.team):
                    out[pl.pid] = e
                    continue
            still.append(pl)
        missing = still
    return out, missing


def stand_in_id(platform_id: str) -> int:
    """A stable negative ID for a player ESPN's pool doesn't have."""
    import zlib

    return -(zlib.crc32(str(platform_id).encode()) or 1)
