"""Player-name normalization so ESPN and stats.nba.com names join (e.g. 'Nikola Jokić' == 'Nikola Jokic')."""
from __future__ import annotations

import re
import unicodedata

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'\-]", " ", s)
    s = _SUFFIX.sub("", s)
    return " ".join(s.split())


def match_names(names: list[str], pool: list[str], cutoff: int = 88) -> dict[str, str]:
    """Map each name in `names` to its best match in `pool` (exact normalized first, then fuzzy)."""
    from rapidfuzz import fuzz, process

    by_norm = {norm_name(p): p for p in pool}
    out = {}
    for n in names:
        k = norm_name(n)
        if k in by_norm:
            out[n] = by_norm[k]
            continue
        hit = process.extractOne(k, list(by_norm), scorer=fuzz.token_sort_ratio, score_cutoff=cutoff)
        if hit:
            out[n] = by_norm[hit[0]]
    return out
