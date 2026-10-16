"""Prior FPPG from ADP (section 4.2).

ADP is a draft position, not a points projection, so we fit a monotone decreasing
curve rank -> FPPG on a previous season (that season's ADP vs. its actual FPPG) and
read each player's prior off the curve. If a points projection exists, the prior is
the average of the curve value and the projection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from tradecalc.model.names import norm_name


@dataclass
class AdpCurve:
    iso: IsotonicRegression
    tail_rank: int

    def __call__(self, rank) -> np.ndarray:
        r = np.minimum(np.atleast_1d(np.asarray(rank, dtype=float)), self.tail_rank)
        return self.iso.predict(r)

    @property
    def tail_value(self) -> float:
        return float(self(self.tail_rank)[0])


def fit_adp_curve(adp: pd.DataFrame, actual: pd.DataFrame, min_games: int = 10) -> AdpCurve:
    """Fit rank -> FPPG.

    adp: columns name, adp_rank (from ingest.adp.load_adp).
    actual: columns player_name, live_fppg, n (season FPPG and games, from scoring.live_stats).
    Players with fewer than `min_games` are dropped; the rest are weighted by games played.
    """
    act = actual.assign(key=actual["player_name"].map(norm_name))
    act = act[act["n"] >= min_games].drop_duplicates("key").set_index("key")
    a = adp.assign(key=adp["name"].map(norm_name)).join(act[["live_fppg", "n"]], on="key", how="inner")
    if len(a) < 30:
        raise ValueError(f"only {len(a)} ADP players matched game logs; check names / seasons")
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    # Smooth a little before the isotonic fit: rolling mean over neighbouring ranks.
    a = a.sort_values("adp_rank")
    y = a["live_fppg"].rolling(9, center=True, min_periods=1).mean()
    iso.fit(a["adp_rank"], y, sample_weight=a["n"])
    return AdpCurve(iso, int(a["adp_rank"].max()))


def prior_table(adp: pd.DataFrame, curve: AdpCurve, projections: dict[str, float] | None = None) -> pd.DataFrame:
    """Prior FPPG per player (indexed by normalized name).

    projections: {player name: projected FPPG under league scoring}. Players with a projection
    but no ADP get the average of the tail curve value and their projection.
    """
    t = pd.DataFrame({
        "name": adp["name"],
        "adp": adp["adp"],
        "adp_rank": adp["adp_rank"],
        "curve_fppg": curve(adp["adp_rank"]),
    })
    t["key"] = t["name"].map(norm_name)
    t = t.drop_duplicates("key").set_index("key")
    if projections:
        proj = pd.Series({norm_name(k): v for k, v in projections.items()}, name="proj_fppg")
        t = t.join(proj, how="outer")
        t["name"] = t["name"].fillna(pd.Series({norm_name(k): k for k in projections}))
        t["curve_fppg"] = t["curve_fppg"].fillna(curve.tail_value)
        t["prior_fppg"] = np.where(t["proj_fppg"].notna(), (t["curve_fppg"] + t["proj_fppg"]) / 2, t["curve_fppg"])
    else:
        t["proj_fppg"] = np.nan
        t["prior_fppg"] = t["curve_fppg"]
    return t
