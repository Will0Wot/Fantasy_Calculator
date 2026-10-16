"""Load config/league.yaml. All league-specific values come from here."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PKG_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PKG_DIR / "config" / "league.yaml"


class Config(dict):
    """Dict with attribute access and path resolution relative to the package dir."""

    def __getattr__(self, name: str) -> Any:
        try:
            v = self[name]
        except KeyError as e:
            raise AttributeError(name) from e
        return Config(v) if isinstance(v, dict) and not isinstance(v, Config) else v

    def path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else PKG_DIR / p


@lru_cache(maxsize=None)
def _load(path: str) -> Config:
    with open(path) as f:
        return Config(yaml.safe_load(f))


def load_config(path: str | Path | None = None) -> Config:
    return _load(str(path or DEFAULT_CONFIG))


def save_model_value(key: str, value: Any, path: str | Path | None = None) -> None:
    """Persist a tuned model value (e.g. k) back into league.yaml, preserving the rest of the file."""
    import re

    p = Path(path or DEFAULT_CONFIG)
    text = p.read_text()
    pattern = re.compile(rf"^(\s+{re.escape(key)}:\s*)([^#\n]*?)(\s*(#.*)?)$", re.M)
    new, n = pattern.subn(lambda m: f"{m.group(1)}{value}{m.group(3)}", text, count=1)
    if n != 1:
        raise KeyError(f"{key} not found in {p}")
    p.write_text(new)
    _load.cache_clear()
