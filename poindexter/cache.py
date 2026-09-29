"""Response cache: sqlite, keyed by (model, temperature, system, user, sample_index).

The default path is `.poindexter/cache.sqlite` under the working directory, or the file
named by POINDEXTER_CACHE. WAL mode and a busy timeout let several processes share it.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import sqlite3
from pathlib import Path

BUSY_TIMEOUT_S = 60


def key(model: str, temperature: float | None, system: str, user: str, sample_index: int) -> str:
    temp = "default" if temperature is None else repr(float(temperature))
    parts = json.dumps([model, temp, system, user, sample_index])
    return hashlib.sha256(parts.encode()).hexdigest()


class Cache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=BUSY_TIMEOUT_S, isolation_level=None)
        self.db.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_S * 1000}")
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )

    def get(self, k: str) -> str | None:
        row = self.db.execute("SELECT value FROM responses WHERE key = ?", (k,)).fetchone()
        return row[0] if row else None

    def put(self, k: str, value: str) -> str:
        """Store value unless another writer got there first. Returns the stored value,
        so every process sees the same sample for the same key."""
        self.db.execute("INSERT OR IGNORE INTO responses (key, value) VALUES (?, ?)", (k, value))
        stored = self.get(k)
        assert stored is not None
        return stored


def default_path() -> Path:
    env = os.environ.get("POINDEXTER_CACHE")
    return Path(env) if env else Path.cwd() / ".poindexter" / "cache.sqlite"


def open_default() -> Cache:
    return _open(str(default_path().resolve()))


@functools.cache
def _open(path: str) -> Cache:
    return Cache(path)
