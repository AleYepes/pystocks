from __future__ import annotations

import sqlite3
from pathlib import Path

from .schema import apply_migrations, current_schema_version


def connect_sqlite(path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def initialize_operational_store(path: Path) -> int:
    with connect_sqlite(path) as conn:
        apply_migrations(conn)
        return current_schema_version(conn)
