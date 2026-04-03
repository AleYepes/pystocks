import sqlite3
from pathlib import Path

from ..config import SQLITE_DB_PATH


def normalize_sqlite_path(sqlite_path=SQLITE_DB_PATH) -> Path:
    path = Path(sqlite_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def configure_connection(
    conn: sqlite3.Connection, row_factory=None
) -> sqlite3.Connection:
    if row_factory is not None:
        conn.row_factory = row_factory
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


def open_connection(
    sqlite_path=SQLITE_DB_PATH, *, row_factory=None
) -> sqlite3.Connection:
    conn = sqlite3.connect(str(normalize_sqlite_path(sqlite_path)))
    return configure_connection(conn, row_factory=row_factory)
