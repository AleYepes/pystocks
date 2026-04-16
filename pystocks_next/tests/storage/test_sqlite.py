from __future__ import annotations

from pathlib import Path

from pystocks_next.storage.schema import current_schema_version
from pystocks_next.storage.sqlite import connect_sqlite, initialize_operational_store


def test_initialize_operational_store_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "storage.sqlite"

    first_version = initialize_operational_store(db_path)
    second_version = initialize_operational_store(db_path)

    assert first_version == 1
    assert second_version == 1

    with connect_sqlite(db_path, read_only=True) as conn:
        assert current_schema_version(conn) == 1
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        foreign_keys = conn.execute("PRAGMA foreign_keys").fetchone()[0]

    assert journal_mode == "wal"
    assert foreign_keys == 1
