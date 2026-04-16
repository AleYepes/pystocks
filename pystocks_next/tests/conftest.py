from __future__ import annotations

from pathlib import Path

import pytest

from pystocks_next.storage.sqlite import connect_sqlite, initialize_operational_store


@pytest.fixture
def temp_store_path(tmp_path: Path) -> Path:
    db_path = tmp_path / "pystocks_next.sqlite"
    initialize_operational_store(db_path)
    return db_path


@pytest.fixture
def temp_store(temp_store_path: Path):
    with connect_sqlite(temp_store_path) as conn:
        yield conn
