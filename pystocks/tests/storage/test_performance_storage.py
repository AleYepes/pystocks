import sqlite3
import tempfile
from pathlib import Path

from pystocks.storage.fundamentals_store import FundamentalsStore


def _make_store():
    tmp = tempfile.TemporaryDirectory()
    db_path = Path(tmp.name) / "pystocks.sqlite"
    store = FundamentalsStore(sqlite_path=db_path)
    return tmp, db_path, store


def test_legacy_performance_payload_is_ignored_and_tables_are_absent():
    tmp, db_path, store = _make_store()
    try:
        snapshot = {
            "conid": "perf_1",
            "scraped_at": "2026-02-23T12:00:00+00:00",
            "ratios": {"as_of_date": "2026-02-23"},
            "performance": {
                "risk": [{"name": "Std Dev", "name_tag": "std_dev", "value": 12.2}],
            },
        }

        result = store.persist_combined_snapshot(snapshot)
        assert result["status"] == "ok"

        con = sqlite3.connect(db_path)
        try:
            table_names = {
                row[0]
                for row in con.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            assert "performance" not in table_names
            assert "performance_snapshots" not in table_names
        finally:
            con.close()
    finally:
        tmp.cleanup()
