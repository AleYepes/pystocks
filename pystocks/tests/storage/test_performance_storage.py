import sqlite3
import tempfile
from pathlib import Path

import pytest

from pystocks.storage.fundamentals_store import FundamentalsStore


def _make_store():
    tmp = tempfile.TemporaryDirectory()
    db_path = Path(tmp.name) / "pystocks.sqlite"
    store = FundamentalsStore(sqlite_path=db_path)
    return tmp, db_path, store


def _table_columns(con, table):
    return [r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()]


def test_performance_table_rename_and_column_reduction():
    tmp, db_path, store = _make_store()
    try:
        snapshot = {
            "conid": "perf_1",
            "scraped_at": "2026-02-23T12:00:00+00:00",
            "ratios": {"as_of_date": "2026-02-23"},
            "performance": {
                "title_vs": "peer universe",
                "cumulative": [
                    {
                        "name": "YTD",
                        "name_tag": "ytd",
                        "name_tag_arg": "1",
                        "value": 10.5,
                        "value_fmt": "10.5%",
                        "vs": 0.15,
                        "min": -3.0,
                        "max": 15.0,
                        "avg": 8.0,
                        "percentile": None,
                        "min_fmt": "-3%",
                        "max_fmt": "15%",
                        "avg_fmt": "8%",
                    }
                ],
                "risk": [
                    {
                        "name": "Std Dev",
                        "name_tag": "std_dev",
                        "value": 12.2,
                        "vs": -0.21,
                        "min": 9.5,
                        "max": 18.2,
                        "avg": 13.4,
                    }
                ],
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
            assert "performance_metrics" not in table_names
            assert "performance" in table_names

            perf_row = con.execute(
                """
                SELECT section, metric_id, value_num, vs_num, min_num, max_num, avg_num
                FROM performance
                WHERE conid = ? AND section = ?
                """,
                ["perf_1", "cumulative"],
            ).fetchone()
            assert perf_row == (
                "cumulative",
                "ytd",
                pytest.approx(10.5),
                pytest.approx(0.15),
                pytest.approx(-3.0),
                pytest.approx(15.0),
                pytest.approx(8.0),
            )

            perf_cols = _table_columns(con, "performance")
            assert "metric_name" not in perf_cols
            assert "name_tag_arg" not in perf_cols
            assert "value_fmt" not in perf_cols
            assert "percentile_num" not in perf_cols
            assert "min_fmt" not in perf_cols
            assert "max_fmt" not in perf_cols
            assert "avg_fmt" not in perf_cols

            snapshot_cols = _table_columns(con, "performance_snapshots")
            assert "title_vs" not in snapshot_cols
        finally:
            con.close()
    finally:
        tmp.cleanup()
