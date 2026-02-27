import sqlite3
import tempfile
from pathlib import Path

from pystocks.fundamentals_store import FundamentalsStore


def _make_store():
    tmp = tempfile.TemporaryDirectory()
    db_path = Path(tmp.name) / "pystocks.sqlite"
    store = FundamentalsStore(sqlite_path=db_path)
    return tmp, db_path, store


def _table_columns(con, table):
    return [r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()]


def test_lipper_multi_universe_rows_and_snapshot_schema():
    tmp, db_path, store = _make_store()
    try:
        snapshot = {
            "conid": "lipper_1",
            "scraped_at": "2026-02-23T12:00:00+00:00",
            "ratios": {"as_of_date": "2026-02-23"},
            "lipper_ratings": {
                "universes": [
                    {
                        "name": "Sweden",
                        "as_of_date": 1769749200000,
                        "overall": [
                            {
                                "name": "Total Return",
                                "name_tag": "total_return",
                                "rating": {"name": "236 funds", "value": 5},
                            }
                        ],
                        "3_year": [
                            {
                                "name": "Total Return",
                                "name_tag": "total_return",
                                "rating": {"name": "236 funds", "value": 4},
                            }
                        ],
                    },
                    {
                        "name": "Denmark",
                        "as_of_date": 1769749200000,
                        "overall": [
                            {
                                "name": "Expense",
                                "name_tag": "expense",
                                "rating": {"name": "179 funds", "value": 5},
                            }
                        ],
                    },
                ]
            },
        }

        result = store.persist_combined_snapshot(snapshot)
        assert result["status"] == "ok"

        con = sqlite3.connect(db_path)
        try:
            snapshot_cols = _table_columns(con, "lipper_ratings_snapshots")
            assert "source_file" not in snapshot_cols

            snapshot_row = con.execute(
                """
                SELECT universe_count
                FROM lipper_ratings_snapshots
                WHERE conid = ?
                """,
                ["lipper_1"],
            ).fetchone()
            assert snapshot_row == (2,)

            rows = con.execute(
                """
                SELECT period, metric_id, rating_value, rating_label, universe_name, universe_as_of_date
                FROM lipper_ratings
                WHERE conid = ?
                ORDER BY universe_name, period, metric_id
                """,
                ["lipper_1"],
            ).fetchall()
            assert rows == [
                ("overall", "expense", 5.0, "179 funds", "Denmark", "2026-01-30"),
                ("3_year", "total_return", 4.0, "236 funds", "Sweden", "2026-01-30"),
                ("overall", "total_return", 5.0, "236 funds", "Sweden", "2026-01-30"),
            ]

            ratings_cols = _table_columns(con, "lipper_ratings")
            assert "metric_name" not in ratings_cols
        finally:
            con.close()
    finally:
        tmp.cleanup()
