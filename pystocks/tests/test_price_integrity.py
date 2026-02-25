import sqlite3
import tempfile
import unittest
from pathlib import Path

from pystocks.fundamentals_store import FundamentalsStore


class PriceIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.sqlite_path = self.root / "pystocks.sqlite"
        self.store = FundamentalsStore(sqlite_path=self.sqlite_path)

    def tearDown(self):
        self.tmp.cleanup()

    def _conn(self):
        return sqlite3.connect(str(self.sqlite_path))

    def test_price_date_priority_x_over_debug_y(self):
        timestamp_ms = 1708689600000
        expected_date = "2024-02-23"

        snapshot = {
            "conid": "priority_test",
            "scraped_at": "2024-02-23T12:00:00+00:00",
            "price_chart": {
                "plot": {
                    "series": [
                        {
                            "name": "price",
                            "plotData": [
                                {
                                    "x": timestamp_ms,
                                    "y": 100.0,
                                    "debugY": 20240224,
                                }
                            ],
                        }
                    ]
                }
            },
        }

        self.store.persist_combined_snapshot(snapshot, refresh_views=False)

        con = self._conn()
        try:
            row = con.execute(
                """
                SELECT trade_date, timestamp_ms
                FROM price_chart_series_latest
                WHERE conid = ?
                """,
                ["priority_test"],
            ).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], expected_date)
            self.assertEqual(row[1], timestamp_ms)
        finally:
            con.close()

    def test_price_date_fallback_to_debug_y(self):
        expected_date = "2024-02-23"

        snapshot = {
            "conid": "fallback_test",
            "scraped_at": "2024-02-23T12:00:00+00:00",
            "price_chart": {
                "plot": {
                    "series": [
                        {
                            "name": "price",
                            "plotData": [
                                {
                                    "y": 100.0,
                                    "debugY": 20240223,
                                }
                            ],
                        }
                    ]
                }
            },
        }

        self.store.persist_combined_snapshot(snapshot, refresh_views=False)

        con = self._conn()
        try:
            row = con.execute(
                """
                SELECT trade_date
                FROM price_chart_series_latest
                WHERE conid = ?
                """,
                ["fallback_test"],
            ).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], expected_date)
        finally:
            con.close()


if __name__ == "__main__":
    unittest.main()
