import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pystocks.ops_state as database


class DatabaseStatusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "pystocks.sqlite"
        self._original_db_path = database.DB_PATH
        self._original_initialized = database._INITIALIZED_DB_PATH
        database.DB_PATH = self.db_path
        database._INITIALIZED_DB_PATH = None
        database.init_db()
        database.upsert_instruments_from_products(
            pd.DataFrame(
                [
                    {
                        "conid": "123",
                        "symbol": "ABC",
                        "exchangeId": "TEST",
                        "isin": "",
                        "currency": "USD",
                    }
                ]
            )
        )

    def tearDown(self):
        database.DB_PATH = self._original_db_path
        database._INITIALIZED_DB_PATH = self._original_initialized
        self.tmp.cleanup()

    def _instrument_state(self):
        conn = database.get_connection()
        try:
            return conn.execute(
                """
                SELECT last_scraped_fundamentals, last_status_fundamentals
                FROM products
                WHERE conid = ?
                """,
                ["123"],
            ).fetchone()
        finally:
            conn.close()

    def test_instrument_starts_unscraped(self):
        last_scraped, status = self._instrument_state()
        self.assertIsNone(last_scraped)
        self.assertIsNone(status)

    def test_status_update_marks_scraped_only_on_success(self):
        database.update_instrument_fundamentals_status("123", "auth_error", mark_scraped=False)
        last_scraped, status = self._instrument_state()
        self.assertIsNone(last_scraped)
        self.assertEqual(status, "auth_error")

        database.update_instrument_fundamentals_status("123", "success", mark_scraped=True)
        last_scraped, status = self._instrument_state()
        self.assertIsNotNone(last_scraped)
        self.assertEqual(status, "success")


if __name__ == "__main__":
    unittest.main()
