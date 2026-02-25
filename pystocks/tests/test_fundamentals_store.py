import sqlite3
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from pystocks.fundamentals_store import FundamentalsStore


class FundamentalsStoreSqliteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.sqlite_path = self.root / "data" / "pystocks.sqlite"
        self.store = FundamentalsStore(sqlite_path=self.sqlite_path)

    def tearDown(self):
        self.tmp.cleanup()

    def _conn(self):
        return sqlite3.connect(str(self.sqlite_path))

    def test_non_series_insert_unchanged_overwrite(self):
        snapshot = {
            "conid": "12345",
            "scraped_at": "2026-02-23T12:00:00+00:00",
            "ratios": {
                "as_of_date": 20260131,
                "ratios": [
                    {
                        "name": "Price/Sales",
                        "name_tag": "price_sales",
                        "value": 3.2,
                        "vs": 0.11,
                        "percentile": 44.0,
                    }
                ],
            },
        }

        first = self.store.persist_combined_snapshot(snapshot)
        self.assertEqual(first["inserted_events"], 1)
        self.assertEqual(first["overwritten_events"], 0)
        self.assertEqual(first["unchanged_events"], 0)

        second = self.store.persist_combined_snapshot(snapshot)
        self.assertEqual(second["inserted_events"], 0)
        self.assertEqual(second["overwritten_events"], 0)
        self.assertEqual(second["unchanged_events"], 1)

        changed = deepcopy(snapshot)
        changed["ratios"]["ratios"][0]["value"] = 3.5
        third = self.store.persist_combined_snapshot(changed)
        self.assertEqual(third["inserted_events"], 0)
        self.assertEqual(third["overwritten_events"], 1)
        self.assertEqual(third["unchanged_events"], 0)

        con = self._conn()
        try:
            n_snapshots = con.execute("SELECT COUNT(*) FROM ratios_snapshots").fetchone()[0]
            self.assertEqual(n_snapshots, 1)

            value = con.execute(
                """
                SELECT value_num
                FROM ratios_metrics
                WHERE conid = ? AND metric_id = ?
                """,
                ["12345", "price_sales"],
            ).fetchone()[0]
            self.assertAlmostEqual(float(value), 3.5)

            blob_count = con.execute("SELECT COUNT(*) FROM raw_payload_blobs").fetchone()[0]
            self.assertEqual(blob_count, 2)
        finally:
            con.close()

    def test_price_series_raw_append_and_latest_upsert(self):
        first_snapshot = {
            "conid": "extend_123",
            "scraped_at": "2026-02-23T12:00:00+00:00",
            "price_chart": {
                "plot": {
                    "series": [
                        {
                            "name": "price",
                            "plotData": [
                                {"x": 1704067200000, "y": 100.0, "close": 100.0, "debugY": 20240101},
                                {"x": 1704153600000, "y": 101.0, "close": 101.0, "debugY": 20240102},
                            ],
                        }
                    ]
                }
            },
        }

        second_snapshot = {
            "conid": "extend_123",
            "scraped_at": "2026-02-24T12:00:00+00:00",
            "price_chart": {
                "plot": {
                    "series": [
                        {
                            "name": "price",
                            "plotData": [
                                {"x": 1704067200000, "y": 105.0, "close": 105.0, "debugY": 20240101},
                                {"x": 1704153600000, "y": 101.0, "close": 101.0, "debugY": 20240102},
                                {"x": 1704240000000, "y": 103.0, "close": 103.0, "debugY": 20240103},
                            ],
                        }
                    ]
                }
            },
        }

        self.store.persist_combined_snapshot(first_snapshot)
        self.store.persist_combined_snapshot(second_snapshot)

        con = self._conn()
        try:
            raw_count = con.execute(
                "SELECT COUNT(*) FROM price_chart_series_raw WHERE conid = ?",
                ["extend_123"],
            ).fetchone()[0]
            latest_count = con.execute(
                "SELECT COUNT(*) FROM price_chart_series_latest WHERE conid = ?",
                ["extend_123"],
            ).fetchone()[0]
            self.assertEqual(raw_count, 5)
            self.assertEqual(latest_count, 3)

            day1_price = con.execute(
                """
                SELECT close
                FROM price_chart_series_latest
                WHERE conid = ? AND trade_date = ?
                """,
                ["extend_123", "2024-01-01"],
            ).fetchone()[0]
            self.assertAlmostEqual(float(day1_price), 105.0)
        finally:
            con.close()

    def test_endpoint_flattening_writes_child_tables(self):
        snapshot = {
            "conid": "flat_123",
            "scraped_at": "2026-02-23T12:00:00+00:00",
            "profile_and_fees": {
                "symbol": "SPY",
                "objective": "Test objective",
                "jap_fund_warning": False,
                "fund_and_profile": [
                    {"name": "Asset Type", "name_tag": "Asset_Type", "value": "Equity"},
                    {"name": "Total Expense Ratio", "name_tag": "Total_Expense_Ratio", "value": "0.09%"},
                ],
                "reports": [
                    {
                        "name": "Annual Report",
                        "as_of_date": 1759204800000,
                        "fields": [{"name": "Total Expense", "value": "0.0893%", "is_summary": True}],
                    }
                ],
                "expenses_allocation": [{"name": "Management", "ratio": "52.5%", "value": "52.5%"}],
                "themes": ["Value"],
            },
            "holdings": {
                "as_of_date": 20260131,
                "allocation_self": [{"name": "Equity", "weight": 99.8, "rank": 1, "vs": 0.1}],
                "top_10": [{"name": "NVIDIA", "ticker": "NVDA", "rank": 1, "assets_pct": "7.83%", "conids": [4815747]}],
                "geographic": {"us": "95%"},
            },
            "morningstar": {
                "as_of_date": "20260131",
                "summary": [{"id": "medalist_rating", "title": "Medalist", "value": "Silver", "publish_date": "20260128", "q": False}],
                "commentary": [{"id": "summary", "title": "Summary", "text": "Best in class", "publish_date": "20260128", "q": False}],
                "q_full_report_id": "abc123",
            },
            "esg": {
                "asOfDate": "20251231",
                "coverage": 0.83,
                "source": "REFINITIV_LIPPER",
                "symbol": "SPY",
                "no_settings": True,
                "content": [
                    {
                        "name": "TRESGS",
                        "value": 5,
                        "children": [{"name": "TRESGENRRS", "value": 5}],
                    }
                ],
            },
        }

        result = self.store.persist_combined_snapshot(snapshot)
        self.assertEqual(result["status"], "ok")

        con = self._conn()
        try:
            self.assertGreater(
                con.execute("SELECT COUNT(*) FROM profile_fees_fund_profile_fields").fetchone()[0],
                0,
            )
            self.assertGreater(
                con.execute("SELECT COUNT(*) FROM profile_fees_report_fields").fetchone()[0],
                0,
            )
            self.assertGreater(
                con.execute("SELECT COUNT(*) FROM holdings_bucket_weights").fetchone()[0],
                0,
            )
            self.assertGreater(
                con.execute("SELECT COUNT(*) FROM holdings_top10_conids").fetchone()[0],
                0,
            )
            self.assertGreater(
                con.execute("SELECT COUNT(*) FROM morningstar_summary").fetchone()[0],
                0,
            )
            self.assertGreater(
                con.execute("SELECT COUNT(*) FROM esg_nodes").fetchone()[0],
                0,
            )
        finally:
            con.close()

    def test_persist_ingest_run_writes_rollups(self):
        run_id = self.store.persist_ingest_run(
            run_stats={
                "run_started_at": "2026-02-23T12:00:00+00:00",
                "run_finished_at": "2026-02-23T12:10:00+00:00",
                "total_targeted_conids": 10,
                "processed_conids": 10,
                "saved_snapshots": 8,
                "inserted_events": 80,
                "overwritten_events": 3,
                "unchanged_events": 5,
                "series_raw_rows_written": 1200,
                "series_latest_rows_upserted": 900,
                "auth_retries": 1,
                "aborted": False,
            },
            endpoint_summary=[
                {
                    "endpoint": "landing",
                    "call_count": 10,
                    "useful_payload_count": 8,
                    "useful_payload_rate": 0.8,
                    "status_codes": {"200": 8, "404": 1, "500": 1},
                }
            ],
        )

        con = self._conn()
        try:
            stored = con.execute("SELECT COUNT(*) FROM ingest_runs WHERE run_id = ?", [run_id]).fetchone()[0]
            self.assertEqual(stored, 1)

            rollup = con.execute(
                """
                SELECT status_2xx, status_4xx, status_5xx, status_other
                FROM ingest_run_endpoint_rollups
                WHERE run_id = ? AND endpoint = 'landing'
                """,
                [run_id],
            ).fetchone()
            self.assertEqual(tuple(rollup), (8, 1, 1, 0))
        finally:
            con.close()

    def test_ownership_holders_accept_dict_type(self):
        snapshot = {
            "conid": "owner_123",
            "scraped_at": "2026-02-23T12:00:00+00:00",
            "ownership": {
                "owners_types": [],
                "institutional_owners": [
                    {
                        "name": "Test Inst",
                        "type": {"type": "Institutional", "display_type": "Institutional Owner"},
                        "display_value": "1.0M",
                        "display_shares": "10K",
                        "display_pct": "1.2%",
                    }
                ],
                "insider_owners": [],
                "trade_log": [],
                "institutional_total": {"display_value": "1.0M", "display_shares": "10K", "display_pct": "1.2%"},
                "insider_total": {"display_value": "0", "display_shares": "0", "display_pct": "0%"},
                "ownership_history": {},
            },
        }

        result = self.store.persist_combined_snapshot(snapshot)
        self.assertEqual(result["status"], "ok")

        con = self._conn()
        try:
            row = con.execute(
                """
                SELECT holder_type
                FROM ownership_holders
                WHERE conid = ?
                LIMIT 1
                """,
                ["owner_123"],
            ).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], "Institutional")
        finally:
            con.close()


if __name__ == "__main__":
    unittest.main()
