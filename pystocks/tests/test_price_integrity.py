import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

from pystocks.fundamentals_store import FundamentalsStore

class PriceIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.store = FundamentalsStore(
            fundamentals_dir=root / "fundamentals",
            blobs_dir=root / "fundamentals" / "blobs",
            parquet_dir=root / "fundamentals" / "parquet",
            factor_features_parquet_dir=root / "factors" / "ibkr_factor_features",
            price_chart_parquet_dir=root / "prices" / "ibkr_mf_performance_chart",
            sentiment_search_parquet_dir=root / "sentiment" / "ibkr_sma_search",
            ownership_trade_log_parquet_dir=root / "ownership" / "ibkr_ownership_trade_log",
            dividends_events_parquet_dir=root / "dividends" / "ibkr_dividends_events",
            events_db_path=root / "fundamentals" / "events.db",
            duckdb_path=root / "fundamentals" / "fundamentals.duckdb",
        )
        self.root = root

    def tearDown(self):
        self.tmp.cleanup()

    def test_price_date_priority_x_over_debug_y(self):
        # Friday, Feb 23, 2024 at 12:00:00 UTC -> 1708689600000
        # debugY set to next day (Saturday) to simulate mismatch
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
                                    "debugY": 20240224  # Saturday
                                }
                            ]
                        }
                    ]
                }
            }
        }

        self.store.persist_combined_snapshot(snapshot, refresh_duckdb=False)
        
        files = list((self.root / "prices" / "ibkr_mf_performance_chart").glob("conid=priority_test/*.parquet"))
        self.assertEqual(len(files), 1)
        df = pd.read_parquet(files[0])
        
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["trade_date"], expected_date)
        self.assertEqual(df.iloc[0]["timestamp_ms"], timestamp_ms)

    def test_price_date_fallback_to_debug_y(self):
        # No 'x', only 'debugY' = 20240223
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
                                    # x is missing
                                    "y": 100.0,
                                    "debugY": 20240223
                                }
                            ]
                        }
                    ]
                }
            }
        }

        self.store.persist_combined_snapshot(snapshot, refresh_duckdb=False)
        
        files = list((self.root / "prices" / "ibkr_mf_performance_chart").glob("conid=fallback_test/*.parquet"))
        self.assertEqual(len(files), 1)
        df = pd.read_parquet(files[0])
        
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["trade_date"], expected_date)

if __name__ == "__main__":
    unittest.main()
