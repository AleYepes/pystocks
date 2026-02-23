import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd

from pystocks.fundamentals_store import FundamentalsStore


class FundamentalsStoreIntegrationTests(unittest.TestCase):
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

    def test_persist_snapshot_writes_factor_and_series(self):
        snapshot = {
            "conid": "12345",
            "scraped_at": "2026-02-23T12:00:00+00:00",
            "landing": {"widgets": []},
            "profile_and_fees": {
                "fund_and_profile": [
                    {"name_tag": "Asset_Type", "value": "Equity"},
                    {"name_tag": "Total_Expense_Ratio", "value": "0.09%"},
                    {"name_tag": "Domicile", "value": "USA"},
                ],
                "themes": ["Value", "Growth"],
            },
            "holdings": {
                "as_of_date": 20260131,
                "allocation_self": [{"name": "Equity", "weight": 99.8}],
                "industry": [{"name": "Technology", "weight": 44.5}],
                "currency": [{"name": "USD", "weight": "98.1%"}],
            },
            "ratios": {
                "as_of_date": 20260131,
                "ratios": [
                    {
                        "name_tag": "price_sales",
                        "value": 3.2,
                        "vs": 0.11,
                        "percentile": 44.0,
                    }
                ],
            },
            "lipper_ratings": {
                "universes": [
                    {
                        "as_of_date": 20260131,
                        "overall": [{"name_tag": "total_return", "rating": {"value": 5}}],
                    }
                ]
            },
            "dividends": {
                "as_of_date": 20260131,
                "industry_average": {"dividend_yield": "1.22%", "annual_dividend": "2.2"},
                "industry_comparison": {"content": [{"name_tag": "div_yield", "value": "1.20%"}]},
                "history": {
                    "series": [
                        {
                            "name": "dividends",
                            "plotData": [
                                {
                                    "x": 1704067200000,
                                    "amount": 1.23,
                                    "type": "ACTUAL",
                                    "ex_dividend_date": {"d": 1, "m": "JAN", "y": 2024},
                                    "formatted_amount": "1.23 USD",
                                }
                            ],
                        }
                    ]
                },
            },
            "morningstar": {
                "as_of_date": "20230430",
                "summary": [
                    {"id": "medalist_rating", "value": "Bronze", "publish_date": "20230331"},
                    {"id": "sustainability_rating", "value": "Above_Average", "publish_date": "20251231"},
                ],
            },
            "performance": {
                "cumulative": [{"name_tag": "1_Year", "value": 16.2, "vs": 0.7}],
                "risk": [{"name_tag": "std_dev", "value": 18.4}],
            },
            "price_chart": {
                "plot": {
                    "series": [
                        {
                            "name": "price",
                            "plotData": [
                                {
                                    "x": 1704067200000,
                                    "y": 100.0,
                                    "open": 99.0,
                                    "high": 101.0,
                                    "low": 98.0,
                                    "close": 100.0,
                                    "debugY": 20240101,
                                },
                                {
                                    "x": 1704153600000,
                                    "y": 102.0,
                                    "open": 100.0,
                                    "high": 103.0,
                                    "low": 99.0,
                                    "close": 102.0,
                                    "debugY": 20240102,
                                },
                            ],
                        }
                    ]
                }
            },
            "sentiment_search": {
                "sentiment": [
                    {
                        "datetime": 1704067200000,
                        "sscore": 0.5,
                        "sdelta": 0.1,
                        "svolatility": 1.2,
                        "price": 100.0,
                        "open": 99.0,
                        "high": 101.0,
                        "low": 98.0,
                        "close": 100.0,
                    }
                ]
            },
            "ownership": {
                "trade_log": [
                    {"action": "NO CHANGE", "shares": 0, "displayDate": {"t": "2026-01-31"}},
                    {
                        "action": "PURCHASE",
                        "shares": 10,
                        "value": 1000.0,
                        "holding": 200.0,
                        "party": "Test Manager",
                        "source": "13F",
                        "insider": False,
                        "displayDate": {"t": "2026-01-31"},
                    },
                ],
                "owners_types": [{"type": {"type": "Institutional"}, "float": 75.0}],
                "institutional_owners": [],
                "insider_owners": [],
            },
            "esg": {
                "asOfDate": "20251231",
                "coverage": 0.83,
                "content": [{"name": "TRESGS", "value": 5}],
            },
        }

        result = self.store.persist_combined_snapshot(snapshot, refresh_duckdb=True)
        self.assertEqual(result["inserted_events"], 12)
        self.assertGreater(result["factor_rows_written"], 0)
        self.assertGreater(result["series_rows_written"], 0)

        factor_files = list((self.root / "factors" / "ibkr_factor_features").glob("endpoint=*/conid=*/*.parquet"))
        self.assertGreater(len(factor_files), 0)

        sentiment_files = list((self.root / "sentiment" / "ibkr_sma_search").glob("conid=*/*.parquet"))
        self.assertEqual(len(sentiment_files), 1)
        sentiment_df = pd.read_parquet(sentiment_files[0])
        forbidden = {"price", "open", "high", "low", "close", "price_change", "price_change_p"}
        self.assertTrue(forbidden.isdisjoint(set(sentiment_df.columns)))

        ownership_files = list((self.root / "ownership" / "ibkr_ownership_trade_log").glob("conid=*/*.parquet"))
        self.assertEqual(len(ownership_files), 1)
        ownership_df = pd.read_parquet(ownership_files[0])
        self.assertNotIn("NO CHANGE", set(ownership_df["action"].dropna().astype(str).str.upper()))

        dividends_files = list((self.root / "dividends" / "ibkr_dividends_events").glob("conid=*/*.parquet"))
        self.assertEqual(len(dividends_files), 1)

        con = duckdb.connect(str(self.root / "fundamentals" / "fundamentals.duckdb"))
        try:
            n_features = con.execute("SELECT COUNT(*) FROM factor_features_all").fetchone()[0]
            n_catalog = con.execute("SELECT COUNT(*) FROM factor_features_catalog").fetchone()[0]
            n_panel = con.execute("SELECT COUNT(*) FROM factor_panel_long_daily").fetchone()[0]
            legacy_views = con.execute(
                """
                SELECT table_name
                FROM information_schema.views
                WHERE table_name LIKE 'analytics_%'
                """
            ).fetchall()
            self.assertGreater(n_features, 0)
            self.assertGreater(n_catalog, 0)
            self.assertGreater(n_panel, 0)
            self.assertEqual(legacy_views, [])
        finally:
            con.close()


if __name__ == "__main__":
    unittest.main()
