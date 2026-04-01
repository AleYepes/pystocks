import sqlite3
import tempfile
from pathlib import Path

import pytest

from pystocks.fundamentals_store import FundamentalsStore


def _make_store():
    tmp = tempfile.TemporaryDirectory()
    db_path = Path(tmp.name) / "pystocks.sqlite"
    store = FundamentalsStore(sqlite_path=db_path)
    return tmp, db_path, store


def _table_columns(con, table):
    return [r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()]


def test_ratios_split_into_section_tables_and_drop_fmt_fields():
    tmp, db_path, store = _make_store()
    try:
        snapshot = {
            "conid": "ratios_1",
            "scraped_at": "2026-02-23T12:00:00+00:00",
            "ratios": {
                "as_of_date": "2026-02-23",
                "title_vs": "vs category",
                "ratios": [
                    {
                        "name": "Price/Sales",
                        "name_tag": "price_sales",
                        "value": 3.63,
                        "vs": 0.0146,
                        "min": 2.08,
                        "max": 5.59,
                        "avg": 3.60,
                        "percentile": 40.47,
                        "value_fmt": "3.63",
                        "min_fmt": "2.08",
                        "max_fmt": "5.59",
                        "avg_fmt": "3.60",
                    }
                ],
                "financials": [
                    {
                        "name": "Sales Growth 1 Year",
                        "name_tag": "sales_growth_1_year",
                        "value": 5.04,
                        "vs": -0.15,
                        "min": 1.63,
                        "max": 10.66,
                        "avg": 5.67,
                        "percentile": 59.05,
                    }
                ],
                "fixed_income": [
                    {
                        "name": "Current Yield",
                        "name_tag": "current_yield",
                        "value": 3.17,
                        "vs": 0.05,
                        "min": 1.84,
                        "max": 5.41,
                        "avg": 3.05,
                        "percentile": 37.2,
                    }
                ],
                "dividend": [
                    {
                        "name": "Dividend Yield",
                        "name_tag": "dividend_yield",
                        "value": 2.35,
                        "vs": -0.08,
                        "min": 0.88,
                        "max": 4.43,
                        "avg": 2.58,
                        "percentile": 45.0,
                    }
                ],
                "zscore": [
                    {
                        "name": "1 Month",
                        "name_tag": "1_month",
                        "value": -0.04,
                        "value_fmt": "-0.04",
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
            assert "ratios_metrics" not in table_names
            assert "ratios_key_ratios" in table_names
            assert "ratios_financials" in table_names
            assert "ratios_fixed_income" in table_names
            assert "ratios_dividend" in table_names
            assert "ratios_zscore" in table_names

            ratios_row = con.execute(
                """
                SELECT metric_id, value_num, vs_num, min_num, max_num, avg_num, percentile_num
                FROM ratios_key_ratios
                WHERE conid = ?
                """,
                ["ratios_1"],
            ).fetchone()
            assert ratios_row == (
                "price_sales",
                pytest.approx(3.63),
                pytest.approx(0.0146),
                pytest.approx(2.08),
                pytest.approx(5.59),
                pytest.approx(3.60),
                pytest.approx(40.47),
            )

            zscore_row = con.execute(
                """
                SELECT metric_id, value_num, vs_num, min_num, max_num, avg_num, percentile_num
                FROM ratios_zscore
                WHERE conid = ?
                """,
                ["ratios_1"],
            ).fetchone()
            assert zscore_row == (
                "1_month",
                pytest.approx(-0.04),
                None,
                None,
                None,
                None,
                None,
            )

            for table in [
                "ratios_key_ratios",
                "ratios_financials",
                "ratios_fixed_income",
                "ratios_dividend",
                "ratios_zscore",
            ]:
                cols = _table_columns(con, table)
                assert "metric_name" not in cols
                assert "value_fmt" not in cols
                assert "min_fmt" not in cols
                assert "max_fmt" not in cols
                assert "avg_fmt" not in cols

            ratios_snapshot_cols = _table_columns(con, "ratios_snapshots")
            assert "source_file" not in ratios_snapshot_cols
            assert "title_vs" not in ratios_snapshot_cols

            for table in [
                "dividends_snapshots",
                "performance_snapshots",
                "ownership_snapshots",
                "esg_snapshots",
                "sentiment_snapshots",
            ]:
                assert "source_file" not in _table_columns(con, table)
        finally:
            con.close()
    finally:
        tmp.cleanup()
