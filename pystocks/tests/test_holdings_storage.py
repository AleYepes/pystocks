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


def test_holdings_schema_uses_new_table_layout():
    tmp, db_path, store = _make_store()
    try:
        snapshot = {
            "conid": "holdings_schema_1",
            "scraped_at": "2026-02-24T12:00:00+00:00",
            "ratios": {
                "as_of_date": "2026-02-24",
            },
            "holdings": {
                "as_of_date": "2026-02-24",
                "top_10_weight": "38.39%",
                "top_10": [],
                "geographic": {"us": "100%"},
            },
        }
        result = store.persist_combined_snapshot(snapshot, source_file="holdings_schema.json")
        assert result["status"] == "ok"

        con = sqlite3.connect(db_path)
        try:
            snapshot_cols = _table_columns(con, "holdings_snapshots")
            assert "as_of_date" in snapshot_cols
            assert "source_file" not in snapshot_cols
            assert "top_10_weight" not in snapshot_cols

            table_names = {
                row[0]
                for row in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            }
            assert "holdings_asset_type" in table_names
            assert "holdings_industry" in table_names
            assert "holdings_currency" in table_names
            assert "holdings_investor_country" in table_names
            assert "holdings_debt_type" in table_names
            assert "holdings_debtor_quality" in table_names
            assert "holdings_maturity" in table_names
            assert "holdings_top10" in table_names
            assert "holdings_geographic_weights" in table_names
            assert "holdings_bucket_weights" not in table_names
            assert "holdings_top10_conids" not in table_names
        finally:
            con.close()
    finally:
        tmp.cleanup()


def test_holdings_top10_stores_weight_and_conids_in_single_table():
    tmp, db_path, store = _make_store()
    try:
        snapshot = {
            "conid": "holdings_top10_1",
            "scraped_at": "2026-02-24T12:00:00+00:00",
            "ratios": {
                "as_of_date": "2026-02-24",
            },
            "holdings": {
                "as_of_date": "2026-02-24",
                "top_10": [
                    {
                        "name": "NVIDIA CORPORATION",
                        "ticker": "NVDA",
                        "rank": 1,
                        "assets_pct": "7.83%",
                        "conids": [4815747, 13104788, 84223567],
                    }
                ],
            },
        }
        result = store.persist_combined_snapshot(snapshot)
        assert result["status"] == "ok"

        con = sqlite3.connect(db_path)
        try:
            cols = _table_columns(con, "holdings_top10")
            assert "holding_weight_num" in cols
            assert "holding_conids" in cols
            assert "assets_pct_text" not in cols
            assert "assets_pct_num" not in cols
            assert "rank_num" not in cols

            row = con.execute(
                """
                SELECT name, ticker, holding_weight_num, holding_conids
                FROM holdings_top10
                WHERE conid = ?
                """,
                ["holdings_top10_1"],
            ).fetchone()
            assert row[0] == "NVIDIA CORPORATION"
            assert row[1] == "NVDA"
            assert row[2] == pytest.approx(0.0783)
            assert row[3] == "4815747,13104788,84223567"
        finally:
            con.close()
    finally:
        tmp.cleanup()


def test_holdings_debt_maturity_asset_type_use_static_columns_with_direct_mapping():
    tmp, db_path, store = _make_store()
    try:
        snapshot_1 = {
            "conid": "holdings_pivot_1",
            "scraped_at": "2026-02-24T12:00:00+00:00",
            "ratios": {
                "as_of_date": "2026-02-24",
            },
            "holdings": {
                "as_of_date": "2026-02-24",
                "allocation_self": [
                    {"name": "Equity", "weight": 99.9088},
                    {"name": "Cash", "formatted_weight": "0.05%"},
                    {"name": "Fixed Income", "formatted_weight": "0.03%"},
                    {"name": "Alternatives", "formatted_weight": "0.01%"},
                ],
                "industry": [
                    {"name": "Technology", "weight": 44.8681, "vs": 48.34125625},
                ],
                "currency": [
                    {"name": "US Dollar", "weight": 99.9604, "vs": 107.984995833333, "code": "USD"},
                ],
                "investor_country": [
                    {"name": "United States", "weight": 97.3418, "vs": 104.736089583333, "country_code": "US"},
                ],
                "debt_type": [
                    {"name": "Sovereign Bond", "weight": "20%", "vs": "19.5%"},
                ],
                "debtor": [
                    {"name": "% Quality/AA", "weight": "15%", "vs": "14.4%"},
                    {"name": "% Quality/BBB", "weight": "8%", "vs": "9.1%"},
                    {"name": "% Quality Not Rated", "weight": "2%", "vs": "3.3%"},
                ],
                "maturity": [
                    {"name": "% Maturity 1 to 3 Years", "weight": "12.5%", "vs": "11.3%"},
                    {"name": "% Maturity Less than 1 Year", "weight": "5.4%", "vs": "4.1%"},
                ],
                "geographic": {
                    "us": "97.34%",
                    "eu": "1.89%",
                },
            },
        }
        snapshot_2 = {
            "conid": "holdings_pivot_1",
            "scraped_at": "2026-02-25T12:00:00+00:00",
            "ratios": {
                "as_of_date": "2026-02-25",
            },
            "holdings": {
                "as_of_date": "2026-02-25",
                "currency": [
                    {"name": "Euro", "weight": "4.5%", "vs": "5.2%", "code": "EUR"},
                ],
                "investor_country": [
                    {"name": "Ireland", "weight": "1.2%", "vs": "1.0%", "country_code": "IE"},
                ],
                "geographic": {
                    "apac": "0.77%",
                },
            },
        }

        result_1 = store.persist_combined_snapshot(snapshot_1)
        result_2 = store.persist_combined_snapshot(snapshot_2)
        assert result_1["status"] == "ok"
        assert result_2["status"] == "ok"

        con = sqlite3.connect(db_path)
        try:
            asset_cols = _table_columns(con, "holdings_asset_type")
            assert "equity" in asset_cols
            assert "cash" in asset_cols
            assert "fixed_income" in asset_cols
            assert "other" in asset_cols
            assert "equity_vs_num" not in asset_cols

            industry_cols = _table_columns(con, "holdings_industry")
            assert "industry" in industry_cols
            assert "value_num" in industry_cols
            assert "industry_avg" in industry_cols
            assert "technology" not in industry_cols

            currency_cols = _table_columns(con, "holdings_currency")
            assert "currency" in currency_cols
            assert "value_num" in currency_cols
            assert "industry_avg" in currency_cols
            assert "code" in currency_cols
            assert "us_dollar" not in currency_cols
            assert "euro" not in currency_cols

            investor_cols = _table_columns(con, "holdings_investor_country")
            assert "country" in investor_cols
            assert "country_code" in investor_cols
            assert "value_num" in investor_cols
            assert "industry_avg" in investor_cols
            assert "united_states" not in investor_cols
            assert "ireland" not in investor_cols

            debt_type_cols = _table_columns(con, "holdings_debt_type")
            assert "debt_type" in debt_type_cols
            assert "value_num" in debt_type_cols
            assert "industry_avg" in debt_type_cols
            assert "quality_aa" not in debt_type_cols

            debtor_quality_cols = _table_columns(con, "holdings_debtor_quality")
            assert "quality_aa" in debtor_quality_cols
            assert "quality_aa_industry_avg" in debtor_quality_cols
            assert "quality_bbb" in debtor_quality_cols
            assert "quality_not_rated" in debtor_quality_cols
            assert "debtor" not in debtor_quality_cols
            assert "value_num" not in debtor_quality_cols

            maturity_cols = _table_columns(con, "holdings_maturity")
            assert "maturity_1_to_3_years" in maturity_cols
            assert "maturity_1_to_3_years_industry_avg" in maturity_cols
            assert "maturity_less_than_1_year" in maturity_cols
            assert "maturity_other" in maturity_cols
            assert "maturity" not in maturity_cols
            assert "value_num" not in maturity_cols

            geographic_cols = _table_columns(con, "holdings_geographic_weights")
            assert "region" in geographic_cols
            assert "value_num" in geographic_cols
            assert "us" not in geographic_cols
            assert "apac" not in geographic_cols

            asset_row = con.execute(
                """
                SELECT equity, cash, fixed_income, other
                FROM holdings_asset_type
                WHERE conid = ? AND effective_at = ?
                """,
                ["holdings_pivot_1", "2026-02-24"],
            ).fetchone()
            assert asset_row[0] == pytest.approx(0.999088)
            assert asset_row[1] == pytest.approx(0.0005)
            assert asset_row[2] == pytest.approx(0.0003)
            assert asset_row[3] == pytest.approx(0.0001)

            debt_type_row = con.execute(
                """
                SELECT debt_type, value_num, industry_avg
                FROM holdings_debt_type
                WHERE conid = ? AND effective_at = ?
                """,
                ["holdings_pivot_1", "2026-02-24"],
            ).fetchone()
            assert debt_type_row[0] == "Sovereign Bond"
            assert debt_type_row[1] == pytest.approx(0.2)
            assert debt_type_row[2] == pytest.approx(0.195)

            debtor_quality_row = con.execute(
                """
                SELECT quality_aa, quality_aa_industry_avg, quality_bbb, quality_bbb_industry_avg,
                       quality_not_rated, quality_not_rated_industry_avg
                FROM holdings_debtor_quality
                WHERE conid = ? AND effective_at = ?
                """,
                ["holdings_pivot_1", "2026-02-24"],
            ).fetchone()
            assert debtor_quality_row[0] == pytest.approx(0.15)
            assert debtor_quality_row[1] == pytest.approx(0.144)
            assert debtor_quality_row[2] == pytest.approx(0.08)
            assert debtor_quality_row[3] == pytest.approx(0.091)
            assert debtor_quality_row[4] == pytest.approx(0.02)
            assert debtor_quality_row[5] == pytest.approx(0.033)

            maturity_row = con.execute(
                """
                SELECT maturity_1_to_3_years, maturity_1_to_3_years_industry_avg,
                       maturity_less_than_1_year, maturity_less_than_1_year_industry_avg
                FROM holdings_maturity
                WHERE conid = ? AND effective_at = ?
                """,
                ["holdings_pivot_1", "2026-02-24"],
            ).fetchone()
            assert maturity_row[0] == pytest.approx(0.125)
            assert maturity_row[1] == pytest.approx(0.113)
            assert maturity_row[2] == pytest.approx(0.054)
            assert maturity_row[3] == pytest.approx(0.041)

            currency_row = con.execute(
                """
                SELECT currency, value_num, industry_avg, code
                FROM holdings_currency
                WHERE conid = ? AND effective_at = ?
                """,
                ["holdings_pivot_1", "2026-02-24"],
            ).fetchone()
            assert currency_row[0] == "US Dollar"
            assert currency_row[1] == pytest.approx(0.999604)
            assert currency_row[2] == pytest.approx(1.07984995833333)
            assert currency_row[3] == "USD"

            currency_row_2 = con.execute(
                """
                SELECT currency, value_num, industry_avg, code
                FROM holdings_currency
                WHERE conid = ? AND effective_at = ?
                """,
                ["holdings_pivot_1", "2026-02-25"],
            ).fetchone()
            assert currency_row_2[0] == "Euro"
            assert currency_row_2[1] == pytest.approx(0.045)
            assert currency_row_2[2] == pytest.approx(0.052)
            assert currency_row_2[3] == "EUR"

            geographic_rows = con.execute(
                """
                SELECT region, value_num
                FROM holdings_geographic_weights
                WHERE conid = ? AND effective_at = ?
                ORDER BY region
                """,
                ["holdings_pivot_1", "2026-02-24"],
            ).fetchall()
            assert geographic_rows[0][0] == "eu"
            assert geographic_rows[0][1] == pytest.approx(0.0189)
            assert geographic_rows[1][0] == "us"
            assert geographic_rows[1][1] == pytest.approx(0.9734)
        finally:
            con.close()
    finally:
        tmp.cleanup()


def test_holdings_snapshot_accepts_camel_case_as_of_date():
    tmp, db_path, store = _make_store()
    try:
        snapshot = {
            "conid": "holdings_asof_camel_1",
            "scraped_at": "2026-02-26T12:00:00+00:00",
            "ratios": {
                "as_of_date": "2026-02-21",
            },
            "holdings": {
                "asOfDate": "2026-02-20",
                "currency": [
                    {"name": "US Dollar", "weight": "100%", "vs": "95%", "code": "USD"},
                ],
            },
        }
        result = store.persist_combined_snapshot(snapshot)
        assert result["status"] == "ok"

        con = sqlite3.connect(db_path)
        try:
            row = con.execute(
                """
                SELECT effective_at, as_of_date
                FROM holdings_snapshots
                WHERE conid = ?
                """,
                ["holdings_asof_camel_1"],
            ).fetchone()
            assert row[0] == "2026-02-21"
            assert row[1] == "2026-02-20"
        finally:
            con.close()
    finally:
        tmp.cleanup()
