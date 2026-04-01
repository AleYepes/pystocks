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


def test_dividends_series_and_metrics_are_refactored_to_new_schema():
    tmp, db_path, store = _make_store()
    try:
        snapshot = {
            "conid": "divs_1",
            "scraped_at": "2026-03-01T10:00:00+00:00",
            "ratios": {"as_of_date": "2026-02-27"},
            "dividends": {
                "industry_average": {
                    "dividend_yield": "1.22%",
                    "annual_dividend": "25.65",
                    "paying_companies": 12,
                    "paying_companies_percent": "33.3%",
                },
                "industry_comparison": {
                    "content": [
                        {
                            "search_id": "div_yield",
                            "value": 0.0085,
                            "formatted_value": "0.85%",
                        },
                        {
                            "search_id": "div_per_share",
                            "value": 5.585599,
                            "formatted_value": "5.59",
                        },
                    ]
                },
                "last_payed_dividend_currency": "USD",
                "history": {
                    "series": [
                        {
                            "name": "dividends",
                            "plotData": [
                                {
                                    "x": "2024-01-10",
                                    "amount": 1.11,
                                    "formatted_amount": "1.11 EUR",
                                    "type": "ACTUAL",
                                    "description": "Regular Dividend",
                                    "ex_dividend_date": {
                                        "y": 2024,
                                        "m": "JAN",
                                        "d": 10,
                                    },
                                    "declaration_date": {"y": 2024, "m": "JAN", "d": 9},
                                    "record_date": {"y": 2024, "m": "JAN", "d": 11},
                                    "payment_date": {"y": 2024, "m": "FEB", "d": 1},
                                },
                                {
                                    "x": "2024-04-10",
                                    "amount": 1.22,
                                    "type": "ACTUAL",
                                    "description": "Regular Dividend",
                                    "ex_dividend_date": {
                                        "y": 2024,
                                        "m": "APR",
                                        "d": 10,
                                    },
                                    "declaration_date": {"y": 2024, "m": "APR", "d": 9},
                                    "record_date": {"y": 2024, "m": "APR", "d": 11},
                                    "payment_date": {"y": 2024, "m": "MAY", "d": 1},
                                },
                            ],
                        },
                        {
                            "name": "price",
                            "plotData": [
                                {
                                    "x": "2024-01-10",
                                    "y": 100.0,
                                    "open": 99.0,
                                    "high": 101.0,
                                    "low": 98.0,
                                },
                            ],
                        },
                    ]
                },
            },
        }

        result = store.persist_combined_snapshot(snapshot)
        assert result["status"] == "ok"
        assert result["series_latest_rows_upserted"] == 0
        assert result["series_raw_rows_written"] == 2

        con = sqlite3.connect(db_path)
        try:
            table_names = {
                row[0]
                for row in con.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            assert "dividends_events_series_latest" not in table_names
            assert "dividends_events_series" in table_names

            snapshot_cols = _table_columns(con, "dividends_snapshots")
            assert "response_type" not in snapshot_cols
            assert "has_history" not in snapshot_cols
            assert "history_points" not in snapshot_cols
            assert "embedded_price_points" not in snapshot_cols
            assert "no_div_data_marker" not in snapshot_cols
            assert "no_div_data_period" not in snapshot_cols
            assert "no_dividend_text" not in snapshot_cols
            assert "last_paid_date" not in snapshot_cols
            assert "last_paid_amount" not in snapshot_cols
            assert "last_paid_currency" not in snapshot_cols
            assert "dividend_yield" not in snapshot_cols
            assert "annual_dividend" not in snapshot_cols
            assert "paying_companies" not in snapshot_cols
            assert "paying_companies_percent" not in snapshot_cols
            assert "dividend_ttm" not in snapshot_cols
            assert "dividend_yield_ttm" not in snapshot_cols

            metrics_cols = _table_columns(con, "dividends_industry_metrics")
            assert "metric_id" not in metrics_cols
            assert "value_num" not in metrics_cols
            assert "formatted_value" not in metrics_cols

            metrics = con.execute(
                """
                SELECT dividend_yield, annual_dividend, dividend_ttm, dividend_yield_ttm, currency
                FROM dividends_industry_metrics
                WHERE conid = ? AND effective_at = ?
                """,
                ["divs_1", "2026-02-27"],
            ).fetchone()
            assert metrics[0] == pytest.approx(0.0122)
            assert metrics[1] == pytest.approx(25.65)
            assert metrics[2] == pytest.approx(5.585599)
            assert metrics[3] == pytest.approx(0.0085)
            assert metrics[4] == "USD"

            series_cols = _table_columns(con, "dividends_events_series")
            assert "row_key" not in series_cols
            assert "observed_at" not in series_cols
            assert "payload_hash" not in series_cols
            assert "inserted_at" not in series_cols
            assert "trade_date" not in series_cols
            assert "event_date" not in series_cols

            rows = con.execute(
                """
                SELECT effective_at, amount, currency, description, event_type,
                       declaration_date, record_date, payment_date
                FROM dividends_events_series
                WHERE conid = ?
                ORDER BY effective_at
                """,
                ["divs_1"],
            ).fetchall()
            assert rows == [
                (
                    "2024-01-10",
                    1.11,
                    "EUR",
                    "Regular Dividend",
                    "ACTUAL",
                    "2024-01-09",
                    "2024-01-11",
                    "2024-02-01",
                ),
                (
                    "2024-04-10",
                    1.22,
                    "USD",
                    "Regular Dividend",
                    "ACTUAL",
                    "2024-04-09",
                    "2024-04-11",
                    "2024-05-01",
                ),
            ]

            payload_hash = con.execute(
                """
                SELECT payload_hash
                FROM dividends_snapshots
                WHERE conid = ? AND effective_at = ?
                """,
                ["divs_1", "2026-02-27"],
            ).fetchone()[0]
            payload = store._load_blob_payload(payload_hash)
            history = payload.get("history", {})
            series_names = [
                str(s.get("name") or s.get("title") or "").strip().lower()
                for s in history.get("series", [])
                if isinstance(s, dict)
            ]
            assert "price" not in series_names
        finally:
            con.close()
    finally:
        tmp.cleanup()


def test_dividends_mismatch_logging_only_triggers_for_gt_one_day(caplog):
    tmp, _, store = _make_store()
    try:
        caplog.set_level("WARNING")
        snapshot = {
            "conid": "divs_warn",
            "scraped_at": "2026-03-01T10:00:00+00:00",
            "ratios": {"as_of_date": "2026-02-27"},
            "dividends": {
                "history": {
                    "series": [
                        {
                            "name": "dividends",
                            "plotData": [
                                {
                                    "x": "2024-01-10",
                                    "amount": 1.0,
                                    "type": "ACTUAL",
                                    "ex_dividend_date": {"y": 2024, "m": "JAN", "d": 9},
                                },
                                {
                                    "x": "2024-02-10",
                                    "amount": 1.1,
                                    "type": "ACTUAL",
                                    "ex_dividend_date": {"y": 2024, "m": "FEB", "d": 7},
                                },
                            ],
                        }
                    ]
                }
            },
        }

        result = store.persist_combined_snapshot(snapshot)
        assert result["status"] == "ok"

        mismatch_logs = [
            rec.message
            for rec in caplog.records
            if "dividends date mismatch between trade_date and event_date"
            in rec.message
        ]
        assert len(mismatch_logs) == 1
    finally:
        tmp.cleanup()


def test_dividends_events_series_is_idempotent_across_changed_snapshot_dates():
    tmp, db_path, store = _make_store()
    try:
        base_dividends = {
            "history": {
                "series": [
                    {
                        "name": "dividends",
                        "plotData": [
                            {
                                "x": "2024-01-10",
                                "amount": 1.0,
                                "type": "ACTUAL",
                                "description": "Regular Dividend",
                                "ex_dividend_date": {"y": 2024, "m": "JAN", "d": 10},
                            },
                            {
                                "x": "2024-04-10",
                                "amount": 1.2,
                                "type": "ACTUAL",
                                "description": "Regular Dividend",
                                "ex_dividend_date": {"y": 2024, "m": "APR", "d": 10},
                            },
                        ],
                    }
                ]
            }
        }
        snapshot_1 = {
            "conid": "divs_repeat",
            "scraped_at": "2026-03-01T10:00:00+00:00",
            "ratios": {"as_of_date": "2026-02-27"},
            "dividends": base_dividends,
        }
        snapshot_2 = {
            "conid": "divs_repeat",
            "scraped_at": "2026-04-01T10:00:00+00:00",
            "ratios": {"as_of_date": "2026-03-31"},
            "dividends": {
                **base_dividends,
                "industry_average": {"annual_dividend": "2.2"},
            },
        }

        result_1 = store.persist_combined_snapshot(snapshot_1)
        result_2 = store.persist_combined_snapshot(snapshot_2)
        assert result_1["status"] == "ok"
        assert result_2["status"] == "ok"

        con = sqlite3.connect(db_path)
        try:
            rows = con.execute(
                """
                SELECT effective_at, amount, description
                FROM dividends_events_series
                WHERE conid = ?
                ORDER BY effective_at
                """,
                ["divs_repeat"],
            ).fetchall()
            assert rows == [
                ("2024-01-10", 1.0, "Regular Dividend"),
                ("2024-04-10", 1.2, "Regular Dividend"),
            ]
        finally:
            con.close()
    finally:
        tmp.cleanup()
