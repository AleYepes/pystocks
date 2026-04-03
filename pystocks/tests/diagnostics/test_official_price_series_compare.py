import sqlite3
import tempfile
from pathlib import Path

import pandas as pd

from pystocks.diagnostics.official_price_series_compare import (
    OfficialPriceStore,
    _load_conids_from_file,
    build_comparison_frames,
)


def test_load_conids_file_deduplicates_and_preserves_order():
    tmp = tempfile.TemporaryDirectory()
    try:
        path = Path(tmp.name) / "conids.txt"
        path.write_text("100\n\n200\n100\n 300 \n")
        assert _load_conids_from_file(path) == ["100", "200", "300"]
    finally:
        tmp.cleanup()


def test_official_store_upserts_series_and_snapshot_metrics():
    tmp = tempfile.TemporaryDirectory()
    try:
        db_path = Path(tmp.name) / "official.sqlite"
        store = OfficialPriceStore(sqlite_path=db_path)

        rows_1 = [
            {
                "effective_at": "2025-01-02",
                "price": 10.0,
                "open": 10.1,
                "high": 10.2,
                "low": 9.9,
                "close": 10.0,
            },
            {
                "effective_at": "2025-01-03",
                "price": 10.5,
                "open": 10.0,
                "high": 10.6,
                "low": 9.8,
                "close": 10.5,
            },
        ]
        raw_rows_1 = [{"date": "2025-01-02"}, {"date": "2025-01-03"}]
        result_1 = store.persist_price_data(
            conid="abc",
            observed_at="2026-02-27T18:00:00+00:00",
            request_payload={"duration_str": "10 Y"},
            raw_rows=raw_rows_1,
            rows=rows_1,
        )
        assert result_1["rows_written"] == 2
        assert result_1["payload_hash"] is not None

        rows_2 = [
            {
                "effective_at": "2025-01-02",
                "price": 10.2,
                "open": 10.1,
                "high": 10.4,
                "low": 9.9,
                "close": 10.2,
            },
            {
                "effective_at": "2025-01-03",
                "price": 10.5,
                "open": 10.0,
                "high": 10.6,
                "low": 9.8,
                "close": 10.5,
            },
        ]
        raw_rows_2 = [{"date": "2025-01-02"}, {"date": "2025-01-03"}]
        result_2 = store.persist_price_data(
            conid="abc",
            observed_at="2026-02-27T18:01:00+00:00",
            request_payload={"duration_str": "10 Y"},
            raw_rows=raw_rows_2,
            rows=rows_2,
        )
        assert result_2["rows_written"] == 2

        con = sqlite3.connect(db_path)
        try:
            series_count = con.execute(
                "SELECT COUNT(*) FROM price_chart_series WHERE conid = ?",
                ["abc"],
            ).fetchone()[0]
            assert series_count == 2

            updated_close = con.execute(
                """
                SELECT close
                FROM price_chart_series
                WHERE conid = ? AND effective_at = ?
                """,
                ["abc", "2025-01-02"],
            ).fetchone()[0]
            assert updated_close == 10.2

            snap = con.execute(
                """
                SELECT points_count, min_trade_date, max_trade_date
                FROM price_chart_snapshots
                WHERE conid = ? AND effective_at = ?
                """,
                ["abc", "2025-01-03"],
            ).fetchone()
            assert snap[0] == 2
            assert snap[1] == "2025-01-02"
            assert snap[2] == "2025-01-03"
        finally:
            con.close()
    finally:
        tmp.cleanup()


def test_build_comparison_frames_outputs_expected_diffs():
    official = pd.DataFrame(
        [
            {
                "conid": "x",
                "effective_at": "2025-01-02",
                "price": 10.0,
                "open": 9.8,
                "high": 10.2,
                "low": 9.7,
                "close": 10.0,
            },
            {
                "conid": "x",
                "effective_at": "2025-01-03",
                "price": 10.5,
                "open": 10.0,
                "high": 10.6,
                "low": 9.9,
                "close": 10.5,
            },
        ]
    )
    fundamentals = pd.DataFrame(
        [
            {
                "conid": "x",
                "effective_at": "2025-01-02",
                "price": 9.5,
                "open": 9.6,
                "high": 9.7,
                "low": 9.2,
                "close": 9.5,
            },
            {
                "conid": "x",
                "effective_at": "2025-01-03",
                "price": 10.0,
                "open": 9.9,
                "high": 10.2,
                "low": 9.8,
                "close": 10.0,
            },
        ]
    )

    detail, summary = build_comparison_frames(official, fundamentals)
    assert len(detail) == 2
    assert len(summary) == 1

    first = detail.iloc[0]
    assert first["close_official"] == 10.0
    assert first["close_fundamentals"] == 9.5
    assert first["close_diff"] == 0.5
    assert first["close_diff_abs"] == 0.5

    summary_row = summary.iloc[0]
    assert summary_row["overlap_rows"] == 2
    assert summary_row["close_mean_abs_diff"] == 0.5
