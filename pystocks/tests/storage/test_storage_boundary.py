import sqlite3

import pandas as pd
import pytest

from pystocks.storage.schema import init_storage
from pystocks.storage.txn import transaction


def test_init_storage_creates_schema_and_is_idempotent(tmp_path):
    db_path = tmp_path / "storage.sqlite"

    init_storage(db_path)
    init_storage(db_path)

    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

    assert "products" in tables
    assert "raw_payload_blobs" in tables
    assert "price_chart_series" in tables
    assert "dividends_events_series" in tables
    assert "ingest_runs" not in tables
    assert "ingest_run_endpoint_rollups" not in tables
    assert "endpoint_scalar_extras" not in tables


def test_storage_transaction_commits_multiple_writes(tmp_path):
    db_path = tmp_path / "storage.sqlite"

    with transaction(db_path) as tx:
        tx.execute(
            """
            INSERT INTO products (conid, symbol, updated_at)
            VALUES (?, ?, ?)
            """,
            ["1001", "TEST", "2026-01-01T00:00:00+00:00"],
        )
        tx.execute(
            """
            INSERT INTO price_chart_series (conid, effective_at, close)
            VALUES (?, ?, ?)
            """,
            ["1001", "2026-01-02", 101.5],
        )

    with sqlite3.connect(db_path) as conn:
        product_count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        price_count = conn.execute(
            "SELECT COUNT(*) FROM price_chart_series"
        ).fetchone()[0]

    assert product_count == 1
    assert price_count == 1


def test_storage_transaction_rolls_back_on_error(tmp_path):
    db_path = tmp_path / "storage.sqlite"

    with pytest.raises(RuntimeError, match="boom"):
        with transaction(db_path) as tx:
            tx.execute(
                """
                INSERT INTO products (conid, symbol, updated_at)
                VALUES (?, ?, ?)
                """,
                ["1002", "FAIL", "2026-01-01T00:00:00+00:00"],
            )
            raise RuntimeError("boom")

    with sqlite3.connect(db_path) as conn:
        product_count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]

    assert product_count == 0


def test_storage_transaction_write_frame_writes_analysis_outputs(tmp_path):
    db_path = tmp_path / "storage.sqlite"
    frame = pd.DataFrame({"factor_id": ["value"], "score": [1.25]})

    with transaction(db_path) as tx:
        tx.write_frame(
            "analysis_factor_scores", frame, if_exists="replace", index=False
        )

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT factor_id, score FROM analysis_factor_scores"
        ).fetchall()

    assert rows == [("value", 1.25)]
