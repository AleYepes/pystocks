from __future__ import annotations

import sqlite3
from pathlib import Path

from pystocks_next.storage.schema import (
    MIGRATIONS,
    apply_migrations,
    current_schema_version,
)
from pystocks_next.storage.sqlite import connect_sqlite, initialize_operational_store


def test_initialize_operational_store_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "storage.sqlite"

    first_version = initialize_operational_store(db_path)
    second_version = initialize_operational_store(db_path)

    assert first_version == 8
    assert second_version == 8

    with connect_sqlite(db_path, read_only=True) as conn:
        assert current_schema_version(conn) == 8
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        foreign_keys = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(raw_payload_observations)")
        }
        price_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(price_chart_series)")
        }
        dividend_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(dividends_events_series)")
        }
        profile_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(profile_and_fees_factors)")
        }
        holdings_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(holdings_asset_type_factors)")
        }
        holdings_quality_columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(holdings_debtor_quality_factors)"
            )
        }
        ratios_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(ratios_key_ratios)")
        }
        dividends_columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(dividends_industry_metrics_factors)"
            )
        }
        morningstar_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(morningstar_summary_factors)")
        }
        lipper_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(lipper_ratings)")
        }

    assert journal_mode == "wal"
    assert foreign_keys == 1
    assert "capture_batch_id" in columns
    assert "observed_at" in price_columns
    assert "event_signature" in dividend_columns
    assert "field_id" in profile_columns
    assert "bucket_id" in holdings_columns
    assert "bucket_id" in holdings_quality_columns
    assert "vs_num" in ratios_columns
    assert "metric_id" in dividends_columns
    assert "metric_id" in morningstar_columns
    assert "universe_name" in lipper_columns


def test_apply_migrations_upgrades_v1_store_through_supplementary_tables(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy_v1.sqlite"

    with sqlite3.connect(db_path) as conn:
        for statement in MIGRATIONS[0].statements:
            conn.execute(statement)
        conn.execute(
            """
            INSERT INTO schema_migrations (version, description, applied_at)
            VALUES (1, 'initial operational schema', CURRENT_TIMESTAMP)
            """
        )
        conn.execute(
            """
            INSERT INTO raw_payload_blobs (
                payload_hash, payload_bytes, payload_size, created_at
            ) VALUES ('hash-1', X'7B7D', 2, CURRENT_TIMESTAMP)
            """
        )
        conn.execute(
            """
            INSERT INTO raw_payload_observations (
                payload_hash, source_family, endpoint, conid, observed_at, source_as_of_date
            ) VALUES (
                'hash-1', 'ibkr', 'holdings', '123', '2026-01-02T00:00:00+00:00', '2025-12-31'
            )
            """
        )

    with connect_sqlite(db_path) as conn:
        assert apply_migrations(conn) == [2, 3, 4, 5, 6, 7, 8]
        assert current_schema_version(conn) == 8
        row = conn.execute(
            """
            SELECT source_as_of_date, capture_batch_id
            FROM raw_payload_observations
            WHERE payload_hash = 'hash-1'
            """
        ).fetchone()
        table_names = {
            table_row[0]
            for table_row in conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                """
            ).fetchall()
        }

    assert row["source_as_of_date"] == "2025-12-31"
    assert row["capture_batch_id"] is None
    assert "price_chart_series" in table_names
    assert "dividends_events_series" in table_names
    assert "supplementary_risk_free_daily" not in table_names
    assert "supplementary_world_bank_country_features" not in table_names
    assert "profile_and_fees_factors" in table_names
    assert "holdings_asset_type_factors" in table_names
    assert "holdings_debtor_quality_factors" in table_names
    assert "ratios_key_ratios" in table_names
    assert "dividends_industry_metrics_factors" in table_names
    assert "morningstar_summary_factors" in table_names
    assert "lipper_ratings" in table_names
