from __future__ import annotations

import sqlite3
from pathlib import Path

from pystocks_next.storage.schema import (
    apply_migrations,
    current_schema_version,
)
from pystocks_next.storage.sqlite import connect_sqlite, initialize_operational_store


def test_initialize_operational_store_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "storage.sqlite"

    first_version = initialize_operational_store(db_path)
    second_version = initialize_operational_store(db_path)

    assert first_version == 12
    assert second_version == 12

    with connect_sqlite(db_path, read_only=True) as conn:
        assert current_schema_version(conn) == 12
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        foreign_keys = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        universe_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(universe_instruments)")
        }
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
            row[1] for row in conn.execute("PRAGMA table_info(profile_and_fees)")
        }
        holdings_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(holdings_asset_type)")
        }
        holdings_quality_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(holdings_debtor_quality)")
        }
        holdings_top10_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(holdings_top10)")
        }
        ratios_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(ratios_key_ratios)")
        }
        dividends_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(dividends_industry_metrics)")
        }
        morningstar_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(morningstar_summary)")
        }
        lipper_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(lipper_ratings)")
        }

    assert journal_mode == "wal"
    assert foreign_keys == 1
    assert "local_symbol" in universe_columns
    assert "cusip" in universe_columns
    assert "country" in universe_columns
    assert "under_conid" in universe_columns
    assert "is_prime_exch_id" in universe_columns
    assert "is_new_pdt" in universe_columns
    assert "assoc_entity_id" in universe_columns
    assert "fc_conid" in universe_columns
    assert "capture_batch_id" in columns
    assert "observed_at" in price_columns
    assert "event_signature" in dividend_columns
    assert "field_id" in profile_columns
    assert "bucket_id" in holdings_columns
    assert "bucket_id" in holdings_quality_columns
    assert "ticker" in holdings_top10_columns
    assert "rank" in holdings_top10_columns
    assert "conids_json" in holdings_top10_columns
    assert "vs_peers" in holdings_columns
    assert "vs_peers" in holdings_top10_columns
    assert "vs_peers" in ratios_columns
    assert "metric_id" in dividends_columns
    assert "metric_id" in morningstar_columns
    assert "universe_name" in lipper_columns


def test_apply_migrations_marks_partial_legacy_store_with_canonical_schema(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy_v1.sqlite"

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                description TEXT NOT NULL,
                applied_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE raw_payload_blobs (
                payload_hash TEXT PRIMARY KEY,
                payload_bytes BLOB NOT NULL,
                payload_size INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE raw_payload_observations (
                observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                payload_hash TEXT NOT NULL,
                source_family TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                conid TEXT,
                observed_at TEXT NOT NULL,
                source_as_of_date TEXT,
                UNIQUE(payload_hash, source_family, endpoint, conid, observed_at)
            )
            """
        )
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
        assert apply_migrations(conn) == [9, 10, 11, 12]
        assert current_schema_version(conn) == 12
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
    assert "profile_and_fees" in table_names
    assert "holdings_asset_type" in table_names
    assert "holdings_debtor_quality" in table_names
    assert "ratios_key_ratios" in table_names
    assert "dividends_industry_metrics" in table_names
    assert "morningstar_summary" in table_names
    assert "lipper_ratings" in table_names
