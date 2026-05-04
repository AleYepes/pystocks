from __future__ import annotations

from pathlib import Path

from pystocks_next.storage.schema import (
    current_schema_version,
)
from pystocks_next.storage.sqlite import connect_sqlite, initialize_operational_store


def test_initialize_operational_store_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "storage.sqlite"

    first_version = initialize_operational_store(db_path)
    second_version = initialize_operational_store(db_path)

    assert first_version == 1
    assert second_version == 1

    with connect_sqlite(db_path, read_only=True) as conn:
        assert current_schema_version(conn) == 1
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
            row[1] for row in conn.execute("PRAGMA table_info(profile_fields)")
        }
        profile_overview_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(profile_overview)")
        }
        profile_annual_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(profile_annual_report)")
        }
        profile_prospectus_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(profile_prospectus_report)")
        }
        profile_stylebox_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(profile_stylebox)")
        }
        holdings_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(holdings_asset_type)")
        }
        holdings_quality_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(holdings_debtor_quality)")
        }
        holdings_industry_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(holdings_industry)")
        }
        holdings_country_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(holdings_investor_country)")
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
    assert "management_expenses_ratio" in profile_overview_columns
    assert "field_id" in profile_annual_columns
    assert "field_id" in profile_prospectus_columns
    assert "stylebox_id" in profile_stylebox_columns
    assert "bucket_id" in holdings_columns
    assert "bucket_id" in holdings_quality_columns
    assert "industry_id" in holdings_industry_columns
    assert "code" in holdings_country_columns
    assert "ticker" in holdings_top10_columns
    assert "name" in holdings_top10_columns
    assert "rank" in holdings_top10_columns
    assert "conids_json" in holdings_top10_columns
    assert "vs_peers" in holdings_columns
    assert "vs_peers" in holdings_top10_columns
    assert "vs_peers" in ratios_columns
    assert "metric_id" in dividends_columns
    assert "metric_id" in morningstar_columns
    assert "title" in morningstar_columns
    assert "derived_quantitatively" in morningstar_columns
    assert "publish_date" in morningstar_columns
    assert "universe_name" in lipper_columns
