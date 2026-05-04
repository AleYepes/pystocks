from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    description: str
    statements: tuple[str, ...]


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version=1,
        description="canonical operational schema",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                description TEXT NOT NULL,
                applied_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS raw_payload_blobs (
                payload_hash TEXT PRIMARY KEY,
                payload_bytes BLOB NOT NULL,
                payload_size INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS raw_payload_observations (
                observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                payload_hash TEXT NOT NULL REFERENCES raw_payload_blobs(payload_hash),
                source_family TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                conid TEXT,
                observed_at TEXT NOT NULL,
                source_as_of_date TEXT,
                capture_batch_id TEXT,
                UNIQUE(payload_hash, source_family, endpoint, conid, observed_at)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS universe_instruments (
                conid TEXT PRIMARY KEY,
                symbol TEXT,
                name TEXT,
                exchange TEXT,
                isin TEXT,
                currency TEXT,
                product_type TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL,
                local_symbol TEXT,
                primary_listing_exchange TEXT,
                cusip TEXT,
                country TEXT,
                under_conid TEXT,
                is_prime_exch_id TEXT,
                is_new_pdt TEXT,
                assoc_entity_id TEXT,
                fc_conid TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS universe_exclusions (
                conid TEXT PRIMARY KEY REFERENCES universe_instruments(conid),
                reason TEXT NOT NULL,
                effective_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS price_chart_series (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                payload_hash TEXT NOT NULL REFERENCES raw_payload_blobs(payload_hash),
                capture_batch_id TEXT,
                price REAL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                debug_mismatch INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (conid, effective_at)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS dividends_events_series (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                event_signature TEXT NOT NULL,
                effective_at TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                payload_hash TEXT NOT NULL REFERENCES raw_payload_blobs(payload_hash),
                capture_batch_id TEXT,
                amount REAL,
                currency TEXT,
                description TEXT,
                event_type TEXT,
                declaration_date TEXT,
                record_date TEXT,
                payment_date TEXT,
                event_date TEXT,
                PRIMARY KEY (conid, event_signature)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS holdings_snapshots (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                payload_hash TEXT NOT NULL REFERENCES raw_payload_blobs(payload_hash),
                capture_batch_id TEXT,
                as_of_date TEXT,
                PRIMARY KEY (conid, effective_at)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS holdings_asset_type (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                bucket_id TEXT NOT NULL,
                value_num REAL,
                vs_peers REAL,
                PRIMARY KEY (conid, effective_at, bucket_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS holdings_debtor_quality (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                bucket_id TEXT NOT NULL,
                value_num REAL,
                vs_peers REAL,
                PRIMARY KEY (conid, effective_at, bucket_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS holdings_maturity (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                bucket_id TEXT NOT NULL,
                value_num REAL,
                vs_peers REAL,
                PRIMARY KEY (conid, effective_at, bucket_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS holdings_industry (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                industry_id TEXT NOT NULL,
                value_num REAL,
                vs_peers REAL,
                PRIMARY KEY (conid, effective_at, industry_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS holdings_currency (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                code TEXT,
                name TEXT NOT NULL,
                value_num REAL,
                vs_peers REAL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS holdings_investor_country (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                code TEXT,
                name TEXT NOT NULL,
                value_num REAL,
                vs_peers REAL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS holdings_geographic_weights (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                region_id TEXT NOT NULL,
                value_num REAL,
                vs_peers REAL,
                PRIMARY KEY (conid, effective_at, region_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS holdings_debt_type (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                debt_type_id TEXT NOT NULL,
                value_num REAL,
                vs_peers REAL,
                PRIMARY KEY (conid, effective_at, debt_type_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS holdings_top10 (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                name TEXT NOT NULL,
                ticker TEXT,
                rank INTEGER,
                holding_weight_num REAL,
                vs_peers REAL,
                conids_json TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ratios_snapshots (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                payload_hash TEXT NOT NULL REFERENCES raw_payload_blobs(payload_hash),
                capture_batch_id TEXT,
                as_of_date TEXT,
                PRIMARY KEY (conid, effective_at)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ratios_key_ratios (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                metric_id TEXT NOT NULL,
                value_num REAL,
                vs_peers REAL,
                PRIMARY KEY (conid, effective_at, metric_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ratios_financials (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                metric_id TEXT NOT NULL,
                value_num REAL,
                vs_peers REAL,
                PRIMARY KEY (conid, effective_at, metric_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ratios_fixed_income (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                metric_id TEXT NOT NULL,
                value_num REAL,
                vs_peers REAL,
                PRIMARY KEY (conid, effective_at, metric_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ratios_dividend (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                metric_id TEXT NOT NULL,
                value_num REAL,
                vs_peers REAL,
                PRIMARY KEY (conid, effective_at, metric_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ratios_zscore (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                metric_id TEXT NOT NULL,
                value_num REAL,
                vs_peers REAL,
                PRIMARY KEY (conid, effective_at, metric_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS dividends_snapshots (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                payload_hash TEXT NOT NULL REFERENCES raw_payload_blobs(payload_hash),
                capture_batch_id TEXT,
                as_of_date TEXT,
                PRIMARY KEY (conid, effective_at)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS dividends_industry_metrics (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                metric_id TEXT NOT NULL,
                value_num REAL,
                currency TEXT,
                PRIMARY KEY (conid, effective_at, metric_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS morningstar_snapshots (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                payload_hash TEXT NOT NULL REFERENCES raw_payload_blobs(payload_hash),
                capture_batch_id TEXT,
                as_of_date TEXT,
                q_full_report_id TEXT,
                PRIMARY KEY (conid, effective_at)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS morningstar_summary (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                metric_id TEXT NOT NULL,
                value_text TEXT,
                value_num REAL,
                title TEXT,
                derived_quantitatively INTEGER,
                publish_date TEXT,
                PRIMARY KEY (conid, effective_at, metric_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lipper_ratings_snapshots (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                payload_hash TEXT NOT NULL REFERENCES raw_payload_blobs(payload_hash),
                capture_batch_id TEXT,
                universe_count INTEGER NOT NULL,
                PRIMARY KEY (conid, effective_at)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS lipper_ratings (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                period TEXT NOT NULL,
                metric_id TEXT NOT NULL,
                value_num REAL,
                rating_label TEXT,
                universe_name TEXT,
                universe_as_of_date TEXT,
                PRIMARY KEY (conid, effective_at, universe_name, period, metric_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS supplementary_fetch_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                status TEXT NOT NULL,
                record_count INTEGER NOT NULL,
                min_key TEXT,
                max_key TEXT,
                notes TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS supplementary_risk_free_sources (
                series_id TEXT NOT NULL,
                source_name TEXT NOT NULL,
                economy_code TEXT,
                trade_date TEXT NOT NULL,
                nominal_rate REAL,
                observed_at TEXT NOT NULL,
                PRIMARY KEY (series_id, trade_date)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS supplementary_world_bank_raw (
                economy_code TEXT NOT NULL,
                indicator_id TEXT NOT NULL,
                year INTEGER NOT NULL,
                value REAL,
                observed_at TEXT NOT NULL,
                PRIMARY KEY (economy_code, indicator_id, year)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS profile_snapshots (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                payload_hash TEXT NOT NULL REFERENCES raw_payload_blobs(payload_hash),
                capture_batch_id TEXT,
                source_as_of_date TEXT,
                PRIMARY KEY (conid, effective_at)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS profile_overview (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                symbol TEXT,
                objective TEXT,
                jap_fund_warning INTEGER,
                management_expenses_ratio REAL,
                PRIMARY KEY (conid, effective_at)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS profile_fields (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                field_id TEXT NOT NULL,
                value_text TEXT,
                value_num REAL,
                value_date TEXT,
                PRIMARY KEY (conid, effective_at, field_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS profile_annual_report (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                field_id TEXT NOT NULL,
                value_num REAL,
                is_summary INTEGER,
                PRIMARY KEY (conid, effective_at, field_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS profile_prospectus_report (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                field_id TEXT NOT NULL,
                value_num REAL,
                is_summary INTEGER,
                PRIMARY KEY (conid, effective_at, field_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS profile_themes (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                theme_id TEXT NOT NULL,
                PRIMARY KEY (conid, effective_at, theme_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS profile_stylebox (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                stylebox_id TEXT NOT NULL,
                x_index INTEGER,
                y_index INTEGER,
                x_label TEXT,
                y_label TEXT,
                x_tag TEXT,
                y_tag TEXT,
                PRIMARY KEY (conid, effective_at, stylebox_id)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_raw_payload_observations_endpoint
            ON raw_payload_observations(endpoint, observed_at)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_raw_payload_observations_batch
            ON raw_payload_observations(capture_batch_id, endpoint, observed_at)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_universe_instruments_active
            ON universe_instruments(is_active, conid)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_price_chart_series_effective_at
            ON price_chart_series(effective_at, conid)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_dividends_events_series_effective_at
            ON dividends_events_series(effective_at, conid)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_holdings_snapshots_observed_at
            ON holdings_snapshots(observed_at, conid)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_ratios_snapshots_observed_at
            ON ratios_snapshots(observed_at, conid)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_dividends_snapshots_observed_at
            ON dividends_snapshots(observed_at, conid)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_morningstar_snapshots_observed_at
            ON morningstar_snapshots(observed_at, conid)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_lipper_ratings_snapshots_observed_at
            ON lipper_ratings_snapshots(observed_at, conid)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_supplementary_fetch_log_dataset
            ON supplementary_fetch_log(dataset, observed_at)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_supplementary_risk_free_sources_trade_date
            ON supplementary_risk_free_sources(trade_date, series_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_supplementary_world_bank_raw_year
            ON supplementary_world_bank_raw(year, economy_code)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_profile_snapshots_observed_at
            ON profile_snapshots(observed_at, conid)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_profile_fields_effective_at
            ON profile_fields(effective_at, conid)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_profile_annual_report_effective_at
            ON profile_annual_report(effective_at, conid)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_profile_prospectus_report_effective_at
            ON profile_prospectus_report(effective_at, conid)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_holdings_industry_effective_at
            ON holdings_industry(effective_at, conid)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_holdings_currency_effective_at
            ON holdings_currency(effective_at, conid)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_holdings_investor_country_effective_at
            ON holdings_investor_country(effective_at, conid)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_holdings_geographic_weights_effective_at
            ON holdings_geographic_weights(effective_at, conid)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_holdings_debt_type_effective_at
            ON holdings_debt_type(effective_at, conid)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_holdings_top10_effective_at
            ON holdings_top10(effective_at, conid)
            """,
        ),
    ),
)


def _ensure_migration_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            description TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
        """
    )


def applied_migration_versions(conn: sqlite3.Connection) -> set[int]:
    _ensure_migration_table(conn)
    rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
    return {int(row[0]) for row in rows}


def current_schema_version(conn: sqlite3.Connection) -> int:
    versions = applied_migration_versions(conn)
    return max(versions) if versions else 0


def apply_migrations(conn: sqlite3.Connection) -> list[int]:
    _ensure_migration_table(conn)
    applied = applied_migration_versions(conn)
    newly_applied: list[int] = []

    for migration in MIGRATIONS:
        if migration.version in applied:
            continue

        for statement in migration.statements:
            conn.execute(statement)
        conn.execute(
            """
            INSERT INTO schema_migrations (version, description, applied_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
            (migration.version, migration.description),
        )
        newly_applied.append(migration.version)

    return newly_applied
