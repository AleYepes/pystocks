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
        version=9,
        description="canonical tall-only operational schema",
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
                updated_at TEXT NOT NULL
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
                industry TEXT NOT NULL,
                value_num REAL,
                vs_peers REAL,
                PRIMARY KEY (conid, effective_at, industry)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS holdings_currency (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                code TEXT,
                currency TEXT,
                value_num REAL,
                vs_peers REAL,
                PRIMARY KEY (conid, effective_at, code, currency)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS holdings_investor_country (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                country_code TEXT,
                country TEXT,
                value_num REAL,
                vs_peers REAL,
                PRIMARY KEY (conid, effective_at, country_code, country)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS holdings_geographic_weights (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                region TEXT NOT NULL,
                value_num REAL,
                vs_peers REAL,
                PRIMARY KEY (conid, effective_at, region)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS holdings_debt_type (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                debt_type TEXT NOT NULL,
                value_num REAL,
                vs_peers REAL,
                PRIMARY KEY (conid, effective_at, debt_type)
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
        ),
    ),
    Migration(
        version=10,
        description="add supplementary source tables",
        statements=(
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
        ),
    ),
    Migration(
        version=11,
        description="add profile and fees snapshot capture",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS profile_and_fees_snapshots (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                payload_hash TEXT NOT NULL REFERENCES raw_payload_blobs(payload_hash),
                capture_batch_id TEXT,
                PRIMARY KEY (conid, effective_at)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS profile_and_fees (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                profile_json TEXT,
                PRIMARY KEY (conid, effective_at)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_profile_and_fees_snapshots_observed_at
            ON profile_and_fees_snapshots(observed_at, conid)
            """,
        ),
    ),
    Migration(
        version=12,
        description="retain full IBKR product catalog fields",
        statements=(
            """
            ALTER TABLE universe_instruments
            ADD COLUMN local_symbol TEXT
            """,
            """
            ALTER TABLE universe_instruments
            ADD COLUMN primary_listing_exchange TEXT
            """,
            """
            ALTER TABLE universe_instruments
            ADD COLUMN cusip TEXT
            """,
            """
            ALTER TABLE universe_instruments
            ADD COLUMN country TEXT
            """,
            """
            ALTER TABLE universe_instruments
            ADD COLUMN under_conid TEXT
            """,
            """
            ALTER TABLE universe_instruments
            ADD COLUMN is_prime_exch_id TEXT
            """,
            """
            ALTER TABLE universe_instruments
            ADD COLUMN is_new_pdt TEXT
            """,
            """
            ALTER TABLE universe_instruments
            ADD COLUMN assoc_entity_id TEXT
            """,
            """
            ALTER TABLE universe_instruments
            ADD COLUMN fc_conid TEXT
            """,
        ),
    ),
    Migration(
        version=13,
        description="preserve morningstar summary metadata",
        statements=(
            """
            ALTER TABLE morningstar_summary
            ADD COLUMN title TEXT
            """,
            """
            ALTER TABLE morningstar_summary
            ADD COLUMN is_quantitative INTEGER
            """,
            """
            ALTER TABLE morningstar_summary
            ADD COLUMN publish_date TEXT
            """,
        ),
    ),
    Migration(
        version=14,
        description="rename morningstar summary is_quantitative to derived_quantitatively",
        statements=(
            """
            ALTER TABLE morningstar_summary
            RENAME COLUMN is_quantitative TO derived_quantitatively
            """,
        ),
    ),
    Migration(
        version=15,
        description="split profile and fees endpoint into source-shaped profile tables",
        statements=(
            """
            DROP TABLE IF EXISTS profile_and_fees
            """,
            """
            DROP TABLE IF EXISTS profile_and_fees_snapshots
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
            CREATE TABLE IF NOT EXISTS profile_expense_allocations (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                expense_id TEXT NOT NULL,
                value_text TEXT,
                ratio REAL,
                PRIMARY KEY (conid, effective_at, expense_id)
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
        ),
    ),
    Migration(
        version=16,
        description="finalize profile report split",
        statements=(
            """
            ALTER TABLE morningstar_snapshots ADD COLUMN q_full_report_id TEXT
            """,
            """
            ALTER TABLE dividends_industry_metrics ADD COLUMN currency TEXT
            """,
            """
            DROP TABLE IF EXISTS lipper_ratings
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
        ),
    ),
    Migration(
        version=17,
        description="standardize holdings child-table identifiers",
        statements=(
            """
            DROP INDEX IF EXISTS idx_holdings_industry_effective_at
            """,
            """
            DROP INDEX IF EXISTS idx_holdings_currency_effective_at
            """,
            """
            DROP INDEX IF EXISTS idx_holdings_investor_country_effective_at
            """,
            """
            DROP INDEX IF EXISTS idx_holdings_geographic_weights_effective_at
            """,
            """
            DROP INDEX IF EXISTS idx_holdings_debt_type_effective_at
            """,
            """
            DROP INDEX IF EXISTS idx_holdings_top10_effective_at
            """,
            """
            ALTER TABLE holdings_industry RENAME TO holdings_industry_v16
            """,
            """
            ALTER TABLE holdings_currency RENAME TO holdings_currency_v16
            """,
            """
            ALTER TABLE holdings_investor_country
            RENAME TO holdings_investor_country_v16
            """,
            """
            ALTER TABLE holdings_geographic_weights
            RENAME TO holdings_geographic_weights_v16
            """,
            """
            ALTER TABLE holdings_debt_type RENAME TO holdings_debt_type_v16
            """,
            """
            ALTER TABLE holdings_top10 RENAME TO holdings_top10_v16
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
            INSERT INTO holdings_industry (
                conid,
                effective_at,
                industry_id,
                value_num,
                vs_peers
            )
            SELECT
                conid,
                effective_at,
                LOWER(
                    REPLACE(
                        REPLACE(REPLACE(TRIM(industry), ' ', '_'), '-', '_'),
                        '/',
                        '_'
                    )
                ),
                value_num,
                vs_peers
            FROM holdings_industry_v16
            """,
            """
            INSERT INTO holdings_currency (
                conid,
                effective_at,
                code,
                name,
                value_num,
                vs_peers
            )
            SELECT
                conid,
                effective_at,
                code,
                currency,
                value_num,
                vs_peers
            FROM holdings_currency_v16
            WHERE currency IS NOT NULL
            """,
            """
            INSERT INTO holdings_investor_country (
                conid,
                effective_at,
                code,
                name,
                value_num,
                vs_peers
            )
            SELECT
                conid,
                effective_at,
                country_code,
                country,
                value_num,
                vs_peers
            FROM holdings_investor_country_v16
            WHERE country IS NOT NULL
            """,
            """
            INSERT INTO holdings_geographic_weights (
                conid,
                effective_at,
                region_id,
                value_num,
                vs_peers
            )
            SELECT
                conid,
                effective_at,
                region,
                value_num,
                vs_peers
            FROM holdings_geographic_weights_v16
            """,
            """
            INSERT INTO holdings_debt_type (
                conid,
                effective_at,
                debt_type_id,
                value_num,
                vs_peers
            )
            SELECT
                conid,
                effective_at,
                LOWER(
                    REPLACE(
                        REPLACE(REPLACE(TRIM(debt_type), ' ', '_'), '-', '_'),
                        '/',
                        '_'
                    )
                ),
                value_num,
                vs_peers
            FROM holdings_debt_type_v16
            """,
            """
            INSERT INTO holdings_top10 (
                conid,
                effective_at,
                name,
                ticker,
                rank,
                holding_weight_num,
                vs_peers,
                conids_json
            )
            SELECT
                conid,
                effective_at,
                name,
                ticker,
                rank,
                holding_weight_num,
                vs_peers,
                conids_json
            FROM holdings_top10_v16
            """,
            """
            DROP TABLE IF EXISTS holdings_industry_v16
            """,
            """
            DROP TABLE IF EXISTS holdings_currency_v16
            """,
            """
            DROP TABLE IF EXISTS holdings_investor_country_v16
            """,
            """
            DROP TABLE IF EXISTS holdings_geographic_weights_v16
            """,
            """
            DROP TABLE IF EXISTS holdings_debt_type_v16
            """,
            """
            DROP TABLE IF EXISTS holdings_top10_v16
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


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}


def _prepare_v9_legacy_upgrade(conn: sqlite3.Connection) -> None:
    table_names = {
        str(row[0])
        for row in conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            """
        ).fetchall()
    }
    if "raw_payload_observations" not in table_names:
        return
    if "capture_batch_id" in _table_columns(conn, "raw_payload_observations"):
        return
    conn.execute(
        """
        ALTER TABLE raw_payload_observations
        ADD COLUMN capture_batch_id TEXT
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
        if migration.version == 9:
            _prepare_v9_legacy_upgrade(conn)

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
