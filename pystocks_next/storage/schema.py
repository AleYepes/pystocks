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
        description="initial operational schema",
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
            CREATE INDEX IF NOT EXISTS idx_raw_payload_observations_endpoint
            ON raw_payload_observations(endpoint, observed_at)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_universe_instruments_active
            ON universe_instruments(is_active, conid)
            """,
        ),
    ),
    Migration(
        version=2,
        description="add raw capture batch identity",
        statements=(
            """
            ALTER TABLE raw_payload_observations
            ADD COLUMN capture_batch_id TEXT
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_raw_payload_observations_batch
            ON raw_payload_observations(capture_batch_id, endpoint, observed_at)
            """,
        ),
    ),
    Migration(
        version=3,
        description="add canonical price and dividend series tables",
        statements=(
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
            CREATE INDEX IF NOT EXISTS idx_price_chart_series_effective_at
            ON price_chart_series(effective_at, conid)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_dividends_events_series_effective_at
            ON dividends_events_series(effective_at, conid)
            """,
        ),
    ),
    Migration(
        version=4,
        description="add supplementary storage foundations",
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
            CREATE TABLE IF NOT EXISTS supplementary_risk_free_daily (
                trade_date TEXT PRIMARY KEY,
                nominal_rate REAL,
                daily_nominal_rate REAL,
                source_count INTEGER NOT NULL,
                observed_at TEXT NOT NULL
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
            CREATE TABLE IF NOT EXISTS supplementary_world_bank_country_features (
                economy_code TEXT NOT NULL,
                effective_at TEXT NOT NULL,
                feature_year INTEGER NOT NULL,
                population_level REAL,
                population_growth REAL,
                population_acceleration REAL,
                gdp_pcap_level REAL,
                gdp_pcap_growth REAL,
                gdp_pcap_acceleration REAL,
                economic_output_gdp_level REAL,
                economic_output_gdp_growth REAL,
                economic_output_gdp_acceleration REAL,
                foreign_direct_investment_level REAL,
                foreign_direct_investment_growth REAL,
                foreign_direct_investment_acceleration REAL,
                share_trade_volume_level REAL,
                share_trade_volume_growth REAL,
                share_trade_volume_acceleration REAL,
                observed_at TEXT NOT NULL,
                PRIMARY KEY (economy_code, feature_year)
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
            """
            CREATE INDEX IF NOT EXISTS idx_supplementary_world_bank_country_features_effective_at
            ON supplementary_world_bank_country_features(effective_at, economy_code)
            """,
        ),
    ),
    Migration(
        version=5,
        description="add first snapshot storage tables",
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
                asset_type TEXT,
                classification TEXT,
                distribution_details TEXT,
                domicile TEXT,
                fiscal_date TEXT,
                fund_category TEXT,
                fund_management_company TEXT,
                fund_manager_benchmark TEXT,
                fund_market_cap_focus TEXT,
                geographical_focus TEXT,
                inception_date TEXT,
                management_approach TEXT,
                management_expenses REAL,
                manager_tenure TEXT,
                maturity_date TEXT,
                objective_type TEXT,
                portfolio_manager TEXT,
                redemption_charge_actual REAL,
                redemption_charge_max REAL,
                scheme TEXT,
                total_expense_ratio REAL,
                total_net_assets_value TEXT,
                total_net_assets_date TEXT,
                objective TEXT,
                jap_fund_warning INTEGER,
                theme_name TEXT,
                PRIMARY KEY (conid, effective_at)
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
                equity REAL,
                cash REAL,
                fixed_income REAL,
                other REAL,
                PRIMARY KEY (conid, effective_at)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_profile_and_fees_snapshots_observed_at
            ON profile_and_fees_snapshots(observed_at, conid)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_holdings_snapshots_observed_at
            ON holdings_snapshots(observed_at, conid)
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
