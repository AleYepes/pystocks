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
    Migration(
        version=6,
        description="add holdings breadth and ratios snapshot storage tables",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS holdings_debtor_quality (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                quality_aaa REAL,
                quality_aa REAL,
                quality_a REAL,
                quality_bbb REAL,
                quality_bb REAL,
                quality_b REAL,
                quality_ccc REAL,
                quality_cc REAL,
                quality_c REAL,
                quality_d REAL,
                quality_not_rated REAL,
                quality_not_available REAL,
                PRIMARY KEY (conid, effective_at)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS holdings_maturity (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                maturity_less_than_1_year REAL,
                maturity_1_to_3_years REAL,
                maturity_3_to_5_years REAL,
                maturity_5_to_10_years REAL,
                maturity_10_to_20_years REAL,
                maturity_20_to_30_years REAL,
                maturity_greater_than_30_years REAL,
                maturity_other REAL,
                PRIMARY KEY (conid, effective_at)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS holdings_industry (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                industry TEXT NOT NULL,
                value_num REAL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS holdings_currency (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                code TEXT,
                currency TEXT,
                value_num REAL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS holdings_investor_country (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                country_code TEXT,
                country TEXT,
                value_num REAL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS holdings_geographic_weights (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                region TEXT NOT NULL,
                value_num REAL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS holdings_debt_type (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                debt_type TEXT NOT NULL,
                value_num REAL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS holdings_top10 (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                name TEXT NOT NULL,
                holding_weight_num REAL
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
                vs_num REAL,
                PRIMARY KEY (conid, effective_at, metric_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ratios_financials (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                metric_id TEXT NOT NULL,
                value_num REAL,
                vs_num REAL,
                PRIMARY KEY (conid, effective_at, metric_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ratios_fixed_income (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                metric_id TEXT NOT NULL,
                value_num REAL,
                vs_num REAL,
                PRIMARY KEY (conid, effective_at, metric_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ratios_dividend (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                metric_id TEXT NOT NULL,
                value_num REAL,
                vs_num REAL,
                PRIMARY KEY (conid, effective_at, metric_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ratios_zscore (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                metric_id TEXT NOT NULL,
                value_num REAL,
                vs_num REAL,
                PRIMARY KEY (conid, effective_at, metric_id)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_ratios_snapshots_observed_at
            ON ratios_snapshots(observed_at, conid)
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
    Migration(
        version=7,
        description="add consistent tall snapshot factor tables and backfill current wide data",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS profile_and_fees_factors (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                field_id TEXT NOT NULL,
                value_text TEXT,
                value_num REAL,
                value_date TEXT,
                value_bool INTEGER,
                PRIMARY KEY (conid, effective_at, field_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS holdings_asset_type_factors (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                bucket_id TEXT NOT NULL,
                value_num REAL,
                PRIMARY KEY (conid, effective_at, bucket_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS holdings_debtor_quality_factors (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                bucket_id TEXT NOT NULL,
                value_num REAL,
                PRIMARY KEY (conid, effective_at, bucket_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS holdings_maturity_factors (
                conid TEXT NOT NULL REFERENCES universe_instruments(conid),
                effective_at TEXT NOT NULL,
                bucket_id TEXT NOT NULL,
                value_num REAL,
                PRIMARY KEY (conid, effective_at, bucket_id)
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
            CREATE TABLE IF NOT EXISTS dividends_industry_metrics_factors (
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
            CREATE TABLE IF NOT EXISTS morningstar_summary_factors (
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
            INSERT OR IGNORE INTO profile_and_fees_factors (
                conid, effective_at, field_id, value_text, value_num, value_date, value_bool
            )
            SELECT conid, effective_at, 'asset_type', asset_type, NULL, NULL, NULL
            FROM profile_and_fees
            WHERE asset_type IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO profile_and_fees_factors (
                conid, effective_at, field_id, value_text, value_num, value_date, value_bool
            )
            SELECT conid, effective_at, 'classification', classification, NULL, NULL, NULL
            FROM profile_and_fees
            WHERE classification IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO profile_and_fees_factors (
                conid, effective_at, field_id, value_text, value_num, value_date, value_bool
            )
            SELECT conid, effective_at, 'distribution_details', distribution_details, NULL, NULL, NULL
            FROM profile_and_fees
            WHERE distribution_details IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO profile_and_fees_factors (
                conid, effective_at, field_id, value_text, value_num, value_date, value_bool
            )
            SELECT conid, effective_at, 'domicile', domicile, NULL, NULL, NULL
            FROM profile_and_fees
            WHERE domicile IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO profile_and_fees_factors (
                conid, effective_at, field_id, value_text, value_num, value_date, value_bool
            )
            SELECT conid, effective_at, 'fiscal_date', fiscal_date, NULL, NULL, NULL
            FROM profile_and_fees
            WHERE fiscal_date IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO profile_and_fees_factors (
                conid, effective_at, field_id, value_text, value_num, value_date, value_bool
            )
            SELECT conid, effective_at, 'fund_category', fund_category, NULL, NULL, NULL
            FROM profile_and_fees
            WHERE fund_category IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO profile_and_fees_factors (
                conid, effective_at, field_id, value_text, value_num, value_date, value_bool
            )
            SELECT conid, effective_at, 'fund_management_company', fund_management_company, NULL, NULL, NULL
            FROM profile_and_fees
            WHERE fund_management_company IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO profile_and_fees_factors (
                conid, effective_at, field_id, value_text, value_num, value_date, value_bool
            )
            SELECT conid, effective_at, 'fund_manager_benchmark', fund_manager_benchmark, NULL, NULL, NULL
            FROM profile_and_fees
            WHERE fund_manager_benchmark IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO profile_and_fees_factors (
                conid, effective_at, field_id, value_text, value_num, value_date, value_bool
            )
            SELECT conid, effective_at, 'fund_market_cap_focus', fund_market_cap_focus, NULL, NULL, NULL
            FROM profile_and_fees
            WHERE fund_market_cap_focus IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO profile_and_fees_factors (
                conid, effective_at, field_id, value_text, value_num, value_date, value_bool
            )
            SELECT conid, effective_at, 'geographical_focus', geographical_focus, NULL, NULL, NULL
            FROM profile_and_fees
            WHERE geographical_focus IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO profile_and_fees_factors (
                conid, effective_at, field_id, value_text, value_num, value_date, value_bool
            )
            SELECT conid, effective_at, 'inception_date', NULL, NULL, inception_date, NULL
            FROM profile_and_fees
            WHERE inception_date IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO profile_and_fees_factors (
                conid, effective_at, field_id, value_text, value_num, value_date, value_bool
            )
            SELECT conid, effective_at, 'management_approach', management_approach, NULL, NULL, NULL
            FROM profile_and_fees
            WHERE management_approach IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO profile_and_fees_factors (
                conid, effective_at, field_id, value_text, value_num, value_date, value_bool
            )
            SELECT conid, effective_at, 'management_expenses', NULL, management_expenses, NULL, NULL
            FROM profile_and_fees
            WHERE management_expenses IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO profile_and_fees_factors (
                conid, effective_at, field_id, value_text, value_num, value_date, value_bool
            )
            SELECT conid, effective_at, 'manager_tenure', NULL, NULL, manager_tenure, NULL
            FROM profile_and_fees
            WHERE manager_tenure IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO profile_and_fees_factors (
                conid, effective_at, field_id, value_text, value_num, value_date, value_bool
            )
            SELECT conid, effective_at, 'maturity_date', NULL, NULL, maturity_date, NULL
            FROM profile_and_fees
            WHERE maturity_date IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO profile_and_fees_factors (
                conid, effective_at, field_id, value_text, value_num, value_date, value_bool
            )
            SELECT conid, effective_at, 'objective_type', objective_type, NULL, NULL, NULL
            FROM profile_and_fees
            WHERE objective_type IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO profile_and_fees_factors (
                conid, effective_at, field_id, value_text, value_num, value_date, value_bool
            )
            SELECT conid, effective_at, 'portfolio_manager', portfolio_manager, NULL, NULL, NULL
            FROM profile_and_fees
            WHERE portfolio_manager IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO profile_and_fees_factors (
                conid, effective_at, field_id, value_text, value_num, value_date, value_bool
            )
            SELECT conid, effective_at, 'redemption_charge_actual', NULL, redemption_charge_actual, NULL, NULL
            FROM profile_and_fees
            WHERE redemption_charge_actual IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO profile_and_fees_factors (
                conid, effective_at, field_id, value_text, value_num, value_date, value_bool
            )
            SELECT conid, effective_at, 'redemption_charge_max', NULL, redemption_charge_max, NULL, NULL
            FROM profile_and_fees
            WHERE redemption_charge_max IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO profile_and_fees_factors (
                conid, effective_at, field_id, value_text, value_num, value_date, value_bool
            )
            SELECT conid, effective_at, 'scheme', scheme, NULL, NULL, NULL
            FROM profile_and_fees
            WHERE scheme IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO profile_and_fees_factors (
                conid, effective_at, field_id, value_text, value_num, value_date, value_bool
            )
            SELECT conid, effective_at, 'total_expense_ratio', NULL, total_expense_ratio, NULL, NULL
            FROM profile_and_fees
            WHERE total_expense_ratio IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO profile_and_fees_factors (
                conid, effective_at, field_id, value_text, value_num, value_date, value_bool
            )
            SELECT conid, effective_at, 'total_net_assets_value', total_net_assets_value, NULL, NULL, NULL
            FROM profile_and_fees
            WHERE total_net_assets_value IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO profile_and_fees_factors (
                conid, effective_at, field_id, value_text, value_num, value_date, value_bool
            )
            SELECT conid, effective_at, 'total_net_assets_date', NULL, NULL, total_net_assets_date, NULL
            FROM profile_and_fees
            WHERE total_net_assets_date IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO profile_and_fees_factors (
                conid, effective_at, field_id, value_text, value_num, value_date, value_bool
            )
            SELECT conid, effective_at, 'objective', objective, NULL, NULL, NULL
            FROM profile_and_fees
            WHERE objective IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO profile_and_fees_factors (
                conid, effective_at, field_id, value_text, value_num, value_date, value_bool
            )
            SELECT conid, effective_at, 'jap_fund_warning', NULL, NULL, NULL, jap_fund_warning
            FROM profile_and_fees
            WHERE jap_fund_warning IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO profile_and_fees_factors (
                conid, effective_at, field_id, value_text, value_num, value_date, value_bool
            )
            SELECT conid, effective_at, 'theme_name', theme_name, NULL, NULL, NULL
            FROM profile_and_fees
            WHERE theme_name IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO holdings_asset_type_factors (conid, effective_at, bucket_id, value_num)
            SELECT conid, effective_at, 'equity', equity
            FROM holdings_asset_type
            WHERE equity IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO holdings_asset_type_factors (conid, effective_at, bucket_id, value_num)
            SELECT conid, effective_at, 'cash', cash
            FROM holdings_asset_type
            WHERE cash IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO holdings_asset_type_factors (conid, effective_at, bucket_id, value_num)
            SELECT conid, effective_at, 'fixed_income', fixed_income
            FROM holdings_asset_type
            WHERE fixed_income IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO holdings_asset_type_factors (conid, effective_at, bucket_id, value_num)
            SELECT conid, effective_at, 'other', other
            FROM holdings_asset_type
            WHERE other IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO holdings_debtor_quality_factors (conid, effective_at, bucket_id, value_num)
            SELECT conid, effective_at, 'quality_aaa', quality_aaa
            FROM holdings_debtor_quality
            WHERE quality_aaa IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO holdings_debtor_quality_factors (conid, effective_at, bucket_id, value_num)
            SELECT conid, effective_at, 'quality_aa', quality_aa
            FROM holdings_debtor_quality
            WHERE quality_aa IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO holdings_debtor_quality_factors (conid, effective_at, bucket_id, value_num)
            SELECT conid, effective_at, 'quality_a', quality_a
            FROM holdings_debtor_quality
            WHERE quality_a IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO holdings_debtor_quality_factors (conid, effective_at, bucket_id, value_num)
            SELECT conid, effective_at, 'quality_bbb', quality_bbb
            FROM holdings_debtor_quality
            WHERE quality_bbb IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO holdings_debtor_quality_factors (conid, effective_at, bucket_id, value_num)
            SELECT conid, effective_at, 'quality_bb', quality_bb
            FROM holdings_debtor_quality
            WHERE quality_bb IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO holdings_debtor_quality_factors (conid, effective_at, bucket_id, value_num)
            SELECT conid, effective_at, 'quality_b', quality_b
            FROM holdings_debtor_quality
            WHERE quality_b IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO holdings_debtor_quality_factors (conid, effective_at, bucket_id, value_num)
            SELECT conid, effective_at, 'quality_ccc', quality_ccc
            FROM holdings_debtor_quality
            WHERE quality_ccc IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO holdings_debtor_quality_factors (conid, effective_at, bucket_id, value_num)
            SELECT conid, effective_at, 'quality_cc', quality_cc
            FROM holdings_debtor_quality
            WHERE quality_cc IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO holdings_debtor_quality_factors (conid, effective_at, bucket_id, value_num)
            SELECT conid, effective_at, 'quality_c', quality_c
            FROM holdings_debtor_quality
            WHERE quality_c IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO holdings_debtor_quality_factors (conid, effective_at, bucket_id, value_num)
            SELECT conid, effective_at, 'quality_d', quality_d
            FROM holdings_debtor_quality
            WHERE quality_d IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO holdings_debtor_quality_factors (conid, effective_at, bucket_id, value_num)
            SELECT conid, effective_at, 'quality_not_rated', quality_not_rated
            FROM holdings_debtor_quality
            WHERE quality_not_rated IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO holdings_debtor_quality_factors (conid, effective_at, bucket_id, value_num)
            SELECT conid, effective_at, 'quality_not_available', quality_not_available
            FROM holdings_debtor_quality
            WHERE quality_not_available IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO holdings_maturity_factors (conid, effective_at, bucket_id, value_num)
            SELECT conid, effective_at, 'maturity_less_than_1_year', maturity_less_than_1_year
            FROM holdings_maturity
            WHERE maturity_less_than_1_year IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO holdings_maturity_factors (conid, effective_at, bucket_id, value_num)
            SELECT conid, effective_at, 'maturity_1_to_3_years', maturity_1_to_3_years
            FROM holdings_maturity
            WHERE maturity_1_to_3_years IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO holdings_maturity_factors (conid, effective_at, bucket_id, value_num)
            SELECT conid, effective_at, 'maturity_3_to_5_years', maturity_3_to_5_years
            FROM holdings_maturity
            WHERE maturity_3_to_5_years IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO holdings_maturity_factors (conid, effective_at, bucket_id, value_num)
            SELECT conid, effective_at, 'maturity_5_to_10_years', maturity_5_to_10_years
            FROM holdings_maturity
            WHERE maturity_5_to_10_years IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO holdings_maturity_factors (conid, effective_at, bucket_id, value_num)
            SELECT conid, effective_at, 'maturity_10_to_20_years', maturity_10_to_20_years
            FROM holdings_maturity
            WHERE maturity_10_to_20_years IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO holdings_maturity_factors (conid, effective_at, bucket_id, value_num)
            SELECT conid, effective_at, 'maturity_20_to_30_years', maturity_20_to_30_years
            FROM holdings_maturity
            WHERE maturity_20_to_30_years IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO holdings_maturity_factors (conid, effective_at, bucket_id, value_num)
            SELECT conid, effective_at, 'maturity_greater_than_30_years', maturity_greater_than_30_years
            FROM holdings_maturity
            WHERE maturity_greater_than_30_years IS NOT NULL
            """,
            """
            INSERT OR IGNORE INTO holdings_maturity_factors (conid, effective_at, bucket_id, value_num)
            SELECT conid, effective_at, 'maturity_other', maturity_other
            FROM holdings_maturity
            WHERE maturity_other IS NOT NULL
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_profile_and_fees_factors_effective_at
            ON profile_and_fees_factors(effective_at, conid)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_holdings_asset_type_factors_effective_at
            ON holdings_asset_type_factors(effective_at, conid)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_holdings_debtor_quality_factors_effective_at
            ON holdings_debtor_quality_factors(effective_at, conid)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_holdings_maturity_factors_effective_at
            ON holdings_maturity_factors(effective_at, conid)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_dividends_snapshots_observed_at
            ON dividends_snapshots(observed_at, conid)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_dividends_industry_metrics_factors_effective_at
            ON dividends_industry_metrics_factors(effective_at, conid)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_morningstar_snapshots_observed_at
            ON morningstar_snapshots(observed_at, conid)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_morningstar_summary_factors_effective_at
            ON morningstar_summary_factors(effective_at, conid)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_lipper_ratings_snapshots_observed_at
            ON lipper_ratings_snapshots(observed_at, conid)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_lipper_ratings_effective_at
            ON lipper_ratings(effective_at, conid)
            """,
        ),
    ),
    Migration(
        version=8,
        description="retire persisted supplementary derivation tables",
        statements=(
            """
            DROP INDEX IF EXISTS idx_supplementary_world_bank_country_features_effective_at
            """,
            """
            DROP TABLE IF EXISTS supplementary_risk_free_daily
            """,
            """
            DROP TABLE IF EXISTS supplementary_world_bank_country_features
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
