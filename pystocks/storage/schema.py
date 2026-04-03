import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from ..config import SQLITE_DB_PATH
from ._sqlite import open_connection

_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    schema_version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS products (
    conid TEXT PRIMARY KEY,
    symbol TEXT,
    exchange TEXT,
    isin TEXT,
    currency TEXT,
    name TEXT,
    last_scraped_fundamentals TEXT,
    last_status_fundamentals TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS raw_payload_blobs (
    payload_hash TEXT PRIMARY KEY,
    compression TEXT NOT NULL,
    raw_size_bytes INTEGER NOT NULL,
    compressed_size_bytes INTEGER NOT NULL,
    payload_blob BLOB NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ingest_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_started_at TEXT NOT NULL,
    run_finished_at TEXT NOT NULL,
    total_targeted_conids INTEGER NOT NULL,
    processed_conids INTEGER NOT NULL,
    saved_snapshots INTEGER NOT NULL,
    inserted_events INTEGER NOT NULL,
    overwritten_events INTEGER NOT NULL,
    unchanged_events INTEGER NOT NULL,
    series_raw_rows_written INTEGER NOT NULL,
    series_latest_rows_upserted INTEGER NOT NULL,
    auth_retries INTEGER NOT NULL,
    aborted INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS ingest_run_endpoint_rollups (
    run_id INTEGER NOT NULL,
    endpoint TEXT NOT NULL,
    call_count INTEGER NOT NULL,
    useful_payload_count INTEGER NOT NULL,
    useful_payload_rate REAL NOT NULL,
    status_2xx INTEGER NOT NULL,
    status_4xx INTEGER NOT NULL,
    status_5xx INTEGER NOT NULL,
    status_other INTEGER NOT NULL,
    PRIMARY KEY (run_id, endpoint),
    FOREIGN KEY (run_id) REFERENCES ingest_runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS endpoint_scalar_extras (
    endpoint TEXT NOT NULL,
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    path TEXT NOT NULL,
    value_text TEXT,
    value_num REAL,
    value_bool INTEGER,
    value_date TEXT,
    PRIMARY KEY (endpoint, conid, effective_at, path),
    FOREIGN KEY (conid) REFERENCES products(conid)
);

CREATE TABLE IF NOT EXISTS profile_and_fees_snapshots (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    inserted_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (conid, effective_at),
    FOREIGN KEY (conid) REFERENCES products(conid),
    FOREIGN KEY (payload_hash) REFERENCES raw_payload_blobs(payload_hash)
);

CREATE TABLE IF NOT EXISTS profile_and_fees (
    conid TEXT NOT NULL,
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
    PRIMARY KEY (conid, effective_at),
    FOREIGN KEY (conid) REFERENCES products(conid)
);

CREATE TABLE IF NOT EXISTS profile_and_fees_reports (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    report_name TEXT,
    report_as_of_date TEXT,
    administrator_expenses REAL,
    advisor_expenses REAL,
    audit_and_legal_expenses REAL,
    audit_expenses REAL,
    custodian_expenses REAL,
    director_expense REAL,
    management_fees REAL,
    misc_expenses REAL,
    non_management_expenses REAL,
    other_expense REAL,
    other_non_management_fees REAL,
    postage_and_printing_expenses REAL,
    prospectus_gross_expense_ratio REAL,
    prospectus_gross_management_fee_ratio REAL,
    prospectus_gross_other_expense_ratio REAL,
    prospectus_net_expense_ratio REAL,
    prospectus_net_management_fee_ratio REAL,
    prospectus_net_other_expense_ratio REAL,
    registration_expenses REAL,
    total_expense REAL,
    total_gross_expense REAL,
    total_net_expense REAL,
    FOREIGN KEY (conid) REFERENCES products(conid)
);

CREATE TABLE IF NOT EXISTS profile_and_fees_stylebox (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    value_large INTEGER,
    value_multi INTEGER,
    value_mid INTEGER,
    value_small INTEGER,
    core_large INTEGER,
    core_multi INTEGER,
    core_mid INTEGER,
    core_small INTEGER,
    growth_large INTEGER,
    growth_multi INTEGER,
    growth_mid INTEGER,
    growth_small INTEGER,
    PRIMARY KEY (conid, effective_at),
    FOREIGN KEY (conid) REFERENCES products(conid)
);

CREATE TABLE IF NOT EXISTS holdings_snapshots (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    inserted_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    as_of_date TEXT,
    FOREIGN KEY (conid) REFERENCES products(conid),
    FOREIGN KEY (payload_hash) REFERENCES raw_payload_blobs(payload_hash)
);

CREATE TABLE IF NOT EXISTS holdings_asset_type (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    equity REAL,
    cash REAL,
    fixed_income REAL,
    other REAL,
    FOREIGN KEY (conid) REFERENCES products(conid)
);

CREATE TABLE IF NOT EXISTS holdings_industry (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    industry TEXT,
    value_num REAL,
    industry_avg REAL,
    FOREIGN KEY (conid) REFERENCES products(conid)
);

CREATE TABLE IF NOT EXISTS holdings_currency (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    currency TEXT,
    code TEXT,
    value_num REAL,
    industry_avg REAL,
    FOREIGN KEY (conid) REFERENCES products(conid)
);

CREATE TABLE IF NOT EXISTS holdings_investor_country (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    country TEXT,
    country_code TEXT,
    value_num REAL,
    industry_avg REAL,
    FOREIGN KEY (conid) REFERENCES products(conid)
);

CREATE TABLE IF NOT EXISTS holdings_debt_type (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    debt_type TEXT,
    value_num REAL,
    industry_avg REAL,
    FOREIGN KEY (conid) REFERENCES products(conid)
);

CREATE TABLE IF NOT EXISTS holdings_debtor_quality (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    quality_aaa REAL,
    quality_aaa_industry_avg REAL,
    quality_aa REAL,
    quality_aa_industry_avg REAL,
    quality_a REAL,
    quality_a_industry_avg REAL,
    quality_bbb REAL,
    quality_bbb_industry_avg REAL,
    quality_bb REAL,
    quality_bb_industry_avg REAL,
    quality_b REAL,
    quality_b_industry_avg REAL,
    quality_ccc REAL,
    quality_ccc_industry_avg REAL,
    quality_cc REAL,
    quality_cc_industry_avg REAL,
    quality_c REAL,
    quality_c_industry_avg REAL,
    quality_d REAL,
    quality_d_industry_avg REAL,
    quality_not_rated REAL,
    quality_not_rated_industry_avg REAL,
    quality_not_available REAL,
    quality_not_available_industry_avg REAL,
    FOREIGN KEY (conid) REFERENCES products(conid)
);

CREATE TABLE IF NOT EXISTS holdings_maturity (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    maturity_less_than_1_year REAL,
    maturity_less_than_1_year_industry_avg REAL,
    maturity_1_to_3_years REAL,
    maturity_1_to_3_years_industry_avg REAL,
    maturity_3_to_5_years REAL,
    maturity_3_to_5_years_industry_avg REAL,
    maturity_5_to_10_years REAL,
    maturity_5_to_10_years_industry_avg REAL,
    maturity_10_to_20_years REAL,
    maturity_10_to_20_years_industry_avg REAL,
    maturity_20_to_30_years REAL,
    maturity_20_to_30_years_industry_avg REAL,
    maturity_greater_than_30_years REAL,
    maturity_greater_than_30_years_industry_avg REAL,
    maturity_other REAL,
    maturity_other_industry_avg REAL,
    FOREIGN KEY (conid) REFERENCES products(conid)
);

CREATE TABLE IF NOT EXISTS holdings_top10 (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    name TEXT,
    ticker TEXT,
    holding_weight_num REAL,
    holding_conids TEXT,
    FOREIGN KEY (conid) REFERENCES products(conid)
);

CREATE TABLE IF NOT EXISTS holdings_geographic_weights (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    region TEXT,
    value_num REAL,
    FOREIGN KEY (conid) REFERENCES products(conid)
);

CREATE TABLE IF NOT EXISTS ratios_snapshots (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    inserted_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    as_of_date TEXT,
    PRIMARY KEY (conid, effective_at),
    FOREIGN KEY (conid) REFERENCES products(conid),
    FOREIGN KEY (payload_hash) REFERENCES raw_payload_blobs(payload_hash)
);

CREATE TABLE IF NOT EXISTS ratios_key_ratios (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    metric_id TEXT,
    value_num REAL,
    vs_num REAL,
    min_num REAL,
    max_num REAL,
    avg_num REAL,
    percentile_num REAL,
    FOREIGN KEY (conid) REFERENCES products(conid)
);

CREATE TABLE IF NOT EXISTS ratios_financials (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    metric_id TEXT,
    value_num REAL,
    vs_num REAL,
    min_num REAL,
    max_num REAL,
    avg_num REAL,
    percentile_num REAL,
    FOREIGN KEY (conid) REFERENCES products(conid)
);

CREATE TABLE IF NOT EXISTS ratios_fixed_income (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    metric_id TEXT,
    value_num REAL,
    vs_num REAL,
    min_num REAL,
    max_num REAL,
    avg_num REAL,
    percentile_num REAL,
    FOREIGN KEY (conid) REFERENCES products(conid)
);

CREATE TABLE IF NOT EXISTS ratios_dividend (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    metric_id TEXT,
    value_num REAL,
    vs_num REAL,
    min_num REAL,
    max_num REAL,
    avg_num REAL,
    percentile_num REAL,
    FOREIGN KEY (conid) REFERENCES products(conid)
);

CREATE TABLE IF NOT EXISTS ratios_zscore (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    metric_id TEXT,
    value_num REAL,
    vs_num REAL,
    min_num REAL,
    max_num REAL,
    avg_num REAL,
    percentile_num REAL,
    FOREIGN KEY (conid) REFERENCES products(conid)
);

CREATE TABLE IF NOT EXISTS lipper_ratings_snapshots (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    inserted_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    universe_count INTEGER,
    PRIMARY KEY (conid, effective_at),
    FOREIGN KEY (conid) REFERENCES products(conid),
    FOREIGN KEY (payload_hash) REFERENCES raw_payload_blobs(payload_hash)
);

CREATE TABLE IF NOT EXISTS lipper_ratings (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    period TEXT,
    metric_id TEXT,
    rating_value REAL,
    rating_label TEXT,
    universe_name TEXT,
    universe_as_of_date TEXT,
    FOREIGN KEY (conid) REFERENCES products(conid)
);

CREATE TABLE IF NOT EXISTS dividends_snapshots (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    inserted_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (conid, effective_at),
    FOREIGN KEY (conid) REFERENCES products(conid),
    FOREIGN KEY (payload_hash) REFERENCES raw_payload_blobs(payload_hash)
);

CREATE TABLE IF NOT EXISTS dividends_industry_metrics (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    dividend_yield REAL,
    annual_dividend REAL,
    dividend_ttm REAL,
    dividend_yield_ttm REAL,
    currency TEXT,
    PRIMARY KEY (conid, effective_at),
    FOREIGN KEY (conid) REFERENCES products(conid)
);

CREATE TABLE IF NOT EXISTS morningstar_snapshots (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    inserted_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    as_of_date TEXT,
    q_full_report_id TEXT,
    PRIMARY KEY (conid, effective_at),
    FOREIGN KEY (conid) REFERENCES products(conid),
    FOREIGN KEY (payload_hash) REFERENCES raw_payload_blobs(payload_hash)
);

CREATE TABLE IF NOT EXISTS morningstar_summary (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    medalist_rating TEXT,
    process TEXT,
    people TEXT,
    parent TEXT,
    morningstar_rating REAL,
    sustainability_rating TEXT,
    category TEXT,
    category_index TEXT,
    PRIMARY KEY (conid, effective_at),
    FOREIGN KEY (conid) REFERENCES products(conid)
);

CREATE TABLE IF NOT EXISTS morningstar_commentary (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    item_id TEXT,
    subsection_id TEXT,
    publish_date TEXT,
    text TEXT,
    author_name TEXT,
    FOREIGN KEY (conid) REFERENCES products(conid)
);

CREATE TABLE IF NOT EXISTS performance_snapshots (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    inserted_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (conid, effective_at),
    FOREIGN KEY (conid) REFERENCES products(conid),
    FOREIGN KEY (payload_hash) REFERENCES raw_payload_blobs(payload_hash)
);

CREATE TABLE IF NOT EXISTS performance (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    section TEXT,
    metric_id TEXT,
    value_num REAL,
    vs_num REAL,
    min_num REAL,
    max_num REAL,
    avg_num REAL,
    FOREIGN KEY (conid) REFERENCES products(conid)
);

CREATE TABLE IF NOT EXISTS ownership_snapshots (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    inserted_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    owners_types_count INTEGER,
    institutional_owners_count INTEGER,
    insider_owners_count INTEGER,
    trade_log_count_raw INTEGER,
    trade_log_count_kept INTEGER,
    has_ownership_history INTEGER,
    ownership_history_price_points INTEGER,
    institutional_total_value_text TEXT,
    institutional_total_shares_text TEXT,
    institutional_total_pct_text TEXT,
    institutional_total_pct_num REAL,
    insider_total_value_text TEXT,
    insider_total_shares_text TEXT,
    insider_total_pct_text TEXT,
    insider_total_pct_num REAL,
    PRIMARY KEY (conid, effective_at),
    FOREIGN KEY (conid) REFERENCES products(conid),
    FOREIGN KEY (payload_hash) REFERENCES raw_payload_blobs(payload_hash)
);

CREATE TABLE IF NOT EXISTS ownership_owners_types (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    type TEXT,
    display_type TEXT,
    float_value REAL,
    display_float TEXT,
    FOREIGN KEY (conid) REFERENCES products(conid)
);

CREATE TABLE IF NOT EXISTS ownership_holders (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    holder_group TEXT,
    holder_name TEXT,
    holder_type TEXT,
    display_value TEXT,
    display_shares TEXT,
    display_pct TEXT,
    pct_num REAL,
    FOREIGN KEY (conid) REFERENCES products(conid)
);

CREATE TABLE IF NOT EXISTS esg_snapshots (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    inserted_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    as_of_date TEXT,
    PRIMARY KEY (conid, effective_at),
    FOREIGN KEY (conid) REFERENCES products(conid),
    FOREIGN KEY (payload_hash) REFERENCES raw_payload_blobs(payload_hash)
);

CREATE TABLE IF NOT EXISTS esg (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    coverage REAL,
    source TEXT,
    esg_overall_score REAL,
    esg_combined_score REAL,
    esg_controversies_score REAL,
    environmental_overall_score REAL,
    environmental_resource_use_score REAL,
    environmental_emissions_score REAL,
    environmental_innovation_score REAL,
    social_overall_score REAL,
    social_workforce_score REAL,
    social_human_rights_score REAL,
    social_community_score REAL,
    social_product_responsibility_score REAL,
    governance_overall_score REAL,
    governance_management_score REAL,
    governance_shareholders_score REAL,
    governance_csr_strategy_score REAL,
    PRIMARY KEY (conid, effective_at),
    FOREIGN KEY (conid) REFERENCES products(conid)
);

CREATE TABLE IF NOT EXISTS price_chart_snapshots (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    inserted_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    points_count INTEGER,
    min_trade_date TEXT,
    max_trade_date TEXT,
    PRIMARY KEY (conid, effective_at),
    FOREIGN KEY (conid) REFERENCES products(conid),
    FOREIGN KEY (payload_hash) REFERENCES raw_payload_blobs(payload_hash)
);

CREATE TABLE IF NOT EXISTS sentiment_snapshots (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    inserted_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    points_count INTEGER,
    min_trade_date TEXT,
    max_trade_date TEXT,
    PRIMARY KEY (conid, effective_at),
    FOREIGN KEY (conid) REFERENCES products(conid),
    FOREIGN KEY (payload_hash) REFERENCES raw_payload_blobs(payload_hash)
);

CREATE TABLE IF NOT EXISTS price_chart_series (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    price REAL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    PRIMARY KEY (conid, effective_at),
    FOREIGN KEY (conid) REFERENCES products(conid)
);

CREATE TABLE IF NOT EXISTS sentiment_series (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    sscore REAL,
    sdelta REAL,
    svolatility REAL,
    sdispersion REAL,
    svscore REAL,
    svolume REAL,
    smean REAL,
    sbuzz REAL,
    PRIMARY KEY (conid, effective_at),
    FOREIGN KEY (conid) REFERENCES products(conid)
);

CREATE TABLE IF NOT EXISTS ownership_trade_log_series_raw (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    inserted_at TEXT NOT NULL,
    row_key TEXT NOT NULL,
    trade_date TEXT,
    action TEXT,
    shares REAL,
    value REAL,
    holding REAL,
    party TEXT,
    source TEXT,
    insider TEXT,
    FOREIGN KEY (conid) REFERENCES products(conid)
);

CREATE TABLE IF NOT EXISTS ownership_trade_log_series_latest (
    conid TEXT NOT NULL,
    row_key TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    trade_date TEXT,
    action TEXT,
    shares REAL,
    value REAL,
    holding REAL,
    party TEXT,
    source TEXT,
    insider TEXT,
    PRIMARY KEY (conid, row_key),
    FOREIGN KEY (conid) REFERENCES products(conid)
);

CREATE TABLE IF NOT EXISTS dividends_events_series (
    conid TEXT NOT NULL,
    effective_at TEXT NOT NULL,
    amount REAL,
    currency TEXT,
    description TEXT,
    event_type TEXT,
    declaration_date TEXT,
    record_date TEXT,
    payment_date TEXT,
    FOREIGN KEY (conid) REFERENCES products(conid)
);
"""

_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_profile_and_fees_snapshots_hash ON profile_and_fees_snapshots(payload_hash);
CREATE INDEX IF NOT EXISTS idx_holdings_snapshots_hash ON holdings_snapshots(payload_hash);
CREATE INDEX IF NOT EXISTS idx_ratios_snapshots_hash ON ratios_snapshots(payload_hash);
CREATE INDEX IF NOT EXISTS idx_lipper_snapshots_hash ON lipper_ratings_snapshots(payload_hash);
CREATE INDEX IF NOT EXISTS idx_dividends_snapshots_hash ON dividends_snapshots(payload_hash);
CREATE INDEX IF NOT EXISTS idx_morningstar_snapshots_hash ON morningstar_snapshots(payload_hash);
CREATE INDEX IF NOT EXISTS idx_performance_snapshots_hash ON performance_snapshots(payload_hash);
CREATE INDEX IF NOT EXISTS idx_ownership_snapshots_hash ON ownership_snapshots(payload_hash);
CREATE INDEX IF NOT EXISTS idx_esg_snapshots_hash ON esg_snapshots(payload_hash);
CREATE INDEX IF NOT EXISTS idx_price_snapshots_hash ON price_chart_snapshots(payload_hash);
CREATE INDEX IF NOT EXISTS idx_sentiment_snapshots_hash ON sentiment_snapshots(payload_hash);

CREATE INDEX IF NOT EXISTS idx_price_series_effective_at ON price_chart_series(effective_at);
CREATE INDEX IF NOT EXISTS idx_sentiment_series_effective_at ON sentiment_series(effective_at);
CREATE INDEX IF NOT EXISTS idx_ownership_latest_trade_date ON ownership_trade_log_series_latest(trade_date);
CREATE INDEX IF NOT EXISTS idx_dividends_events_series_effective_at ON dividends_events_series(effective_at);
"""

_INITIALIZED_PATHS: set[str] = set()


def apply_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA_DDL)
    conn.execute(
        """
        INSERT OR IGNORE INTO schema_meta (schema_version, applied_at)
        VALUES (1, ?)
        """,
        [datetime.now(UTC).isoformat()],
    )
    conn.executescript(_INDEX_DDL)


def init_storage(sqlite_path=SQLITE_DB_PATH) -> None:
    db_path = Path(sqlite_path)
    db_key = str(db_path)
    if db_key in _INITIALIZED_PATHS and db_path.exists():
        return

    with open_connection(db_path) as conn:
        apply_schema(conn)

    _INITIALIZED_PATHS.add(db_key)
