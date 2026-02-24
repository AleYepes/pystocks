from pathlib import Path

# Project Root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR = ROOT_DIR / "data"
FUNDAMENTALS_DIR = DATA_DIR / "fundamentals"
FUNDAMENTALS_BLOBS_DIR = FUNDAMENTALS_DIR / "blobs"
FUNDAMENTALS_PARQUET_DIR = FUNDAMENTALS_DIR / "parquet"
PRICES_DIR = DATA_DIR / "prices"
PRICE_CHART_PARQUET_DIR = PRICES_DIR / "ibkr_mf_performance_chart"
PRICE_CHART_CLEAN_PARQUET_DIR = PRICES_DIR / "ibkr_mf_performance_chart_clean"
SENTIMENT_DIR = DATA_DIR / "sentiment"
SENTIMENT_SEARCH_PARQUET_DIR = SENTIMENT_DIR / "ibkr_sma_search"
OWNERSHIP_DIR = DATA_DIR / "ownership"
OWNERSHIP_TRADE_LOG_PARQUET_DIR = OWNERSHIP_DIR / "ibkr_ownership_trade_log"
DIVIDENDS_DIR = DATA_DIR / "dividends"
DIVIDENDS_EVENTS_PARQUET_DIR = DIVIDENDS_DIR / "ibkr_dividends_events"
FACTORS_DIR = DATA_DIR / "factors"
FACTOR_FEATURES_PARQUET_DIR = FACTORS_DIR / "ibkr_factor_features"
RESEARCH_DIR = DATA_DIR / "research"

# File Paths
CONTRACTS_DB_PATH = DATA_DIR / "contract_details.csv"
SESSION_STATE_PATH = DATA_DIR / "auth_state.json"
FUNDAMENTALS_ARCHIVE_PATH = FUNDAMENTALS_DIR / "fundamentals_archive.parquet"
FUNDAMENTALS_EVENTS_DB_PATH = FUNDAMENTALS_DIR / "events.db"
FUNDAMENTALS_DUCKDB_PATH = FUNDAMENTALS_DIR / "fundamentals.duckdb"

# Create directories if they don't exist
for d in [
    DATA_DIR,
    FUNDAMENTALS_DIR,
    FUNDAMENTALS_BLOBS_DIR,
    FUNDAMENTALS_PARQUET_DIR,
    PRICES_DIR,
    PRICE_CHART_PARQUET_DIR,
    PRICE_CHART_CLEAN_PARQUET_DIR,
    SENTIMENT_DIR,
    SENTIMENT_SEARCH_PARQUET_DIR,
    OWNERSHIP_DIR,
    OWNERSHIP_TRADE_LOG_PARQUET_DIR,
    DIVIDENDS_DIR,
    DIVIDENDS_EVENTS_PARQUET_DIR,
    FACTORS_DIR,
    FACTOR_FEATURES_PARQUET_DIR,
    RESEARCH_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)
