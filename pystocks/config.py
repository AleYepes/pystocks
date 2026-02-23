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
RESEARCH_DIR = DATA_DIR / "research"

# File Paths
CONTRACTS_DB_PATH = DATA_DIR / "contract_details.csv"
IB_PRODUCTS_PATH = DATA_DIR / "ib_products.csv"
SESSION_STATE_PATH = DATA_DIR / "auth_state.json"
SQLITE_DB_PATH = DATA_DIR / "pystocks.db"
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
    RESEARCH_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)
