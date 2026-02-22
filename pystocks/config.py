from pathlib import Path

# Project Root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
FUNDAMENTALS_DIR = DATA_DIR / "fundamentals"
FUNDAMENTALS_BLOBS_DIR = FUNDAMENTALS_DIR / "blobs"
FUNDAMENTALS_PARQUET_DIR = FUNDAMENTALS_DIR / "parquet"
TRADES_DIR = DATA_DIR / "daily-trades"
RESEARCH_DIR = DATA_DIR / "research"

# File Paths
CONTRACTS_DB_PATH = DATA_DIR / "contract_details.csv"
IB_PRODUCTS_PATH = DATA_DIR / "ib_products.csv"
SESSION_STATE_PATH = DATA_DIR / "auth_state.json"
SQLITE_DB_PATH = DATA_DIR / "pystocks.db"
FUNDAMENTALS_ARCHIVE_PATH = FUNDAMENTALS_DIR / "fundamentals_archive.parquet"
FUNDAMENTALS_EVENTS_DB_PATH = FUNDAMENTALS_DIR / "events.db"
FUNDAMENTALS_DUCKDB_PATH = FUNDAMENTALS_DIR / "fundamentals.duckdb"
RESEARCH_YIELDS_PATH = RESEARCH_DIR / "research_yields.csv"
RESEARCH_CORR_SUMMARY_PATH = RESEARCH_DIR / "research_correlations_summary.csv"

# Create directories if they don't exist
for d in [
    DATA_DIR,
    RAW_DIR,
    PREPROCESSED_DIR,
    FUNDAMENTALS_DIR,
    FUNDAMENTALS_BLOBS_DIR,
    FUNDAMENTALS_PARQUET_DIR,
    TRADES_DIR,
    RESEARCH_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)
