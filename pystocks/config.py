from pathlib import Path

# Project Root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
FUNDAMENTALS_DIR = DATA_DIR / "fundamentals"
TRADES_DIR = DATA_DIR / "daily-trades"

# File Paths
CONTRACTS_DB_PATH = DATA_DIR / "contract_details.csv"
IB_PRODUCTS_PATH = DATA_DIR / "ib_products.csv"
SESSION_STATE_PATH = DATA_DIR / "auth_state.json"

# Create directories if they don't exist
for d in [DATA_DIR, RAW_DIR, PREPROCESSED_DIR, FUNDAMENTALS_DIR, TRADES_DIR]:
    d.mkdir(parents=True, exist_ok=True)
