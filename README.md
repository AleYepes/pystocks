# pystocks v0.2.0
A modular system to extract IBKR fundamental data and perform factor-based portfolio analysis.

## Project Structure
- `pystocks/`: Core package containing refactored logic.
  - `cli.py`: Unified command-line interface.
  - `scraper.py`: Playwright-based web scraping for product discovery.
  - `contracts.py`: IBKR API interaction for contract details.
  - `fundamentals.py`: JSON-based fundamental data extraction from IBKR portal.
  - `preprocess.py`: Data cleaning and feature engineering pipeline.
  - `analysis.py`: Factor analysis and portfolio optimization.
- `data/`: Centralized data storage.
  - `raw/`: Raw fundamental snapshots (time-series).
  - `preprocessed/`: Cleaned data ready for analysis.
  - `daily-trades/`: Historical price series.

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
playwright install
```
2. Ensure Trader Workstation (TWS) is running and API access is enabled.

## Usage
The system is now powered by a unified CLI:

```bash
# Scrape the initial product list from IBKR website
python3 -m pystocks.cli scrape_products

# Update contract details (conIds, ISINs) via TWS API
python3 -m pystocks.cli update_contracts

# Fetch fundamental data via web portal proxy (JSON)
python3 -m pystocks.cli scrape_fundamentals --limit 100

# Preprocess raw data for analysis
python3 -m pystocks.cli preprocess

# Run factor analysis
python3 -m pystocks.cli analyze
```

## Legacy Data
The system is designed to be backward compatible with previous OCR-based captures. New JSON fetches are appended to existing monthly CSVs in `data/raw/`, allowing for long-term time-series analysis of fundamental data.
