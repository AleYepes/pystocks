# Data Schema Documentation

The data layer is currently transitioning from a large set of wide-format CSVs to a hybrid SQLite/Parquet architecture to manage the history of fundamental snapshots.

## Current CSV Schema (Legacy)

### 1. `ib_products.csv` (The Universe)
| Column | Description |
| --- | --- |
| SYMBOL | The ticker symbol. |
| IBKR SYMBOL | IBKR internal symbol (may differ). |
| CURRENCY | Trading currency (e.g., EUR, USD). |
| EXCHANGE | Primary exchange code. |

### 2. `contract_details.csv` (The Registry)
| Column | Description |
| --- | --- |
| conId | **Primary Key.** IBKR internal unique identifier. |
| isin | International Securities Identification Number. |
| longName | Full instrument name (crucial for OCR/Fuzzy matching). |
| validExchanges | Comma-separated list of all exchanges for the contract. |

### 3. `fundamentals_*.csv` (The Snapshots)
| Column Prefix | Description |
| --- | --- |
| countries_* | Percentage weights of geographic exposure. |
| industries_* | Percentage weights of sector exposure. |
| fundamentals_* | Key ratios (P/E, P/B, Debt/Equity). |
| holding_types_* | Asset class weights (Equity, Cash, Bond). |
| style_* | Morningstar style box (e.g., Large-Cap Value). |

## Proposed Schema Transition

### 1. SQLite Database (`pystocks.db`)
*   **Table: `contracts`** - Registry of instruments and scraping metadata.
*   **Table: `scraper_logs`** - Historical success/failure of API calls.

### 2. Parquet Archive (`fundamentals_archive.parquet`)
*   **Columns:** `conId`, `as_of_date`, `data_blob` (nested JSON structure).
*   **Goal:** Maintain a Point-in-Time (PIT) history of fundamentals to prevent look-ahead bias in regressions.

### 3. TimeSeries Directory (`data/daily-trades/series/*.csv`)
*   **Format:** `SYMBOL-EXCHANGE-CURRENCY.csv`.
*   **Goal:** Standardize price series for regression analysis.
