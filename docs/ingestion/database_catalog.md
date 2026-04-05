# Pystocks Database Catalog

This document describes the structure of the SQLite database used for storing fundamental data. The system uses a prefix-based naming convention to associate tables with specific API endpoints, supported by a set of core meta tables.

## 1. Special Meta Tables (Non-Prefixed)

These tables manage the state of the ingestion process, raw data storage, and instrument metadata.

| Table | Description |
| :--- | :--- |
| **`schema_meta`** | Tracks the current schema version and migration history. |
| **`products`** | Central registry of instruments (`conid`). Stores ISIN, names, and last scrape status. |
| **`raw_payload_blobs`** | Content-addressable storage. Stores Zstd-compressed unique JSON responses to prevent duplication. |
| **`sqlite_sequence`** | SQLite internal table for managing `AUTOINCREMENT` primary keys. |

---

## 2. Endpoint Table Mapping

Data is partitioned by "family." Most families include a `_snapshots` table for metadata and one or more tables for the actual data.

### Profile & Fees
*   **`profile_and_fees_snapshots`**
*   **`profile_and_fees`** (Wide: Static metadata, expense ratios)
*   **`profile_and_fees_reports`** (Tall: Expense breakdowns)
*   **`profile_and_fees_stylebox`** (Wide: Bitmasked Morningstar Style Box coordinates)

### Holdings
*   **`holdings_snapshots`**
*   **`holdings_asset_type`** (Wide: Equity vs. Cash vs. Fixed Income)
*   **`holdings_industry`** (Tall: Industry breakdown)
*   **`holdings_currency`** (Tall: Currency exposure)
*   **`holdings_investor_country`** (Tall: Geographic exposure)
*   **`holdings_debt_type`** (Tall: Debt instruments)
*   **`holdings_debtor_quality`** (Tall: Credit ratings)
*   **`holdings_maturity`** (Tall: Bond maturity buckets)
*   **`holdings_top10`** (Tall: Specific security holdings)
*   **`holdings_geographic_weights`** (Tall: Regional exposure)

### Ratios
*   **`ratios_snapshots`**
*   **`ratios_key_ratios`**, **`ratios_financials`**, **`ratios_fixed_income`**, **`ratios_dividend`**, **`ratios_zscore`** (Tall: Metrics vs. industry averages)

### Ratings & Commentary
*   **`lipper_ratings_snapshots`**
*   **`lipper_ratings`** (Tall: Peer-group rankings)
*   **`morningstar_snapshots`**
*   **`morningstar_summary`** (Wide: Analyst ratings)
*   **`morningstar_commentary`** (Tall: Verbatim analyst text)

### ESG
*   **`esg_snapshots`**
*   **`esg`** (Wide: Comprehensive Environmental, Social, and Governance scores)

### Time Series Data
*   **`price_chart_snapshots`**
*   **`price_chart_series`** (OHLC historical data)
*   **`sentiment_snapshots`**
*   **`sentiment_series`** (Social sentiment metrics over time)
*   **`dividends_snapshots`**
*   **`dividends_industry_metrics`**
*   **`dividends_events_series`** (Specific dividend payout events)
*   **`ownership_snapshots`**
*   **`ownership_owners_types`**, **`ownership_holders`**
*   **`ownership_trade_log_series_raw`**, **`ownership_trade_log_series_latest`** (Insider/Institutional trade history)
