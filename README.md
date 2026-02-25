# pystocks v0.3.0

SQLite-first ETF ingestion pipeline for IBKR fundamentals and series snapshots.

## Current runtime scope
- Auth/session: `pystocks/session.py`
- Product universe scrape: `pystocks/product_scraper.py`
- Fundamentals + series scrape: `pystocks/fundamentals.py`
- Endpoint-centric SQLite storage: `pystocks/fundamentals_store.py`

Postprocessing and analysis are intentionally deferred in this refactor.

## CLI
```bash
python -m pystocks.cli scrape_products
python -m pystocks.cli scrape_fundamentals --limit 100
python -m pystocks.cli refresh_fundamentals_views
python -m pystocks.cli run_pipeline --limit 100
```

`run_pipeline` now executes only:
1. `scrape_products`
2. `scrape_fundamentals`

## Data layout
- Canonical DB: `data/pystocks.sqlite`
- Telemetry JSON artifacts: `data/research/fundamentals_run_telemetry_*.json`

## Storage model
- `products` table keyed by `conid`
- Per-endpoint snapshot tables keyed by `(conid, effective_at)`
- Endpoint child tables for nested payload structures
- Raw payload blob table (`raw_payload_blobs`) keyed by payload hash
- Series stored as:
1. append-only `*_series_raw`
2. deduped `*_series_latest`

## Legacy note
- Existing DuckDB/parquet stores are now legacy and not used by ingestion runtime.
- They remain on disk until explicitly removed.
