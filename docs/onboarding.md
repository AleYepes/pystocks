# Agent Onboarding

This repo is an IBKR ETF ingestion pipeline with SQLite-first storage.

## Runtime modules
1. `pystocks/session.py`: validates saved auth state and supports interactive reauth.
2. `pystocks/product_scraper.py`: loads ETF universe into SQLite `products`.
3. `pystocks/fundamentals.py`: scrapes per-conid fundamentals/series payloads.
4. `pystocks/fundamentals_store.py`: raw payload blobs + flattened endpoint tables + series raw/latest tables.

`src/` and `notebooks/` are historical/reference paths.

## Canonical data store
- `data/pystocks.sqlite`

Key table groups:
- `products`
- `raw_payload_blobs`
- endpoint snapshot tables (`*_snapshots`)
- endpoint child tables (e.g. `ratios_metrics`, `holdings_top10`)
- series raw/latest tables (`*_series_raw`, `*_series_latest`)
- run telemetry (`ingest_runs`, `ingest_run_endpoint_rollups`)

## Ingestion behavior
- Endpoint overwrite key for non-series snapshots: `(conid, effective_at)`
- Same hash on same key: unchanged no-op
- Changed hash on same key: overwrite snapshot + replace child rows
- Series writes on changed/inserted events:
1. append into raw table
2. upsert latest table by natural row key

## Fast start commands
```bash
./venv/bin/python -m pystocks.cli scrape_products
./venv/bin/python -m pystocks.cli scrape_fundamentals --limit 100 --verbose
./venv/bin/python -m pystocks.cli refresh_fundamentals_views
./venv/bin/python -m pystocks.cli run_pipeline --limit 100
```

`run_pipeline` currently runs only products + fundamentals.

## Deferred modules
- `pystocks/price_preprocess.py`
- `pystocks/analysis.py`

These are intentionally out of scope for the SQLite-first refactor.

## Validation before handoff
```bash
./venv/bin/python -m pytest -q
```
