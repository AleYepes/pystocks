# Agent Onboarding

This repo is an IBKR ETF ingestion + analysis pipeline. For production behavior, treat `pystocks/` as source of truth.

## What Matters First
- Runtime modules:
1. `pystocks/session.py`: validates saved auth state and supports interactive reauth.
2. `pystocks/product_scraper.py`: loads ETF universe into DuckDB `instruments`.
3. `pystocks/fundamentals.py`: scrapes per-conid fundamentals and series payloads.
4. `pystocks/fundamentals_store.py`: CAS blobs + parquet materialization + DuckDB views.
5. `pystocks/price_preprocess.py`: dedupe/quality/eligibility + clean price views.
6. `pystocks/analysis.py`: factor returns + per-asset elastic-net betas.
- `src/` and `notebooks/` are historical/reference paths.

## Critical Runtime Facts
- `./venv/bin/python -m pystocks.cli run_pipeline` defaults to `limit=100`.
- `scrape_fundamentals` skips conids already scraped today unless `--force` is set.
- Follow-up endpoint fanout is skipped when `landing.key_profile.data.total_net_assets` is missing.
- Endpoint dedupe key is `(conid, endpoint, effective_at, payload_hash)`.
- `preprocess_prices` rewrites `data/prices/ibkr_mf_performance_chart_clean/`.

## Data Contract
- Manifest + dedupe metadata: `data/fundamentals/events.db`
- DuckDB query layer: `data/fundamentals/fundamentals.duckdb`
- Raw payload CAS blobs: `data/fundamentals/blobs/`
- Endpoint snapshots: `data/fundamentals/parquet/endpoint=*/year=*/month=*/*.parquet`
- Factor rows: `data/factors/ibkr_factor_features/endpoint=*/conid=*/*.parquet`
- Series stores:
1. `data/prices/ibkr_mf_performance_chart/`
2. `data/sentiment/ibkr_sma_search/`
3. `data/ownership/ibkr_ownership_trade_log/`
4. `data/dividends/ibkr_dividends_events/`
- Research artifacts: `data/research/`

## Current Local Snapshot (As Of 2026-02-24 UTC)
- Last telemetry run: 100 conids processed, 854 endpoint events, 31,717 factor rows, 357,252 series rows.
- `instruments`: 20,790 total, 100 marked `success` on 2026-02-24.
- Current tail outputs: 89 conids in `price_quality_catalog`, 59 eligible, 53 regressed assets.
- Built factors in latest analysis: `Mkt-RF`, `SMB`.

## Fast Start Commands
```bash
./venv/bin/python -m pystocks.cli scrape_products
./venv/bin/python -m pystocks.cli scrape_fundamentals --limit 100 --verbose
./venv/bin/python -m pystocks.cli refresh_fundamentals_views
./venv/bin/python -m pystocks.cli preprocess_prices
./venv/bin/python -m pystocks.cli run_analysis
```

One-command run (still limit-bound unless overridden):
```bash
./venv/bin/python -m pystocks.cli run_pipeline --limit 100
```
