# pystocks v0.2.0

IBKR-first ETF fundamentals and series ingestion pipeline with a lean tail-end analysis workflow.

## What this repo currently does
- Authenticated IBKR portal session handling (`pystocks/session.py`)
- ETF universe discovery (`pystocks/product_scraper.py`)
- Fundamentals + series scraping (`pystocks/fundamentals.py`)
- Raw+normalized storage and DuckDB views (`pystocks/fundamentals_store.py`)
- Price cleaning and eligibility gating (`pystocks/price_preprocess.py`)
- Lean daily factor analysis (`pystocks/analysis.py`)

## Current CLI commands
```bash
# discovery + ingestion
python -m pystocks.cli scrape_products
python -m pystocks.cli scrape_fundamentals --limit 100

# storage maintenance
python -m pystocks.cli backfill_fundamentals_normalized
python -m pystocks.cli refresh_fundamentals_views

# tail-end pipeline
python -m pystocks.cli preprocess_prices
python -m pystocks.cli run_analysis
python -m pystocks.cli run_pipeline
```

`run_pipeline` executes:
1. `preprocess_prices`
2. `run_analysis`

## Data locations (current)
- Manifest/event store: `data/fundamentals/events.db`
- DuckDB query layer: `data/fundamentals/fundamentals.duckdb`
- Raw CAS blobs: `data/fundamentals/blobs/`
- Endpoint parquet: `data/fundamentals/parquet/endpoint=*/...`
- Factor features parquet: `data/factors/ibkr_factor_features/`
- Price series parquet: `data/prices/ibkr_mf_performance_chart/`
- Cleaned price parquet: `data/prices/ibkr_mf_performance_chart_clean/`
- Research outputs: `data/research/`

## Notes for new contributors
- `src/` notebooks/scripts are historical references, not the production runtime path.
- Primary production modules live in `pystocks/`.
- Start with docs in this order:
  1. `docs/1.OVERVIEW.md`
  2. `docs/2.REFACTOR_SUMMARY.md`
  3. `docs/3.IBKR_PORTAL_API.md`
  4. `docs/4.DATA_SCHEMA.md`
  5. `docs/5.IMPLEMENTATION_REPORT.md`
  6. `docs/6.PLAN.md`
