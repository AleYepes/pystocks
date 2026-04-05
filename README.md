# pystocks v0.3.0

ETF factor analysis pipeline to calculate efficient frontier portfolios.

`pystocks/` is the active production codebase. Treat `src/` and `notebooks/` as
historical/reference material unless a task explicitly targets them.

## Current runtime scope
- Auth/session: `pystocks/ingest/session.py`
- Product universe scrape: `pystocks/ingest/product_scraper.py`
- Fundamentals + series scrape: `pystocks/ingest/fundamentals.py`
- Endpoint-centric SQLite storage: `pystocks/storage/fundamentals_store.py`
- Preprocessing: `pystocks/preprocess/`
- Analysis: `pystocks/analysis/`

## CLI
```bash
./venv/bin/python -m pystocks.cli scrape_products
./venv/bin/python -m pystocks.cli scrape_fundamentals --limit 100
./venv/bin/python -m pystocks.cli preprocess_prices
./venv/bin/python -m pystocks.cli preprocess_dividends
./venv/bin/python -m pystocks.cli preprocess_snapshots
./venv/bin/python -m pystocks.cli build_analysis_panel
./venv/bin/python -m pystocks.cli run_factor_research
./venv/bin/python -m pystocks.cli run_analysis
./venv/bin/python -m pystocks.cli run_pipeline --limit 100
```

`run_pipeline` executes:
1. `scrape_products`
2. `scrape_fundamentals`
3. `run_analysis`

Price and snapshot preprocessing are performed inside the analysis invocation for that run. Standalone preprocess commands remain available for explicit export, debugging, and inspection workflows.

Use `--conids_file docs/sample_conids.txt` with `scrape_fundamentals` or `run_pipeline` to target a fixed conid list.

## Development workflow
Install dependencies with:

```bash
./venv/bin/pip install -r requirements.txt
```

Install commit hooks with:

```bash
./venv/bin/pre-commit install
```

Run the code quality stack locally before committing:

```bash
./venv/bin/python -m ruff check . --fix
./venv/bin/python -m ruff format .
./venv/bin/python -m pyright
./venv/bin/python -m pytest -q
```

`ruff` is the formatter, linter, and import sorter. `pyright` handles fast static type checking for the active `pystocks/` codebase.
Pyright is intentionally scoped away from a small set of existing pandas-heavy modules with known type debt so the check is enforceable in day-to-day development.

If you change storage or SQLite-backed analysis outputs, also run:

```bash
./venv/bin/python -m pystocks.cli refresh_fundamentals_views
```

## Data layout
- Canonical DB: `data/pystocks.sqlite`
- Telemetry JSON artifacts: `data/research/fundamentals_run_telemetry_*.json`

## Storage model
- `products` table keyed by `conid`
- Per-endpoint snapshot tables keyed by `(conid, effective_at)`
- Endpoint child tables for nested payload structures
- Raw payload blob table (`raw_payload_blobs`) keyed by payload hash
- JSON telemetry is file-based only; ingest telemetry is not persisted in SQLite
- Series tables are endpoint-specific; use the schema as the source of truth rather than assuming every endpoint has both `*_series_raw` and `*_series_latest`

## Data semantics
- Effective dates for persisted endpoint snapshots are currently anchored from `ratios.as_of_date`.
- Raw `*_snapshots` tables are storage metadata, not analysis-ready features.
- Snapshot features are rebuilt from normalized SQLite tables in `pystocks/preprocess/snapshots.py`.
- Price, dividend, sentiment, and ownership histories are handled as series data; analysis currently consumes price and snapshot preprocess outputs directly.
