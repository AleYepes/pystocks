# Pystocks

This repository is an ETF ingestion and analysis pipeline.

## Refactoring from the ground up
- Use `pystocks/` modules as the new production source of truth.
- Treat `src/` and `notebooks/` as historical/reference unless explicitly requested.

## New Execution Flow
1. Validate/login session: `pystocks/session.py`
2. Scrape product universe: `pystocks/product_scraper.py`
3. Scrape fundamentals/series payloads: `pystocks/fundamentals.py`
4. CAS + parquet + DuckDB materialization: `pystocks/fundamentals_store.py`
5. Price preprocessing and eligibility: `pystocks/price_preprocess.py`
6. Daily factor analysis: `pystocks/analysis.py`

- One-command full run: `python -m pystocks.cli run_pipeline`

## Code Guidelines
- Do not write comments unless the code requires critical info that cannot be easily infered.
- Do not add backfill or migration logic unless explicitly requested.
- Prefer DuckDB + parquet flows over CSV duplication.
- Never run destructive git commands unless explicitly requested.

## Validation Before Handoff
- Run: `./venv/bin/python -m pytest -q`
- If storage/view logic changed, run: `python -m pystocks.cli refresh_fundamentals_views`
