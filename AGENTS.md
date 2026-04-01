# Pystocks

This repository is an ETF ingestion and factor analysis pipeline to calculate efficient frontier portfolios.

## Refactoring from the ground up
- Use `pystocks/` modules as the new production source of truth.
- Treat `src/` and `notebooks/` as historical/reference unless explicitly requested.

## New Execution Flow
1. Validate/login session: `pystocks/session.py`
2. Scrape product universe: `pystocks/product_scraper.py`
3. Scrape fundamentals/series payloads: `pystocks/fundamentals.py`
4. SQLite materialization + maintenance: `pystocks/fundamentals_store.py`
5. Series preprocessing:
   - prices: `pystocks/preprocess/price.py`
   - dividends: `pystocks/preprocess/dividends.py`
6. Snapshot-feature preprocessing: `pystocks/preprocess/snapshots.py`
7. Daily factor analysis: `pystocks/analysis.py`

- Compatibility wrappers may still exist, but new work should land in `pystocks/preprocess/`.
- One-command full run: `python -m pystocks.cli run_pipeline`
- Standalone preprocess entrypoints:
  - `python -m pystocks.cli preprocess_prices`
  - `python -m pystocks.cli preprocess_dividends`
  - `python -m pystocks.cli preprocess_snapshots`

## Data Semantics
- Treat raw `*_snapshots` tables as storage metadata, not analysis-ready features.
- Treat dated endpoint tables keyed by `(conid, effective_at)` as snapshot features for rebalance-date analysis.
- Treat prices, dividends, and sentiment as series features.
- Snapshot feature work belongs in `pystocks/preprocess/snapshots.py`, not in `fundamentals_store.py` unless the task is about storage normalization.

## Code Guidelines
- Do not write comments unless the code requires critical info that cannot be easily infered.
- Do not add backfill or migration logic unless explicitly requested.
- Prefer SQLite + parquet flows over CSV duplication.
- Prefer extending `pystocks/preprocess/` over adding new root-level preprocessing scripts.
- Keep price, dividend, sentiment, and snapshot preprocessing concerns separate unless the task explicitly requires integration.
- Never run destructive git commands unless explicitly requested.

## Validation Before Handoff
- Run: `./venv/bin/python -m pytest -q`
- If storage/view logic changed, run: `python -m pystocks.cli refresh_fundamentals_views`
