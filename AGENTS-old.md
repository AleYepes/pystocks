# Repository Guidelines

## Project Structure & Module Organization
`pystocks/` is the active codebase. Keep ingestion logic in `pystocks/ingest/` (`session.py`, `product_scraper.py`, `fundamentals.py`), SQLite schema and normalization in `pystocks/storage/`, preprocessing in `pystocks/preprocess/`, and factor research in `pystocks/analysis/`. Tests live under `pystocks/tests/` and mirror the runtime modules: `ingest`, `storage`, `preprocess`, `analysis`, and `diagnostics`. Use `docs/` for operational notes and samples. Treat `src/` and `notebooks/` as historical reference unless a task explicitly targets them.

## Build, Test, and Development Commands
Install dependencies with `./venv/bin/pip install -r requirements.txt` and hooks with `./venv/bin/pre-commit install`.

Common CLI entrypoints:
- `./venv/bin/python -m pystocks.cli scrape_products`
- `./venv/bin/python -m pystocks.cli scrape_fundamentals --limit 100 --conids_file docs/sample_conids.txt`
- `./venv/bin/python -m pystocks.cli preprocess_prices`
- `./venv/bin/python -m pystocks.cli preprocess_dividends`
- `./venv/bin/python -m pystocks.cli preprocess_snapshots`
- `./venv/bin/python -m pystocks.cli run_analysis`
- `./venv/bin/python -m pystocks.cli run_pipeline --limit 100`

Quality checks:
- `./venv/bin/python -m ruff check . --fix`
- `./venv/bin/python -m ruff format .`
- `./venv/bin/python -m pyright`
- `./venv/bin/python -m pytest -q`

Run `./venv/bin/python -m pystocks.cli refresh_fundamentals_views` after storage or SQLite view changes.

## Coding Style & Naming Conventions
Target Python 3.12. Use 4-space indentation, double quotes, and snake_case for modules, functions, variables, and test names. Let Ruff handle formatting and import sorting; do not hand-format around it. Prefer extending the existing pipeline modules over adding root-level scripts. Keep price, dividend, snapshot, and analysis responsibilities separate unless the task explicitly requires integration.

## Testing Guidelines
Use `pytest`. Add tests near the affected area, for example `pystocks/tests/storage/test_holdings_storage.py`. Name files `test_*.py` and prefer behavior-focused test names. Cover schema shape, effective-date rules, idempotency, and pipeline wiring when touching storage or CLI flow.

## Commit & Pull Request Guidelines
Recent commits use short conventional prefixes such as `refactor:`, `build:`, `docs:`, and `chore:`. Keep each commit scoped to one concern and write an imperative summary. PRs should state which pipeline stage changed, note schema or artifact impacts, and list the validation commands you ran.

## Data & Architecture Notes
The canonical store is `data/pystocks.sqlite`. Raw payload blobs are retained, but analysis should use normalized endpoint tables and preprocess outputs. Effective dates are currently anchored from `ratios.as_of_date`; do not add migrations or backfill logic unless explicitly requested.
