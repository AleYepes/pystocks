# Repository Guidelines

## Project Structure & Module Organization
`pystocks/` is the production codebase. Follow the pipeline stages there: session auth under `pystocks/ingest/`, product scraping under `pystocks/ingest/`, fundamentals ingestion under `pystocks/ingest/`, SQLite storage under `pystocks/storage/`, preprocessing under `pystocks/preprocess/`, and factor analysis under `pystocks/analysis/`. Put new preprocessing work in `pystocks/preprocess/`, not in new root scripts. Tests live in `pystocks/tests/`. Reference material lives in `docs/` and image assets in `assets/`. Treat `src/` and `notebooks/` as historical unless a task explicitly targets them.

## Build, Test, and Development Commands
Install dependencies with `./venv/bin/pip install -r requirements.txt`.
Install hooks with `./venv/bin/pre-commit install`.

Use the CLI for local runs:
- `./venv/bin/python -m pystocks.cli run_pipeline --limit 100` runs products, fundamentals, price preprocessing, and analysis.
- `./venv/bin/python -m pystocks.cli preprocess_prices`
- `./venv/bin/python -m pystocks.cli preprocess_dividends`
- `./venv/bin/python -m pystocks.cli preprocess_snapshots`
- `./venv/bin/python -m pystocks.cli run_analysis`
- `./venv/bin/python -m pystocks.cli refresh_fundamentals_views` runs lightweight SQLite maintenance after storage/view changes.
- `./venv/bin/python -m ruff check . --fix` lints, sorts imports, and applies safe fixes.
- `./venv/bin/python -m ruff format .` formats Python code.
- `./venv/bin/python -m pyright` runs fast type checking on the currently enforced `pystocks/` subset.

## Coding Style & Naming Conventions
Use Python with 4-space indentation, snake_case for modules, functions, and variables, and short, focused functions. Let `ruff format` own formatting and import ordering. Prefer extending existing modules over introducing parallel workflows. Keep storage concerns in `pystocks/storage/fundamentals_store.py`, snapshot feature assembly in `pystocks/preprocess/snapshots.py`, and series preprocessing separated by domain. Avoid unnecessary comments, backfill logic, or destructive git operations.

## Testing Guidelines
The test suite uses `pytest`. Name new tests `test_*.py` and keep them near the affected area under `pystocks/tests/`. Before handoff, run `./venv/bin/python -m ruff check . --fix`, `./venv/bin/python -m ruff format .`, `./venv/bin/python -m pyright`, and `./venv/bin/python -m pytest -q`. Pyright currently excludes a few legacy type-debt modules, so keep new work inside the checked surface when possible or tighten the config as you pay that debt down. If you change SQLite storage or view behavior, also run `./venv/bin/python -m pystocks.cli refresh_fundamentals_views`.

## Commit & Pull Request Guidelines
Recent history uses short conventional prefixes such as `build:`, `refactor:`, `docs:`, and `fix:` followed by an imperative summary. Keep commits scoped to one concern. Pull requests should describe the pipeline stage touched, note any schema or artifact changes, link the related issue when applicable, and include the exact validation commands run.

## Data & Architecture Notes
Treat raw `*_snapshots` tables as storage metadata, not analysis-ready features. Snapshot features are point-in-time tables keyed by `(conid, effective_at)`, while prices, dividends, and sentiment are series features. The canonical store is `data/pystocks.sqlite`.
