# Agent Project Context: pystocks

## Project Overview
`pystocks` is an ETF factor analysis pipeline designed to calculate efficient frontier portfolios. It automates the lifecycle of financial data: from scraping product lists and fundamental data (via IBKR web proxies) to storage in a normalized SQLite database, preprocessing into analysis-ready features, and executing factor research.

### Core Technologies
- Data Storage: SQLite (primary database at `data/pystocks.sqlite`)
- CLI Framework: [Python Fire](https://github.com/google/python-fire)
- Quality Tools: Ruff (linting/formatting), Pyright (static typing), Pytest (testing)
- Data Science Stack: pandas (heavy usage in preprocessing and analysis)

## Project Structure
- `pystocks/`: Active production codebase.
    - `ingest/`: Authentication (`session.py`), product scraping (`product_scraper.py`), and fundamentals scraping (`fundamentals.py`).
    - `storage/`: SQLite schema (`schema.py`), connection management (`_sqlite.py`), and normalization logic (`fundamentals_store.py`).
    - `preprocess/`: Clean and transform raw snapshots into point-in-time features (prices, dividends, snapshots).
    - `analysis/`: Research engine for analysis panels, factor returns, and ETF betas.
    - `tests/`: Mirrored structure of the runtime modules (e.g., `tests/ingest/`, `tests/storage/`).
- `docs/`: Operational notes, plans, and sample data (e.g., `sample_conids.txt`).
- `data/`: SQLite database and research/telemetry artifacts.
- `src/` & `notebooks/`: Historical/reference material; do not modify unless explicitly requested.

## Development Workflow

### Environment Setup
```bash
./venv/bin/pip install -r requirements.txt
./venv/bin/pre-commit install
```

### Key CLI Commands
Main entry point: `pystocks/cli.py`.
- Full Pipeline: `./venv/bin/python -m pystocks.cli run_pipeline --limit 100`
- Scraping: `./venv/bin/python -m pystocks.cli scrape_fundamentals --limit 100 --conids_file docs/sample_conids.txt`
- Preprocessing: `./venv/bin/python -m pystocks.cli preprocess_prices`
- Analysis: `./venv/bin/python -m pystocks.cli run_analysis`
- Maintenance: `./venv/bin/python -m pystocks.cli refresh_fundamentals_views` (Run after storage or view changes).

### Code Quality & Testing
Always run these before committing:
- Lint & Format: `./venv/bin/python -m ruff check . --fix && ./venv/bin/python -m ruff format .`
- Type Check: `./venv/bin/python -m pyright`
- Test: `./venv/bin/python -m pytest -q`

## Technical Conventions

### Coding Style
- Formatting: 4-space indentation, double quotes, snake_case for all symbols.
- Tooling: Let Ruff handle all formatting and import sorting.
- Typing: Pyright is enforced for `pystocks/`. New code must be fully typed.

### Testing Guidelines
- Use `pytest` and name files `test_*.py`.
- Add tests in the mirrored `pystocks/tests/` directory.
- Prioritize behavior-focused tests covering schema shape, effective-date rules, and pipeline wiring.

### Commit Guidelines
- Use conventional prefixes: `refactor:`, `build:`, `docs:`, `chore:`, `feat:`, `fix:`.
- Use imperative summary lines.

### Data & Architecture
- Storage Model: Snapshot-based. Raw payloads are hashed and stored in `raw_payload_blobs`; analysis should target normalized endpoint tables.
- Point-in-Time Semantics: Treat `observed_at`, endpoint-specific `as_of_date`, storage `effective_at`, and analysis rebalance/join dates as distinct concepts. Do not anchor unrelated endpoint data to `ratios.as_of_date` by default.
- Asynchronous I/O: Ingestion tasks are primarily `asyncio`-based.

## Architectural Guardrails

### Stage Ownership
- Put each change in the stage that owns the concern.
- `ingest` fetches and hands off source data.
- `storage` owns canonical persistence and reader contracts.
- `preprocess` owns feature shaping, pivoting, and diagnostics.
- `analysis` consumes stable preprocess outputs and produces derived research outputs.

### Dependency Direction
- Keep dependency direction strict: `ingest -> storage`, `preprocess -> storage`, `analysis -> preprocess`.
- Do not introduce reverse dependencies.
- Do not let `ingest` call `preprocess`.
- Do not let `analysis` depend on raw storage shape when a preprocess contract should exist.

### Data Contracts
- Define each cross-stage contract in one authoritative place.
- Prefer explicit contracts over string-typed conventions spread across modules.
- Add new analysis features through preprocess-owned contracts, not direct storage-column reach-ins.
- Keep reader APIs consumer-oriented and explicit.

### Time Semantics
- Keep distinct time concepts distinct.
- Preserve source observation time, source as-of date, storage effective date, and analysis join dates when they carry different meaning.
- Do not collapse time semantics for convenience.
- Add or update behavior-focused tests when changing joins, effective dates, or publication timing.

### Canonical Storage
- Storage should preserve stable facts, not analysis panel shape.
- Keep open-ended or variable-cardinality data tall unless there is a strong semantic reason not to.
- Only keep canonical tables wide when the record shape is closed-world and stable.
- Prefer schema-enforced integrity over Python choreography when the semantics are known.

### Derived Outputs
- Keep derived research outputs separate from operational source-of-truth storage whenever practical.
- Do not introduce new derived persistence without clear ownership and lifecycle.

### Change Checklist
- Confirm which stage owns the change before editing.
- Confirm the change does not create a reverse dependency.
- Confirm the contract is authoritative in one place.
- Confirm timestamp semantics are explicit.
- Confirm the change does not move analysis shaping into canonical storage.
- When in doubt, prefer clearer boundaries over convenience shortcuts.
