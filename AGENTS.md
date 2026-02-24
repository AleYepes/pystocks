# AGENTS.md

## Canonical Runtime Path
- Use `pystocks/` modules as the production source of truth.
- Treat `src/` and `notebooks/` as historical/reference unless explicitly requested.

## Standard Run Order
1. `python -m pystocks.cli scrape_products`
2. `python -m pystocks.cli scrape_fundamentals --limit 100`
3. `python -m pystocks.cli refresh_fundamentals_views`
4. `python -m pystocks.cli preprocess_prices`
5. `python -m pystocks.cli run_analysis`

- One-command full run: `python -m pystocks.cli run_pipeline --limit 100`

## Data Expectations
- `data/fundamentals/events.db`: endpoint event/manifest log.
- `data/fundamentals/fundamentals.duckdb`: analytics/query views.
- Series stores outside `data/fundamentals/` must keep one `series.parquet` per `conid` and extend it (no per-run file fanout).

## Guardrails
- Do not add backfill or migration logic unless explicitly requested.
- Prefer DuckDB + parquet flows over CSV duplication.
- Never run destructive git commands unless explicitly requested.

## Validation Before Handoff
- Run: `./venv/bin/python -m pytest -q`
- If storage/view logic changed, run: `python -m pystocks.cli refresh_fundamentals_views`
- Report exactly what you ran and any failures/limits.
