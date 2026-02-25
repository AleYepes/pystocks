## SQLite-First Refactor Plan (Single DB, Endpoint-Centric, No Postprocessing)

### Summary
- Replace the current DuckDB + parquet + events.db storage stack with one SQLite database at `data/pystocks.sqlite`.
- Keep ingestion and endpoint storage only in this refactor; explicitly defer `price_preprocess.py` and `analysis.py`.
- Store each endpoint in explicit relational tables, endpoint by endpoint, with no stringified dict/list columns in normalized tables.
- Keep exact raw payload retention via a dedicated compressed blob table.
- Store series using a dual-table pattern: append-only raw rows plus deduped latest rows for fast querying and true time expansion.

### Scope and Non-Scope
- In scope: `product_scraper.py`, `ops_state.py`, `fundamentals.py`, `fundamentals_store.py`, `cli.py`, tests for ingestion/storage, onboarding and operational docs.
- In scope: telemetry persisted to SQLite (run summary + endpoint rollups) and still emitted to JSON artifacts.
- Out of scope: refactoring `price_preprocess.py` and `analysis.py` internals; factor panel / factor regression pipelines.
- Out of scope: historical migration from existing data files (fresh start only).

### Canonical SQLite File
- Path: `data/pystocks.sqlite`.
- SQLite pragmas on every write connection: `journal_mode=WAL`, `synchronous=NORMAL`, `foreign_keys=ON`, `temp_store=MEMORY`.
- Add `schema_meta` table with `schema_version` and `applied_at` for deterministic upgrades.

### Core Global Schema
1. `products`
- PK: `conid`.
- Columns: `symbol`, `exchange`, `isin`, `currency`, `name`, `last_scraped_fundamentals`, `last_status_fundamentals`, `updated_at`.
- Replaces instrument state currently stored in DuckDB.

2. `raw_payload_blobs`
- PK: `payload_hash` (sha256 canonical JSON).
- Columns: `compression` (`zstd`), `raw_size_bytes`, `compressed_size_bytes`, `payload_blob`, `created_at`.
- No endpoint table stores full JSON text.

3. `ingest_runs`
- PK: `run_id`.
- Columns: `run_started_at`, `run_finished_at`, `total_targeted_conids`, `processed_conids`, `saved_snapshots`, `inserted_events`, `overwritten_events`, `unchanged_events`, `series_raw_rows_written`, `series_latest_rows_upserted`, `auth_retries`, `aborted`.

4. `ingest_run_endpoint_rollups`
- PK: `(run_id, endpoint)`.
- Columns: `call_count`, `useful_payload_count`, `useful_payload_rate`, `status_2xx`, `status_4xx`, `status_5xx`, `status_other`.
- Replaces per-request log persistence; telemetry JSON remains.

### Endpoint Storage Contract
- Every endpoint main snapshot table uses PK `(conid, effective_at)`, FK to `products(conid)`, FK to `raw_payload_blobs(payload_hash)`.
- Shared metadata columns on each main snapshot table: `observed_at`, `payload_hash`, `source_file`, `inserted_at`, `updated_at`.
- Overwrite rule for non-series endpoints: same `(conid, effective_at)` updates row and replaces child rows when payload hash changes; identical hash is no-op (`unchanged_events`).

### Endpoint Tables (Explicit, No Dict/Stringified Nested Payloads)
1. `landing_snapshots`
- Core columns: `total_net_assets_text`, `has_mstar`, `has_ownership`, `has_mf_esg`.
- Child tables: `landing_key_profile_fields`, `landing_section_metrics`, `landing_top10_holdings`, `landing_top10_holding_conids`.

2. `profile_fees_snapshots`
- Core columns: `symbol`, `objective`, `jap_fund_warning`.
- Child tables: `profile_fees_fund_profile_fields`, `profile_fees_report_fields`, `profile_fees_expense_allocations`, `profile_fees_themes`.

3. `holdings_snapshots`
- Core columns: `as_of_date`, `top_10_weight`.
- Child tables: `holdings_bucket_weights` (allocation/industry/currency/investor_country/debt_type/debtor/maturity via `bucket_type`), `holdings_top10`, `holdings_top10_conids`, `holdings_geographic_weights`.

4. `ratios_snapshots`
- Core columns: `as_of_date`, `title_vs`.
- Child table: `ratios_metrics` with `section` (`ratios|financials|fixed_income|dividend|zscore`) and normalized metric columns (`value`, `vs`, `min`, `max`, `avg`, `percentile`, formatted strings).

5. `lipper_ratings_snapshots`
- Child table: `lipper_ratings_values` with `period` (`overall|3_year|5_year|10_year`), `metric_id`, `rating_value`, `rating_label`, `universe_as_of_date`.

6. `dividends_snapshots`
- Core columns from canonical normalization: `response_type`, `has_history`, `history_points`, `embedded_price_points`, `no_div_data_marker`, `no_div_data_period`, `no_dividend_text`, `last_paid_date`, `last_paid_amount`, `last_paid_currency`, `dividend_yield`, `annual_dividend`, `paying_companies`, `paying_companies_percent`, `dividend_ttm`, `dividend_yield_ttm`.
- Child table: `dividends_industry_metrics`.

7. `morningstar_snapshots`
- Core columns: `as_of_date`, `q_full_report_id`.
- Child tables: `morningstar_summary`, `morningstar_commentary`.

8. `performance_snapshots`
- Core columns: `title_vs`.
- Child table: `performance_metrics` with `section` (`cumulative|annualized|yield|risk|statistic`) and normalized numeric/format fields.

9. `ownership_snapshots`
- Core columns: ownership totals and normalized summary counts.
- Child tables: `ownership_owners_types`, `ownership_holders` (`holder_group` = `institutional|insider`).

10. `esg_snapshots`
- Core columns: `as_of_date`, `coverage`, `source`, `symbol`, `no_settings`.
- Child table: `esg_nodes` with hierarchical `node_path`, `parent_path`, `depth`, `node_value`.

11. `price_chart_snapshots`
- Core columns: `points_count`, `min_trade_date`, `max_trade_date`.

12. `sentiment_search_snapshots`
- Core columns: `points_count`, `min_trade_date`, `max_trade_date`.

13. Optional scalar overflow table for evolving fields
- `endpoint_scalar_extras(endpoint, conid, effective_at, path, value_text, value_num, value_bool, value_date)`.
- Strict rule: scalar leaves only; no serialized dict/list payloads.

### Series Storage Design (Expansion + Append Guaranteed)
- Pattern for each series endpoint: `*_series_raw` + `*_series_latest`.
- `*_series_raw` is append-only and stores lineage columns (`conid`, `effective_at`, `observed_at`, `payload_hash`, `inserted_at`) plus parsed series fields.
- `*_series_latest` is deduped by natural key for query speed.
- Upsert into latest uses recency precedence: newer `(effective_at, observed_at)` wins; equal timestamps prefer newer `payload_hash` lexical tie-break.
- Natural keys:
  - Price: `(conid, trade_date)` with fallback row key if date missing.
  - Sentiment: `(conid, datetime_ms)`.
  - Ownership trade log: deterministic composite key from trade_date + action + party + source + insider + shares + value + holding.
  - Dividends events: deterministic composite key from event_date + amount + currency + event_type + declaration/record/payment dates + description.
- Result: history is always appendable in raw tables; latest tables always represent current canonical state.

### Refactor Steps (Implementation Sequence)
1. Create SQLite schema bootstrap in `fundamentals_store.py` or dedicated `sqlite_schema.py`; add schema versioning.
2. Replace `ops_state.py` DuckDB logic with sqlite3 CRUD/upsert against `products`.
3. Refactor `product_scraper.py` to write `products` in SQLite and return SQLite path in result payload.
4. Rebuild `FundamentalsStore.persist_combined_snapshot` for SQLite:
- Keep effective date resolution logic.
- Store raw blob once by hash.
- Upsert endpoint main row by `(conid, effective_at)`.
- Replace endpoint child rows transactionally on overwrite.
- Append raw series rows and upsert latest rows.
5. Refactor `fundamentals.py` ingestion counters:
- Replace `duplicate_events`/factor counters with `inserted_events`, `overwritten_events`, `unchanged_events`, `series_raw_rows_written`, `series_latest_rows_upserted`.
- Persist run telemetry both to SQLite rollup tables and existing JSON files.
6. Update `cli.py`:
- `run_pipeline` now runs only `scrape_products` then `scrape_fundamentals`.
- Keep `preprocess_prices` and `run_analysis` commands but return deferred status message.
- Repurpose `refresh_fundamentals_views` into a SQLite maintenance command (`VACUUM` + stats) while retaining command name for compatibility.
7. Update docs (`README.md`, `docs/onboarding.md`, `docs/operational_checklist.md`) to SQLite architecture and commands.
8. Mark old parquet/duckdb notebook flow as legacy; add a SQLite inspection notebook replacing parquet-centric walkthrough.

### Public API / Interface Changes
- Config:
- Add `SQLITE_DB_PATH = data/pystocks.sqlite`.
- Deprecate runtime dependence on `FUNDAMENTALS_DUCKDB_PATH` and parquet directory constants for ingestion flow.
- `FundamentalsStore.persist_combined_snapshot` return payload:
- New counters: `inserted_events`, `overwritten_events`, `unchanged_events`, `series_raw_rows_written`, `series_latest_rows_upserted`.
- CLI:
- `run_pipeline` no longer executes preprocess/analysis steps.
- `refresh_fundamentals_views` becomes SQLite maintenance/health command.
- Telemetry JSON schema:
- Keep existing file outputs; include new overwrite/unchanged counters.

### Test Plan and Acceptance Scenarios
1. Unit tests for schema and upsert semantics.
- Inserts first snapshot row.
- Same hash same `(conid,effective_at)` is unchanged.
- Changed hash same `(conid,effective_at)` overwrites row and replaces child rows.

2. Endpoint flattening tests by fixture.
- One fixture payload per endpoint from `docs/fundamentals_samples`.
- Assert core table row count and critical child table rows/columns.

3. Series behavior tests.
- Re-ingesting full history with one extra day appends raw and expands latest by one new key.
- Re-ingesting corrected values for existing dates updates latest but keeps raw history.
- Ownership `NO CHANGE` drop behavior remains enforced for normalized trade-log storage.

4. Product/ops state tests.
- `products` upsert by `conid`.
- `update_instrument_fundamentals_status` behavior unchanged from current tests.

5. Telemetry tests.
- One ingest run writes exactly one `ingest_runs` row and endpoint rollups.
- JSON telemetry file still emitted and keys populated.

6. CLI contract tests.
- `run_pipeline` ends after fundamentals ingest and returns success payload.
- Deferred postprocessing commands return deterministic deferred status.

7. Full regression test run.
- Execute `./venv/bin/python -m pytest -q`.
- Validate no test references to DuckDB/parquet remain in ingestion/storage suite.

### Rollout and Cutover
- Cutover is immediate and fresh-start: SQLite starts empty; old DuckDB/parquet artifacts remain on disk but are no longer read by ingestion flow.
- No destructive cleanup command is run automatically.
- Add explicit “legacy storage unused” note in docs and onboarding.

### Assumptions and Defaults Locked
- Full replacement now, not dual-write.
- Fresh-start history; no migration from existing events/parquet/duckdb.
- Non-series dedupe key is `(conid, effective_at)` with overwrite on changed payload.
- Raw payloads retained in dedicated compressed blob table.
- Series stored as append-only raw plus deduped latest tables.
- Telemetry kept in both SQLite and JSON.
- Postprocessing and analysis are deferred; pipeline stops after fundamentals ingestion.
- Endpoint design is explicit per resource, with scalar overflow allowed only for scalar leaves; no dict/list stringification in normalized tables.
