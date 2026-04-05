# Simplify V2 Orchestration First, Then Prune Unused Storage Surface

**Summary**
- Keep the current canonical persisted inputs exactly where they are now: the semi-raw SQLite tables written by ingestion.
- Do not add new post-processed input tables to SQLite.
- During a single `run_pipeline` or `run_analysis` invocation, preprocess price and snapshot inputs in memory once and reuse them within that invocation.
- Keep `preprocess_prices`, `preprocess_snapshots`, and `preprocess_dividends` as standalone utility/export commands.
- Remove SQLite telemetry persistence and keep JSON telemetry as the only ingest telemetry sink.
- Do this in two passes: first fix duplicate orchestration work and telemetry duplication, then remove unused generic storage/config surface.

**Implementation Changes**
- Pass 1: make `run_factor_research` in [`pystocks/analysis/__init__.py`](/Users/alex/Documents/pystocks/pystocks/analysis/__init__.py) the canonical full-analysis entrypoint.
- Pass 1: change `run_analysis_pipeline` to delegate to `run_factor_research` instead of separately calling `build_analysis_panel` first.
- Pass 1: change `run_pipeline` in [`pystocks/cli.py`](/Users/alex/Documents/pystocks/pystocks/cli.py) so it no longer runs `preprocess_prices` as a separate pipeline stage. The pipeline becomes products -> fundamentals -> analysis, and the analysis stage owns one in-memory price preprocess and one snapshot-feature preprocess.
- Pass 1: keep `build_analysis_panel` as a standalone panel-only command. It may still preprocess inputs locally when called directly, because that is a separate invocation.
- Pass 1: keep `run_snapshot_preprocess` and `run_dividend_preprocess` as standalone export utilities. Analysis must not depend on their parquet outputs.
- Pass 1: remove SQLite ingest telemetry writes from [`pystocks/ingest/fundamentals.py`](/Users/alex/Documents/pystocks/pystocks/ingest/fundamentals.py) and delete the matching storage/schema code for `persist_ingest_run`, `ingest_runs`, and `ingest_run_endpoint_rollups`. JSON telemetry files remain unchanged.
- Pass 1: simplify CLI/result messaging so price preprocessing is described as part of analysis, not as a separate persisted pipeline stage.

- Pass 2: remove `endpoint_scalar_extras` writes and schema if there is still no in-repo reader after Pass 1.
- Pass 2: remove the dead `include_deferred_families` flag and make the current snapshot-family inclusion unconditional.
- Pass 2: narrow the public storage surface by stopping broad re-exports from `pystocks.storage`; import concrete modules directly instead.
- Pass 2: remove or inline the tiny `replace_table` wrapper if it is only serving as an extra abstraction layer.
- Pass 2: update docs/tests to reflect the reduced orchestration and schema surface. Do not touch diagnostics duplication in this pass.

**Public API / Interface Decisions**
- No new CLI commands.
- `preprocess_prices`, `preprocess_snapshots`, `preprocess_dividends`, `build_analysis_panel`, `run_factor_research`, and `run_analysis` remain available.
- `run_analysis_pipeline` stays as a compatibility wrapper, but becomes a thin alias to the canonical research path.
- No new SQLite tables are introduced for preprocessed inputs.
- Standalone preprocess commands continue writing parquet artifacts, but those artifacts are not treated as required inputs for downstream commands.
- It is acceptable for `run_pipeline` to drop the separate `prices` stage/result and fold that work into `analysis`, because preprocessing is now an internal analysis concern rather than a distinct pipeline dependency.

**Test Plan**
- Add an orchestration test that proves a single `run_analysis` invocation triggers one price preprocess and one snapshot-feature preprocess.
- Add a pipeline test that proves `run_pipeline` no longer calls `preprocess_prices` separately and still completes end-to-end.
- Keep standalone command coverage for `preprocess_prices`, `preprocess_snapshots`, `preprocess_dividends`, and `build_analysis_panel`, including parquet output generation.
- Add/adjust ingest tests so JSON telemetry is still written and SQLite telemetry persistence is gone.
- Add schema/storage tests so fresh initialization no longer creates telemetry tables, and Pass 2 no longer creates `endpoint_scalar_extras` once that removal lands.
- Run `ruff`, `pyright`, `pytest`, and `refresh_fundamentals_views` after the schema changes.

**Assumptions and Defaults**
- Preprocessed inputs are ephemeral in-memory data for a single invocation; they are not a new persistence layer.
- The current analysis output persistence can stay as-is for now; this plan does not simplify analysis result storage yet.
- The standalone preprocess commands are retained because they are useful utilities, not because the main pipeline should depend on their outputs.
- Pass 2 removes only code and schema surface that has no active in-repo consumer.
