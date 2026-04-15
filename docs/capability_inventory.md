# Pystocks Capability Inventory

This document inventories concrete system capabilities discovered from the current codebase.

It is not a target architecture.

Its purpose is to capture:

- what the repo can currently do
- what behavior appears intentional versus incidental
- what likely needs to survive a rebuild
- where the current implementation mixes concerns

## Inventory Conventions

- `Current location` identifies where the behavior is implemented today, not where it should live later.
- `Keep / redesign / drop` is a working recommendation for the rebuild discussion.
- `Time semantics` only names time concepts actually relevant to the capability.

## CLI-Ordered Inventory

### Capability: Bootstrap or refresh the product universe from IBKR

- Current entrypoint: [pystocks/cli.py](/Users/alex/Documents/pystocks/pystocks/cli.py)
  - `PyStocksCLI.scrape_products()`
- Current implementation: [pystocks/ingest/product_scraper.py](/Users/alex/Documents/pystocks/pystocks/ingest/product_scraper.py)
  - `fetch_api_direct()`
  - `scrape_ibkr_products()`
- Current persistence boundary: [pystocks/storage/ops_state.py](/Users/alex/Documents/pystocks/pystocks/storage/ops_state.py)
  - `upsert_instruments_from_products()`
- Canonical schema touched: [pystocks/storage/schema.py](/Users/alex/Documents/pystocks/pystocks/storage/schema.py)
  - `products`

Observed behavior:

- Calls the IBKR product search endpoint directly over HTTP.
- Requests ETF products only.
- Pages through the upstream catalog using `pageNumber` and a fixed page size.
- Retries timeouts, request errors, and HTTP `429` responses with backoff.
- Stops when a page is empty or shorter than the configured page size.
- Accumulates all returned product records for the run.
- Discards malformed records that are not dicts or do not contain a usable `conid`.
- Deduplicates by `conid`, keeping the last record seen in the run.
- Normalizes and upserts canonical product fields into SQLite.
- Returns a machine-readable status payload including upsert count and SQLite path.

Inputs:

- No explicit user arguments at the CLI today.
- Implicit upstream source: IBKR product search API.
- Implicit local destination: configured SQLite database.

Outputs:

- Canonical `products` rows keyed by `conid`.
- CLI/run status payload:
  - `status`
  - `products_upserted`
  - `sqlite_path`

Canonical fields currently persisted:

- `conid`
- `symbol`
- `exchange`
- `isin`
- `currency`
- `name`
- `updated_at`

Time semantics:

- `updated_at`: local persistence/update timestamp for the canonical product row

Notably absent:

- No immutable raw payload capture for the product-catalog response
- No separate fetch-run log for product-universe refreshes
- No explicit distinction between product master data and operational scrape state
- No tests directly covering the product scraper path yet

Intent assessment:

- Intentional requirements likely worth keeping:
  - universe bootstrap from upstream catalog
  - repeated refresh via upsert
  - canonical identity on `conid`
  - minimum metadata preservation for later joins
  - transient-failure retry behavior
  - machine-readable CLI result
- Current implementation details that may be redesigned:
  - direct hardcoded endpoint payload in the scrape function
  - tqdm progress handling inside the fetch loop
  - storing operational scrape status in the same `products` table
  - lack of raw capture for universe-source payloads

Keep / redesign / drop:

- Keep the capability.
- Redesign the ownership split between product master data, refresh telemetry, and operational scrape state.
- Redesign raw-capture treatment so the rebuild can decide whether product-catalog payloads deserve the same auditability guarantees as other sources.

Tests informing this inventory:

- [pystocks/tests/test_cli_contract.py](/Users/alex/Documents/pystocks/pystocks/tests/test_cli_contract.py)
- [pystocks/tests/storage/test_ops_state.py](/Users/alex/Documents/pystocks/pystocks/tests/storage/test_ops_state.py)
- [pystocks/tests/storage/test_storage_boundary.py](/Users/alex/Documents/pystocks/pystocks/tests/storage/test_storage_boundary.py)

### Capability: Run repeated fundamentals collection across the known universe

- Current entrypoint: [pystocks/cli.py](/Users/alex/Documents/pystocks/pystocks/cli.py)
  - `PyStocksCLI.scrape_fundamentals()`
- Current orchestration: [pystocks/ingest/fundamentals.py](/Users/alex/Documents/pystocks/pystocks/ingest/fundamentals.py)
  - `_select_conids_to_scrape()`
  - `FundamentalScraper`
  - `main()`
  - `run_fundamentals_update()`
- Current auth/session boundary: [pystocks/ingest/session.py](/Users/alex/Documents/pystocks/pystocks/ingest/session.py)
  - `IBKRSession`
- Current persistence boundary: [pystocks/storage/fundamentals_store.py](/Users/alex/Documents/pystocks/pystocks/storage/fundamentals_store.py)
  - `persist_combined_snapshot()`
  - raw blob storage
  - endpoint-specific persistence
- Current operational state boundary: [pystocks/storage/ops_state.py](/Users/alex/Documents/pystocks/pystocks/storage/ops_state.py)
  - recent-scrape selection
  - per-instrument scrape status updates

Observed behavior:

- Selects target instruments from the canonical product universe or from an explicit conid file.
- Supports skipping recently scraped instruments using a bounded recency window unless forced.
- Supports limiting and offsetting the target set for partial runs.
- Validates or creates an authenticated IBKR session before collection starts.
- Reauthenticates and resumes when authentication expires during a run, up to a bounded retry count.
- Fetches a landing payload first and uses it as a gate before fanning out to endpoint-specific requests.
- Treats some instruments as landing-only skips rather than hard failures when required payload signals are absent.
- Fans out across multiple endpoint families for each conid:
  - profile and fees
  - holdings
  - ratios
  - lipper ratings
  - dividends
  - morningstar detail
  - price chart
  - sentiment search
  - ownership
  - ESG when an account id is available
- Uses endpoint-specific usefulness heuristics so empty or weak payloads do not automatically count as meaningful data.
- Chooses the price-chart request window based on the latest stored series point for the instrument, allowing incremental extension instead of always refetching max history.
- Persists the combined snapshot through a storage layer that:
  - decomposes the combined payload into endpoint payloads
  - stores raw payload blobs by content hash
  - resolves endpoint effective dates
  - writes canonical snapshot and series tables
  - distinguishes inserted, overwritten, unchanged, and latest-upserted series results
- Updates per-instrument operational statuses such as success, auth error, empty payload, and structured skip reasons.
- Produces structured run telemetry summarizing endpoint call counts, useful payload rates, and status codes.
- Returns a machine-readable run result with counts for targeted instruments, processed instruments, saved snapshots, event writes, series writes, auth retries, and abort state.

Inputs:

- Optional `limit`
- Optional `start_index`
- Optional `conids_file`
- Optional `force`
- Optional auth retry and reauth mode settings
- Implicit canonical universe from SQLite
- Implicit authenticated IBKR portal session state

Outputs:

- Canonical endpoint snapshot and series rows for targeted instruments
- Deduplicated raw payload blobs
- Updated per-instrument operational scrape state
- Telemetry JSON artifacts in the research directory
- CLI/run status payload with aggregate counters

Time semantics observed:

- `scraped_at` / `observed_at`: acquisition time for the combined run or endpoint payload
- endpoint `as_of_date`: source-declared date when present
- `effective_at`: canonical storage date chosen per endpoint by the storage layer
- recent-scrape window: operational recency used to skip instruments already processed within the last seven days

Important sub-capabilities embedded in this flow:

- universe-target selection for recurring runs
- session validation and recovery
- landing-page screening before deeper fanout
- endpoint usefulness classification
- incremental historical series extension
- raw capture plus canonical normalization
- per-run telemetry and diagnostics

Notably absent or likely problematic:

- Endpoint fanout, usefulness heuristics, and skip policy are tightly embedded in one runner.
- Product operational state and scrape-state policy remain coupled to the `products` table.
- Telemetry is file-based only and not obviously modeled as a first-class run history.
- The capability boundary is broad enough that a rebuild may want to split targeting, acquisition, persistence, and telemetry into cleaner concerns.

Intent assessment:

- Intentional requirements likely worth keeping:
  - recurring fundamentals collection over a maintained universe
  - support for explicit target lists and partial runs
  - recency-based skipping plus force override
  - authenticated session reuse and reauthentication
  - structured skip outcomes distinct from fatal failures
  - multi-endpoint collection per instrument
  - incremental extension of historical price series
  - raw payload preservation for reparsing and auditability
  - canonical endpoint persistence with explicit write outcomes
  - machine-readable run results and run telemetry
- Current implementation details that may be redesigned:
  - landing-page `total_net_assets` as the hardcoded fanout gate
  - endpoint usefulness heuristics embedded directly in the scraper class
  - direct coupling between runner logic and per-instrument status strings
  - tight bundling of all endpoint families into one monolithic orchestration loop
  - telemetry lifecycle implemented only through JSON files in the research directory

Keep / redesign / drop:

- Keep the capability.
- Redesign the capability boundary into smaller concerns during the rebuild.
- Redesign targeting policy, endpoint policy, and telemetry ownership so they are explicit and easier to test independently.

Tests informing this inventory:

- [pystocks/tests/ingest/test_fundamentals_runner.py](/Users/alex/Documents/pystocks/pystocks/tests/ingest/test_fundamentals_runner.py)
- [pystocks/tests/ingest/test_fundamentals_payload_heuristics.py](/Users/alex/Documents/pystocks/pystocks/tests/ingest/test_fundamentals_payload_heuristics.py)
- [pystocks/tests/ingest/test_fundamentals_telemetry.py](/Users/alex/Documents/pystocks/pystocks/tests/ingest/test_fundamentals_telemetry.py)
- [pystocks/tests/ingest/test_price_chart_period_selection.py](/Users/alex/Documents/pystocks/pystocks/tests/ingest/test_price_chart_period_selection.py)

### Capability: Build cleaned price and return artifacts for downstream analysis

- Current entrypoint: [pystocks/cli.py](/Users/alex/Documents/pystocks/pystocks/cli.py)
  - `PyStocksCLI.preprocess_prices()`
- Current implementation: [pystocks/preprocess/price.py](/Users/alex/Documents/pystocks/pystocks/preprocess/price.py)
  - `load_price_history()`
  - `preprocess_price_history()`
  - `_compute_eligibility()`
  - `save_price_preprocess_results()`
  - `run_price_preprocess()`
- Current storage read boundary:
  - [pystocks/storage/readers.py](/Users/alex/Documents/pystocks/pystocks/storage/readers.py)
  - `load_price_history()`

Observed behavior:

- Loads canonical daily price history from storage.
- Derives a usable price value from available price columns.
- Flags invalid prices such as nonpositive values and malformed high/low ranges.
- Detects long stale runs where prices repeat beyond an allowed threshold.
- Detects extreme return outliers using a robust median/MAD-based rule.
- Detects bridge-style local price anomalies where a short internal stretch deviates implausibly between surrounding clean anchors.
- Produces both raw returns and cleaned returns.
- Produces row-level clean/flag columns rather than silently dropping bad data.
- Computes per-instrument eligibility summaries based on:
  - minimum clean history length
  - missing business-day ratio
  - maximum internal gap size
- Writes two downstream artifacts:
  - daily cleaned return data
  - price eligibility summary
- Returns a machine-readable status payload summarizing row count and eligible versus ineligible instruments.

Inputs:

- Canonical price history with instrument id and trade date
- Configurable thresholds for history length, missingness, stale runs, outlier detection, and local anomaly detection

Outputs:

- Row-level price artifact containing:
  - raw prices
  - chosen price value
  - cleaned price
  - raw return
  - clean return
  - validity and anomaly flags
- Per-instrument eligibility artifact
- Parquet outputs for downstream analysis reuse

Time semantics observed:

- `trade_date`: canonical daily price date used for return and coverage calculations

Intent assessment:

- Intentional requirements likely worth keeping:
  - explicit daily price cleaning before research use
  - separation of raw returns from cleaned returns
  - row-level diagnostics instead of silent filtering
  - per-instrument eligibility as a first-class output
  - persisted downstream artifacts for reuse
- Current implementation details that may be redesigned:
  - exact anomaly heuristics and thresholds
  - parquet filenames as the current materialization convention
  - whether this capability remains a standalone command or becomes part of a bundle builder

Keep / redesign / drop:

- Keep the capability.
- Redesign only the owning boundary and materialization strategy if needed.
- Preserve the distinction between raw price history, cleaned price series, and eligibility diagnostics.

Tests informing this inventory:

- [pystocks/tests/preprocess/test_price_preprocess.py](/Users/alex/Documents/pystocks/pystocks/tests/preprocess/test_price_preprocess.py)
- [pystocks/tests/preprocess/test_preprocess_commands.py](/Users/alex/Documents/pystocks/pystocks/tests/preprocess/test_preprocess_commands.py)

### Capability: Vet dividend events for total-return use

- Current entrypoint: [pystocks/cli.py](/Users/alex/Documents/pystocks/pystocks/cli.py)
  - `PyStocksCLI.preprocess_dividends()`
- Current implementation: [pystocks/preprocess/dividends.py](/Users/alex/Documents/pystocks/pystocks/preprocess/dividends.py)
  - `load_dividend_events()`
  - `_build_clean_price_reference()`
  - `preprocess_dividend_events()`
  - `_summarize_dividend_events()`
  - `run_dividend_preprocess()`
- Current storage read boundary:
  - [pystocks/storage/readers.py](/Users/alex/Documents/pystocks/pystocks/storage/readers.py)
  - `load_dividend_events()`
  - `load_price_history()` via clean-price reference construction

Observed behavior:

- Loads canonical dividend event history and associated instrument currency metadata.
- Reuses cleaned price history as a reference for evaluating dividend usability.
- Joins each dividend event to the most recent prior clean price.
- Computes price-reference age, implied yield versus prior price, and trailing 365-day dividend sums.
- Flags dividend events that are unsuitable for total-return adjustment, including:
  - missing amount
  - nonpositive amount
  - missing currency
  - dividend/product currency mismatch
  - duplicate event signatures
  - missing price reference
  - stale price reference
  - suspicious implied yield
- Produces a row-level dividend-event artifact with usability flags.
- Produces a per-instrument summary artifact describing usable coverage and failure reasons.
- Writes both artifacts to parquet for downstream analysis use.
- Returns a machine-readable status payload with row counts, conid counts, and usable event counts.

Inputs:

- Canonical dividend event history
- Instrument metadata needed for symbol and product currency context
- Cleaned price history or equivalent clean price reference
- Configurable thresholds for stale price references and suspicious implied yields

Outputs:

- Row-level dividend event artifact with usability and quality flags
- Per-instrument dividend summary artifact
- Parquet outputs for downstream reuse

Time semantics observed:

- `event_date`: canonical dividend event date
- prior `trade_date`: most recent clean-price reference date used to assess event usability
- trailing 365-day event window for rolling dividend totals

Intent assessment:

- Intentional requirements likely worth keeping:
  - dividend events must be vetted before use in total-return adjustment
  - dividend usability depends on both dividend integrity and price-reference integrity
  - row-level diagnostics plus per-instrument summaries are first-class outputs
  - cleaned price outputs are an upstream dependency for dividend usability
- Current implementation details that may be redesigned:
  - exact usability heuristics and thresholds
  - whether trailing dividend sums belong in this stage or a later feature stage
  - parquet filenames and standalone command shape

Keep / redesign / drop:

- Keep the capability.
- Preserve the dependency that dividend usability is evaluated against cleaned price context, not raw price history.
- Redesign only the exact owning boundary and materialization strategy if needed.

Tests informing this inventory:

- [pystocks/tests/preprocess/test_dividend_preprocess.py](/Users/alex/Documents/pystocks/pystocks/tests/preprocess/test_dividend_preprocess.py)
- [pystocks/tests/preprocess/test_preprocess_commands.py](/Users/alex/Documents/pystocks/pystocks/tests/preprocess/test_preprocess_commands.py)

### Capability: Build merged snapshot feature tables and snapshot diagnostics

- Current entrypoint: [pystocks/cli.py](/Users/alex/Documents/pystocks/pystocks/cli.py)
  - `PyStocksCLI.preprocess_snapshots()`
- Current implementation: [pystocks/preprocess/snapshots.py](/Users/alex/Documents/pystocks/pystocks/preprocess/snapshots.py)
  - `load_snapshot_feature_tables()`
  - `preprocess_snapshot_features()`
  - pivoting and prefixing helpers
  - holdings and ratio diagnostic builders
  - `run_snapshot_preprocess()`
- Current storage read boundary:
  - [pystocks/storage/readers.py](/Users/alex/Documents/pystocks/pystocks/storage/readers.py)
  - `load_snapshot_feature_tables()`

Observed behavior:

- Loads a curated set of canonical snapshot tables from storage rather than exposing raw storage shape directly.
- Normalizes those tables into a stable downstream feature-building input set.
- Converts selected snapshot values into analysis-facing scalar columns, including:
  - prefixed profile fields
  - wide holdings allocations
  - pivoted long holdings categories
  - pivoted ratio, dividend, morningstar, and lipper metrics
- Parses scaled asset-size strings into numeric total-net-assets values.
- Merges multiple snapshot tables into one feature table keyed by `conid` and `effective_at`.
- Assigns a coarse instrument sleeve classification such as equity, bond, commodity, or other.
- Produces diagnostics for holdings tables, including:
  - whether category weights approximately sum to one
  - whether weights materially exceed one
  - category-count sparsity
  - concentration summaries for top holdings
- Produces diagnostics for ratio-like tables, including:
  - duplicate metric keys
  - duplicate row counts
  - null versus nonnull value coverage
  - whether all values are null
- Produces a summary of source snapshot table coverage and row counts.
- Drops storage-only columns that are not part of the downstream feature contract.
- Writes feature and diagnostic artifacts to parquet for downstream analysis use.
- Exposes a convenience loader that returns only the merged snapshot features.

Inputs:

- Canonical snapshot tables curated by storage readers
- Configurable tolerances for holdings-sum checks and sparse-category coverage

Outputs:

- Merged snapshot feature table keyed by `conid` and `effective_at`
- Holdings diagnostics table
- Ratio diagnostics table
- Snapshot source-table summary
- Parquet outputs for downstream reuse

Time semantics observed:

- `effective_at`: canonical snapshot date used to merge feature rows across snapshot tables

Intent assessment:

- Intentional requirements likely worth keeping:
  - a stable downstream snapshot feature contract distinct from raw storage tables
  - consistent prefixed feature naming
  - explicit pivoting of long categorical snapshot data into analysis-facing features
  - diagnostics that describe whether holdings and ratio inputs are trustworthy
  - source-table summary as a lightweight coverage diagnostic
- Current implementation details that may be redesigned:
  - exact feature prefixes and sleeve heuristics
  - which tables belong in the merged feature contract
  - whether diagnostics stay in the same stage or become separate quality outputs

Keep / redesign / drop:

- Keep the capability.
- Preserve the separation between canonical snapshot storage and a consumer-oriented snapshot feature contract.
- Redesign the exact contract only deliberately, because this is one of the core cross-stage boundaries in the repo.

Tests informing this inventory:

- [pystocks/tests/preprocess/test_snapshot_preprocess.py](/Users/alex/Documents/pystocks/pystocks/tests/preprocess/test_snapshot_preprocess.py)
- [pystocks/tests/preprocess/test_preprocess_commands.py](/Users/alex/Documents/pystocks/pystocks/tests/preprocess/test_preprocess_commands.py)

### Capability: Refresh and derive supplementary macro and risk-free datasets

- Current entrypoint: [pystocks/cli.py](/Users/alex/Documents/pystocks/pystocks/cli.py)
  - `PyStocksCLI.refresh_supplementary_data()`
- Current refresh implementation: [pystocks/ingest/supplementary.py](/Users/alex/Documents/pystocks/pystocks/ingest/supplementary.py)
  - `fetch_risk_free_sources()`
  - `fetch_world_bank_raw()`
  - `refresh_supplementary_data()`
- Current derivation implementation: [pystocks/preprocess/supplementary.py](/Users/alex/Documents/pystocks/pystocks/preprocess/supplementary.py)
  - `load_risk_free_country_weights()`
  - `build_risk_free_series_weights()`
  - `derive_risk_free_daily()`
  - `preprocess_world_bank_country_features()`

Observed behavior:

- Fetches external short-rate series from FRED.
- Fetches annual macro indicator rows from the World Bank.
- Normalizes economy identifiers to a canonical country-code representation.
- Derives risk-free series weights from the latest stored country allocations in ETF holdings.
- Aggregates multiple risk-free source series into a weighted daily nominal rate and daily trading-day rate.
- Preprocesses World Bank annual indicators into country-level feature tables with:
  - levels
  - year-over-year growth
  - acceleration
  - global-share style derived metrics
- Persists both raw supplementary inputs and derived supplementary features.
- Records fetch-log rows summarizing refresh status and key ranges.
- Returns a machine-readable status payload with row counts for refreshed datasets.

Inputs:

- External FRED rate series
- External World Bank macro indicators
- Latest holdings country allocations from canonical storage
- Configurable interpolation and extrapolation behavior for supplementary derivations

Outputs:

- Raw supplementary risk-free source rows
- Derived daily risk-free rate table
- Raw World Bank indicator rows
- Derived country feature table
- Fetch-log rows

Time semantics observed:

- `trade_date`: daily date for risk-free source and derived daily rates
- `year` / `feature_year`: annual World Bank indicator and feature year
- `effective_at`: annual effective date for derived country features
- `observed_at` / `fetched_at`: supplementary acquisition timestamp

Intent assessment:

- Intentional requirements likely worth keeping:
  - support for supplementary datasets beyond IBKR fundamentals
  - raw-plus-derived persistence for supplementary data
  - derived risk-free daily series for excess-return analysis
  - country-level macro feature derivation for analysis inputs
  - refresh logging for supplementary datasets
- Current implementation details that may be redesigned:
  - the current mixing of external refresh logic and downstream derivation in one runtime path
  - the specific weighting scheme tying risk-free rates to holdings-country weights
  - full-table replace semantics during refresh

Keep / redesign / drop:

- Keep the capability.
- Redesign the ownership boundary, because this currently spans both acquisition and feature derivation.
- Preserve the distinction between raw supplementary inputs and derived supplementary analysis inputs.

Tests informing this inventory:

- [pystocks/tests/ingest/test_supplementary_refresh.py](/Users/alex/Documents/pystocks/pystocks/tests/ingest/test_supplementary_refresh.py)
- [pystocks/tests/preprocess/test_supplementary_preprocess.py](/Users/alex/Documents/pystocks/pystocks/tests/preprocess/test_supplementary_preprocess.py)

### Capability: Build point-in-time analysis panels from processed inputs

- Current entrypoint: [pystocks/cli.py](/Users/alex/Documents/pystocks/pystocks/cli.py)
  - `PyStocksCLI.build_analysis_panel()`
- Current implementation: [pystocks/analysis/__init__.py](/Users/alex/Documents/pystocks/pystocks/analysis/__init__.py)
  - `build_analysis_panel_data()`
  - `build_analysis_panel()`

Observed behavior:

- Loads processed snapshot features, cleaned price outputs, and optional supplementary macro features.
- Builds rebalance dates from available snapshots and price history.
- Selects the latest snapshot at or before each rebalance date for each eligible instrument.
- Carries forward snapshot age information into the panel.
- Merges processed price-derived features into the rebalance panel.
- Adds derived composite, geographic, sector, currency-bloc, and macro features.
- Requires supplementary macro inputs when macro features are configured as mandatory.
- Persists the resulting analysis panel as a derived artifact.
- Returns a machine-readable status payload with panel row and rebalance-date counts.

Inputs:

- Snapshot feature contract
- Cleaned price and eligibility outputs
- Optional supplementary country features
- Analysis configuration controlling rebalance cadence and feature inclusion

Outputs:

- Point-in-time rebalance panel keyed by instrument and rebalance date
- Derived panel artifact persisted to parquet and SQLite output tables

Time semantics observed:

- `effective_at`: snapshot feature date
- `trade_date`: cleaned price date
- `rebalance_date`: downstream analysis join date
- `snapshot_age_days`: lag between snapshot feature date and rebalance date

Intent assessment:

- Intentional requirements likely worth keeping:
  - panel construction must respect point-in-time joins
  - analysis consumes processed inputs rather than raw storage tables directly
  - eligibility gating should be applied before panel inclusion
  - macro augmentation is an optional but real analysis input family
- Current implementation details that may be redesigned:
  - exact feature families added inside panel construction
  - whether panel persistence belongs in analysis or an outputs boundary

Keep / redesign / drop:

- Keep the capability.
- Preserve point-in-time snapshot selection and rebalance-date semantics.
- Redesign the monolithic feature augmentation flow if needed, but keep the panel contract explicit.

Tests informing this inventory:

- [pystocks/tests/analysis/test_analysis_pipeline.py](/Users/alex/Documents/pystocks/pystocks/tests/analysis/test_analysis_pipeline.py)

### Capability: Run factor research and persist research artifacts

- Current entrypoints: [pystocks/cli.py](/Users/alex/Documents/pystocks/pystocks/cli.py)
  - `PyStocksCLI.run_factor_research()`
  - `PyStocksCLI.run_walk_forward_research()`
  - `PyStocksCLI.run_analysis()`
- Current implementation: [pystocks/analysis/__init__.py](/Users/alex/Documents/pystocks/pystocks/analysis/__init__.py)
  - `run_factor_research_data()`
  - `run_factor_research()`
  - `run_analysis_pipeline()`

Observed behavior:

- Uses the analysis panel as the primary research input.
- Builds candidate factor contexts and factor return series.
- Clusters and reduces correlated factor families.
- Computes factor distinctness, screening decisions, selection scores, and diagnostics.
- Runs walk-forward model fitting across sleeves and research windows.
- Produces model telemetry, persistence diagnostics, expected return outputs, and factor beta outputs.
- Writes a broad set of research artifacts to parquet and SQLite output tables, including:
  - snapshot panel
  - factor returns
  - factor clusters and cluster membership
  - factor registry and candidate diagnostics
  - factor distinctness and screening decisions
  - model results and telemetry
  - factor persistence
  - asset expected returns
  - asset factor betas
  - baseline member sets
- Returns a machine-readable status payload including factor counts and output paths.

Inputs:

- Point-in-time analysis panel
- Cleaned price history
- Optional derived risk-free daily series
- Analysis configuration controlling factor selection, clustering, windowing, and regression behavior

Outputs:

- Factor return series
- Factor diagnostics and registries
- Walk-forward research/model results
- Persistence and expected-return artifacts
- Persisted research outputs in parquet and SQLite

Time semantics observed:

- rebalance windows for factor formation
- training and test windows for walk-forward research
- daily return dates for factor returns and regression fitting

Intent assessment:

- Intentional requirements likely worth keeping:
  - factor research is a first-class system responsibility
  - research outputs include both returns and diagnostics explaining factor selection
  - walk-forward evaluation is part of the intended analysis workflow
  - multiple persisted research artifacts are expected outputs, not incidental side effects
- Current implementation details that may be redesigned:
  - the exact decomposition of candidate generation, clustering, modeling, and persistence
  - writing research outputs back into the same SQLite database as operational source data
  - whether `run_analysis` is just an alias for factor research in the future design

Keep / redesign / drop:

- Keep the capability.
- Redesign the ownership and output boundary, because this is currently too tightly coupled to the monolithic analysis module.
- Preserve the distinction between research results and diagnostics that justify them.

Tests informing this inventory:

- [pystocks/tests/analysis/test_analysis_pipeline.py](/Users/alex/Documents/pystocks/pystocks/tests/analysis/test_analysis_pipeline.py)

## Storage And Contract Inventory

### Capability: Persist endpoint payloads into canonical snapshot and series storage

- Current implementation:
  - [pystocks/storage/fundamentals_store.py](/Users/alex/Documents/pystocks/pystocks/storage/fundamentals_store.py)
  - [pystocks/storage/schema.py](/Users/alex/Documents/pystocks/pystocks/storage/schema.py)
  - [pystocks/fundamentals_normalizers.py](/Users/alex/Documents/pystocks/pystocks/fundamentals_normalizers.py)

Observed behavior:

- Accepts a combined per-instrument fundamentals snapshot.
- Splits the combined payload into endpoint payloads.
- Stores immutable raw endpoint blobs by content hash.
- Writes endpoint snapshot metadata rows.
- Writes canonical child tables for each endpoint.
- Writes series tables for endpoints that contain historical rows.
- Distinguishes inserted, overwritten, unchanged, and skipped endpoint persistence outcomes.
- Owns substantial source-to-canonical parsing logic rather than merely storing raw JSON.

Current canonical table families:

- product and operational-state table
- raw payload blob table
- endpoint snapshot tables
- endpoint child fact tables
- series tables
- supplementary raw and derived tables

Current time behavior:

- all current snapshot endpoints are anchored to `ratios.as_of_date` for `effective_at`
- endpoint-specific source dates may also be stored separately
- raw observation time is stored as `observed_at`

Intent assessment:

- Intentional requirements likely worth keeping:
  - raw-plus-canonical persistence
  - endpoint-specific canonical tables
  - separate series handling where source payloads are historical
  - source-to-canonical scalar parsing in storage when semantics are clear
- Current implementation details that may be redesigned:
  - single-class monolith for all endpoint persistence
  - global `ratios.as_of_date` anchoring
  - coupling of product metadata and scrape state in `products`

Keep / redesign / drop:

- Keep the capability.
- Redesign the time contract and internal decomposition.

Tests informing this inventory:

- [pystocks/tests/storage/test_effective_date_resolution.py](/Users/alex/Documents/pystocks/pystocks/tests/storage/test_effective_date_resolution.py)
- [pystocks/tests/storage/test_holdings_storage.py](/Users/alex/Documents/pystocks/pystocks/tests/storage/test_holdings_storage.py)
- [pystocks/tests/storage/test_ratios_storage.py](/Users/alex/Documents/pystocks/pystocks/tests/storage/test_ratios_storage.py)
- [pystocks/tests/storage/test_dividends_storage.py](/Users/alex/Documents/pystocks/pystocks/tests/storage/test_dividends_storage.py)
- [pystocks/tests/storage/test_morningstar_storage.py](/Users/alex/Documents/pystocks/pystocks/tests/storage/test_morningstar_storage.py)
- [pystocks/tests/storage/test_storage_boundary.py](/Users/alex/Documents/pystocks/pystocks/tests/storage/test_storage_boundary.py)

### Capability: Expose consumer-oriented reader contracts over canonical storage

- Current implementation:
  - [pystocks/storage/readers.py](/Users/alex/Documents/pystocks/pystocks/storage/readers.py)

Observed behavior:

- Exposes explicit read contracts for major downstream consumers.
- Normalizes dates and numeric columns for pandas consumers.
- Returns curated snapshot-table groups rather than leaking arbitrary raw schema shape.
- Provides stable downstream inputs for:
  - cleaned price processing
  - dividend preprocessing
  - snapshot feature building
  - supplementary feature loading

Intent assessment:

- Intentional requirements likely worth keeping:
  - readers should be consumer-oriented and explicit
  - analysis-facing code should not depend on `SELECT *` or ad hoc schema reach-ins
- Current implementation details that may be redesigned:
  - exact set of reader functions
  - whether readers return multiple frames, bundles, or typed objects

Keep / redesign / drop:

- Keep the capability.
- Redesign the API shape only if the replacement remains explicit and stable.

Tests informing this inventory:

- [pystocks/tests/storage/test_storage_boundary.py](/Users/alex/Documents/pystocks/pystocks/tests/storage/test_storage_boundary.py)

### Capability: Estimate current instrument betas to persistent factors

- Current entrypoint: [pystocks/cli.py](/Users/alex/Documents/pystocks/pystocks/cli.py)
  - `PyStocksCLI.compute_factor_betas()`
- Current implementation: [pystocks/analysis/__init__.py](/Users/alex/Documents/pystocks/pystocks/analysis/__init__.py)
  - `compute_current_betas_data()`
  - `compute_current_betas()`

Observed behavior:

- Uses the latest analysis panel, cleaned returns, factor series, and persistent-factor selections.
- Fits trailing regressions per instrument within sleeve-specific persistent factor sets.
- Produces current alpha, `r2`, observation counts, and per-factor beta exposures.
- Persists current beta outputs as research artifacts.
- Currently runs through the broader factor-research path rather than as a truly independent pipeline.

Inputs:

- Latest panel and sleeve assignments
- Cleaned daily returns
- Reduced factor return series
- Persistent-factor selections

Outputs:

- Current beta table with alpha and fit diagnostics
- Persisted current-beta artifact

Time semantics observed:

- latest rebalance date for current panel membership
- trailing return window for beta estimation

Intent assessment:

- Intentional requirements likely worth keeping:
  - current factor exposures for instruments are a user-facing analysis output
  - beta estimation depends on persistent factor definitions rather than all raw factor candidates
- Current implementation details that may be redesigned:
  - routing this command through the full factor-research pipeline
  - exact trailing-window length and regression specification

Keep / redesign / drop:

- Keep the capability.
- Redesign the runtime path so it can depend on previously computed research artifacts or a cleaner analysis bundle.

Tests informing this inventory:

- [pystocks/tests/analysis/test_analysis_pipeline.py](/Users/alex/Documents/pystocks/pystocks/tests/analysis/test_analysis_pipeline.py)
