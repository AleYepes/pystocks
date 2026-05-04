# Pystocks Data Contracts And Time Semantics

This document summarizes the current cross-stage data contracts discovered in `/pystocks`.

It is descriptive, not prescriptive.

Its purpose is to make the current storage and reader semantics explicit before a rebuild changes them.

## Scope

This document covers:

- canonical identifiers
- current storage table families
- reader-facing contracts
- current time semantics
- places where current semantics are likely wrong or too coupled

It does not define the future package layout.

## Canonical Identity

The current system uses `conid` as the primary canonical instrument identifier.

Observed usage:

- product universe bootstrap writes canonical `products` rows keyed by `conid`
- all endpoint snapshot tables key instrument rows by `conid`
- all endpoint child tables key rows by `conid` plus endpoint-specific date semantics
- preprocess and analysis contracts expect `conid` as the instrument join key

Other identifiers such as symbol, ISIN, exchange, and country codes are supporting attributes, not primary identity.

## Current Table Families

The SQLite schema currently falls into five broad groups.

### 1. Product Master And Operational State

Primary table:

- `products`

Current contents are mixed:

- canonical instrument identity and metadata
- latest fundamentals scrape status
- latest fundamentals scrape date
- latest metadata refresh timestamp

This is an important current coupling. Product master data and scrape-operational state are stored together.

### 2. Raw Capture

Primary table:

- `raw_payload_blobs`

Current behavior:

- endpoint payloads are canonicalized to JSON bytes
- payloads are deduplicated by content hash
- compressed payload blobs are stored once and referenced by snapshot tables

Notable exception:

- product-universe catalog refresh currently does not appear to use the raw blob pattern

### 3. Canonical Snapshot Endpoints

Each snapshot endpoint currently has a snapshot table plus one or more child tables.

Snapshot tables:

- `profile_and_fees_snapshots`
- `holdings_snapshots`
- `ratios_snapshots`
- `lipper_ratings_snapshots`
- `dividends_snapshots`
- `morningstar_snapshots`
- `ownership_snapshots`
- `esg_snapshots`
- `price_chart_snapshots`
- `sentiment_snapshots`

Representative child tables:

- `profile_and_fees`
- `profile_and_fees_reports`
- `profile_and_fees_stylebox`
- `holdings_asset_type`
- `holdings_industry`
- `holdings_currency`
- `holdings_investor_country`
- `holdings_debt_type`
- `holdings_debtor_quality`
- `holdings_maturity`
- `holdings_top10`
- `holdings_geographic_weights`
- `ratios_key_ratios`
- `ratios_financials`
- `ratios_fixed_income`
- `ratios_dividend`
- `ratios_zscore`
- `lipper_ratings`
- `dividends_industry_metrics`
- `morningstar_summary`
- `morningstar_commentary`
- `ownership_owners_types`
- `ownership_holders`
- `esg`

Current design pattern:

- snapshot table stores endpoint-level metadata plus `payload_hash`
- child tables store canonicalized facts derived from that payload
- many child tables intentionally avoid synthetic ids and instead remain keyed by natural row shape

### 4. Canonical Series Endpoints

Series tables:

- `price_chart_series`
- `sentiment_series`
- `ownership_trade_log_series_raw`
- `ownership_trade_log_series_latest`
- `dividends_events_series`

Current design pattern:

- price and sentiment are keyed by `(conid, effective_at)` and upserted directly
- ownership keeps both append-only raw history and a latest-per-row-key projection
- dividends events are append-only deduplicated event rows keyed effectively by event signature

### 5. Supplementary Data

Tables:

- `supplementary_fetch_log`
- `supplementary_risk_free_sources`
- `supplementary_risk_free_daily`
- `supplementary_world_bank_raw`
- `supplementary_world_bank_country_features`

Current design pattern:

- raw external-source rows are stored
- derived supplementary analysis inputs are also stored
- fetch logging exists for supplementary refreshes

## Reader Contracts

The current storage readers expose consumer-oriented contracts rather than raw `SELECT *` access for analysis-facing code.

Important readers:

- `load_price_history()`
- `load_dividend_events()`
- `load_snapshot_feature_tables()`
- `load_risk_free_daily()`
- `load_world_bank_country_features()`

Important current contract qualities:

- dates are normalized to pandas datetimes
- numeric columns are coerced to numeric
- reader outputs are already shaped around downstream use
- snapshot readers expose a curated table set rather than the full schema

This is one of the healthier boundaries in the current repo.

## Current Time Semantics

The repo currently uses at least four distinct time concepts, but the storage implementation does not preserve them cleanly enough yet.

### `observed_at`

Meaning:

- when the system fetched or observed the payload

Current storage behavior:

- snapshot tables store `observed_at`
- supplementary derived tables also keep `observed_at`
- some series tables preserve `observed_at` only in raw/latest ownership projections, not in all series tables

### Source `as_of_date`

Meaning:

- the source-declared or payload-derived date the payload describes

Current behavior:

- some snapshot tables store their own endpoint `as_of_date`
- example: holdings, ratios, morningstar, esg

### Canonical `effective_at`

Meaning:

- the canonical date used to key stored facts for downstream joins. It must be
  derived from source payload contents and must never be collection time.

Legacy current-package behavior:

- for snapshot endpoints, the storage layer currently anchors all endpoints to `ratios.as_of_date`
- this is encoded directly in `FundamentalsStore._resolve_effective_dates()`
- storage tests confirm that holdings and profile rows are currently stored at the ratios anchor date
- if `ratios.as_of_date` is missing, snapshot persistence is skipped entirely for the snapshot payload

This is the single most important current time-semantic weakness.

Rebuild rule:

- each endpoint owns an explicit source/payload date hierarchy
- `observed_at`, current date, and collection date are never valid
  `effective_at` fallbacks
- if no valid source/payload date can be resolved, the endpoint's canonical
  write is unresolved and should be skipped or quarantined with telemetry

### Analysis Join / Rebalance Date

Meaning:

- the downstream date at which processed features are joined for research

Current behavior:

- analysis panel construction selects the latest snapshot with `effective_at <= rebalance_date`
- `snapshot_age_days` is explicitly computed and carried downstream

This part is conceptually correct, but it depends on a weak upstream `effective_at` contract.

## Current Canonicalization Rules

The storage layer currently owns more than simple persistence. It also owns substantial source-to-canonical parsing.

Examples:

- parsing total-net-assets strings into canonical fields
- mapping holdings payload sections into fixed and tall child tables
- converting ratio sections into section-specific metric tables
- normalizing Morningstar summary and commentary shape
- extracting dividend events from nested history payloads
- extracting ownership trade-log series and ownership summary shape
- parsing ESG tree content into scalar columns

This means the current storage layer is already the authoritative owner of many source-to-fact transformations.

## Current Contract Strengths

- `conid` is consistently treated as the canonical instrument key
- raw payload deduplication exists for fundamentals endpoints
- canonical endpoint tables are more consumer-stable than the raw payloads
- reader outputs are explicit and analysis-oriented
- snapshot preprocess consumes readers instead of reaching into arbitrary storage tables

## Current Contract Weaknesses

### 1. Snapshot `effective_at` is globally anchored to ratios

This is the most serious legacy semantic distortion.

It collapses endpoint-specific dates into one shared anchor even when the source data may have different publication timing.

### 2. Product master and operational scrape state are coupled

The `products` table currently mixes:

- instrument master data
- scrape recency
- scrape status

These are related, but not the same contract.

### 3. Research outputs share the same SQLite database as canonical source data

This blurs the boundary between:

- canonical operational/source-of-truth storage
- recomputable research artifacts

### 4. Supplementary refresh mixes acquisition and derivation

The current supplementary runtime path combines:

- external fetch
- raw persistence
- derived-feature creation
- fetch logging

That is workable, but ownership is muddled.

### 5. Some runtime policies are encoded in storage semantics

Examples:

- skipping all snapshot persistence when `ratios.as_of_date` is missing
- endpoint-specific shape choices embedded directly in the write path

These policies may be correct or incorrect, but they are important system contracts and should be explicit in the rebuild.

## Rebuild Implications

The rebuild should preserve:

- canonical `conid` identity
- raw-capture auditability
- explicit reader contracts
- separation of snapshot and series storage shapes where the source demands it
- distinct handling of `observed_at`, source dates, canonical dates, and analysis join dates

The rebuild should likely change:

- global `ratios.as_of_date` anchoring
- coupling of product metadata and scrape state
- research artifact persistence boundary
- the ownership split between external fetch, canonical persistence, and derived supplementary feature construction
