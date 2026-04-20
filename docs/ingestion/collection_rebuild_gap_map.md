# Collection Rebuild Gap Map

## Purpose

This document translates the current review of `pystocks_next/` into a concrete plan for rebuilding the missing collection layer before downstream phase-4 work continues.

It answers:

- which legacy collection capabilities still need to survive
- which ones are already represented in `pystocks_next/`
- which owner should receive each missing responsibility
- what public contracts the `collection` concern should expose
- what should be built first so later panel/research work sits on a stable base

This is a rebuild-specific note. It does not replace the older SQLite ingestion plan in `docs/ingestion/PLAN.md`, which predates the current `pystocks_next` concern split.

## Current Repo State

As of the current review:

- `pystocks_next/storage/` has meaningful operational foundations.
- `pystocks_next/universe/` exists and cleanly separates canonical instruments from exclusions.
- `pystocks_next/feature_inputs/` has price, dividend, and snapshot builders plus an `AnalysisInputBundle`.
- `pystocks_next/collection/` is still a placeholder package.
- `pystocks_next/panel/`, `research/`, `exposures/`, and `outputs/` are still placeholders.

That means the rebuild has not truly completed phase 2C, and phase 3 is only partially complete.

The main structural risk is continuing into panel/research work before the collection boundary is rebuilt. Doing that would force either:

- direct reuse of legacy ingest paths, or
- ad hoc fetch logic spread across storage, feature inputs, and CLI code

Both would recreate the mixed ownership the rebuild is supposed to remove.

## Main Architectural Findings To Lock

### 1. Collection is the main missing concern

The old collection path in `pystocks/ingest/` contained the following real capabilities:

- product-universe fetch from IBKR
- repeated fundamentals collection over governed targets
- session validation and reauthentication
- per-endpoint fetch fanout
- landing-payload gating
- payload usefulness heuristics
- incremental price-chart window selection
- per-run telemetry and machine-readable results
- supplementary raw-source refresh

Those capabilities are still required by the FRD and roadmap. They have not yet been modeled in `pystocks_next/`.

### 2. Storage fixed the old time-semantics flaw

This is a real improvement and should remain intact:

- `pystocks_next/storage/time.py` centralizes endpoint-family `effective_at` rules.
- The rebuild no longer globally anchors snapshot dates to `ratios.as_of_date`.

That means the next collection work should hand explicit `observed_at`, endpoint payloads, and source dates to storage. It should not start re-implementing time semantics inside the fetch runner.

### 3. Supplementary ownership still needs cleanup

The rebuild currently stores supplementary derived outputs in canonical storage tables. That is acceptable only as an explicit cache policy owned by feature inputs.

The intended split should be:

- `collection` fetches raw supplementary source data
- `storage` persists raw supplementary source data and fetch logs
- `feature_inputs` derives risk-free and macro analysis inputs

If this is not cleaned up, supplementary logic will continue to blur the `collection -> storage -> feature_inputs` boundary.

### 4. One bundle builder is still missing

`feature_inputs` already defines `AnalysisInputBundle`, but it still exposes stage fragments rather than one authoritative bundle builder.

That missing surface is adjacent to the collection gap because phase 4 should not begin until:

- collection can produce canonical writes cleanly
- storage can read consumer-oriented contracts
- feature inputs can build one bundle without panel/research reaching into lower layers

## Legacy Capability Map

The table below identifies the main missing collection capabilities and where they should move.

| Legacy capability | Legacy location | Current `pystocks_next` state | Target owner |
| --- | --- | --- | --- |
| Product catalog fetch and paging | `pystocks/ingest/product_scraper.py` | Missing | `collection/products.py` |
| Session state validation, login, reauth, account resolution | `pystocks/ingest/session.py` | Missing | `collection/session.py` |
| Recurring fundamentals run orchestration | `pystocks/ingest/fundamentals.py` | Missing | `collection/fundamentals.py` |
| Endpoint URL building and endpoint-specific request policy | `pystocks/ingest/fundamentals.py` | Missing | `collection/fundamentals.py` with small local helpers |
| Landing-page gating before deeper fanout | `pystocks/ingest/fundamentals.py` | Missing | `collection/fundamentals.py` |
| Payload usefulness heuristics | `pystocks/ingest/fundamentals.py` | Missing | `collection/fundamentals.py` |
| Incremental price-chart period selection | `pystocks/ingest/fundamentals.py` | Missing | `collection/fundamentals.py` using storage readers |
| Per-run endpoint telemetry | `pystocks/ingest/fundamentals.py` | Missing | `collection/telemetry.py` |
| Structured per-conid outcomes and skip reasons | `pystocks/ingest/fundamentals.py` | Missing | `collection/fundamentals.py` |
| Supplementary raw-source refresh | `pystocks/ingest/supplementary.py` | Missing | `collection/supplementary.py` |

## Recommended `pystocks_next/collection` Layout

The rebuild does not need many modules, but it does need the right ones.

Recommended layout:

```text
pystocks_next/collection/
  __init__.py
  session.py
  telemetry.py
  products.py
  fundamentals.py
  supplementary.py
```

### `collection/session.py`

Own:

- authenticated source session lifecycle
- persisted auth state
- account-id resolution for account-scoped endpoints
- login, validation, and reauthentication behavior

Do not own:

- target selection
- endpoint payload parsing
- storage writes

Public surface should be small and direct, for example:

- `CollectionSession`
- `validate_auth_state()`
- `login()`
- `reauthenticate()`
- `get_primary_account_id()`
- `get_client()`

### `collection/telemetry.py`

Own:

- in-memory run telemetry state
- endpoint call counts
- useful-payload counts
- per-endpoint status-code tallies
- run-summary shaping

Do not own:

- canonical storage schema
- JSON-file path policy for derived outputs

This module should expose typed telemetry objects and summary builders. File materialization can stay thin and stage-local until the top-level `outputs/` concern is implemented.

### `collection/products.py`

Own:

- product catalog API requests
- paging and retry policy
- basic product-record filtering and dedupe within a run
- translation into a write input for `universe` persistence

Do not own:

- canonical instrument schema
- exclusion policy
- scrape-recency policy

This module should call `universe.upsert_instruments(...)` after shaping collection results into canonical universe rows.

### `collection/fundamentals.py`

Own:

- repeated fundamentals collection runs over a target set
- landing fetch before deeper endpoint fanout
- endpoint URL construction
- endpoint usefulness heuristics
- session recovery during a run
- incremental price-chart request-window choice
- per-conid structured outcomes
- collection-facing write inputs handed to storage

Do not own:

- canonical parsing of endpoint payloads into endpoint tables
- `effective_at` resolution
- downstream feature names
- analysis-facing diagnostics

This should be the main runtime owner for the rebuilt fundamentals path, but it should stay thinner than the legacy monolith by delegating:

- auth to `collection/session.py`
- telemetry shaping to `collection/telemetry.py`
- canonical persistence to `storage/writes.py`
- target-set sourcing to `universe/targeting.py`

### `collection/supplementary.py`

Own:

- raw-source refresh from external non-IBKR sources
- source fetch retries and error classification
- fetch-log write inputs

Do not own:

- risk-free weighting and aggregation
- macro feature derivation
- ETF-to-country derived mapping logic

Those derived steps belong in `feature_inputs/supplementary.py`, not in collection.

## What Should Not Be Introduced

The rebuild docs explicitly reject extra abstraction layers. For collection, that means:

- no standalone `contracts/` package
- no metadata registry for endpoints
- no CLI wiring dictionary that restates runtime flow
- no generic orchestration framework
- no endpoint parser layer in collection that duplicates storage ownership

Prefer a few direct typed result objects and explicit functions.

## Public Contracts The Collection Layer Should Expose

The collection concern should hand storage structured write inputs rather than raw uncontrolled blobs and ad hoc status strings.

Recommended public result types:

### `ProductCollectionResult`

Fields:

- `fetched_products`
- `deduped_products`
- `products_upserted`
- `status`

### `CollectedEndpointPayload`

Fields:

- `endpoint`
- `conid`
- `observed_at`
- `payload`
- `status_code`
- `is_useful`

### `FundamentalsCollectionResult`

Run-level fields:

- `status`
- `total_targeted_conids`
- `processed_conids`
- `saved_snapshots`
- `inserted_events`
- `overwritten_events`
- `unchanged_events`
- `series_raw_rows_written`
- `series_latest_rows_upserted`
- `auth_retries`
- `aborted`

### `FundamentalsConidOutcome`

Per-instrument fields:

- `conid`
- `status`
- `skip_reason`
- `observed_at`
- `endpoint_payloads`
- `storage_result`

These names are illustrative. The important part is that:

- per-conid outcomes are explicit
- run summaries are explicit
- storage receives structured inputs
- downstream code never depends on free-form strings hidden in one runner

## Legacy Behavior That Must Be Preserved

The rebuild should preserve these legacy behaviors unless the design changes them intentionally.

### Product refresh

- direct IBKR product fetch
- paging until empty or short page
- retry/backoff for timeouts and HTTP `429`
- dedupe by `conid`
- machine-readable run result

### Fundamentals collection

- target from explicit conid list or governed universe
- bounded partial runs via limit/offset
- recency skip with force override
- landing request before endpoint fanout
- structured skip outcome for landing-only instruments
- per-endpoint usefulness heuristics
- session validation before run and reauth during run
- incremental price-chart windows based on latest stored canonical point
- machine-readable run summary
- endpoint-level telemetry

### Supplementary refresh

- raw external-source fetch support
- fetch-log persistence
- machine-readable run result

The rebuild should not automatically preserve these exact implementation shapes:

- one monolithic scraper class
- telemetry only as JSON files
- hardcoded mixed status strings as implicit contracts
- supplementary derivation inside the collection step

## Specific Legacy Behaviors To Port Early

These are easy to miss because they are small, but they materially affect correctness or operability.

### 1. Landing-only skip classification

Legacy behavior:

- a landing payload without the required signal is treated as a structured skip, not a hard failure

Why it matters:

- this keeps recurring runs operational over heterogeneous instruments
- it prevents a weak or irrelevant product from aborting the run

### 2. Incremental price-chart window selection

Legacy behavior:

- the chart-period request scales to the missing window instead of always requesting `MAX`

Why it matters:

- lower network cost
- lower source load
- fewer large repeated payloads

### 3. ESG account-id recovery from persisted session state

Legacy behavior:

- account-scoped endpoints can recover a primary account id from saved state

Why it matters:

- ESG and similar endpoints should not degrade into avoidable permanent skips

### 4. Usefulness heuristics for sparse but real payloads

Legacy behavior:

- holdings and ratios payloads can still count as meaningful even when they only expose a thin but valid source signal such as `as_of_date`

Why it matters:

- avoids throwing away valid low-density observations
- prevents collection policy from becoming stricter than source reality

## Recommended Implementation Sequence

The safest order is:

1. `collection/session.py`
2. `collection/telemetry.py`
3. `collection/products.py`
4. `collection/fundamentals.py` with only:
   - target selection
   - landing fetch
   - a first endpoint subset
   - structured outcomes
   - storage handoff
5. `collection/supplementary.py` for raw-source refresh only
6. `feature_inputs/supplementary.py`
7. one authoritative `build_analysis_input_bundle()`

Do not wait for all endpoint families before proving the first fundamentals path. A lean first slice is better:

- product refresh
- session validation
- one fundamentals run path
- price chart series
- dividend events
- one or two snapshot endpoints needed by current feature-input tests

That keeps the proof vertical and avoids re-porting the entire legacy fanout at once.

## Suggested First Acceptance Tests

The first `pystocks_next/collection` tests should cover:

### Session

- primary account-id recovery from saved state
- validation probe behavior
- reauth retry path

### Products

- paging stop conditions
- `429` and timeout retry behavior
- malformed-product filtering
- dedupe by `conid`

### Fundamentals

- explicit target-list vs governed target-list selection
- landing-only skip outcome
- per-endpoint usefulness classification
- price-chart window selection by latest stored point
- structured run summary plus telemetry

### Supplementary

- raw-source fetch rows written
- fetch-log rows written
- no derived risk-free or macro feature logic in collection tests

## Adjacent Prerequisites Before Phase 4

Even after collection is rebuilt, two follow-on fixes should happen before panel/research work:

### 1. Add `feature_inputs/supplementary.py`

Move ownership of:

- risk-free daily derivation
- macro feature derivation
- ETF-to-supplementary explicit mappings

into feature inputs.

### 2. Add `build_analysis_input_bundle()`

`feature_inputs` should expose one authoritative bundle builder that:

- loads the required storage reader contracts
- builds prices, dividends, snapshots, and supplementary inputs
- returns one `AnalysisInputBundle`

Panel construction should depend on that builder or its returned bundle, not on lower-level fragment builders.

## Definition Of Done For The Collection Rebuild

Collection is ready for downstream work only when:

- the `collection` package owns source access and run control clearly
- storage does not need to know fetch policy or auth policy
- feature inputs do not know endpoint URLs or session details
- per-run and per-conid outcomes are explicit typed surfaces
- the first recurring fundamentals run is test-covered end to end
- supplementary raw fetch is separated from supplementary derivation
- a later agent can implement new endpoint families without modifying panel or feature-input code

At that point, the rebuild will have a coherent upstream boundary and phase-4 work can proceed on top of real stage contracts rather than placeholders.
