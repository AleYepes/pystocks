# Pystocks Architecture Design

## Purpose

This document proposes the target architecture for the rebuilt `pystocks` system.

It translates the functional requirements into a concrete design for:

- concern boundaries
- dependency direction
- runtime flows
- data contracts
- time semantics
- output boundaries
- migration strategy

It is intentionally more concrete than the FRD, but it is still a design document rather than a file-by-file implementation checklist.

## Status

This design assumes:

- the current `/pystocks` package remains the live reference implementation during the rebuild
- the rebuild is free to change package boundaries and runtime flows
- the functional baseline is defined by [functional_requirements.md](/Users/alex/Documents/pystocks/docs/functional_requirements.md)
- the current storage/time findings are summarized in [data_contracts_and_time_semantics.md](/Users/alex/Documents/pystocks/docs/data_contracts_and_time_semantics.md)

Working name for the rebuilt package:

- `pystocks_next`

That name is temporary. Its only purpose is to make the migration boundary explicit.

## Design Goals

The rebuilt architecture must be:

- clear: one concern, one owner
- point-in-time correct: distinct time concepts remain distinct
- durable: cross-stage contracts are explicit and not restated in multiple places
- pragmatic: direct functions and explicit data structures over framework-style abstraction
- performant: avoid repeated full-table scans, repeated reparsing, and unnecessary round-trips
- auditable: raw capture, canonical facts, diagnostics, and derived outputs have clear boundaries

## Non-Goals

The rebuilt architecture should not:

- recreate notebook-era flat-table workflows
- add registries or metadata layers that duplicate reader and feature contracts
- mix research artifacts into canonical source-of-truth storage without a deliberate exception
- depend on one endpoint's date as a universal anchor for unrelated endpoint data

## Top-Level Concern Map

The rebuilt system should at least have the following nine concern families:

### 1. Universe

Responsibilities:

- maintain canonical instrument identity
- refresh the product universe
- apply explicit universe-governance rules
- provide target lists for collection runs

Outputs:

- canonical instrument master dataset
- target instrument selections

### 2. Collection

Responsibilities:

- manage source access, auth, retries, and run control
- collect raw source payloads
- classify fetch outcomes

Outputs:

- raw payloads plus source metadata
- collection telemetry
- normalized write inputs for canonical storage

### 3. Canonical Storage

Responsibilities:

- persist raw payloads and canonical endpoint facts
- own endpoint-specific source-to-canonical parsing where semantics are clear
- define stable reader contracts
- own canonical time resolution rules

Outputs:

- raw capture store
- canonical snapshot facts
- canonical series facts
- reader contracts for downstream consumers

### 4. Feature Inputs

Responsibilities:

- transform canonical facts into stable analysis input contracts
- define reusable feature vocabulary
- produce diagnostics explaining input trustworthiness

Outputs:

- cleaned price contract
- dividend-usability contract
- snapshot feature contract
- supplementary derived input contract

### 5. Panel Construction

Responsibilities:

- join stable input contracts into point-in-time analysis panels
- apply eligibility rules at join dates
- carry forward join and feature-age metadata

Outputs:

- rebalance or analysis panel

### 6. Research

Responsibilities:

- build candidate factors
- estimate factor returns
- screen, cluster, and reduce factors
- run walk-forward evaluation
- produce expected-return and persistence outputs

Outputs:

- factor returns
- research diagnostics
- expected-return outputs
- persistence and model results

### 7. Exposure Estimation

Responsibilities:

- estimate current instrument exposures to accepted or persistent factors

Outputs:

- current beta or exposure datasets

### 8. Portfolio Construction

Responsibilities:

- consume expected returns, risk inputs, and constraint inputs
- run efficient-frontier and optimizer workflows
- produce portfolio solutions and diagnostics

Outputs:

- optimizer-ready inputs
- portfolio solutions
- portfolio diagnostics

### 9. Outputs

Responsibilities:

- materialize derived artifacts
- keep canonical and derived outputs separate
- manage reusable caches where helpful

Outputs:

- parquet or equivalent artifacts
- optional lightweight output indices or manifests

## Dependency Direction

Two different relationships matter here and should not be conflated:

- runtime dataflow
- code-level dependency direction

### Runtime Dataflow

Runtime dataflow should stay strict:

- `universe -> collection`
- `collection -> canonical_storage`
- `canonical_storage -> feature_inputs`
- `feature_inputs -> panel_construction`
- `panel_construction -> research`
- `research -> exposure_estimation`
- `research -> portfolio_construction`
- `exposure_estimation -> portfolio_construction`
- derived stages may emit artifacts through `outputs`

### Code-Level Dependency Direction

Modules may only import from upstream concerns, not downstream ones.

- `collection` may import from `universe`
- `canonical_storage` may import from `collection`
- `feature_inputs` may import from `canonical_storage`
- `panel_construction` may import from `feature_inputs`
- `research` may import from `panel_construction`
- `exposure_estimation` may import from `research`
- `portfolio_construction` may import from `research` and, when needed, from `exposure_estimation`

The reverse is never permitted. For example, `canonical_storage` must not import from `feature_inputs`, `panel_construction`, or any other downstream concern.

## Proposed Package Layout

One viable package layout is:

```text
pystocks_next/
  __init__.py
  cli.py
  config.py
  universe/
    __init__.py
    products.py
    governance.py
    targeting.py
  collection/
    __init__.py
    session.py
    fundamentals.py
    supplementary.py
    telemetry.py
  storage/
    __init__.py
    schema.py
    sqlite.py
    writes.py
    reads.py
    time.py
    raw_capture.py
  feature_inputs/
    __init__.py
    prices.py
    dividends.py
    snapshots.py
    supplementary.py
    bundle.py
  panel/
    __init__.py
    rebalance.py
    build.py
  research/
    __init__.py
    factors.py
    screening.py
    walk_forward.py
    outputs.py
  exposures/
    __init__.py
    current_betas.py
  portfolio/
    __init__.py
    inputs.py
    optimize.py
    frontier.py
    outputs.py
  outputs/
    __init__.py
    persist.py
    manifest.py
  tests/
    ...
```

This layout is illustrative, not mandatory. The important part is the concern split, not the exact folder names.

Within this layout:

- stage-local `outputs.py` modules should define stage-specific result shaping and any stage-owned serialization helpers
- the top-level `outputs/` package should own filesystem materialization, manifests, cache bookkeeping, and cross-stage artifact naming conventions

That separation avoids duplicating persistence responsibilities across stages.

## Runtime Model

The system should support a few explicit end-to-end runtime flows.

### Flow 1: Universe Refresh

1. refresh product universe from source
2. apply universe-governance rules
3. upsert canonical universe master rows

### Flow 2: Fundamentals Collection

1. select target instruments
2. validate or create source session
3. collect endpoint payloads
4. classify empty, skipped, failed, and successful outcomes
5. persist raw payloads and canonical facts
6. update collection telemetry and per-run summaries

### Flow 3: Analysis Input Build

1. load canonical reader contracts
2. build cleaned price inputs and diagnostics
3. build dividend usability inputs and diagnostics
4. build snapshot feature inputs and diagnostics
5. build supplementary derived inputs
6. return a single analysis input bundle

### Flow 4: Panel Build

1. load the analysis input bundle
2. define join or rebalance dates
3. join latest eligible features available at each date
4. attach price-derived targets and feature-age metadata
5. emit the analysis panel

### Flow 5: Research

1. load the analysis panel
2. construct candidate factors
3. build factor returns
4. screen and cluster factors
5. run walk-forward evaluation
6. emit research outputs and diagnostics

### Flow 6: Portfolio Construction

1. load research outputs
2. assemble optimizer inputs
3. run efficient-frontier or constrained optimization workflows
4. emit portfolio solutions and portfolio diagnostics

## Canonical Data Design

The rebuilt system should keep three major storage layers distinct.

### 1. Canonical Operational Storage

This is the source-of-truth operational store.

It contains:

- canonical instrument master data
- raw payload captures
- canonical endpoint facts
- supplementary raw data

It does not contain:

- merged analysis panels
- factor returns as source-of-truth inputs
- optimizer solutions as canonical operational data

SQLite remains a good default for this layer unless a later scale constraint proves otherwise.

### 2. Derived Analysis Inputs

These are recomputable contracts built from canonical storage.

They include:

- cleaned prices and returns
- price eligibility
- dividend usability artifacts
- snapshot feature tables
- supplementary derived inputs

These can be materialized to parquet or similar cache artifacts for speed and reproducibility, but the contract owner remains the feature-input stage.

### 3. Derived Research And Portfolio Outputs

These are downstream artifacts, not canonical facts.

They include:

- analysis panels
- factor returns
- research diagnostics
- current exposures
- optimizer inputs
- portfolio solutions

These should be written to a separate output boundary, even if the same local directory tree is used.

## Time Semantics Design

Time semantics must be explicit and owned centrally.

### Required Time Fields

The architecture must preserve these four concepts:

- `observed_at`: when the system fetched or observed the data
- `source_as_of_date`: the source-declared date when available
- `effective_at`: the canonical date used for storage and downstream joins
- `join_date` or `rebalance_date`: the downstream analysis date

### Design Rules

1. `observed_at` is never a substitute for `effective_at`.
2. `source_as_of_date` is preserved when the source provides it.
3. `effective_at` is resolved per endpoint family according to explicit rules.
4. unrelated endpoint data must not be globally anchored to one endpoint's date.
5. analysis joins use `effective_at <= join_date`, not source-specific ad hoc comparisons.

### High-Level `effective_at` Resolution Rules

The architecture should adopt explicit fallback hierarchies per endpoint family.

For snapshot endpoints that provide a source date describing the payload:

- first choice: the endpoint's own source `as_of_date`
- fallback: another endpoint-specific source date from the same payload only if the source contract explicitly states they describe the same publication snapshot
- final fallback: `observed_at` only when no reliable source date exists and the endpoint is still worth storing as an observed snapshot

For series endpoints:

- `effective_at` is the point-level event date or trade date carried by each series row
- the enclosing snapshot's `observed_at` remains separate metadata and is never substituted for the row date

For endpoints with neither a trustworthy source date nor a justified observed-date fallback:

- the write path should classify the payload as unresolved rather than silently borrowing another endpoint's date

Every endpoint-family rule should be implemented in one place and covered by behavior-focused tests.

### Ownership

Effective-date resolution should be owned by a dedicated `storage.time` module plus behavior-focused tests.

That module should expose endpoint-family rules explicitly instead of hiding them inside a large monolithic write path.

## Public Cross-Stage Contracts

The rebuilt system should use a small number of explicit public contracts.

### 1. Universe Contract

Minimum fields:

- `conid`
- core metadata needed for collection and downstream joins
- universe status and any governance flags that are truly canonical

### 2. Canonical Reader Contracts

These should expose consumer-oriented reads, such as:

- price history
- dividend events
- curated snapshot tables
- supplementary raw or derived datasets

Reader contracts must be explicit and stable. Downstream stages should not use `SELECT *` against arbitrary tables.

### 3. Analysis Input Bundle

The feature-input stage should return one explicit bundle, for example:

```python
AnalysisInputBundle(
    prices=...,
    price_eligibility=...,
    dividends=...,
    dividend_summary=...,
    snapshot_features=...,
    snapshot_diagnostics=...,
    risk_free_daily=...,
    macro_features=...,
)
```

The exact type may be a dataclass, typed dict, or similar explicit structure.

### 4. Analysis Panel Contract

The panel contract should include:

- `conid`
- `rebalance_date`
- point-in-time feature columns
- eligibility state
- feature age diagnostics
- price-derived targets or return context

### 5. Research Output Bundle

The research stage should return one explicit result bundle containing:

- factor returns
- factor metadata and screening decisions
- cluster outputs
- walk-forward model results
- expected-return outputs
- persistence outputs

### 6. Portfolio Input Contract

Portfolio construction should consume explicit inputs, such as:

- expected returns
- covariance or other risk model outputs
- factor exposure matrix
- eligibility constraints
- optimizer configuration

The factor exposure matrix may come from research outputs, explicit exposure-estimation outputs, or both, but the contract must state which source is authoritative for each portfolio workflow.

## CLI Design

The CLI should expose honest runtime flows rather than a misleading umbrella DAG.

Suggested command groups:

- `refresh_universe`
- `collect_fundamentals`
- `refresh_supplementary`
- `build_inputs`
- `build_panel`
- `run_research`
- `estimate_exposures`
- `run_portfolio`
- `run_pipeline`

`run_pipeline` should call the same stage functions as the individual commands. No separate hidden orchestration dictionary should exist.

## Collection Concurrency Model

Collection should use bounded `asyncio` concurrency.

Design rules:

- one run owns one authenticated session context at a time unless the source contract explicitly supports multiple independent sessions
- target instruments may be processed concurrently up to a configured concurrency limit
- endpoint fanout within a single instrument may also be concurrent when the endpoints are independent
- concurrency must be bounded so rate limits, auth churn, and SQLite write contention remain controlled

Practical implication:

- collection gathers raw payloads concurrently
- canonical persistence should be serialized through a bounded write path or writer queue rather than allowing unconstrained concurrent SQLite writes

## SQLite Connection And Transaction Model

SQLite remains the default operational store, with explicit connection and transaction rules.

Required rules:

- WAL mode enabled
- foreign keys enabled
- one transaction per logical write unit, normally per instrument snapshot or per bounded write batch
- read flows use independent read-only or ordinary read connections
- write flows avoid sharing one connection across unrelated concurrent tasks

With bounded async collection:

- collection tasks should hand off canonical write inputs to a writer boundary
- the writer boundary should own SQLite transactions so write semantics stay deterministic

## Configuration And Secrets

The rebuild should use a layered configuration model.

Recommended structure:

- environment variables for secrets and environment-specific overrides
- a local config file for non-secret defaults when useful
- typed config objects inside the codebase

Configuration must cover at least:

- SQLite path and output paths
- collection concurrency and retry settings
- source base URLs and timeouts
- optional feature flags
- portfolio and research defaults

Secrets such as IBKR credentials must not be stored in committed project files.

## Error Propagation Strategy

Stage boundaries should distinguish between:

- fatal run-level errors
- per-item recoverable errors

Design rules:

- invalid configuration, missing credentials, or unrecoverable schema failures should raise fatal errors
- per-instrument collection failures should be captured as structured result states and diagnostics rather than aborting the whole run
- storage write failures should be attached to the affected write unit and surfaced in run summaries
- CLI commands should return machine-readable summaries and exit nonzero only for true run-level failure

## Diagnostics Design

Diagnostics should be emitted at every major stage.

Required diagnostic families:

- collection telemetry
- per-dataset usability diagnostics
- feature integrity diagnostics
- panel join diagnostics
- factor-selection diagnostics
- optimizer diagnostics

Diagnostics must be queryable artifacts, not only log lines.

## Derived Artifact Caching And Invalidation

Derived artifacts may be cached, but caches must be invalidated explicitly.

At minimum, cache identity should depend on:

- canonical source watermark or last-updated state for the relevant inputs
- feature or research configuration hash
- schema or contract version

Design rules:

- feature-input artifacts are recomputed when canonical inputs or relevant config change
- panel and research artifacts are recomputed when upstream input contracts or relevant config change
- caches should never be treated as authoritative when their manifest or watermark no longer matches current inputs

## Performance Design

Performance is a design constraint, not a later optimization pass.

The design should minimize:

- repeated full-table scans
- repeated reparsing of identical payloads
- repeated wide pivots when the same contract can be cached once
- unnecessary write-read-write artifact loops
- Python loops where SQL or vectorized pandas is clearer and faster

Concrete implications:

- canonical parsing happens once per raw payload version
- reader contracts expose already-curated shapes
- feature-input builders operate on canonical reader outputs, not raw payloads
- materialized derived artifacts are caches, not the primary architecture boundary

## Testing Strategy

The architecture should be supported by behavior-focused tests at the contract boundaries.

High-priority test families:

- endpoint effective-date resolution
- canonical write semantics per endpoint family
- reader contract shape and type guarantees
- price, dividend, snapshot, and supplementary input contracts
- point-in-time panel correctness
- factor-selection and walk-forward behavior
- optimizer input and constraint behavior

The tests should validate behavior and semantics, not only file existence.

### Fixture And Test Data Strategy

The rebuilt test suite should use a mix of:

- captured raw payload fixtures for endpoint canonicalization tests
- synthetic minimal payloads for edge-case time and parsing tests
- SQLite temp databases for write/read contract tests
- stubbed collection clients or local fake services for collection orchestration tests

The goal is to test:

- collection behavior without depending on live IBKR or World Bank services
- canonical write behavior against stable payload fixtures
- panel, research, and portfolio behavior against small deterministic datasets

## Schema Migration Strategy

Schema evolution should be explicit.

Recommended approach:

- keep a schema version table in the operational store
- apply ordered migrations on startup or command entry before stage logic runs
- store migrations as explicit versioned steps rather than relying only on `CREATE TABLE IF NOT EXISTS`

The exact migration mechanism can remain lightweight, but schema evolution must be deliberate and replayable.

## Idempotency Rules

Repeated runs are expected, so idempotency should be explicit.

Target behavior:

- `refresh_universe`: idempotent for unchanged upstream product data
- `collect_fundamentals`: idempotent at the canonical persistence layer for unchanged payloads, while still recording run telemetry
- `refresh_supplementary`: deterministic full refresh or deterministic replace semantics per dataset
- `build_inputs`: deterministic recompute from canonical inputs
- `build_panel`: deterministic recompute from stable input contracts
- `run_research`: deterministic recompute for fixed inputs, config, and seeds
- `run_portfolio`: deterministic recompute for fixed inputs, config, and solver settings where the solver supports it

## Migration Strategy

The rebuild should proceed incrementally.

### Phase 1: Lock Contracts

- finalize FRD
- finalize time-semantics and contract rules
- finalize the concern map

### Phase 2: Build New Canonical Boundaries

- implement the new package skeleton
- implement universe, collection, and storage foundations
- port reader contracts

### Phase 3: Build New Analysis Inputs

- implement cleaned price contract
- implement dividend usability contract
- implement snapshot feature contract
- implement supplementary derived input contract

### Phase 4: Build Panel And Research

- implement panel construction
- implement factor research and diagnostics
- implement exposure estimation

### Phase 5: Build Portfolio Construction

- implement optimizer input assembly
- implement efficient-frontier and constrained optimization flows
- emit portfolio outputs and diagnostics

### Phase 6: Cut Over

- compare outputs against the current implementation where meaningful
- migrate CLI entrypoints
- retire old flows once the new stage slice is proven

## Key Decisions This Design Makes

This design deliberately chooses:

- explicit concern ownership over monolithic stage modules
- endpoint-specific canonical storage over flat research-table storage
- reader contracts as the main cross-stage storage boundary
- feature-input bundles as the main analysis boundary
- point-in-time panel construction as its own concern
- portfolio construction as a mandatory top-level capability
- separate output boundaries for derived research and portfolio artifacts

## Open Decisions

These decisions remain open, but bounded:

- exact package name when the rebuild becomes canonical
- exact artifact format and manifest structure for derived outputs
- exact optimizer library choices and supported solver set
- exact scope of optional external enrichment sources beyond the mandatory core system

These are implementation decisions, not blockers to the architecture itself.
