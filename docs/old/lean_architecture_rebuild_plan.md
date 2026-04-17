# Pystocks Lean Architecture Rebuild Plan

## Purpose

This document replaces "refactor by adding more layers" with a constrained rebuild plan.

The goal is not to make `pystocks` more abstract. The goal is to make it:

- lean
- explicit
- testable
- point-in-time correct
- performant on realistic ETF datasets

This plan assumes the current `pystocks/` package remains the production system while a new package is built in parallel from the repo root.

Recommended rebuild target: `pystocks_next/`

That name is intentionally temporary. It makes the migration boundary obvious and avoids pretending the new implementation is already the canonical one.

## Why The Current Refactor Drifted

The current roadmap correctly identified the architectural problems, but the implementation drifted because it solved them with more metadata instead of fewer owners.

The main failure modes were:

- contract metadata added on top of existing reader and preprocess logic instead of replacing it
- docs and code both restating the same contracts
- CLI dictionaries describing the pipeline separately from the pipeline code
- analysis decoupled from storage mostly through mandatory artifact files rather than a cleaner in-memory API
- a large snapshot system still spread across storage, preprocess, and analysis

The result is improved boundaries in some places, but more code paths and more things to keep in sync.

## Rebuild Principles

These rules are non-negotiable.

### 1. One Concern, One Owner

Every cross-stage concern has one owner:

- endpoint fetch behavior: `ingest`
- canonical persistence and time semantics: `storage`
- feature naming and shaping: `features`
- panel construction and research logic: `analysis`
- disposable derived outputs: `outputs`

If two modules define the same feature vocabulary or timestamp policy, the design is wrong.

### 2. No Parallel Contract Layers

Do not create:

- a `contracts/` directory
- metadata registries that mirror SQL readers
- CLI dictionaries that mirror actual stage wiring

The contract should live in the code that owns it:

- storage reader contract: explicit reader function columns
- feature contract: explicit feature builder output columns
- time contract: storage write rules plus tests

Documentation may explain the contract, but must not become a second implementation.

### 3. Analysis Consumes Data, Not Filenames

Analysis should accept in-memory DataFrames or a typed input bundle.

Parquet is a cache and reproducibility layer, not the primary architecture boundary.

Good:

- `bundle = build_feature_bundle(sqlite_path=...)`
- `run_analysis(bundle)`

Acceptable:

- `bundle = load_feature_bundle_from_cache(output_dir)`

Bad:

- analysis only works by knowing specific artifact filenames

### 4. Canonical Storage Stores Facts, Not Panels

Storage keeps:

- raw payloads
- normalized endpoint facts
- explicit source dates
- typed scalar values whenever source meaning is unambiguous

Storage does not keep:

- analysis composites
- convenience pivots for open-ended feature sets
- research outputs mixed with source-of-truth tables unless there is a temporary, documented exception

Implication:

- canonical storage is not "raw strings plus nicer table names"
- endpoint parsing that turns clear source values into typed canonical facts still belongs to storage
- feature-stage logic may derive new metrics, bins, and pivots, but it should not repeatedly re-parse obvious canonical numbers from text

### 5. Pivot Late

Canonical storage should stay tall unless the data shape is truly closed-world and stable.

Wide is acceptable only when all of these are true:

- the columns are semantically fixed
- row identity is stable
- downstream consumers would otherwise pay repeated pivot cost for the same closed-world record

### 6. Prefer Functions Over Framework

Use small modules with direct functions.

Prefer:

- `load_*`
- `build_*`
- `save_*`
- `run_*`

Avoid:

- manager classes
- orchestration registries
- plugin-style hook systems
- generic "contract resolvers"

### 7. Performance Is A Design Constraint

The new architecture must reduce data movement, not only improve conceptual boundaries.

The rebuild should minimize:

- repeated full-table scans
- repeated wide pivots
- unnecessary parquet round-trips
- repeated coercion of the same columns
- Python loops where SQL or vectorized pandas is simpler

## Proposed Package Layout

Create a new root package:

```text
pystocks_next/
  __init__.py
  cli.py
  config.py
  ingest/
    __init__.py
    products.py
    fundamentals.py
    supplementary.py
  storage/
    __init__.py
    schema.py
    sqlite.py
    writes.py
    readers.py
    time.py
  features/
    __init__.py
    bundle.py
    definitions.py
    prices.py
    snapshots.py
    supplementary.py
  analysis/
    __init__.py
    inputs.py
    panel.py
    factors.py
    research.py
    outputs.py
  tests/
    ...
```

Notes:

- rename `preprocess` to `features`
  - "preprocess" is too generic
  - "features" states the stage's real purpose
- do not create a standalone `contracts` package
- do not split `analysis` further until the first end-to-end slice works

## Stage Responsibilities

### ingest

Responsibilities:

- call external services
- handle retries and scrape heuristics
- emit raw payloads and normalized write inputs

Forbidden:

- importing `features`
- running feature shaping
- writing research outputs

### storage

Responsibilities:

- schema bootstrap and migrations
- raw payload persistence
- canonical endpoint writes
- point-in-time date resolution
- narrow reader APIs
- endpoint normalization and scalar parsing where the source semantics are clear

Forbidden:

- feature prefixes
- analysis composites
- reading with `SELECT *` in public reader APIs
- pushing obvious source-to-scalar parsing downstream merely to make storage look "thinner"

### features

Responsibilities:

- transform canonical storage inputs into analysis-ready datasets
- define stable feature names
- compute diagnostics needed to trust those features
- return a single input bundle for analysis
- own derived mappings and reusable analysis-facing definitions

Forbidden:

- scrape logic
- storage mutation except optional cache writes
- factor research logic
- becoming a grab-bag monolith of static taxonomies mixed with pivot code

### analysis

Responsibilities:

- validate feature inputs
- construct rebalance panels
- compute factors and research outputs
- persist research artifacts to a separate output boundary

Forbidden:

- reaching back into raw storage tables when a feature builder should own the transform
- hidden preprocessing

## Core Runtime Model

The real DAG should be:

1. ingest products
2. ingest fundamentals
3. ingest supplementary
4. build price features
5. build snapshot features
6. build supplementary features
7. assemble feature bundle
8. run analysis
9. write outputs

The CLI should expose exactly those steps or clear convenience wrappers around them.

No separate dictionary should be needed to explain the pipeline order.

## Contract Model

### Storage Contract

The storage contract is:

- the schema
- the storage write rules
- the reader function output columns
- the tests asserting those columns and time semantics

Storage normalization rule:

- if the source exposes a scalar that clearly means "number", "percent", "date", or "boolean", parse it into canonical typed storage once
- preserve the original raw payload in `raw_payload_blobs` for backfills and parser fixes
- do not defer obvious scalar parsing to `features`

Example:

- `load_snapshot_inputs()` returns a dict of canonical tables for feature building
- its column shape is asserted in tests
- there is no second metadata file restating the same thing

### Feature Contract

The feature contract is:

- the columns returned by each feature builder
- the bundle returned by `features.bundle.build_feature_bundle()`
- tests that prove those columns and point-in-time joins

Snapshot namespaces should be defined only inside `features/snapshots.py`.

Analysis may rely on those names, but must not regenerate or reinterpret them.

Static feature definitions rule:

- large reusable mappings such as sector groups, country blocs, or feature families should live in `features/definitions.py`
- builder modules such as `features/snapshots.py` should focus on assembling and validating features, not housing every taxonomy constant inline
- `definitions.py` must stay declarative; if it starts accumulating procedural logic, split by domain instead of rebuilding a new monolith

### Time Contract

The time contract is owned by `storage/time.py`.

It should be short and executable:

- one function per endpoint family
- one test module proving expected behavior

Do not create both a code registry and a second explanatory metadata registry unless the second one is purely prose.

## Recommended Public APIs

The new package should bias toward a few obvious entry points.

### storage

- `persist_combined_snapshot(...)`
- `load_price_history(...)`
- `load_snapshot_inputs(...)`
- `load_supplementary_sources(...)`

### features

- `build_price_features(...)`
- `build_snapshot_features(...)`
- `build_supplementary_features(...)`
- `build_feature_bundle(...)`

### analysis

- `build_analysis_panel(bundle, ...)`
- `run_factor_research(bundle, ...)`
- `run_analysis(bundle, ...)`

### outputs

- `save_feature_cache(bundle, output_dir)`
- `load_feature_cache(output_dir)`
- `save_research_outputs(result, output_dir or sqlite_path)`

## Feature Bundle Shape

The new analysis boundary should be one object, not many filenames.

Recommended shape:

```python
@dataclass
class FeatureBundle:
    prices: pd.DataFrame
    price_eligibility: pd.DataFrame
    snapshots: pd.DataFrame
    risk_free_daily: pd.DataFrame
    world_bank_features: pd.DataFrame
```

Keep it minimal.

Do not put configuration, diagnostics, paths, and cache metadata into the same object unless analysis actually needs them.

Diagnostics can be returned separately:

- `FeatureBundle`
- `FeatureDiagnostics`

The bundle must also be operationally sliceable.

Required properties:

- easy date filtering before analysis panel construction
- easy `conid` filtering for focused runs and tests
- ability to build only the datasets needed by the requested analysis mode

Preferred APIs:

- `build_feature_bundle(..., start_date=None, end_date=None, conids=None)`
- `slice_feature_bundle(bundle, start_date=None, end_date=None, conids=None)`

If full-history bundles become too large for memory, the next step is not to make analysis depend on filenames again. The next step is:

- narrower bundle construction
- stage-local caching
- optional lazy readers behind the same bundle interface

## Persistence Strategy

### Operational Store

Keep canonical storage in the existing SQLite DB.

This DB contains:

- products
- raw payload blobs
- endpoint tables
- supplementary raw and canonical inputs

### Research Store

Move research persistence out of the operational DB.

Preferred options, in order:

1. parquet-only for derived research outputs
2. separate SQLite DB under `data/research.sqlite`
3. temporary namespaced tables in the main DB only if migration cost blocks separation

The rebuild should default to option 1 or 2.

## Performance Rules

Every major stage should follow these rules.

### Reader Rules

- read only required columns
- push filtering and grouping into SQL when practical
- avoid returning tables broader than the consumer needs

### Feature Rules

- compute each expensive pivot once
- avoid repeated normalization passes over the same DataFrame
- prefer union-of-dates assembly only where point-in-time semantics require it
- persist caches only for expensive reusable outputs
- keep static mapping definitions out of hot builder paths unless they are actually used
- support partial feature builds by date range or asset subset when analysis does not need the full corpus

### Analysis Rules

- consume prepared inputs directly
- avoid recomputing features inside research loops
- cache intermediate wide return matrices inside the run when reused
- request the narrowest viable bundle for the job instead of assuming "full history, full universe"

### Testing Rules

- include a small fixture dataset for correctness tests
- include at least one timing or scale regression test for the heaviest feature builders

## Migration Strategy

Do not rewrite the whole repo at once.

Build one thin vertical slice at a time inside `pystocks_next/`.

### Phase 0: Freeze More Abstraction

Before building the new package:

- stop adding new contract metadata files to `pystocks/`
- stop adding new analysis-to-storage shortcuts
- stop adding new docs that restate runtime logic

### Phase 1: Build Skeleton And Shared Utilities

Create:

- `pystocks_next/config.py`
- `pystocks_next/storage/sqlite.py`
- `pystocks_next/storage/schema.py`
- `pystocks_next/cli.py`

Goal:

- a minimal package that imports cleanly
- no feature logic yet

### Phase 2: Rebuild Storage Time Semantics First

Port only:

- canonical storage bootstrap
- snapshot persistence
- endpoint time resolution
- endpoint scalar parsing needed to produce typed canonical facts
- narrow readers needed by the first feature slice

Acceptance criteria:

- the new storage layer can persist and read one combined snapshot correctly
- point-in-time tests are clearer than the current implementation

### Phase 3: Build Price Features End-To-End

Prices are the cleanest first vertical slice.

Implement:

- storage price reader
- `features/prices.py`
- optional parquet cache
- analysis input loading from in-memory bundle

Acceptance criteria:

- analysis can consume price features without hidden preprocessing

### Phase 4: Build Snapshot Features With A Single Owner

Implement snapshot shaping entirely in `features/snapshots.py`.

Rules:

- one file owns namespaces and feature naming
- no duplicate metadata registry elsewhere
- diagnostics stay adjacent to the builder
- reusable static maps move to `features/definitions.py`, not inline into another giant module

Acceptance criteria:

- analysis imports snapshot columns from one place only
- adding a new snapshot feature requires editing one builder, not three layers

### Phase 5: Rebuild Supplementary As Two Clean Stages

Implement:

- `ingest/supplementary.py` for raw fetch only
- `features/supplementary.py` for derived risk-free and macro features

Acceptance criteria:

- ingest does not import features
- analysis sees only built supplementary datasets

### Phase 6: Rebuild Analysis On The Bundle Boundary

Implement:

- `analysis/inputs.py`
- `analysis/panel.py`
- `analysis/factors.py`

Rules:

- analysis accepts `FeatureBundle`
- cache loading is optional and thin
- no direct dependency on storage reader details

### Phase 7: Switch CLI To The New Vertical Slices

Only after the first slices are correct:

- add `pystocks_next` CLI commands
- compare outputs side by side with legacy `pystocks`
- migrate users when parity is acceptable

### Phase 8: Retire Legacy Modules Incrementally

Do not delete `pystocks/` all at once.

Retire module families only when:

- replacement behavior is verified
- tests cover the replacement
- output parity is acceptable

## What To Keep From The Current Refactor

Keep the ideas that actually simplify the system:

- honest CLI stage order
- removal of ingest -> preprocess coupling
- endpoint-specific `effective_at` semantics
- loud failures when required analysis inputs are truly missing

## What To Remove Or Avoid

Avoid these in the rebuild:

- `*_contract.py` files that mirror runtime code
- giant pipeline description dictionaries
- artifact filenames as the primary analysis boundary
- more monolithic registries
- more wide canonical storage tables for convenience
- storing obvious numeric facts as raw text in canonical tables
- relocating large hardcoded analysis maps into a new builder monolith without separating definitions from transforms

## Bloat Alarms

Stop and redesign if any of these appear:

- a new module exists mainly to describe another module
- the same column list appears in more than one owner module
- analysis needs to know storage table names directly
- a convenience cache becomes mandatory for correctness
- a new feature requires coordinated edits across storage, metadata, and analysis
- canonical readers return text that every downstream consumer immediately reparses into the same numeric scalar type
- `FeatureBundle` construction always loads full history and full universe even when the caller needs a narrow slice

## Acceptance Criteria For The Rebuild

The rebuild is successful only if all of these are true:

1. A new engineer can explain the DAG from the CLI code alone.
2. Snapshot feature naming has one owner.
3. Time semantics have one executable owner plus tests.
4. Analysis runs from an in-memory feature bundle.
5. Research outputs are separated from canonical operational storage.
6. The new implementation uses fewer moving parts than the current refactor.
7. Performance is at least as good as the legacy pipeline on representative data.

## Immediate Next Steps

1. Approve this plan as the rebuild baseline.
2. Create `pystocks_next/` with only the minimal skeleton.
3. Port storage time semantics first.
4. Implement the price vertical slice before any broader package split.
5. Use output parity tests to decide when each legacy stage can be retired.
