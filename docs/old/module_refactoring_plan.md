# Revised Storage-First Reorganization for pystocks 2.x

## Summary

Proceed with the reorganization, but refine Step 1 so storage is introduced as a full persistence boundary, not just a folder for SQL code. The
implementation should explicitly solve three risks up front:

- Avoid storage/preprocess circular dependencies by classifying normalization by purpose.
- Avoid leaky transaction control by introducing a storage-owned unit-of-work API.
- Avoid partial initialization by centralizing all schema bootstrap in one place.

The first milestone is therefore not “move DB code into storage/”. It is “create a complete persistence boundary with schema, transaction, and
schema-normalization ownership”.

## Key Changes

### 1. Define normalization ownership before moving files

Split normalization into two categories:

- Persistence normalization: convert raw endpoint payloads into rows/shapes needed by SQLite tables.
- Analytical preprocessing: transform persisted tables into analysis-ready features/artifacts.

Rule:

- Persistence normalization lives under pystocks/storage/normalize/.
- Analytical preprocessing lives under pystocks/preprocess/.

For current code:

- The logic in fundamentals_normalizers.py that exists to produce snapshot rows, event rows, trade-log rows, metric rows, and schema-shaped fields
  moves with storage, not preprocess.
- preprocess/ must not import storage/normalize/ directly except through repository/service APIs that return persisted data or storage-ready row
  batches.
- storage may depend on storage.normalize.
- preprocess may depend on storage.readers / storage.repositories, but not on storage internals.

This prevents the circularity you called out:

- ingest -> storage
- storage -> storage.normalize
- preprocess -> storage
- never storage -> preprocess

### 2. Introduce transaction scope as a first-class storage API

Do not hide transaction control behind one-shot save helpers only, and do not expose raw sqlite3.Connection as the main public abstraction.

Add a storage-owned transaction boundary, for example:

- storage.transaction()
- or storage.unit_of_work()

Behavior:

- caller enters a context manager
- context object owns one DB session/connection
- repositories/readers/writers operate through that context
- commit on success, rollback on exception
- nested write flows inside one transaction are supported at the storage layer, not by callers reaching for raw SQLite

Public shape requirements:

- callers can group multiple writes atomically
- callers do not create connections directly
- callers do not need to know PRAGMA/bootstrap details
- raw connection access, if needed at all, is private or clearly marked as escape-hatch internal API

Preferred pattern:

- with storage.transaction() as tx:
- then use storage-scoped repositories/services on tx
- example responsibilities: persist product catalog rows, persist one fundamentals endpoint batch, replace one derived artifact family atomically

Rule for higher layers:

- ingest, preprocess, and analysis may control transaction scope
- they may not control low-level connection creation/configuration

### 3. Centralize schema ownership and bootstrap

All DDL moves into a single storage-owned bootstrap path.

Create:

- pystocks/storage/schema.py for table/index creation and schema version metadata
- one public bootstrap entrypoint such as storage.init_storage() or bootstrap on first transaction/session creation

Rules:

- no domain module creates tables directly
- ops_state.init_db() and FundamentalsStore._init_db() are both retired into the unified schema bootstrap
- bootstrap is idempotent
- bootstrap covers all tables/indexes needed by ingest, preprocess, and analysis outputs
- schema metadata/versioning remains storage-owned

This removes the partial-init risk where products exists but fundamentals tables do not.

### 4. Sequence the refactor around those boundaries

Stage the work in this order:

1. Introduce storage/schema, storage/txn, and storage/normalize without moving the whole package tree yet.
2. Move schema bootstrap out of ops_state and FundamentalsStore into unified storage bootstrap.
3. Move persistence-oriented normalizers out of fundamentals_normalizers.py into storage/normalize.
4. Replace direct sqlite3.connect(...) usage in preprocess and analysis with storage read/write APIs and transaction scopes.
5. After storage APIs are stable, move root modules into ingest/, analysis/, and diagnostics/ with temporary compatibility shims.
6. Split tests to mirror the final boundaries.
7. Remove compatibility shims only after all internal imports and tests are migrated.

### 5. Keep analysis and preprocess honest about their boundaries

Refine the earlier dependency rule:

- preprocess may load persisted source tables through storage readers and write derived artifacts through storage writers
- preprocess owns compute logic only
- analysis may call preprocess APIs and persist outputs through storage writers
- neither layer should embed SQL, table DDL, or connection bootstrap logic

Where a module currently mixes compute and persistence:

- first extract pure compute functions
- then move load/save logic behind storage APIs
- only then move files/packages

## Test Plan

Validate each stage with:

- ./venv/bin/python -m pytest -q
- ./venv/bin/python -m pyright
- ./venv/bin/python -m ruff check . --fix
- ./venv/bin/python -m ruff format .

Add or preserve tests for these scenarios:

- unified storage bootstrap creates all required tables/indexes in an empty DB
- repeated bootstrap is idempotent
- one transaction committing multiple writes persists all changes
- one transaction failing midway rolls back all writes
- preprocess and analysis paths no longer open SQLite connections directly
- persistence normalizers can be used by storage without importing preprocess
- ingest still persists endpoint payloads correctly through storage-owned APIs

## Assumptions

- Persistence normalization and analytical preprocessing are different concerns and must not share a module boundary.
- Transaction control is required by callers, but connection construction/configuration should remain storage-owned.
- SQLite remains the only backing store for now, but the design should avoid exposing raw SQLite setup details as the public interface.
- Schema bootstrap should be unified before any package move, otherwise the reorg will preserve existing initialization ambiguity.
