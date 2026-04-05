# PyStocks Simplification Plan

## Goal

Simplify `pystocks/` as far as possible without losing required behavior, keep modules decoupled, fix the real correctness bugs first, and only then optimize performance.

This plan is based on an end-to-end review of:

- `pystocks/ingest`
- `pystocks/preprocess`
- `pystocks/analysis`
- `pystocks/storage`
- supporting modules used directly by the pipeline

## Guiding Principles

1. Fix correctness before cleanup.
2. Prefer explicit contracts over implicit schema coupling.
3. Keep each stage responsible for one thing:
   ingest fetches data, preprocess shapes data, analysis consumes stable inputs, storage persists canonical artifacts.
4. Avoid duplicate work across CLI commands.
5. Only optimize after module boundaries are clear.

## Current Pipeline Shape

The effective pipeline order from [`pystocks/cli.py`](../pystocks/cli.py) is:

1. `scrape_products`
2. `scrape_fundamentals`
3. `run_analysis`

Standalone preprocess commands exist for prices, dividends, and snapshots, but analysis currently recomputes some of that work internally.

## Review Findings

### Priority 1: Correctness

1. `preprocess/dividends.py`
   Trailing dividend sums are assigned to the wrong rows across conids.
   The rolling calculation is performed on a re-sorted/reset frame and then aligned back by integer index instead of event identity.

2. `analysis/__init__.py`
   `cluster_factor_returns()` crashes when no sleeve has enough history.
   The function can build an empty `cluster_rows` list and then call `sort_values()` on a DataFrame that does not have the expected columns.

3. `ingest/fundamentals.py`
   Instruments intentionally skipped by the landing heuristic are retried forever.
   They are recorded as `empty_payload` with `mark_scraped=False`, so they are never filtered out by the recent-scrape logic.

### Priority 2: Behavior / API / Architecture

4. `analysis/__init__.py`
   Research fitting has an undocumented hard requirement of four snapshot dates.
   The train/test window builder silently no-ops for small datasets even when there may be enough return history to proceed.

5. `ingest/fundamentals.py`
   The fundamentals CLI path computes run stats and telemetry but returns no structured result.

6. `ingest/fundamentals.py`
   `FundamentalScraper` construction has hard I/O side effects.
   Instantiation immediately wires the real session, creates directories, opens the concrete store, initializes SQLite, and seeds telemetry.

7. `analysis/__init__.py`
   Analysis bypasses the standalone snapshot preprocess artifact and recomputes snapshot features from raw SQLite tables.

8. `preprocess/snapshots.py` and `storage/readers.py`
   Snapshot features are implicitly tied to storage schema churn.
   `SELECT *` plus "prefix every non-key column" means storage-only schema changes can silently become analysis features.

### Priority 3: Complexity / Coupling / Waste

9. `preprocess/dividends.py`
   Dividend preprocessing recomputes price preprocessing when a reference is not supplied.
   That is useful as a fallback but creates duplicate work and couples dividends tightly to the price module.

10. `ingest/fundamentals.py`
    The module mixes:
    - target selection
    - auth/session lifecycle
    - endpoint request planning
    - payload usefulness heuristics
    - persistence/status updates
    - telemetry serialization

11. `analysis/__init__.py`
    Analysis owns preprocessing logic, panel building, factor construction, clustering, model research, persistence, and current-beta estimation in one file.

12. `storage/readers.py`
    Reader contracts are too broad and in several places mirror raw table layout rather than stable consumer needs.

13. `config.py`
    Import-time directory creation is a small but real side effect that makes configuration less passive than it should be.

## Recommended Target Architecture

### Ingest

`ingest/session.py`
- Owns authentication state, login flow, and authenticated client creation only.

`ingest/product_scraper.py`
- Fetches product records and returns normalized product rows.
- Does not need dataframe-heavy processing for simple dedupe/upsert preparation.

`ingest/fundamentals.py`
- Selects target conids.
- Builds per-conid endpoint requests.
- Returns structured scrape outcomes.
- Delegates persistence, status recording, and telemetry writing to smaller helpers.

### Storage

`storage/`
- Owns canonical persistence and narrow read models.
- Exposes explicit reader schemas for downstream consumers.
- Stops leaking storage-only columns into preprocessing by default.

### Preprocess

`preprocess/price.py`
- Builds clean price and eligibility artifacts from canonical price series.

`preprocess/dividends.py`
- Consumes dividend events plus a supplied clean-price reference when available.
- Keeps the internal fallback to self-build price reference only as a convenience wrapper.

`preprocess/snapshots.py`
- Converts canonical snapshot tables into an explicit analysis feature contract.
- Does not infer feature columns from arbitrary table columns.

### Analysis

`analysis/`
- Consumes stable preprocessed inputs only.
- Should not need to know raw snapshot storage schema details.
- Keeps panel building, factor construction, and model research modular enough to test independently.

## Restructuring Guidance

Module restructuring should follow the simplification plan, not replace it.

The main issue in the codebase is not simply that some files are large. The deeper problem is that several modules have implicit contracts and mixed responsibilities. Moving code around before fixing those contracts would create larger diffs without much real simplification.

The recommended rule is:

1. fix correctness bugs first
2. define explicit inputs and outputs
3. restructure only the modules whose responsibilities are still too broad after that

### Recommended Restructuring

These are the structural changes most likely to pay off.

#### 1. Split `analysis/__init__.py`

This is the strongest restructuring candidate in the repository.

Today it mixes:

- input preparation
- panel building
- price feature construction
- factor return construction
- factor clustering
- model fitting
- persistence/output writing
- current beta computation

Suggested target shape:

- `analysis/panel.py`
  Panel construction and rebalance-date alignment.
- `analysis/factors.py`
  Price feature engineering, factor construction, and baseline construction.
- `analysis/research.py`
  Clustering, model fitting, persistence scoring, and current beta estimation.
- `analysis/io.py` or `analysis/outputs.py`
  Output writing and command-facing result assembly.
- `analysis/__init__.py`
  Thin public entrypoint layer only.

This split should happen after the snapshot feature contract is made explicit so the new modules do not inherit the same raw-storage coupling.

#### 2. Thin `ingest/fundamentals.py`

This file also has too many responsibilities, but it does not need to become a large package.

Suggested target shape:

- `ingest/fundamentals_runner.py`
  Target selection, session lifecycle, retry handling, telemetry/reporting, CLI-facing orchestration.
- `ingest/fundamentals_fetch.py`
  Per-conid fetch planning, endpoint selection, payload usefulness heuristics, and scrape result shaping.
- `ingest/fundamentals.py`
  Thin compatibility wrapper or public entrypoint module.

This should happen after the fundamentals result contract is made explicit and after the skip-state bug is fixed.

#### 3. Remove or merge trivial indirection modules

Some small modules are only useful if they define a real boundary.

Current candidate:

- `storage/normalize.py`

Right now it is mostly a re-export shim over `fundamentals_normalizers.py`. If it is not serving as a real storage-facing boundary, it should either be removed or replaced by direct imports from the canonical implementation module.

### Restructuring To Avoid Early

These changes are lower priority and should be deferred.

#### 1. Do not split `preprocess/` just for symmetry

The current preprocess modules already map to real domain boundaries:

- `preprocess/price.py`
- `preprocess/dividends.py`
- `preprocess/snapshots.py`

Those modules mostly need contract tightening, not file movement.

#### 2. Do not split `storage/fundamentals_store.py` immediately

It is large, but it is also the canonical persistence implementation. Splitting it too early risks spreading implicit coupling across several files without reducing complexity.

Only consider splitting it after:

- reader contracts are explicit
- snapshot feature boundaries are explicit
- ingest orchestration is simpler

At that point a split by endpoint family may become worthwhile. Before that, it is mostly churn.

### How Restructuring Fits Into The Plan

Restructuring belongs inside the later phases, not before them.

- Phase 1: no broad restructuring, only bug fixes
- Phase 2: contract cleanup first
- Phase 3: light ingest restructuring becomes worthwhile
- Phase 4: analysis restructuring becomes high value
- Phase 5: performance tuning after the new structure is stable

### Practical Recommendation

If only one structural refactor is done in the near term, it should be splitting `analysis/__init__.py`.

If a second structural refactor is done, it should be thinning `ingest/fundamentals.py`.

Everything else is secondary to the correctness and contract work already listed in this plan.

## Change Strategy

### Phase 1: Correctness Fixes

Implement first and release first.

1. Fix dividend trailing-sum alignment in `preprocess/dividends.py`.
2. Fix empty-result handling in `cluster_factor_returns()`.
3. Fix the ingest skip-state bug so intentionally skipped conids are not retried every run.
4. Replace the current research window construction with logic based on available factor-return history rather than fixed snapshot offsets.
5. Add regression tests for each issue.

### Phase 2: Explicit Contracts

Stabilize module boundaries before major cleanup.

1. Define an explicit snapshot feature schema in `preprocess/snapshots.py`.
2. Replace broad `SELECT *` snapshot readers with explicit column lists in `storage/readers.py`.
3. Make analysis consume that explicit snapshot feature contract only.
4. Decide whether the canonical interface is:
   - in-memory preprocessing functions, or
   - persisted preprocess artifacts

Recommendation:
keep the canonical interface as in-memory library functions for main pipeline execution, while retaining standalone preprocess commands as export utilities.

### Phase 3: Ingest Simplification

Break `ingest/fundamentals.py` into simpler units.

1. Separate target selection from scraping.
2. Separate per-conid fetch planning from per-run orchestration.
3. Inject session, store, and telemetry/output dependencies instead of constructing everything in `FundamentalScraper.__init__`.
4. Return a structured result from the fundamentals command path.
5. Resolve ESG account state once per authenticated run, not once per conid.

### Phase 4: Analysis Simplification

Reduce cross-module coupling and shrink the monolith.

1. Keep `_prepare_analysis_inputs()` small and explicit.
2. Split panel construction from factor construction from model fitting into smaller internal modules or clearly separated sections.
3. Make the panel builder consume preprocessed snapshot features instead of rebuilding them from raw storage tables.
4. Make train/test window rules explicit in config or code comments, not hidden in slice arithmetic.

### Phase 5: Performance Pass

Only after the above is stable.

1. Remove repeated work:
   - stop retrying intentionally skipped conids
   - stop recomputing snapshot preprocessing inside analysis if it is already available in-process
   - avoid rebuilding clean price references unnecessarily

2. Reduce unnecessary I/O:
   - avoid one DB lookup per conid for price-chart window selection if batch lookup is straightforward
   - stop re-reading account state per conid for ESG endpoint construction

3. Reduce unnecessary pandas overhead where the logic is simple:
   - replace the product scraper dataframe dedupe path with direct record dedupe

4. Revisit unconditional sleeps in ingest:
   - keep backoff only where endpoint behavior requires it

## Ordered Patch List

This is the recommended implementation order.

1. Patch dividend trailing-sum alignment bug.
2. Patch factor clustering empty-case bug.
3. Patch ingest landing-skip state handling.
4. Patch research window construction to remove the hidden four-snapshot requirement.
5. Return a structured summary from `run_fundamentals_update()` / `main()`.
6. Replace snapshot `SELECT *` readers with explicit column lists.
7. Introduce an explicit snapshot feature allowlist / mapping in `preprocess/snapshots.py`.
8. Make analysis consume that explicit contract only.
9. Refactor `FundamentalScraper` initialization so dependencies can be injected.
10. Cache obvious hot-path lookups and remove duplicate preprocess work.

## File-by-File Focus Areas

### `pystocks/ingest/fundamentals.py`

- Fix skip-state handling.
- Return structured run results.
- Split orchestration responsibilities.
- Inject dependencies instead of hard-wiring them at construction time.

### `pystocks/ingest/product_scraper.py`

- Remove unnecessary dataframe usage for dedupe.
- Keep the module focused on fetch + normalize + handoff.

### `pystocks/preprocess/dividends.py`

- Fix trailing-sum assignment.
- Keep price-reference handling explicit.
- Prefer passing a precomputed clean-price reference from callers that already have one.

### `pystocks/preprocess/snapshots.py`

- Replace implicit feature expansion with explicit feature mapping.
- Keep diagnostics, but separate them conceptually from feature generation.

### `pystocks/storage/readers.py`

- Replace `SELECT *` on analysis-facing snapshot readers.
- Keep readers shaped around consumer contracts, not full table shape.

### `pystocks/analysis/__init__.py`

- Fix `cluster_factor_returns()` empty-case handling.
- Replace research window slice arithmetic with explicit, testable logic.
- Keep analysis dependent on stable preprocess outputs only.

### `pystocks/storage/fundamentals_store.py`

- Leave the core persistence model in place for now.
- Revisit only where it directly blocks decoupling or causes repeated work.

## Non-Goals for the First Pass

These should not be mixed into the initial cleanup unless they become necessary.

- redesigning the full factor research methodology
- replacing SQLite
- moving diagnostics into a new subsystem
- rewriting storage normalization logic without a proven bug or simplification payoff
- broad file moves that make review harder before behavior is stabilized

## Validation Plan

After each phase, run:

```bash
./venv/bin/python -m ruff check . --fix
./venv/bin/python -m ruff format .
./venv/bin/python -m pyright
./venv/bin/python -m pytest -q
```

If storage or schema behavior changes:

```bash
./venv/bin/python -m pystocks.cli refresh_fundamentals_views
```

Add focused regression tests for:

- cross-conid dividend rolling sums
- clustering with no sleeve meeting history threshold
- fundamentals runs where all candidate factors are below `min_train_days`
- skipped landing-only instruments not being retried forever
- snapshot feature generation remaining stable after storage-only column additions

## Success Criteria

The cleanup is successful when:

1. The known correctness bugs are fixed and covered by tests.
2. Analysis no longer depends on implicit raw snapshot table shape.
3. Standalone preprocess commands remain useful utilities, but the main pipeline does not duplicate their work unnecessarily.
4. Ingest modules can be tested without constructing real filesystem/database state unless the test explicitly wants integration coverage.
5. The pipeline stays smaller, clearer, and faster without introducing a new persistence layer or unnecessary abstraction.
