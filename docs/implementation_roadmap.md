# Pystocks Implementation Roadmap

## Purpose

This document turns the current requirements and architecture docs into an execution plan for fresh agents.

It exists to answer:

- what to build first
- what each phase owns
- what must be true before a phase starts
- what counts as done
- which shortcuts are forbidden even if they seem faster

This roadmap is intentionally concrete. It should help an uninformed agent start useful work without rediscovering the architecture from scratch.

## Read This First

Before starting implementation work, read these documents in order:

1. [functional_requirements.md](/Users/alex/Documents/pystocks/docs/functional_requirements.md)
2. [architecture_design.md](/Users/alex/Documents/pystocks/docs/architecture_design.md)
3. [data_contracts_and_time_semantics.md](/Users/alex/Documents/pystocks/docs/data_contracts_and_time_semantics.md)
4. [proposed_concern_map.md](/Users/alex/Documents/pystocks/docs/proposed_concern_map.md)

Use the current `pystocks/` package as the behavioral reference. Do not assume its package boundaries are the target design.

## Roadmap Goals

The rebuild should optimize for:

- one concern, one owner
- explicit cross-stage contracts
- point-in-time correctness
- low module coupling
- low duplication of business rules
- small, testable slices
- simplicity over framework-style abstraction

The rebuild should not optimize for:

- maximum package granularity
- metadata registries
- generic orchestration layers
- porting every old module before proving the new architecture

## Agent Operating Rules

These rules apply in every phase.

### 1. Work In One Concern At A Time

Each slice should have one primary owner:

- universe
- collection
- canonical storage
- feature inputs
- panel construction
- research
- exposure estimation
- portfolio construction
- outputs

If a change needs simultaneous ownership from multiple concerns, the slice is probably too large.

### 2. Do Not Create Parallel Contract Layers

Do not add:

- a standalone `contracts/` package
- registries that restate reader contracts
- metadata files that mirror feature builders
- CLI wiring dictionaries that restate the real runtime flow

Contracts must live in the code that owns them.

### 3. Keep Dependency Direction Strict

Do not conflate runtime flow with import direction.

Runtime progression is:

- `universe -> collection`
- `collection -> storage`
- `storage -> feature_inputs`
- `feature_inputs -> panel`
- `panel -> research`
- `research -> exposures`
- `research -> portfolio`
- `exposures -> portfolio`

Code imports go the opposite way:

- `collection` may import from `universe`
- `storage` may import from `collection`
- `feature_inputs` may import from `storage`
- `panel` may import from `feature_inputs`
- `research` may import from `panel`
- `exposures` may import from `research`
- `portfolio` may import from `research` and `exposures`

Reverse imports are forbidden.

### 4. Keep Time Semantics Explicit

Do not collapse:

- `observed_at`
- `source_as_of_date`
- `effective_at`
- `join_date` or `rebalance_date`

If a slice changes time behavior, it must include behavior-focused tests.

### 5. Prefer Vertical Proof Over Broad Porting

Do not start by copying the whole repo into `pystocks_next/`.

Instead:

- establish one boundary
- prove it with tests
- expose one stable contract
- move to the next stage

### 6. Every Slice Must End With A Stable Public Surface

A slice is not done when code exists. It is done when:

- ownership is clear
- the public API is explicit
- tests cover the intended semantics
- a later agent can build on it without reading unrelated modules

## Definition Of Done For Any Slice

A work slice is complete only if all of these are true:

- the owning concern is obvious from the file layout and imports
- the slice does not introduce a reverse dependency
- the slice exposes one explicit contract or API
- behavior-focused tests cover the main semantics and edge cases
- names are stable enough for downstream code to depend on
- any cache or derived artifact policy is explicit
- docs are updated if the slice changes a public contract

## Delivery Strategy

The rebuild should use two nested levels of planning:

- phases: major architecture milestones
- work units: small agent-sized slices within a phase

Phases define order. Work units define execution.

## Recommended Working Package

Use a new package root during the rebuild:

- `pystocks_next/`

That keeps the migration boundary explicit and reduces accidental coupling to the live implementation.

The current `pystocks/` package remains the reference implementation until cutover.

## Phase 1: Lock Contracts And Build The Harness

### Objective

Make the design executable by locking the contracts that later code will depend on.

### Scope

- finalize the rebuild package name and top-level package layout
- define the minimal configuration surface
- define the migration/bootstrap entrypoint for the operational SQLite store
- define the shared test harness for temp databases and fixture payloads
- lock the endpoint-family `effective_at` rules at a level concrete enough to implement
- lock the first public result types or typed bundles that later stages will return

### Deliverables

- `pystocks_next/` package skeleton
- `config.py`
- test helpers for temp SQLite stores and payload fixtures
- a concrete endpoint-family time-resolution matrix
- explicit type shapes for:
  - universe rows
  - reader outputs
  - analysis input bundle
  - portfolio input bundle

### Acceptance Criteria

- a new agent can identify where configuration, migrations, and tests belong without reading the old codebase
- endpoint families no longer have ambiguous `effective_at` behavior in the docs
- the first downstream contracts are named and bounded tightly enough to implement without inventing new abstractions

### Forbidden Shortcuts

- no broad code port from `pystocks/`
- no full pipeline CLI yet
- no generic contract registry

## Phase 2: Build Canonical Operational Foundations

### Objective

Build the operational source-of-truth layer that every later stage depends on.

### Scope

- SQLite connection management
- WAL and transaction policy
- schema bootstrap and ordered migrations
- raw payload capture
- universe master persistence
- collection telemetry persistence
- endpoint-family canonical writes
- consumer-oriented reader contracts

### Work Units

#### 2A. Storage Core

Own:

- `storage/sqlite.py`
- `storage/schema.py`
- `storage/time.py`
- `storage/raw_capture.py`

Acceptance:

- the operational store can be initialized from scratch
- migrations are replayable
- time-resolution rules are implemented in one place
- raw payload blobs are persisted and deduplicated

#### 2B. Universe And Targeting

Own:

- `universe/products.py`
- `universe/governance.py`
- `universe/targeting.py`

Acceptance:

- canonical instrument identity is stored separately from scrape-operational state
- fixed target lists and governed universe selections are both supported

#### 2C. Collection Foundations

Own:

- `collection/session.py`
- `collection/telemetry.py`
- collection-facing write inputs

Acceptance:

- auth/session lifecycle is isolated from feature logic
- per-instrument fetch outcomes are classified without aborting full runs
- collection can hand structured write inputs to storage without knowing table details

#### 2D. Reader Contracts

Own:

- `storage/reads.py`

Acceptance:

- downstream code can read explicit consumer-oriented datasets without `SELECT *`
- reader outputs are typed and stable enough for feature-input work to begin

### Acceptance Criteria For Phase 2

- the new operational store can ingest and read back canonical data for the first supported endpoint families
- operational facts and derived outputs are still separate
- no downstream concern depends on raw table shape

### Forbidden Shortcuts

- no analysis-facing pivots in storage
- no research outputs in the operational SQLite store
- no product master table that also owns scrape status

## Phase 3: Build Feature Inputs

### Objective

Create the stable analysis-facing contracts that replace ad hoc preprocess logic.

### Scope

- cleaned price inputs
- dividend-usability inputs
- snapshot feature inputs
- supplementary derived inputs
- diagnostics for coverage, trustworthiness, and eligibility

### Work Units

#### 3A. Price Inputs

Acceptance:

- cleaned price history and return paths are reproducible from canonical storage
- price eligibility and coverage diagnostics are explicit outputs

#### 3B. Dividend Inputs

Acceptance:

- dividend events are preserved as history
- dividend usability is evaluated against price context
- row-level and summary diagnostics are emitted

#### 3C. Snapshot Feature Inputs

Acceptance:

- stable feature naming is owned here, not in research
- canonical snapshot facts are converted into an analysis-facing feature contract
- snapshot diagnostics explain missingness, freshness, and suspicious values

#### 3D. Supplementary Inputs

Acceptance:

- raw supplementary sources remain distinct from derived supplementary inputs
- ETF-to-supplementary mappings are explicit

### Acceptance Criteria For Phase 3

- the feature-input stage returns one explicit `AnalysisInputBundle`
- analysis code can consume the bundle without reaching into storage tables directly
- feature vocabulary exists in one authoritative place

### Forbidden Shortcuts

- no feature naming inside panel or research code
- no repeated parsing of canonical storage text that should already be typed facts
- no mandatory artifact-file boundary between feature inputs and analysis

## Phase 4: Build Panel Construction And Research

### Objective

Build the downstream research engine on top of stable input contracts.

### Scope

- rebalance or join-date definition
- point-in-time panel assembly
- factor construction
- factor return estimation
- screening, clustering, and reduction
- walk-forward research outputs
- current exposure estimation

### Work Units

#### 4A. Panel Construction

Acceptance:

- latest eligible features at or before each join date are selected deterministically
- feature age, eligibility state, and join diagnostics are carried forward

#### 4B. Research

Acceptance:

- factor returns and screening decisions are explicit outputs
- walk-forward window semantics are test-covered
- expected-return outputs are clearly separated from canonical storage

#### 4C. Exposure Estimation

Acceptance:

- current factor exposures are produced from accepted research outputs
- the exposure contract is explicit enough for portfolio workflows to consume

### Acceptance Criteria For Phase 4

- panel construction is its own concern, not buried inside research
- research consumes only the analysis input bundle or panel contract
- derived research artifacts can be materialized without becoming canonical facts

### Forbidden Shortcuts

- no direct snapshot-feature assembly inside research
- no mixing of panel-build logic with collection or storage code

## Phase 5: Build Portfolio Construction

### Objective

Implement the mandatory portfolio-construction domain from the FRD.

### Scope

- optimizer input assembly
- covariance or risk input assembly
- factor-aware constraint handling
- efficient-frontier workflows
- constrained portfolio workflows
- portfolio diagnostics and outputs

### Work Units

#### 5A. Portfolio Input Contract

Acceptance:

- expected returns, risk inputs, exposures, and eligibility constraints are explicit
- the authoritative source of each portfolio input is documented in code

#### 5B. Optimization And Frontier Workflows

Acceptance:

- at least one efficient-frontier workflow is reproducible
- at least one constrained optimizer workflow is reproducible
- solver assumptions and constraint semantics are explicit

#### 5C. Portfolio Outputs

Acceptance:

- portfolio solutions and diagnostics are emitted through the derived-output boundary
- output shape is stable enough for comparison and regression testing

### Acceptance Criteria For Phase 5

- portfolio construction is a first-class concern, not an appendix to research
- workflows consume explicit portfolio inputs rather than storage tables or raw research internals

### Forbidden Shortcuts

- no optimizer reading arbitrary research artifacts by filename alone
- no mixing solver-specific assumptions into generic research bundles

## Phase 6: Cut Over

### Objective

Replace the live implementation only after the new one proves the core contracts.

### Scope

- compare new outputs against the current implementation where meaningful
- expose honest CLI entrypoints for the new flows
- migrate operational commands
- retire old flows only after their replacements are verified

### Acceptance Criteria

- the new CLI reflects the real runtime flows
- the rebuild covers the mandatory FRD scope
- old and new outputs have been compared for the slices where equivalence is meaningful
- unsupported or intentionally changed behaviors are documented explicitly

### Forbidden Shortcuts

- no cutover just because the package compiles
- no silent retirement of existing behavior without an explicit decision

## Recommended First Proving Slice

Do not begin with the whole fundamentals pipeline.

The first proving slice should be:

1. `pystocks_next` package skeleton
2. configuration and temp-DB test harness
3. storage core:
   - SQLite bootstrap
   - schema versioning
   - raw payload capture
   - explicit `effective_at` rules
4. universe master and targeting
5. reader contracts for the first supported datasets
6. price-input and dividend-input feature builders
7. a minimal point-in-time panel smoke path

Why this slice first:

- it proves the new ownership model
- it avoids starting with the broadest mixed-responsibility scraper path
- it validates time semantics and reader contracts early
- it gives later snapshot and research work a stable base

Snapshot-heavy feature work should come after this slice is stable, because snapshot semantics are the most coupled and highest-risk part of the current implementation.

## Endpoint Sequencing Guidance

Within collection and storage, do not try to reimplement every endpoint at once.

Recommended order:

1. universe refresh and targeting inputs
2. price and dividend series support
3. snapshot endpoint families needed for stable feature inputs
4. supplementary refresh and derived supplementary inputs
5. remaining endpoint families that are useful but not on the critical path

This ordering supports a lean end-to-end path sooner and reduces the risk of building a broad but unproven storage layer.

## Output And Cache Guidance

Treat artifact files as optional materializations, not primary architecture boundaries.

Rules:

- canonical operational data lives in the operational store
- feature-input outputs may be cached for speed
- panel, research, exposure, and portfolio outputs must stay outside canonical operational storage
- cache identity must depend on source watermark, config, and schema or contract version

## Testing Strategy By Phase

Every phase should add tests that prove semantics, not only wiring.

### Phase 1

- temp SQLite bootstrap tests
- config parsing tests
- endpoint-family time-rule tests

### Phase 2

- raw payload dedup tests
- migration replay tests
- canonical write/read contract tests
- collection outcome-classification tests

### Phase 3

- cleaned price and return-shaping tests
- dividend usability tests
- snapshot feature contract tests
- supplementary mapping tests

### Phase 4

- point-in-time panel tests
- walk-forward window tests
- factor-screening behavior tests
- exposure estimation tests

### Phase 5

- optimizer input assembly tests
- efficient-frontier reproducibility tests
- portfolio-constraint behavior tests

## How To Assign Work To Agents

When assigning a slice to an agent, the task should state:

- the owning concern
- the exact files or modules in scope
- the public contract to implement or change
- the acceptance criteria
- the forbidden shortcuts for that slice

Good task shape:

- "Implement `storage.time` endpoint-family `effective_at` rules and tests. Do not touch feature-input or research modules."

Bad task shape:

- "Refactor fundamentals flow."

## Stop Conditions

An agent should stop and escalate instead of improvising when:

- a slice appears to require a reverse dependency
- a required contract does not have an obvious owner
- a time rule cannot be stated explicitly
- two phases appear to need the same feature vocabulary
- a proposed shortcut would mix canonical and derived storage again

## Success Criteria For The Rebuild

The rebuild is successful when all of these are true:

- fresh agents can locate ownership quickly
- cross-stage contracts exist in one authoritative place
- point-in-time semantics are explicit and test-covered
- portfolio construction is supported as a first-class workflow
- the implementation is smaller and easier to reason about than the current refactor, not merely more abstract
