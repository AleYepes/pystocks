# Pystocks Architecture Refactor Roadmap

## Purpose

This document is the canonical architectural catch-up guide for new agents working in `pystocks`.

Read it before making architecture-affecting changes.

This document exists to:

- explain the repo's currently confirmed architectural problems precisely
- distinguish root causes from symptoms
- define the target architecture and non-negotiable invariants
- provide a refactor sequence that improves correctness before aesthetics
- give new agents a reliable mental model of what belongs where

This roadmap reflects a code review of the live pipeline as of 2026-04-11.

## How To Use This Document

Use this roadmap when you need the explanation, rationale, and sequencing behind the current architectural guidance.

In particular, use it to answer:

- what the intended pipeline is
- what the current pipeline actually does
- which current behaviors are architectural violations versus deliberate tradeoffs
- where a proposed change should live
- which refactors are safe to do before others

This document is intentionally detailed.

`AGENTS.md` serves a different purpose:

- this roadmap explains the problems, boundaries, and sequencing
- `AGENTS.md` gives short, durable guardrails that prevent future drift

If the two ever seem to disagree, treat this roadmap as the canonical explanation and update `AGENTS.md` to match it.

## Executive Summary

The repo's main architectural problem is not simply that some storage tables are tall and others are wide.

The deeper problem is that there is no single canonical contract for snapshot features and point-in-time data across:

- ingestion
- storage
- preprocess
- analysis
- CLI orchestration

Because there is no authoritative contract, multiple layers re-declare the same business semantics in different forms:

- storage schema and upsert logic define one shape
- storage readers define another
- snapshot preprocess renames and pivots again
- analysis hardcodes final column names and composites again

The tall/wide inconsistency is real, but it is best understood as one symptom of this larger contract drift.

The other confirmed architectural problems are:

1. The pipeline DAG exposed by the CLI is not the real dependency graph.
2. The point-in-time time model is too weak and currently over-anchors endpoint dates to `ratios.as_of_date`.
3. Analysis is a monolith that owns too many responsibilities.
4. Research outputs and operational data are mixed too closely in the same SQLite database.
5. Several integrity rules live in Python choreography instead of the schema.

## Current System Map

### Intended Pipeline Shape

The intended stage model is:

1. ingest
   - authenticate
   - fetch raw payloads
   - choose what to request next
2. storage
   - persist raw payload blobs
   - persist canonical normalized endpoint tables
   - expose narrow read models
3. preprocess
   - turn canonical stored data into stable analysis-facing features and diagnostics
4. analysis
   - consume preprocessed inputs
   - build panels, factors, models, and research outputs
5. outputs
   - persist or export derived artifacts with clear ownership

The intended dependency direction is:

- `ingest -> storage`
- `preprocess -> storage`
- `analysis -> preprocess`

### Current Effective Pipeline Shape

The effective runtime DAG is looser and less honest:

1. `run_pipeline` presents:
   - scrape products
   - scrape fundamentals
   - run analysis
2. standalone preprocess commands exist for:
   - prices
   - dividends
   - snapshots
3. `analysis` still performs price preprocessing internally
4. snapshot features are recomputed on demand from storage rather than treated as a stable preprocess contract
5. supplementary refresh performs preprocessing inside the ingest stage
6. analysis writes research tables back into the same SQLite DB as operational source data

This mismatch matters because many modules look cleaner at the import boundary than they are in the real dataflow.

## Core Architectural Vocabulary

Use these terms consistently.

### Raw Payload Blob

The immutable captured source payload stored in `raw_payload_blobs`.

Purpose:

- preserve source capture
- support backfills and reparsing
- avoid losing source fidelity when downstream contracts change

### Canonical Storage

The normalized persistence layer owned by storage.

Purpose:

- store source-derived facts in stable, consumer-independent form
- preserve explicit time semantics
- avoid embedding analysis panel shape in the DB layer

### Consumer-Oriented Reader Contract

A storage reader output designed around downstream consumers, not around whichever raw table columns happen to exist today.

Good reader contracts:

- expose explicit columns
- have stable meaning
- do not silently widen when schema changes

### Preprocess Contract

The documented analysis-facing feature contract emitted by preprocess.

Purpose:

- define which features analysis can rely on
- own feature naming and pivoting
- isolate storage churn from analysis

### Analysis Artifact

A derived output produced by analysis, such as a panel, factor return table, model result, or diagnostics table.

These are not canonical source-of-truth inputs.

### `observed_at`

When the system observed or fetched the data.

This is an acquisition timestamp.

### Endpoint `as_of_date`

The date the source says the data describes.

This is source-defined and endpoint-specific.

### Storage `effective_at`

The canonical row date used for persistence and downstream joins.

This must be chosen deliberately. It should not casually collapse distinct source dates into one global anchor.

### Rebalance / Join Date

The downstream analysis date at which features or returns are joined.

This is an analysis concern, not a storage concern.

### Canonical vs. Derived

Canonical data:

- should be stable
- should support reprocessing
- should not depend on one analysis consumer's convenience shape

Derived data:

- can be recomputed
- can be replaced
- should have explicit lifecycle and ownership

### Tall vs. Wide

Tall:

- feature identity is stored in rows
- typically uses columns like feature key, category, or metric id plus value

Wide:

- feature identity is encoded in columns
- convenient for analysis panels
- risky in canonical storage when features are open-ended or high-cardinality

## Confirmed Current Problems

### 1. No Canonical Snapshot Feature Contract

This is the highest-priority architectural issue.

What it is:

- the same snapshot concepts are declared in multiple layers, each with its own shape and naming

Where it appears:

- `pystocks/storage/schema.py`
- `pystocks/storage/fundamentals_store.py`
- `pystocks/storage/readers.py`
- `pystocks/preprocess/snapshots.py`
- `pystocks/analysis/__init__.py`

Why it exists:

- storage, preprocess, and analysis evolved incrementally
- each stage solved its own immediate shape problem
- no single contract was declared as authoritative

Why it matters:

- adding one metric often requires coordinated edits across several layers
- feature names become string-typed and brittle
- schema churn leaks into preprocess and analysis
- ownership of feature semantics is ambiguous

How a new agent will notice this:

- feature names appear as hardcoded strings in multiple files
- snapshot preprocess contains a registry plus another round of prefixing and pivoting
- analysis contains hardcoded composite feature lists and assumptions

Correct boundary:

- storage should expose canonical source inputs
- preprocess should own the explicit feature contract
- analysis should consume that contract only

Root-cause statement:

`pystocks` currently has storage-defined shapes, preprocess-defined feature names, and analysis-defined semantic groupings, but no single source of truth connecting them.

### 2. The CLI DAG Is Not Honest About Real Dependencies

What it is:

- the user-facing pipeline shape does not match the actual computational DAG

Where it appears:

- `run_pipeline` presents products -> fundamentals -> analysis
- standalone preprocess commands exist but are not represented in the main DAG
- analysis still performs hidden preprocessing

Why it exists:

- convenience-oriented orchestration accumulated before stage boundaries were fully defined

Why it matters:

- stage ownership becomes hard to reason about
- users cannot tell which outputs are required inputs versus convenience exports
- work is duplicated
- architectural violations can hide inside orchestration shortcuts

How a new agent will notice this:

- preprocess commands exist in the CLI but the main pipeline skips over them
- analysis still loads raw-ish inputs and computes some preprocess outputs internally

Correct boundary:

- the main pipeline should reflect the real DAG
- if preprocess work is required, it must appear as preprocess, whether executed as explicit commands or in-process library calls

Root-cause statement:

The orchestration layer does not match the actual dependency graph, so module boundaries look cleaner than they really are.

### 3. Boundary Inversion Between Ingest and Preprocess

What it is:

- `ingest/supplementary.py` imports preprocess helpers and runs preprocessing logic as part of a refresh flow

Where it appears:

- supplementary fetch and supplementary preprocess are currently mixed inside the ingest stage

Why it exists:

- supplementary data was added pragmatically as a single refresh path

Why it matters:

- dependency direction is inverted
- ingest can no longer be treated as a fetch-and-persist stage
- preprocess logic is harder to reason about independently
- circular dependency risk rises

How a new agent will notice this:

- an ingest module imports `preprocess.supplementary`
- refresh code fetches raw data and also emits preprocessed feature tables

Correct boundary:

- ingest may fetch and persist source data
- preprocess may read stored supplementary data and derive analysis-facing features

Rule violation:

The desired direction is:

- `ingest -> storage`
- `preprocess -> storage`
- `analysis -> preprocess`

Not:

- `ingest -> preprocess`

### 4. Weak Point-in-Time Data Model

This is partly a correctness issue and partly an architectural one.

What it is:

- endpoint-specific time meanings are not treated as first-class cross-stage contract elements

Where it appears:

- `FundamentalsStore._resolve_effective_dates()` anchors endpoint `effective_at` values to `ratios.as_of_date` when available
- endpoint-specific `observed_at` and endpoint-specific `as_of_date` are not consistently preserved as distinct downstream concepts

Why it exists:

- one globally useful anchor was used as a convenience during normalization

Why it matters:

- different endpoints have different publication cadences and update semantics
- unrelated data can appear more synchronized than it truly was
- point-in-time reconstruction becomes fragile
- analysis leakage risk increases

How a new agent will notice this:

- multiple endpoint rows share one `effective_at` even when their source semantics differ
- tests and code talk about `effective_at` more often than the source-specific dates that produced it

Correct boundary:

- storage must preserve explicit source time semantics
- preprocess and analysis must document which time field drives each join

Root-cause statement:

The repo stores timestamps, but it does not yet have a rigorous cross-stage time contract defining:

- when data was observed
- what date the source says it describes
- which timestamp downstream point-in-time joins should use

### 5. Tall/Wide Inconsistency At The Wrong Layer

The original user concern is valid.

What it is:

- some canonical storage tables are wide and pivoted in advance
- others remain tall
- preprocess must support both and unify them back into analysis-facing features

Where it appears:

- wide holdings buckets and some fixed-schema endpoint tables coexist with long metric-style tables and long categorical weights

Why it exists:

- wide format is convenient for final analysis panels
- some endpoints were small enough to pivot early
- open-ended endpoints were too large or unstable to pivot wide canonically

Why it matters:

- preprocess contains parallel logic for wide and long inputs
- storage decisions leak directly into feature engineering
- canonical storage mixes "facts as captured" with "analysis-shaped convenience pivots"

How a new agent will notice this:

- snapshot preprocess has both wide-prefix logic and long-pivot logic
- adding one new source feature requires asking whether the storage table is wide or tall before deciding how to consume it

Correct framing:

The main issue is not that wide tables exist at all. The issue is that wide tables are being used inside the canonical ingest/store layer for problems that belong in preprocess.

Preferred rule:

- store canonical facts in the most stable representation
- keep open-ended or high-cardinality feature sets tall
- pivot only in preprocess or semantic feature assembly
- only keep wide canonical tables for truly fixed, closed-world records where one row per `(conid, effective_at)` is semantically stable

### 6. Analysis Is A Monolith

What it is:

- `pystocks/analysis/__init__.py` owns too many unrelated responsibilities

Where it appears:

- input preparation
- price feature engineering
- rebalance logic
- panel construction
- macro joins
- composite feature creation
- factor construction
- clustering
- model fitting
- persistence
- current-beta computation

Why it exists:

- analysis became the place where upstream ambiguity was repaired and downstream outputs were assembled

Why it matters:

- testing isolated behaviors is harder than necessary
- feature logic and output logic are mixed
- architectural cleanup elsewhere is harder to capitalize on

How a new agent will notice this:

- the file is large and contains stage-adjacent responsibilities that should belong elsewhere
- type checks are broadly suppressed at the top of the file

Correct boundary:

- analysis should be decomposed after upstream contracts are cleaned up enough that the split does not just spread ambiguity across more files

### 7. Operational And Research Persistence Are Mixed

What it is:

- operational source tables and replaceable research outputs share the same SQLite persistence boundary

Where it appears:

- analysis writes `analysis_*` tables back into the main DB while also writing parquet outputs

Why it exists:

- convenient local persistence was added without fully separating ownership

Why it matters:

- operational truth and derived research data become harder to distinguish
- schema ownership becomes ambiguous
- migrations and table hygiene become harder
- it becomes less obvious which tables are canonical inputs versus disposable outputs

How a new agent will notice this:

- operational tables and analysis tables live together in one DB
- many analysis outputs are persisted with replace semantics

Correct boundary:

- operational and derived stores should be separated
- if full separation is deferred, the namespace and ownership rules must still be explicit

### 8. DB Integrity Depends Too Much On Python Choreography

What it is:

- several write paths assume key-like behavior that is not fully expressed by primary keys or uniqueness constraints

Where it appears:

- delete-and-reinsert logic
- parent/child replace flows
- tables without PK or uniqueness guarantees even though the write path assumes them

Why it exists:

- the write pipeline evolved procedurally before all table semantics were encoded declaratively

Why it matters:

- correctness depends on every caller following the same sequence
- duplicate or stale rows are easier to introduce during refactors
- failures are harder to localize

How a new agent will notice this:

- tables exist without PK or uniqueness clauses where replace-like behavior is assumed
- Python code deletes rows first to emulate integrity rules the schema could own

Correct boundary:

- integrity that belongs to the schema should live in the schema whenever the semantics are known

## Root Causes vs. Symptoms

### Root Causes

1. No canonical cross-stage data contract.
2. No rigorous point-in-time timestamp semantics.
3. Stage boundaries are not enforced by orchestration or dependency direction.
4. Canonical persistence and analytical shaping are mixed.

### Symptoms

1. Mixed tall/wide storage shapes.
2. Prefix and pivot proliferation in snapshot preprocess.
3. String-typed feature lists in analysis.
4. Hidden preprocessing inside analysis.
5. Research tables in the operational DB.
6. Large monolithic modules.
7. Procedural dedupe and replace logic in write paths.

## Cross-Module Failure Patterns

These are the repeated patterns that create drift across the repo.

### 1. Contract Re-Declaration

Bad pattern:

- storage defines one feature shape
- preprocess renames it ad hoc
- analysis hardcodes the renamed columns again

Desired pattern:

- one stage owns the contract
- downstream stages consume it rather than redefining it

### 2. Hidden Stage Work

Bad pattern:

- a stage looks pure at the CLI or import boundary but performs another stage's work internally

Desired pattern:

- the real computational DAG is visible in orchestration and module ownership

### 3. Storage-Shape Leakage Into Features

Bad pattern:

- analysis or preprocess behavior is implicitly controlled by current storage table shape

Desired pattern:

- storage exposes stable source inputs
- preprocess owns the feature contract explicitly

### 4. Point-in-Time Conflation

Bad pattern:

- one convenient date is reused as the semantic anchor for unrelated endpoint data

Desired pattern:

- `observed_at`, source `as_of_date`, storage `effective_at`, and analysis join dates are treated as distinct until a documented contract says otherwise

### 5. Derived-Output Ownership Drift

Bad pattern:

- operational and research tables accumulate in one store without strong ownership boundaries

Desired pattern:

- canonical inputs and disposable outputs have explicit ownership and lifecycle

### 6. Python-Enforced Integrity

Bad pattern:

- correctness depends on delete/reinsert choreography in code

Desired pattern:

- keys, uniqueness, and replacement semantics live in the schema when the business meaning is known

## Target Architecture

### Ingest

Inputs:

- products to scrape
- authenticated session state
- request-planning heuristics

Responsibilities:

- authenticate
- fetch raw payloads
- perform narrow request-planning and scrape-control heuristics
- hand payloads to storage

Outputs:

- raw source payloads
- scrape status and telemetry

Forbidden knowledge:

- analysis feature naming
- analytical pivots
- preprocess-specific feature contracts

Rules:

- ingest does not import preprocess
- ingest does not perform analytical shaping
- ingest should be able to persist raw payloads even if downstream feature logic changes

### Storage

Inputs:

- raw payloads and normalized persistence-ready rows

Responsibilities:

- own schema bootstrap
- persist raw payload blobs
- persist canonical normalized endpoint tables
- preserve explicit time semantics
- expose narrow read models that are stable for downstream consumers

Outputs:

- canonical stored facts
- stable reader contracts

Forbidden knowledge:

- final analysis panel shape
- research-specific feature composites
- orchestration shortcuts that hide stage boundaries

Rules:

- storage stores canonical facts, not analysis-ready panels
- storage readers must not silently widen feature scope through `SELECT *`
- storage keys and constraints should express integrity directly where possible

### Preprocess

Inputs:

- canonical stored data exposed through storage readers

Responsibilities:

- own pivoting
- own feature naming
- own diagnostics
- own panel input assembly from canonical tables
- produce stable, documented analysis-facing artifacts

Outputs:

- explicit feature contracts
- diagnostics and summaries

Forbidden knowledge:

- raw scrape-control logic
- downstream research persistence decisions
- ad hoc analysis-specific repair logic for upstream ambiguity

Rules:

- preprocess is the semantic shaping layer
- preprocess consumes storage contracts and emits explicit feature contracts
- preprocess must not depend on analysis internals

### Analysis

Inputs:

- stable preprocessed inputs

Responsibilities:

- consume stable preprocessed inputs
- construct panels, factors, and research outputs

Outputs:

- derived research artifacts
- diagnostics
- model outputs

Forbidden knowledge:

- raw endpoint storage details when a preprocess contract exists or should exist
- hidden preprocessing of raw storage tables

Rules:

- analysis must not perform hidden raw-storage preprocessing
- analysis should not be the place where storage ambiguity is repaired
- analysis feature assumptions should come from preprocess artifacts or metadata, not ad hoc string lists scattered across the file

### Persistence Of Derived Outputs

Preferred responsibilities:

- parquet outputs remain acceptable
- SQL persistence for research artifacts should be separate from operational source-of-truth storage

Rules:

- do not couple operational ingestion tables and replaceable research tables more tightly than necessary

## Architectural Invariants

These invariants should guide all future changes, even before the full refactor is complete.

1. Stage direction is strict:
   - `ingest -> storage`
   - `preprocess -> storage`
   - `analysis -> preprocess`
   - never `ingest -> preprocess`
   - never `analysis -> raw storage tables` when a preprocess contract exists or should exist

2. Storage owns canonical persistence, not analysis shape.

3. Preprocess owns pivoting, feature naming, diagnostics, and explicit analysis-facing feature contracts.

4. Analysis consumes stable preprocessed inputs only.

5. `observed_at`, endpoint-specific `as_of_date`, storage `effective_at`, and downstream join dates are distinct concepts.

6. Variable-cardinality endpoint data stays tall in canonical storage unless there is a strong, documented reason otherwise.

7. Wide canonical tables are allowed only for closed-world records with stable columns and stable row identity.

8. New features must be introduced through an explicit preprocess contract, not by reaching directly into storage columns from analysis.

9. Reader APIs should expose consumer-oriented contracts, not whichever raw columns happen to exist today.

10. Derived research outputs should not define the operational schema.

## Decision Rules For Common Changes

### Adding A New Raw Endpoint

Belongs to:

- ingest for fetch planning and payload capture
- storage for canonical persistence

Must not be solved by:

- adding direct analysis logic for raw payload interpretation

### Adding A New Stored Field

Belongs to:

- storage, if the field is part of canonical persistence semantics

Must also consider:

- whether the field needs explicit time semantics
- whether downstream consumers need a reader contract update

### Adding A New Analysis Feature

Belongs to:

- preprocess, if the feature is analysis-facing

Must not be solved by:

- directly hardcoding a new storage column into analysis

### Changing A Point-In-Time Join

Belongs to:

- preprocess or analysis, depending on where the join contract lives

Must also do:

- make the time semantics explicit
- add or update behavior-focused tests

Must not do:

- silently reuse `ratios.as_of_date` as a default anchor for unrelated data

### Adding A New Derived Artifact

Belongs to:

- analysis for compute ownership
- output persistence with explicit lifecycle

Must also decide:

- whether it is parquet-only
- separate research-store SQL
- or temporary namespaced SQL persistence with documented ownership

### Adding A New Reader

Belongs to:

- storage

Must do:

- expose a stable consumer-oriented contract
- choose explicit columns deliberately

Must not do:

- expose raw table churn by default

## Safe vs. Unsafe Change Examples

### Feature Additions

Safe:

- add canonical source fields in storage
- expose them through a narrow reader
- add one preprocess-owned feature mapping
- let analysis consume the documented feature contract

Unsafe:

- add a new storage column
- immediately reference it in analysis by string name
- skip preprocess contract ownership

### Time Semantics

Safe:

- preserve both `observed_at` and endpoint `as_of_date`
- document which one drives downstream joins

Unsafe:

- anchor a new endpoint's `effective_at` to `ratios.as_of_date` just because it is already available

### Storage Shape

Safe:

- keep variable-cardinality or open-ended data tall in canonical storage
- pivot it later in preprocess if analysis needs wide columns

Unsafe:

- pivot canonical storage wide merely because the final panel is wide

### Stage Boundaries

Safe:

- split fetch from preprocess in supplementary flows
- make orchestration show the real stage order

Unsafe:

- let ingest call preprocess for convenience
- let analysis quietly absorb more preprocessing work

## Refactor Strategy

Refactor for correctness and contract clarity first. Do not start by splitting files for aesthetic reasons alone.

Recommended order:

1. lock in invariants and stop making the architecture worse
2. define the missing contracts
3. align orchestration with the real DAG
4. move shaping responsibility into preprocess
5. split analysis once inputs are cleaner
6. tighten persistence integrity and output separation

## Phased Roadmap

### Phase 0: Freeze Further Drift

Goal:

- stop introducing new cross-stage coupling before larger refactors begin

Tasks:

- document the architectural invariants in `AGENTS.md`
- require new feature additions to go through explicit preprocess contracts
- stop adding new hidden preprocessing inside analysis
- stop adding new ingest imports from preprocess

Why this comes first:

- later refactors are much harder if the architecture keeps drifting during cleanup

Acceptance criteria:

- future PRs can be reviewed against clear architectural rules
- no new code deepens the current contract drift

### Phase 1: Define The Canonical Time And Feature Contracts

Goal:

- establish the missing source of truth before moving logic around

Tasks:

- define the canonical timestamp semantics for each endpoint family
- document the meanings of:
  - `observed_at`
  - endpoint `as_of_date`
  - storage `effective_at`
  - analysis join and rebalance dates
- define one explicit snapshot-feature contract owned by preprocess
- decide which endpoint tables are canonical facts versus convenience pivots

Deliverables:

- a written contract doc or metadata structure for snapshot features
- explicit endpoint time-model documentation

Why this comes before stage cleanup:

- without explicit contracts, moving logic only relocates ambiguity

Acceptance criteria:

- a new agent can answer "where does this feature come from?" from one contract
- a new agent can answer "what date does this row mean?" without guessing

### Phase 2: Make Stage Boundaries Honest

Goal:

- align dependency direction and CLI orchestration with the real DAG

Tasks:

- remove preprocess imports from `ingest/supplementary.py`
- split supplementary fetch from supplementary preprocess if needed
- make `run_pipeline` reflect the true stage order
- either:
  - have the pipeline call preprocess stages explicitly, or
  - define analysis as consuming preprocessed artifacts only and fail loudly when they are missing

Why this comes after contract definition:

- the real DAG cannot be made honest until the contract handoff points are explicit

Acceptance criteria:

- no ingest module imports preprocess
- orchestration matches the real dependency graph
- a user can understand which commands produce which required artifacts

### Phase 3: Re-Center Canonical Storage

Goal:

- make storage store stable facts rather than partial analysis shapes

Tasks:

- review each endpoint table and classify it as:
  - canonical tall fact table
  - canonical fixed-schema row table
  - derived analytical artifact that should move out of storage
- migrate open-ended feature sets toward tall canonical storage where appropriate
- preserve raw blobs as the immutable source capture
- add missing keys and uniqueness constraints where semantics are known

Why this comes after boundary cleanup:

- storage shape decisions should be made against explicit contracts, not while boundaries are still blurred

Acceptance criteria:

- canonical storage shape is chosen by data semantics, not convenience for one consumer
- integrity rules move from Python choreography into schema where feasible

### Phase 4: Make Preprocess The Semantic Shaping Layer

Goal:

- centralize pivoting, feature naming, and diagnostics in preprocess

Tasks:

- replace implicit storage-to-analysis coupling with explicit preprocess outputs
- make snapshot preprocess define the analysis-facing feature registry
- keep diagnostics in preprocess, not analysis
- ensure reader APIs supply only the canonical source inputs preprocess needs

Why this comes after storage re-centering:

- preprocess should consume stable canonical inputs, not partially cleaned storage compromises

Acceptance criteria:

- analysis no longer needs raw storage schema knowledge for snapshot features
- feature-name churn is localized to preprocess

### Phase 5: Split The Analysis Monolith

Goal:

- decompose analysis after upstream inputs become stable

Suggested shape:

- `analysis/inputs.py` or equivalent:
  input loading and validation
- `analysis/panel.py`:
  rebalance-date logic and panel assembly
- `analysis/features.py`:
  price-derived and composite feature construction
- `analysis/research.py`:
  factor construction, clustering, model fitting
- `analysis/outputs.py`:
  persistence and result assembly

Why this comes after preprocess cleanup:

- otherwise the split only spreads the same implicit contracts into more files

Acceptance criteria:

- analysis modules have narrower responsibilities
- unit tests can target panel-building, factor-building, and research logic independently

### Phase 6: Separate Operational And Research Persistence

Goal:

- make output ownership explicit and reduce operational-schema ambiguity

Tasks:

- choose one of:
  - separate SQLite DB for research outputs
  - separate schema or strongly namespaced store policy
  - parquet-only outputs for derived research where SQL persistence is not required
- document ownership and lifecycle of every analysis output

Why this comes after analysis decomposition:

- output ownership is easier to cleanly separate once analysis responsibilities and artifact boundaries are explicit

Acceptance criteria:

- a new agent can tell which tables are canonical inputs and which are disposable research outputs

## Practical Priorities

If the refactor must be done incrementally, prioritize in this order:

1. time-model clarification
2. strict stage direction
3. explicit preprocess feature contract
4. orchestration cleanup
5. canonical storage cleanup
6. analysis split
7. research-output separation

## What Not To Do

Do not:

- split large files before defining the contracts they currently blur
- add new analysis features by directly hardcoding more storage column names into analysis
- anchor new endpoint time semantics to `ratios.as_of_date` by default
- add more wide canonical tables just because the final analysis panel is wide
- let ingest call preprocess for convenience
- expose raw table churn directly to analysis
- rely on Python delete/reinsert choreography when the DB can express the integrity rule

## Migration Safety Guidelines

For each architectural phase:

- keep behavior-focused tests around point-in-time joins, schema shape, and pipeline wiring
- preserve raw payload blobs and canonical backfillability
- prefer additive migration steps over large rewrites
- validate old and new outputs side by side where practical
- keep current CLI behavior working until replacement paths are explicit and tested

## Recommended Near-Term Deliverables

Before major code movement, the next high-leverage deliverables are:

1. a documented point-in-time timestamp contract
2. an explicit snapshot feature contract owned by preprocess
3. removal of ingest-to-preprocess supplementary coupling
4. orchestration changes so `run_pipeline` reflects the real stage DAG
5. a plan for which current wide tables remain wide and which should become tall canonical facts

## Architectural Review Checklist

Before making or approving an architecture-affecting change, answer:

1. Which stage owns this concern?
2. Am I introducing a reverse dependency?
3. Am I re-declaring a contract that already exists elsewhere?
4. Are the relevant timestamps explicit and still distinct?
5. Am I solving an analysis-shape problem inside canonical storage?
6. Am I exposing raw storage churn to preprocess or analysis?
7. Am I mixing canonical inputs and disposable outputs more tightly?

If any answer is unclear, stop and resolve it before editing code.

## Quick Start For New Agents

Before making architectural changes:

1. read this roadmap
2. identify whether your change belongs to ingest, storage, preprocess, or analysis
3. verify you are not creating a reverse dependency
4. verify you are not introducing a new implicit feature contract
5. verify your timestamp semantics are explicit
6. verify you are not solving an analysis-shape problem inside canonical storage

If a proposed change violates one of those checks, redesign it before editing code.
