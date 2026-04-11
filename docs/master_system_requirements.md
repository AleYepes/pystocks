# Pystocks Master System Requirements

## Purpose

This document defines what `pystocks` is actually supposed to do at the system level.

It is intended to become the requirements baseline for the next architecture, not a description of the current package layout.

The goal is to answer:

- what capabilities the project must provide
- what data semantics the system must preserve
- which outputs are required for downstream research and portfolio construction
- which non-functional constraints matter enough to shape the architecture

This document is deliberately broader than the current implementation.

## System Mission

`pystocks` is a research pipeline for ETF cross-sectional factor analysis and portfolio construction.

At a high level, it must:

1. build and maintain an ETF universe
2. fetch ETF-related data from IBKR and supplementary external sources
3. persist raw and canonical versions of that data across repeated fetch cycles
4. vet data quality and operability
5. preprocess canonical data into stable analysis inputs
6. estimate factor returns, factor exposures, and related diagnostics
7. construct efficient portfolios from those modeled inputs

## End-To-End Functional Scope

The full system is responsible for six major domains:

1. universe management
2. acquisition
3. persistence and temporal history
4. quality control and operability vetting
5. feature and panel construction
6. modeling and portfolio construction

## 1. Universe Management Requirements

The system must maintain an explicit investable universe of ETFs and ETF-like products.

Required capabilities:

- scrape or ingest the available product universe
- identify products by a stable canonical key such as `conid`
- preserve product metadata needed for later joins:
  - symbol
  - exchange
  - currency
  - ISIN when available
  - descriptive name
- support fixed-universe runs from a supplied conid list
- support incremental refreshes of the product universe over time

The system should also support universe filters used operationally or analytically:

- geography or listing-region filters
- currency filters
- asset-type filters
- explicit exclusion lists

Architecture implication:

- universe definition is a first-class dataset, not an incidental side effect of scraping

## 2. Acquisition Requirements

The system must fetch data from heterogeneous sources and endpoint shapes.

### 2.1 Source Types

Supported source families include:

- IBKR product metadata
- IBKR endpoint snapshots
- IBKR historical or semi-historical series
- supplementary macro and benchmark datasets from external providers

### 2.2 Endpoint Shape Types

The system must support at least these source shapes:

- discrete snapshots
  - one payload observed at one time, possibly containing an `as_of_date`
- series payloads
  - a set of observations across dates, possibly extended over time
- nested snapshot-plus-series payloads
  - for example dividend payloads with both summary metrics and event history

### 2.3 Fetch-Cycle Requirements

The system must support recurring fetch cycles without losing history.

Required behavior:

- repeated fetching of the same endpoint for the same product
- incremental extension of historical series
- repeated observation of snapshots whose content changes over time
- preservation of fetch timestamps and source dates
- ability to resume or retry interrupted runs

Architecture implication:

- acquisition must be designed around repeated observation, not single-shot scraping

## 3. Persistence And Temporal History Requirements

This is one of the most important domains.

The system must persist data in a way that preserves both:

- raw source fidelity
- canonical downstream usability

### 3.1 Raw Capture

The system must store immutable raw payloads or equivalent raw source captures so that:

- parser bugs can be fixed later
- canonical tables can be rebuilt
- source changes can be audited

### 3.2 Canonical Persistence

The system must also persist normalized canonical facts for downstream use.

Canonical persistence must support:

- typed scalar values when source meaning is clear
- endpoint-specific child tables for nested structures
- separation of source capture from downstream convenience shape

### 3.3 Repeated Snapshot History

The system must treat snapshots as data that evolve over time.

Requirements:

- preserve repeated observations of endpoint snapshots over different fetch cycles
- record the source-specific date the snapshot describes when available
- define a canonical storage join key for downstream point-in-time use
- avoid pretending all endpoints update on the same schedule

This is the core problem behind "joining snapshots into time series."

The system must be able to answer:

- when was this data observed?
- what date did the source say it described?
- what canonical date should downstream joins use?
- how many distinct historical observations exist for this endpoint and product?

### 3.4 Series Extension

The system must support incremental extension of time series.

Requirements:

- append new points without duplicating old points
- preserve source trade dates or event dates
- tolerate overlapping fetch windows
- detect mismatches or duplicate points when the same date is refetched with different values

### 3.5 Time Semantics

The system must keep at least four distinct temporal concepts:

- `observed_at`
- source `as_of_date`
- canonical storage `effective_at`
- downstream analysis join or rebalance date

Architecture implication:

- temporal semantics are not a documentation detail; they are a system requirement

## 4. Data Vetting And Operability Requirements

The project is not only a storage pipeline. It must decide which data are usable downstream.

### 4.1 Source Validation

The system must detect and surface malformed or suspicious source data.

Examples:

- impossible prices
- stale series stretches
- duplicate metric rows
- holdings weights far from expected totals
- inconsistent dates inside a payload
- currency inconsistencies
- missing required fields

### 4.2 Coverage And Eligibility

The system must evaluate whether a dataset is fit for downstream use.

Examples:

- whether a price series has sufficient coverage for return modeling
- whether dividend events are reliable enough for total-return adjustment
- whether a snapshot history is dense enough for panel construction
- whether supplementary macro coverage is sufficient for the requested universe

### 4.3 Diagnostics

The system must produce diagnostics that explain why data are accepted, repaired, skipped, or rejected.

Architecture implication:

- diagnostics are first-class outputs, not temporary debugging prints

## 5. Preprocessing And Feature Construction Requirements

The system must derive stable downstream analysis inputs from canonical storage.

This stage is broader than "cleaning."

It includes:

- converting canonical endpoint facts into usable feature datasets
- building time-aligned panels
- deriving analysis-facing features and composites
- creating eligibility and quality artifacts

### 5.1 Price Pipeline

The system must:

- load canonical price history
- validate price integrity
- detect stale periods and anomalies
- generate price returns
- optionally incorporate dividend information into total-return variants
- keep raw price-return and total-return paths distinct

Required outputs likely include:

- cleaned price history
- clean return series
- eligibility summaries
- outlier and quality diagnostics

### 5.2 Dividend Pipeline

The system must:

- persist and expose dividend event history
- assess dividend quality and coverage
- support downstream total-return adjustment when quality permits
- preserve dividend currency and event dates

### 5.3 Snapshot-To-Panel Pipeline

The system must turn repeated endpoint snapshots into analysis-ready time-varying features.

This includes:

- defining stable feature names
- pivoting or reshaping endpoint facts into analysis-facing columns
- carrying features forward appropriately across time
- respecting endpoint-specific update cadences
- joining multiple endpoint families into one point-in-time feature panel

This is likely the central preprocessing stage for the project.

### 5.4 Supplementary Data Pipeline

The system must support supplementary datasets that are not ETF-specific but are needed for analysis.

Examples:

- risk-free series
- macro features
- country-level indicators

These must be:

- fetched independently
- normalized canonically
- transformed into analysis-facing datasets
- joined to ETF features through explicit geographic or economic mappings

### 5.5 Factor-Candidate Feature Construction

The system must derive the final candidate explanatory variables used in regression and factor research.

This may include:

- direct source features
- grouped or themed features
- momentum or trend features
- portfolio-composition aggregates
- macro exposures
- region, bloc, or category mappings

Architecture implication:

- feature naming, grouping, and candidate construction are core system requirements, not ad hoc analysis details

## 6. Panel Assembly Requirements

The system must be able to assemble a point-in-time analysis panel.

Required behavior:

- define rebalance dates
- join latest eligible feature values available at each rebalance date
- join forward return windows or other targets
- preserve enough metadata to audit how each panel row was constructed

The panel must support:

- cross-sectional factor research
- regression fitting
- current-beta computation
- portfolio construction inputs

## 7. Modeling Requirements

The project is not complete at panel generation. It must run factor and exposure modeling.

### 7.1 Factor Return Construction

The system must support building factor return series from the analysis panel.

This likely includes:

- benchmark-style factors
- long-short factor portfolios
- sleeve-specific factor sets
- candidate screening and reduction

### 7.2 Regression And Exposure Estimation

The system must support estimating ETF exposures to factor sets.

Required modeling families include at least:

- cross-sectional or time-series regression workflows
- elastic net or related regularized regressions
- possibly OLS-based diagnostics or benchmark comparisons

Required outputs may include:

- betas
- intercepts or alpha-like terms
- goodness-of-fit diagnostics
- stability diagnostics
- selected-factor metadata

### 7.3 Walk-Forward / Out-Of-Sample Evaluation

The system should support walk-forward or rolling validation of factor models.

This is important because the repo is not just describing data; it is trying to produce investable model outputs.

## 8. Portfolio Construction Requirements

The system must use modeled inputs to construct portfolios.

Required inputs include some combination of:

- expected returns
- factor exposures
- factor covariances
- asset covariances
- eligibility constraints
- universe membership

Required outputs may include:

- efficient portfolios
- portfolio weights
- frontier points
- sleeve-level or global allocations
- portfolio diagnostics and turnover metrics

Architecture implication:

- portfolio optimization is a first-class domain, not just a notebook-side afterthought

## 9. Output Requirements

The system must persist and expose derived outputs clearly.

Output families include:

- quality diagnostics
- preprocessed datasets
- analysis panels
- factor return tables
- factor exposure tables
- model telemetry
- portfolio weights and optimization results

Derived outputs must be distinguishable from canonical operational data.

## 10. Operational Requirements

The system must support real recurring use, not just one-off research scripts.

Required capabilities:

- repeatable CLI workflows
- fixed-universe runs
- partial reruns by stage
- refresh of supplementary data independently of ETF endpoint refresh
- rebuildability from canonical storage
- failure visibility

## 11. Reproducibility Requirements

The system must make research outputs reproducible enough for comparison over time.

This includes:

- stable input datasets for a given run configuration
- deterministic feature-building where practical
- preserved run metadata
- ability to compare old and new outputs after refactors

## 12. Performance Requirements

The architecture must stay practical for a non-trivial ETF universe.

The system should be designed to avoid:

- repeated reparsing of the same canonical data
- repeated full-history pivots when narrow slices suffice
- unnecessary round-trips through file caches for required in-memory work
- duplicated wide intermediate tables that are expensive to maintain

The system should support:

- incremental series extension
- incremental snapshot refresh
- partial feature builds by date or universe slice
- efficient joins for point-in-time panel construction

## 13. Testing Requirements

The system needs tests that correspond to the real domains above.

Minimum required test categories:

- storage schema and idempotency
- endpoint-specific time semantics
- raw-to-canonical normalization behavior
- series extension and deduplication
- snapshot-to-panel carry-forward behavior
- data quality and eligibility decisions
- factor-construction correctness
- regression and output-shape tests
- portfolio-construction sanity tests
- pipeline wiring tests

## 14. Explicit Architecture Drivers

These requirements imply several architecture drivers.

### Driver 1: Repeated Observation Over Time

The system is fundamentally historical and revision-aware.

### Driver 2: Different Endpoint Cadences

The system cannot assume synchronized updates across data sources.

### Driver 3: Quality-Gated Downstream Use

Not every stored dataset is automatically fit for modeling.

### Driver 4: Stable Analysis Contracts

Modeling code needs stable feature datasets, not direct knowledge of raw storage shapes.

### Driver 5: Separation Of Canonical And Derived Data

The system must distinguish source-of-truth persistence from replaceable research artifacts.

### Driver 6: Lean But Scalable Dataflow

The system should stay explicit and small while still supporting incremental history and larger universes.

## 15. Recommended Top-Level Domain Model

If rebuilt cleanly, the project likely wants these durable domains:

1. `universe`
2. `acquisition`
3. `canonical_storage`
4. `quality`
5. `features`
6. `panels`
7. `models`
8. `portfolio`
9. `outputs`

This is a domain model, not necessarily a package layout.

## 16. Questions To Resolve Before Final Architecture

These requirements are already enough to shape the architecture, but several design choices still need explicit decisions:

1. What is the canonical history model for snapshots:
   - every observed version
   - only latest per source date
   - both
2. Which endpoint tables should remain wide in canonical storage, if any?
3. Which derived datasets must be persisted, versus computed on demand?
4. How much of model telemetry needs durable storage?
5. Is portfolio optimization a runtime stage in the main CLI or a separate research command family?
6. What scale target should the new system optimize for:
   - hundreds of ETFs
   - low thousands
   - tens of thousands

## 17. Immediate Next Use Of This Document

This document should now be used to produce:

1. a target architecture proposal
2. a data model proposal for snapshot history and series history
3. a lean package layout for the rebuild
4. a migration plan from the current implementation
