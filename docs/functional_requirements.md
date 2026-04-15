# Pystocks Functional Requirements

## Purpose

This document defines what `pystocks` must do at the system level.

It is the functional baseline for the rebuild. It is not a package layout, implementation plan, or module map.

It should answer:

- what capabilities the system must provide
- what data and time semantics the system must preserve
- what outputs are required for research and portfolio construction
- which remaining scope decisions must be made explicitly

## System Mission

`pystocks` is a research system for ETF cross-sectional factor analysis and portfolio construction.

At a high level, it must:

1. define and maintain an ETF universe
2. collect ETF-related data from IBKR and supplementary external sources
3. preserve both raw source captures and canonical historical facts
4. vet data quality and downstream usability
5. construct stable analysis inputs and point-in-time panels
6. estimate factor returns, exposures, and related diagnostics
7. support portfolio construction from those modeled inputs

## Functional Scope

The full system is responsible for seven functional domains:

1. universe management
2. data collection
3. canonical persistence and temporal history
4. data vetting and operability
5. analysis input construction
6. research and exposure estimation
7. portfolio construction

## 1. Universe Management Requirements

The system must maintain an explicit investable universe of ETFs and ETF-like products.

Required capabilities:

- bootstrap and refresh the product universe from one or more upstream sources
- identify instruments by a stable canonical key such as `conid`
- preserve core product metadata needed for later joins and operational use
- support fixed-universe runs from explicit target lists
- support incremental refreshes over time

The system should also support explicit universe-governance rules, such as:

- exclusion lists
- currency or geography filters
- product-type or structural filters
- tradability or verification rules when those are part of the research process

The universe must be treated as a first-class dataset, not an incidental byproduct of scraping.

## 2. Data Collection Requirements

The system must collect data from heterogeneous source families and support repeated collection cycles.

Supported source families include:

- IBKR product metadata
- IBKR snapshot endpoints
- IBKR historical or semi-historical endpoint series
- supplementary external datasets such as macro and benchmark inputs

The collection layer must support:

- repeated observation of the same endpoint over time
- partial or targeted runs over a selected subset of instruments
- retry and resume behavior for interrupted runs
- authentication lifecycle management where required by the source
- collection telemetry and run diagnostics
- explicit handling of empty, skipped, malformed, and failed fetch outcomes

The system must support at least these source shapes:

- discrete snapshots
- historical series
- nested payloads that contain both snapshot and series data

## 3. Canonical Persistence And Temporal History Requirements

The system must preserve both raw source fidelity and canonical downstream usability.

### 3.1 Raw Capture

The system must store immutable raw payloads or equivalent raw captures for collected source data so that:

- parser bugs can be fixed later
- canonical facts can be rebuilt
- source changes can be audited

### 3.2 Canonical Facts

The system must persist normalized canonical facts for downstream use.

Canonical persistence must support:

- endpoint-specific storage shapes when source semantics differ
- typed scalar values when source meaning is clear
- series history where the source provides dated observations
- stable reader contracts for downstream consumers

Canonical storage should preserve facts, not analysis panel shape.

### 3.3 Temporal History

The system must preserve enough temporal information to answer:

- when was this data observed?
- what date did the source say it described?
- what canonical date should downstream joins use?
- how many historical observations exist for this endpoint and instrument?

The system must preserve at least these distinct temporal concepts:

- `observed_at`
- source `as_of_date` or equivalent source date
- canonical storage `effective_at`
- downstream analysis join or rebalance date

These concepts must not be collapsed casually.

In particular, the rebuilt system must not rely on one endpoint's date as a universal anchor for unrelated endpoint data unless that rule is explicitly justified by source semantics.

### 3.4 Incremental Series Maintenance

The system must support incremental extension of time series.

Required behavior:

- append or upsert new points without unnecessary duplication
- tolerate overlapping fetch windows
- preserve source event or trade dates
- detect or surface mismatches when refetched points disagree

## 4. Data Vetting And Operability Requirements

The system must determine whether collected data are usable downstream.

Required capabilities:

- detect malformed or suspicious source data
- evaluate whether a dataset is sufficiently complete and trustworthy for its intended downstream use
- distinguish acceptable gaps from blocking data failures
- record diagnostics explaining accepted, repaired, skipped, and rejected data

This must apply across multiple data families, including:

- price series
- dividend events
- snapshot histories
- supplementary macro or benchmark inputs

Diagnostics are first-class outputs, not temporary debugging artifacts.

## 5. Analysis Input Construction Requirements

The system must transform canonical stored data into stable analysis inputs.

This domain is broader than generic preprocessing. It is responsible for producing durable downstream contracts that analysis can rely on without reaching back into raw storage shape.

### 5.1 Price Inputs

The system must provide cleaned price and return inputs for downstream research.

Required outputs include:

- cleaned price history
- raw and cleaned return paths
- per-instrument eligibility or coverage summaries
- price-quality diagnostics

### 5.2 Dividend Inputs

The system must provide dividend-event inputs suitable for total-return analysis.

Required capabilities:

- preserve dividend event history
- vet dividend events for downstream use
- evaluate dividend usability against price-reference context
- produce row-level and summary diagnostics

### 5.3 Snapshot Feature Inputs

The system must convert repeated endpoint snapshots into stable analysis-facing feature inputs.

Required capabilities:

- define stable feature naming
- reshape canonical endpoint facts into analysis-facing structures when needed for downstream research, including wide feature tables or other consumer-oriented contracts
- preserve the point-in-time meaning of those features
- provide diagnostics about feature integrity and coverage

This is one of the core cross-stage contracts in the system.

### 5.4 Supplementary Inputs

The system must support supplementary non-ETF-specific analysis inputs, such as:

- risk-free rate series
- country or macroeconomic features
- explicit mappings that connect ETF-level exposures, such as country weights, to supplementary country- or macro-level datasets

The system must preserve the distinction between raw supplementary inputs and derived supplementary analysis inputs.

## 6. Panel Construction Requirements

The system must assemble point-in-time analysis panels from stable input contracts.

Required behavior:

- define rebalance or other analysis join dates
- use only information available at or before each join date
- join the latest eligible feature values available at each join date
- carry forward sufficient metadata to audit feature age, eligibility state, and join behavior

The panel-construction stage must preserve point-in-time correctness as a core requirement.

## 7. Research And Exposure Estimation Requirements

The system must support research over the point-in-time panel.

Required capabilities:

- construct and evaluate candidate factors
- estimate factor return series
- screen, cluster, or reduce redundant factors
- produce diagnostics explaining factor selection and rejection
- estimate instrument exposures to accepted or persistent factors
- produce expected-return estimates and related research outputs needed for downstream portfolio construction or model evaluation

Where research uses walk-forward evaluation, the system must preserve explicit training and test window semantics.

Research outputs are derived artifacts, not canonical source-of-truth facts.

## 8. Portfolio Construction Requirements

The system must support portfolio construction from modeled research outputs.

The rebuilt system must include portfolio optimization and efficient-frontier workflows.

Required capabilities include:

- expected-return inputs for optimization
- covariance or risk inputs
- factor-aware exposure constraints
- efficient-frontier or optimizer workflows
- discrete or constrained portfolio selection workflows

If multiple portfolio-construction workflows are supported, the system must make their input assumptions and constraint semantics explicit.

## 9. Required System Outputs

The system must be able to produce, persist, or expose the following output families:

- universe and targeting datasets
- canonical stored endpoint histories
- diagnostics and operability artifacts
- cleaned analysis-input artifacts
- point-in-time analysis panels
- factor research outputs
- current exposure estimates
- portfolio-construction inputs and portfolio solutions

## 10. Scope Candidates From Legacy `/src`

The rebuild must carry forward the following legacy capability:

- portfolio optimization and efficient-frontier workflows

Other legacy areas should be treated as out of scope unless they are later reintroduced deliberately through a separate scope decision.

## 11. Functional Invariants

Regardless of the future package layout, the rebuilt system must preserve these invariants:

- one canonical instrument identity
- explicit raw-versus-canonical separation
- explicit distinction between source dates, storage dates, and analysis join dates
- explicit downstream input contracts
- explicit diagnostics around data trustworthiness
- clear separation between canonical operational data and derived research outputs
