# Pystocks Proposed Concern Map

This document proposes a future concern map based on the current `/pystocks` inventory.

It is not a package layout yet.

Its purpose is to answer:

- what the durable concern boundaries appear to be
- which current modules are overly broad
- which capabilities should likely stay grouped
- where current ownership is mixed or misleading

## Design Goal

The future concern map should optimize for:

- one concern, one owner
- explicit cross-stage contracts
- point-in-time correctness
- low duplication of feature vocabulary
- practical runtime performance

## Proposed Concern Families

### 1. Universe

Responsibilities:

- bootstrap and refresh the investable instrument universe
- preserve canonical instrument identity and core metadata
- manage explicit target lists and universe selection inputs

Should likely own:

- canonical instrument master data
- universe refresh logic
- fixed-universe targeting inputs

Should likely not own:

- downstream fundamentals scrape status policy
- research eligibility decisions

### 2. Collection

Responsibilities:

- run repeated source collection cycles
- manage session/auth state and retries
- decide what to request from external sources
- capture raw source payloads

Should likely own:

- IBKR session/auth
- fundamentals endpoint fanout
- retry and reauth logic
- source fetch telemetry

Should likely not own:

- canonical child-table shaping for downstream consumers
- analysis-facing feature names

### 3. Canonical Storage

Responsibilities:

- persist canonical facts from source payloads
- preserve raw payload auditability
- define stable reader contracts
- preserve explicit time semantics

Should likely own:

- schema and migrations
- raw blob storage
- endpoint-specific canonical tables
- source-to-canonical scalar parsing where semantics are clear
- consumer-oriented readers

Should likely not own:

- research panel shaping
- factor vocabularies
- model outputs

### 4. Feature Inputs

Responsibilities:

- transform canonical stored facts into stable analysis inputs
- define reusable feature vocabulary
- produce diagnostics explaining input trustworthiness

This is the concern family currently mislabeled as `preprocess`.

The current capability clusters here are real and should likely remain distinct:

- price inputs and eligibility
- dividend usability for total-return adjustment
- snapshot feature contract and diagnostics
- supplementary analysis inputs

This family should likely own:

- merged snapshot feature contract
- cleaned price contract
- dividend-usability contract
- supplementary derived input contract

This family should likely not own:

- source fetch/auth
- canonical persistence
- factor research decisions

### 5. Panel Construction

Responsibilities:

- join feature inputs into point-in-time analysis panels
- apply eligibility rules at rebalance dates
- carry forward feature age and join diagnostics

This is currently buried inside the monolithic analysis module, but it is distinct enough to deserve separate ownership.

### 6. Research

Responsibilities:

- build candidate factors
- estimate factor returns
- screen, cluster, and reduce factors
- run walk-forward evaluation
- derive expected-return and persistence outputs

This is a real concern family, but it is broader than “analysis” in the current code because the current module also owns panel construction and some feature augmentation.

### 7. Exposure Estimation

Responsibilities:

- estimate current instrument exposures to accepted or persistent factors

This is downstream of research, but distinct from factor discovery and walk-forward evaluation.

### 8. Outputs

Responsibilities:

- persist derived research artifacts
- manage artifact naming and materialization
- keep derived outputs separate from canonical source-of-truth storage where practical

This is currently not a clean owner in `/pystocks`.

## Main Current Ownership Problems

### `ingest/fundamentals.py` is too broad

It currently combines:

- target selection
- recency skip logic
- auth/session recovery
- endpoint fanout
- landing-page gating
- payload usefulness heuristics
- run telemetry
- per-instrument status policy

That is too much for one runtime owner.

### `storage/fundamentals_store.py` is both healthy and overloaded

Healthy:

- it clearly owns many source-to-canonical transformations

Overloaded:

- time semantics are globally anchored in one weak rule
- many endpoint-specific policies are embedded in one large class

### `preprocess` is a misleading name

The current modules are not generic cleanup utilities. They define durable downstream contracts.

“Feature inputs”, “features”, or a similar name is likely a better future concern label.

### `analysis/__init__.py` is a monolith

It currently mixes:

- panel construction
- feature augmentation
- factor discovery
- clustering
- walk-forward modeling
- persistence of many output artifacts
- current beta estimation

These are related, but they are not one concern.

### Supplementary data crosses concern boundaries

The current supplementary path includes:

- external refresh
- raw persistence
- derived-feature generation
- fetch logging

That should likely be split across collection, canonical storage, and feature inputs.

## Current Capability Grouping That Looks Durable

These current clusters appear conceptually sound even if the module names or code shape change:

- product-universe bootstrap and refresh
- repeated fundamentals collection over a maintained universe
- cleaned price contract plus eligibility diagnostics
- dividend event vetting for total-return use
- merged snapshot feature contract plus diagnostics
- supplementary derived macro and risk-free input contract
- point-in-time panel construction
- factor research and diagnostics
- current factor beta estimation

## Current Capability Grouping That Looks Accidental

These current groupings look more like implementation drift than durable design:

- product metadata and scrape-operational state in the same `products` table
- all snapshot endpoint `effective_at` values anchored to `ratios.as_of_date`
- supplementary collection and supplementary feature derivation in one runtime function
- panel construction and factor research inside the same monolithic analysis module
- research outputs persisted back into the same SQLite database as canonical source data

## Immediate Rebuild Guidance

Before choosing package names, the rebuild should lock down:

1. canonical data/time semantics
2. stable analysis-input contracts
3. the boundary between panel construction and research
4. the boundary between canonical storage and derived outputs

If those four boundaries are clear, package layout becomes much easier.
