# Factor Research Refactor Plan

## Summary

Status as of April 10, 2026:

- phases 1 through 4 are implemented in `pystocks/analysis/__init__.py` and the supporting supplementary-data pipeline
- the remaining work is ongoing research evaluation and factor review, not additional planned architecture phases

The factor research pipeline is now operational end to end:

- products scrape successfully
- fundamentals ingestion completes
- supplementary risk-free and World Bank data are stored locally
- analysis runs to completion and writes parquet outputs

That is the right foundation, but it is not the end state. The next phase is not to create more factors indiscriminately. The goal is to create fewer, more distinct, more interpretable, and more stable factors while preserving the strengths of the V2 architecture:

- normalized storage
- point-in-time safety
- explicit preprocess artifacts
- testable analysis stages
- stable parquet and SQLite outputs

The legacy V1 analysis and notebook workflow still contain useful ideas that V2 does not yet express strongly enough:

- semantic compression before regression
- country and macro grouping into coherent themes
- notebook-driven factor diagnostics
- manual scrutiny of factor distinctness
- VIF and Elastic Net used as diagnostics rather than as the only selection logic

This document defines the concrete roadmap for merging V1’s factor intelligence into V2’s production architecture without collapsing back into a notebook-style monolith.

## Current State Assessment

### What V2 Already Does Well

The current implementation in `pystocks/analysis/__init__.py` already includes several strong building blocks:

- benchmark factor generation for market excess, SMB, and HML-style value
- composite features such as value, profitability, leverage, momentum, income, duration, credit, and concentration
- supersector aggregation for defensive, cyclical, and commodities
- macro exposure collapse from country weights into World Bank-derived features
- point-in-time rebalance-date panel construction
- walk-forward Elastic Net research windows
- factor return clustering by realized return correlation
- factor persistence and current-beta estimation

### What V2 Still Gets Wrong

The weak point is factor distillation.

V2 currently:

- generates factors mechanically from a broad set of numeric columns
- removes redundancy mainly through return-correlation clustering
- does not yet have an explicit semantic factor registry
- does not score factor usefulness before or during clustering
- does not preserve enough lineage to explain why a factor survived or was dropped

The result is a factor library that is broad but not obviously contentful. A recent realistic run produced a large factor library and zero persistent factors. That is not a catastrophic failure, but it is a clear sign that the factor set is too noisy and too weakly distilled.

### What V1 Still Did Better

The V1 code in `src/analysis.py` and `src/6.analysis.ipynb` did several useful things that remain worth preserving:

- grouped related fundamental ratios into coherent composite factors
- grouped country exposures into more meaningful macro or regional themes
- used World Bank level, growth, and acceleration style features
- recognized that raw currency and raw country factors often encode the same exposure
- used manual factor review informed by Elastic Net zero-frequency and VIF diagnostics

The target is not to restore V1 as-is. The target is to capture those ideas in a deterministic, modular, testable V2 pipeline.

## Design Principles

- Prefer semantic compression before statistical pruning.
- Treat VIF as a diagnostic constraint, not the main selection algorithm.
- Separate candidate generation from candidate admission.
- Preserve point-in-time correctness at every stage.
- Prefer interpretable factors over marginally clever but fragile ones.
- Keep notebooks and production code aligned through shared diagnostics and naming rather than duplicated logic.

## Target Outcomes

The refactor should produce a research pipeline that:

- emits a smaller and more interpretable candidate factor set before clustering
- groups overlapping raw features into coherent semantic families
- reduces final multicollinearity materially relative to the raw candidate universe
- produces nontrivial persistent factors in normal runs
- explains every keep, merge, cluster, and drop decision with persisted diagnostics
- remains optimizer-ready through stable `asset_expected_returns` and `asset_factor_betas` outputs

## Proposed Architecture

### Stage A: Candidate Feature Families

Candidate factors must be organized into explicit blocks before factor series are constructed.

Required blocks:

- benchmark
  - `market_excess`
  - `smb`
  - `hml`
- composite fundamentals
  - value
  - leverage
  - profitability
  - momentum / relative strength
  - income
  - duration
  - credit
  - concentration
- sector structure
  - supersector factors
  - selected industry factors only when not already captured by a stronger grouped feature
- macro-country
  - continent or regional blocs
  - developed / emerging style blocs when supported by holdings data
  - World Bank-derived macro themes
- currency exposure
  - reserve-currency and bloc-level factors before raw single-currency factors
- fixed-income sleeve
  - maturity
  - quality
  - debt type
  - sovereign / corporate distinctions

Raw features should not automatically become production factors if a stronger grouped representation already exists.

### Stage B: Semantic Compression

Introduce an explicit semantic compression pass between panel construction and long/short factor construction.

This pass should:

- map raw snapshot columns into semantic groups
- aggregate overlapping country and currency exposures into blocs
- derive macro themes from stored World Bank features
- register whether a feature is raw, grouped, composite, benchmark, or macro-derived

Examples:

- raw `country__usa` and raw `currency__usd` should not both survive by default when they represent the same structural exposure
- raw industry exposures should often collapse into supersector families
- country weights plus macro data should mostly feed grouped macro themes rather than a one-factor-per-country sprawl

### Stage C: Factor Construction

Long/short factor series should be built from the curated candidate set, not from nearly every numeric column.

Construction rules:

- keep current sleeve-scoped long/short construction
- keep benchmark factors always explicit
- enforce minimum cross-sectional spread
- enforce minimum long and short basket sizes
- preserve size-weighted construction unless a specific factor family requires a different weighting rule

Each factor should emit construction telemetry:

- basket sizes
- effective constituent count
- coverage ratio
- turnover proxy
- family
- source columns

### Stage D: Distinctness Screening

Replace single-layer screening with a layered system:

1. Structural screen
   - empty
   - constant
   - sparse
   - low-coverage
   - zero-spread
2. Semantic sibling screen
   - currency vs country near-duplicates
   - raw vs grouped duplicates
   - industry vs supersector duplicates
3. Statistical duplication screen
   - factor return correlation
   - constituent overlap
   - basket similarity
4. Model usefulness screen
   - Elastic Net nonzero frequency
   - sign consistency
   - mean absolute coefficient
   - incremental out-of-sample value

## Factor Registry

Add a persisted factor registry that acts as the source of truth for all candidate and selected factors.

Minimum fields:

- `factor_id`
- `sleeve`
- `family`
- `semantic_group`
- `kind`
- `source_columns`
- `construction_type`
- `economic_rationale`
- `expected_direction`
- `is_benchmark`
- `is_macro`
- `is_composite`
- `admission_status`
- `rejection_reason`

This registry should be produced by the main analysis pipeline and consumed by notebooks and diagnostics.

## Selection Strategy

The new selection flow must be hierarchical, not greedy.

Required order:

1. generate a broad candidate library
2. apply semantic compression
3. screen structurally weak factors
4. choose provisional representatives within each semantic block
5. perform cross-block redundancy checks
6. run walk-forward Elastic Net on the reduced library
7. score factors using model usefulness diagnostics
8. apply final multicollinearity checks
9. compute persistence and current betas on the reduced set

Do not use iterative highest-VIF dropping as the primary algorithm.

VIF should only be:

- a final-set health check
- a tie-break diagnostic among otherwise similar factors
- a reporting metric in diagnostics outputs

## Representative Scoring

Current clustering chooses representatives primarily by factor kind and coverage. That is too weak.

Representative selection should use a scored ranking combining:

- kind priority
  - benchmark and grouped/composite factors preferred over raw leaf factors
- coverage
- return-series stability
- constituent distinctness
- Elastic Net nonzero frequency
- sign stability
- out-of-sample usefulness
- interpretability

This score should be explicit and persisted.

## Macro And World Bank Enhancements

Keep the current V2 macro base:

- population
- GDP per capita
- GDP share
- foreign direct investment
- share of global trade

Expand carefully rather than broadly.

Planned additions:

- selective derivative features such as growth and acceleration / trend change
- grouped macro themes rather than direct one-factor-per-metric proliferation
- bloc-level macro exposures for region or development-status groupings where supported by holdings data

Candidate macro themes to evaluate:

- demographic scale
- demographic momentum
- trade centrality
- global output share
- external investment intensity

Any added macro theme must justify itself through distinctness and stability, not novelty.

## Notebook Integration

`src/6.analysis.ipynb` should remain a research reference, but the validation workflow needs a V2-aligned notebook that reads production outputs instead of reimplementing the pipeline ad hoc.

The notebook workflow should:

- read stored panel and factor artifacts from V2 outputs
- inspect the factor registry and screening decisions
- visualize within-family and cross-family correlation structures
- visualize factor basket overlap
- compare suspiciously similar factor series directly
- inspect Elastic Net zero-frequency and sign consistency
- inspect final-set VIF

Required visual comparisons:

- USD exposure vs US-country exposure
- raw country factors vs continent or bloc factors
- raw industry factors vs supersector factors
- raw macro factors vs grouped macro themes

Notebook use is for validation and hypothesis testing, not as an alternate production implementation.

## New Outputs And Diagnostics

Add stable parquet and SQLite outputs for Phase 1 diagnostics.

Planned output families:

- `analysis_factor_registry`
- `analysis_factor_candidate_diagnostics`
- `analysis_factor_distinctness`
- `analysis_factor_selection_scores`
- `analysis_factor_screening_decisions`
- `analysis_factor_cluster_membership`
- `analysis_factor_model_telemetry`

Each output should have:

- a stable schema
- deterministic naming
- direct usefulness in notebooks and downstream review

## Progress Bar UX

Analysis progress bars should remain visible after completion by default.

Implementation requirements:

- persist price-feature progress
- persist rebalance-date progress
- persist baseline-window progress
- persist factor-return-window progress
- persist clustering progress
- persist research model-fit progress
- persist current-beta-fit progress

Public interface:

- `AnalysisConfig.persist_progress_bars: bool = True`

This is a usability feature, not part of factor selection, but it improves operator visibility for long analysis runs.

## Phased Implementation Plan

### Phase 1: Instrumentation And Audit

Goals:

- add factor registry
- add candidate diagnostics
- add screening-decision outputs
- add notebook-facing diagnostics tables
- keep current factor construction broadly intact

Acceptance criteria:

- pipeline behavior is mostly unchanged
- diagnostics explain current factor redundancy clearly
- candidate factors can be traced back to source columns and families

### Phase 2: Semantic Compression

Goals:

- add grouped country and currency factors
- add bloc-level macro features
- define admission rules that prefer grouped factors over raw duplicates
- reduce uncontrolled candidate proliferation before clustering

Acceptance criteria:

- candidate factor count declines materially before clustering
- grouped factors replace obvious raw duplicates

### Phase 3: Selection Refactor

Goals:

- replace simplistic representative picking with scored selection
- incorporate Elastic Net usefulness metrics into factor ranking
- use VIF as a final diagnostic constraint

Acceptance criteria:

- final selected factor set is smaller and more interpretable
- final selected set satisfies configured multicollinearity limits
- persistent factors appear in normal runs

### Phase 4: Macro Expansion

Goals:

- add carefully chosen derivative macro features
- evaluate additional World Bank themes inspired by legacy research
- keep macro factors interpretable and distinct

Acceptance criteria:

- macro additions improve distinctness or predictive usefulness
- macro factor families do not explode uncontrollably

Completion notes:

- yearly World Bank acceleration fields are now derived and stored alongside level and growth features
- macro leaves are compressed into curated theme factors such as demographic scale, demographic momentum, development, trade centrality, global output share, and external investment intensity
- bloc-level macro themes are derived deterministically from the same stored features so macro expansion remains grouped rather than leaf-driven

## Public Interfaces

### Required Now

- `AnalysisConfig.persist_progress_bars`

### Planned For Phase 1 Or Later

- `enable_factor_registry`
- `max_final_vif`
- `max_family_redundancy_corr`
- `min_factor_spread`
- `selection_score_weights`
- `semantic_grouping_version`

These should be introduced only when the corresponding behavior is implemented. Do not add placeholder config without code that uses it meaningfully.

## Test Plan

Required test coverage for the refactor:

- factor registry is populated with stable fields
- semantic compression maps raw inputs to expected grouped families
- obvious duplicate country/currency exposures are screened appropriately
- cluster representative selection is deterministic
- final shortlisted factors satisfy configured VIF constraints
- model telemetry lands with stable schemas
- notebook-facing diagnostics tables are emitted
- progress bars remain visible by default and can still be disabled explicitly

Required behavioral scenarios:

- USD-heavy portfolios should not retain both redundant USD and US-country factors without a documented semantic distinction
- continent or bloc factors should dominate raw country factors when they encode the same exposure
- macro themes should survive only when they add distinct information

## Acceptance Criteria

The roadmap should be considered successful when:

- the candidate factor set is materially smaller before clustering
- factor families are explicit and semantically coherent
- final shortlisted factors are more distinct than the raw library
- persistent factors are nontrivial in realistic runs
- every keep/drop/merge decision is explainable from diagnostics artifacts
- notebook validation can trace any factor back to its production lineage

## Assumptions And Defaults

- `pystocks/analysis/__init__.py` remains the production source of truth
- notebooks are research and validation tools, not alternate production code paths
- grouped and composite factors are preferred over raw leaf-level factors when they encode the same information
- VIF remains a final diagnostic, not the primary factor selection method
- portfolio optimization remains out of scope except for preserving optimizer-ready outputs
