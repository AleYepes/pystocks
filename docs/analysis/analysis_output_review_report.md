# Analysis Output Review Report

This report is a guide to the executed output in [notebooks/analysis_output_review.ipynb](/home/alex/Documents/pystocks/notebooks/analysis_output_review.ipynb). It is written for a reader who is new to this repository and wants to understand what the factor-analysis outputs mean, how to read them, and what the current run appears to be doing.

## Purpose Of The Notebook

The notebook is not a second implementation of the analysis pipeline. It is a review layer over the production outputs written by `run_factor_research`.

That distinction matters:

- production code constructs the panel, candidate factors, factor returns, clustering, model telemetry, persistence, and downstream outputs
- the notebook reads those stored artifacts and helps a human inspect them
- if the notebook is confusing, the right fix is usually to improve the diagnostics or the notebook labels, not to duplicate more logic in the notebook

## What The Current Notebook Run Loaded

The executed notebook shows these stored artifacts:

- `panel`: `2188 x 503`
- `factor_returns`: `83 x 273`
- `factor_registry`: `1531 x 14`
- `candidate_diagnostics`: `1522 x 16`
- `distinctness`: `1464 x 7`
- `selection_scores`: `1522 x 9`
- `screening_decisions`: `1578 x 7`
- `factor_diagnostics`: `272 x 25`
- `factor_final_vif_diagnostics`: empty
- `factor_model_telemetry`: empty
- `factor_persistence`: empty
- `model_results`: empty
- `current_betas`: empty
- `asset_expected_returns`: empty
- `asset_factor_betas`: empty

At a high level, this means:

- the pipeline successfully built the analysis panel
- it generated a large candidate library
- it created factor-return series for a reduced set of factors
- it persisted structural and semantic diagnostics
- but this particular run did not produce usable model-stage outputs

That last point is important. In this run, the notebook is mostly showing Stage A and Stage B style outputs: candidate generation, compression, and pre-model diagnostics. It is not yet showing a successful final selected factor set.

## How To Read The Main Tables

### `analysis_snapshot_panel`

This is the point-in-time ETF feature panel. Each row represents an ETF at a rebalance date, with many columns describing the ETF as of that date.

Examples of what can appear here:

- holdings-derived country weights
- currency weights
- sector or industry weights
- bond maturity or quality fields
- raw fundamental ratios
- preprocessed price features
- World Bank macro-derived features

This table is the source material for later factor construction.

### `analysis_factor_registry`

This is the inventory of candidate and constructed factors. A factor here is not yet a regression result. It is a named exposure concept that the pipeline has defined and tracked.

Important columns:

- `factor_id`: the fully qualified factor name
- `sleeve`: the sleeve the factor belongs to, such as `equity`, `bond`, `commodity`, or `other`
- `family`: the broad conceptual bucket
- `semantic_group`: the finer semantic identity used for grouping and duplicate handling
- `kind`: the construction style or maturity of the factor
- `source_columns`: the panel columns used to derive it
- `admission_status`: whether it was admitted, rejected, or constructed
- `rejection_reason`: why it was screened out if rejected

This is the best starting table for understanding what the pipeline thinks exists.

### `analysis_factor_candidate_diagnostics`

This is the table that explains how the candidate library was compressed before factor construction and modeling.

Important columns:

- `coverage_ratio`: how often the source feature is populated in the cross section
- `zero_ratio`: how often the feature is exactly zero
- `unique_count`: how many distinct values it takes
- `cross_sectional_spread`: how much cross-sectional variation it has
- `admitted_for_construction`: whether it survived the pre-construction filters
- `pre_compression_candidate_count`: size of the candidate pool before semantic compression
- `post_compression_candidate_count`: size after compression
- `compression_removed_count`
- `compression_removed_ratio`

This is where a newcomer can answer: “how aggressively did the pipeline reduce the raw candidate universe?”

### `analysis_factor_screening_decisions`

This is the audit trail of keep/drop decisions.

Important columns:

- `stage`: where in the pipeline the decision happened
- `decision`: `keep` or `drop`
- `reason`: the specific decision rule
- `reference_factor_id`: the related factor, when a factor was dropped in favor of another one

Typical reasons visible in the current run:

- `passed_structural_screen`
- `low_coverage`
- `constant`
- `empty`
- `sparse`
- `zero_spread`
- `semantic_duplicate_of_grouped_or_composite`
- `semantic_duplicate_of_grouped_source_overlap`

This is the best table for answering: “why did this factor disappear?”

### `analysis_factor_diagnostics`

This table is about realized factor-return series, not raw panel features.

Important columns:

- `coverage_ratio_x`: how often the factor return exists over time
- `mean_return`
- `volatility`
- `selected_for_model`
- metadata copied from the registry and candidate diagnostics

In the current run, `selected_for_model` is false for everything shown in the notebook fallback view, which is consistent with the model-stage outputs being empty.

### `analysis_factor_selection_scores`

This stores explicit scores used during selection or compression stages.

In the current run, the visible top rows are from `stage = semantic_compression`.

Important columns:

- `kind_priority_component`
- `coverage_component`
- `density_component`
- `uniqueness_component`
- `selection_score`

This is where the pipeline makes ranking explicit rather than relying on hidden heuristics.

### `analysis_factor_distinctness`

This is a pairwise diagnostics table. It records when two factors are suspiciously similar according to some comparison rule.

The visible notebook output shows many high-correlation pairs among price-feature momentum and volatility variants. That is expected and is exactly the kind of clutter this refactor is meant to expose and reduce.

## What “Family” Means In This Repo

The word `family` is easy to misread, because it is not the same thing as `kind` and not the same thing as `semantic_group`.

In this repo:

- `family` means the broad economic or structural theme a factor belongs to
- it answers “what type of thing is this factor trying to measure?”

Examples from the current outputs:

- `market_excess`
- `smb`
- `composite`
- `country`
- `currency`
- `macro_bloc`
- `supersector`
- `ratio_key`
- `price_feature`
- `debt_type`

A useful mental model:

- `family` is the chapter title
- `semantic_group` is the specific topic
- `factor_id` is the exact file name

For example:

- family: `supersector`
- semantic group: `sector_theme__cyclical`
- factor id: `equity__grouped__supersector__cyclical`

## Other Terms The Notebook Assumes You Already Know

### Sleeve

A `sleeve` is a major asset bucket. The code currently uses sleeves such as:

- `equity`
- `bond`
- `commodity`
- `other`

Factor construction and model fitting are sleeve-scoped. An equity factor is compared to other equity factors, not usually to bond factors.

### Kind

`kind` describes how a factor was formed or how mature it is semantically.

Common values:

- `benchmark`: explicit baseline factors like market excess or SMB
- `composite`: a hand-grouped combination of related raw features
- `grouped`: a deterministic grouped exposure such as a supersector or currency bloc
- `raw`: a leaf feature taken more directly from the panel
- `macro_derived`: a factor derived from country weights and macro data

The broad preference order in the refactor is:

- benchmark
- composite or grouped
- macro-derived when interpretable
- raw leaves last

### Semantic Group

`semantic_group` is the identity used to decide whether two candidates are basically expressing the same idea.

Examples:

- `benchmark__market_excess`
- `composite__concentration`
- `fixed_income_credit`

This is more precise than `family`. Multiple factors can belong to the same family but different semantic groups.

### Source Columns

`source_columns` names the panel features from which the factor was derived.

This is especially important for grouped and composite factors. A long list of `country__*` or `industry__*` columns usually means the factor is a grouped exposure synthesized from many leaves.

### Admission Status

`admission_status` answers where the factor currently stands in the pipeline.

Typical meanings:

- `constructed`: the factor made it through to factor-return construction
- `admitted`: it was accepted as a candidate but may not yet have become a live factor-return series
- `rejected`: it was screened out

### Structural Screen

This is the earliest filter layer. It removes features that are not useful enough to justify factor construction.

Examples:

- empty
- constant
- sparse
- low coverage
- zero spread

These are quality-control failures, not economic judgments.

### Semantic Screen

This is where the pipeline prefers a stronger grouped or composite representation over an overly literal raw feature.

Examples:

- a grouped supersector instead of a noisy raw industry leaf
- a currency bloc instead of several single-currency leaves
- a grouped macro theme instead of a large sprawl of raw country-macro combinations

### Distinctness

`distinctness` is about whether two factors are meaningfully different enough to both keep.

In this refactor, distinctness is not only statistical. The broader intent is:

- structural distinctness: does the feature have enough signal to exist?
- semantic distinctness: is it conceptually redundant with a better grouped factor?
- statistical distinctness: do the realized factor returns look too similar?

### Selected For Model

This means a factor survived far enough to be included in the model-stage reduced library. It does not mean the factor was persistent or useful in out-of-sample regressions. It only means it remained alive at model entry.

In the current notebook run, no factors are shown as selected for model in the fallback view.

### Persistence

A factor is `persistent` if it shows up repeatedly enough across walk-forward Elastic Net fits, subject to configured thresholds.

Persistence is meant to capture stability, not just a one-off selection.

### VIF

Variance Inflation Factor is a multicollinearity diagnostic on the final factor set.

In this project, VIF is supposed to be:

- a final health check
- a tie-break tool
- a diagnostic output

It is not supposed to be the main selection algorithm.

## What The Current Run Seems To Be Saying

The current notebook output already tells a useful story even though the final model outputs are empty.

### 1. Semantic compression is doing real work

For many sleeves, the candidate pool shrank dramatically before modeling.

Examples visible in the notebook:

- bond candidates: `460 -> 64`
- other candidates: `354 -> 41`

That is a large reduction, and it is exactly what the Phase 2 refactor was meant to accomplish.

### 2. Structural screening is also removing a lot of noise

The screening summary shows large counts for:

- `low_coverage`
- `constant`
- `empty`

This suggests the raw panel still contains many mechanically generated features that are not robust enough to become factors.

### 3. The current saved run does not yet have model-stage evidence

Because these outputs are empty:

- `factor_model_telemetry`
- `factor_final_vif_diagnostics`
- `factor_persistence`
- `model_results`

the notebook cannot yet show:

- which factors Elastic Net selected repeatedly
- whether any factors became persistent
- the final selected factor set
- final-set VIF
- current betas or expected returns

So this notebook run is more of a compression-and-diagnostics review than a full end-to-end factor selection review.

### 4. Distinctness pressure is concentrated in familiar areas

The distinctness output highlights many similar:

- momentum variants
- volatility variants
- related debt or leverage measures

This is normal. These are exactly the feature families where semantic compression and later statistical screening should be most useful.

### 5. The suspicious-pair helper is useful, but the current run has limited matching examples

The notebook’s string-based pair search found:

- `currency__usd` factors
- some `raw__country__*` factors
- grouped `supersector` factors

But it found no `grouped__continent__*`, no `grouped__bloc__*`, and no `macro_bloc__*` factors in the saved factor-return table. That does not necessarily mean the grouping logic is broken. It may simply mean those candidates did not survive to factor-return construction in this particular run.

## How A New Reader Should Use This Notebook

A practical reading order is:

1. start with artifact sizes to see which stages actually produced output
2. read `factor_registry` to understand what the pipeline thinks the factor universe is
3. read `candidate_diagnostics` to understand how much semantic compression happened
4. read `screening_decisions` to see why factors were dropped
5. inspect `distinctness` to find crowded families
6. only then move to model telemetry, persistence, and VIF if those artifacts are populated

If model-stage outputs are empty, do not over-interpret the final-selection sections. The correct conclusion is simply that this run did not produce usable downstream research artifacts.

## Suggested Improvements To The Notebook

The notebook is already useful, but a newcomer would benefit from a few small additions:

- a glossary cell at the top defining `family`, `kind`, `semantic_group`, `sleeve`, `persistent`, and `selected_for_model`
- a status cell that explicitly says when model-stage outputs are empty and what that prevents the notebook from showing
- a short explanation that `factor_returns` is wide-form in the stored parquet
- a note that “no final selected factors” may mean “no model outputs were produced”, not necessarily “the pipeline is broken”

## Bottom Line

The current notebook output supports a positive reading of the Phase 1 and Phase 2 refactor work:

- factor lineage is visible
- compression is measurable
- screening decisions are explainable
- semantic clutter is now inspectable

But this specific saved run does not yet demonstrate the full Phase 3 success condition, because the model-stage artifacts are empty. For a full review of final factor usefulness, persistence, and multicollinearity, the next notebook run needs populated `model_results`, `factor_model_telemetry`, and `factor_final_vif_diagnostics`.
