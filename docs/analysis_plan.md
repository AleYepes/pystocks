# Factor Research Pipeline Plan

## Summary

- Keep one small ingestion precondition ahead of analysis work: pystocks/fundamentals_store.py:1344 and pystocks/fundamentals_store.py:2617
currently let dividends_events_series append duplicate rows when a changed dividend snapshot is reprocessed. Price and sentiment series are
idempotent; dividend events are not.
- The current DB supports a real v1: 229 ratio snapshots across 8 effective dates, 212 conids in the latest 2026-02-28 snapshot, 209 with price
history, 94 with sentiment, 32 with ownership, and 0 ESG. That means the core model should use price + fundamentals/holdings/performance/
dividends, with sentiment as a sidecar track, ownership as optional event research, and ESG excluded for now.
- Primary objective is explanatory and persistence-oriented: stable factor definitions, persistent factor returns, and current factor betas. Not
pure forecasting.
- The pipeline will cover all ETF holding types in one framework, but it will not force one shared factor library onto all of them. It will run one
common pipeline with sleeve-specific factor families and sleeve-specific regressions.

## Architecture

- pystocks/price_preprocess.py becomes the hard gate for daily return usability.
It should produce a clean daily return panel, an eligibility table, and explicit row-level flags for invalid price, stale run, gap/interpolation,
and outlier removal.
- pystocks/analysis.py becomes a thin orchestrator over four stages:
    1. build_snapshot_panel
    2. build_factor_library
    3. build_factor_returns
    4. run_factor_research and compute_current_betas
- Add one internal module for snapshot assembly.
Its job is to denormalize SQLite tables into one analysis-ready panel keyed by (rebalance_date, conid), using the latest snapshot with
effective_at <= rebalance_date and never looking forward.
- Add one internal module for factor definitions.
It should store factor metadata in a registry: factor_id, family, sleeve applicability, sign convention, source columns, transform, weighting
rule, and whether it is raw, composite, or clustered.
- Add one internal module for model selection and reporting.
It should handle correlation clustering, factor pruning, elastic net selection frequency, stability scoring, and final OLS beta estimation on the
reduced factor set.

## Data Model And Outputs

- Rebalance frequency is monthly, aligned to snapshot availability.
Each monthly rebalance date uses the latest available fundamental snapshot and carries that state forward until the next snapshot arrives.
- Define ETF sleeves once and use them everywhere.
Sleeves are equity, bond, commodity, and other, assigned from profile_and_fees.asset_type with holdings_asset_type as the tie-breaker.
- Build one analysis-ready snapshot panel with one row per (rebalance_date, conid).
It includes:
    - Core shared features: AUM, expense ratio, domicile, classification, payout/dividend metrics, recent performance stats, price-derived momentum/
    volatility/drawdown/liquidity/gap quality, and snapshot age.
    - Equity sleeve features: value, profitability, leverage, sector/country/style concentration, top-10 concentration, regional aggregates, and
    country macro exposure from holdings weights.
    - Bond sleeve features: duration/maturity buckets, debtor quality, debt type, yield/income metrics, fixed-income ratios, and inflation-protected
    flags.
    - Commodity and other sleeves: only shared features in v1 unless there are at least 20 comparable ETFs in a family on a rebalance date.
- Build factor series as tradable mimicking portfolios, not direct raw fundamentals in the regression.
For each candidate factor at each rebalance date:
    - Rank eligible ETFs within the applicable sleeve.
    - Standardize sign so higher score always means “more of the factor”.
    - Form long-short portfolios from top and bottom quantiles.
    - Value-weight by total_net_assets_value when parsable, else equal-weight.
    - Hold until the next rebalance.
- Use a short-duration sovereign bond ETF basket as the excess-return baseline.
The basket is built from the bond sleeve using the shortest maturity buckets, highest quality buckets, and government/treasury-oriented
classifications when available. All ETF returns are modeled as excess returns over this internal baseline.
- Sentiment is a separate research track in v1.
It produces its own daily factor candidates from sscore, sdelta, svolatility, sdispersion, svscore, svolume, smean, and sbuzz, plus rolling z-
scores and moving-average deviations. These are evaluated only on the 94-conid covered subset and compared against the core model as an
incremental overlay, not mixed into the core factor set.
- Ownership is not part of the core regression set in v1.
Use it only for event-style exploratory summaries because coverage is thin and current history is very short.
- ESG is excluded entirely until coverage exists.
- Materialize these outputs:
    - analysis_snapshot_panel
    - analysis_daily_returns
    - analysis_factor_returns
    - analysis_factor_clusters
    - analysis_model_results
    - analysis_factor_persistence
    - analysis_current_betas
- Store durable outputs in SQLite plus parquet extracts under data/analysis/.
SQLite stays the queryable source of truth; parquet is only for model-friendly cached frames.

## Multicollinearity, Stationarity, And Selection

- Do not manually drop factors in code as the legacy script did in src/analysis.py:972.
Replace that with deterministic screening.
- Screening order is fixed:
    1. Coverage filter: drop factor candidates with less than 60% non-null coverage in their applicable sleeve.
    2. Variance filter: drop near-constant factors.
    3. Turnover filter: drop factors whose long/short memberships barely change across rebalances.
    4. Correlation clustering on factor-return series with absolute correlation threshold 0.90.
    5. Within each cluster, keep one canonical representative if one exists; otherwise replace the cluster with the first standardized principal
        component.
    6. Run elastic net on the clustered set.
    7. Refit OLS only on the surviving factors to estimate interpretable betas and diagnostics.
- Persistence is measured across snapshot blocks, not just arbitrary day windows.
Use anchored walk-forward runs where each training set contains whole snapshot vintages and each test set is the next snapshot interval.
- Treat fundamentals as piecewise-constant state variables.
Do not difference or de-trend the raw snapshot features aggressively in v1. Instead:
    - carry them forward between snapshots,
    - add days_since_snapshot,
    - add snapshot-to-snapshot deltas where prior snapshots exist,
    - and judge persistence on whether factor returns and selected betas remain stable across vintages.
- Price-derived factors remain fully daily.
That is where most short-horizon time variation comes from in v1.
- Composite factors are first-class and registry-defined.
Start with canonical composites:
    - value
    - profitability
    - leverage
    - momentum
    - quality
    - income
    - duration
    - credit
    - concentration
    Each composite must list its source metrics and weights explicitly so the same definition is reused in every run.

## Interfaces And Acceptance

- Add CLI entrypoints under pystocks.cli:
    - build_analysis_panel
    - run_factor_research
    - compute_factor_betas
    - run_analysis_pipeline
- run_factor_research should output:
    - factor-return series,
    - cluster map,
    - elastic-net selection frequencies,
    - OLS diagnostics,
    - persistence summary by factor and sleeve.
- compute_factor_betas should output current betas for the latest eligible rebalance date using the persistent factor set chosen by research.
- Required test scenarios:
    - snapshot as-of joining never uses future fundamentals
    - a missing new snapshot correctly carries the previous snapshot forward
    - price cleaning removes invalid rows, stale runs, and extreme spikes without deleting normal jumps
    - factor registry produces the same factor definition every run
    - correlation clustering collapses near-duplicate factor families deterministically
    - sentiment research runs on the covered subset without shrinking the core universe
    - bond baseline selection only uses eligible short-duration sovereign bond ETFs
    - current-beta computation uses the persistent factor set, not all raw candidates
    - duplicate dividend-event ingestion is blocked once the precondition fix is applied

## Assumptions And Defaults

- Primary goal is explanation and persistence, not maximum predictive accuracy.
- All ETF holding types stay in scope, but regressions are sleeve-specific with one optional pooled exploratory report.
- The internal short-government-bond ETF basket is the excess-return baseline.
- Sentiment stays separate from the core factor model in v1.
- Ownership and ESG do not block the first factor-analysis implementation.
- No CSV-centric workflow returns; SQLite plus parquet is the analysis path.
