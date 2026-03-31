# Factor Research Pipeline Plan

## Status

This document describes the current target state for the refactored analysis flow in `pystocks/`.

Some parts already exist:

- `pystocks/price_preprocess.py` builds row-level clean-price flags and eligibility outputs.
- `pystocks/analysis.py` already builds a snapshot panel, factor returns, factor clustering, persistence outputs, and current betas.
- CLI entrypoints already exist in `pystocks/cli.py`.

This plan now focuses on the gaps that still matter after the initial implementation and the SQLite anomaly review.

## Objective

- Primary goal is explanatory and persistence-oriented factor research.
- The pipeline should produce stable factor definitions, persistent factor returns, and current factor betas.
- The pipeline should stay point-in-time correct and avoid mixing future fundamentals into earlier rebalances.
- The pipeline should support multiple ETF sleeves without forcing one shared factor library onto all of them.

## Current Execution Flow

1. Session validation and login
2. Product universe scrape
3. Fundamentals and series scrape
4. SQLite materialization
5. Price series preprocessing
6. Snapshot panel assembly
7. Factor return construction
8. Factor research and persistence scoring
9. Current beta estimation

## What Is Working

### Price preprocessing

`pystocks/price_preprocess.py` now covers:

- invalid prices
- stale runs
- robust return outliers
- short bridge-price anomalies inside decimal-shift pockets
- instrument eligibility based on history length, missing ratio, and internal gaps

This is the correct place for series-level cleaning of `price_chart_series`.

### Analysis orchestration

`pystocks/analysis.py` already handles:

- as-of snapshot selection
- monthly rebalance panel construction
- sleeve assignment
- price-derived features
- raw and composite factor return construction
- correlation clustering
- elastic-net-based factor persistence research
- current beta computation

### Validation

The current test suite covers:

- storage and normalization for major endpoint families
- effective date resolution
- point-in-time panel behavior
- analysis clustering behavior
- price preprocessing edge cases

## Gaps That Still Need Work

### 1. Separate snapshot preprocessing layer

This is the main missing architecture piece.

The current code loads and merges snapshot tables directly inside `pystocks/analysis.py`. That is workable for a first pass, but it is not enough for durable factor research because the holdings and snapshot tables are not scale-consistent.

Required action:

- Add a dedicated snapshot preprocessing module.
- Keep `pystocks/price_preprocess.py` series-only.
- Move snapshot hygiene, table-specific scaling rules, and point-in-time feature normalization into the new snapshot layer.

Reason:

- Price-series anomalies and snapshot-table anomalies have different semantics and should not share one preprocessing module.

### 2. Persist raw price anomaly signals from ingestion

`pystocks/fundamentals_store.py` computes `debug_mismatch` for price rows, but that signal is not persisted into `price_chart_series`.

Required action:

- Persist `debug_mismatch` or an aggregated mismatch signal in SQLite.
- Expose it to preprocessing as another row-level flag.

Reason:

- The current bridge-price anomaly logic is useful, but it is inferential.
- If ingestion already sees a date mismatch signal, analysis should retain it.

### 3. Instrument-level quarantine rules

Some instruments are structurally broken, not just locally noisy.

Examples seen in SQLite:

- repeated zero/nonzero toggling
- long invalid runs from inception
- dense `high < low` rows
- long flat stale runs

Required action:

- Add instrument-level quarantine rules on top of row-level cleaning.
- Quarantine should be based on metrics such as invalid-row density, zero-toggle frequency, and repeated bridge-anomaly pockets.

Reason:

- Eligibility filtering alone is too late and too indirect for clearly broken instruments.

### 4. Decide on regularized business-day return panels

The legacy path regularized prices onto business days before downstream analysis. The current refactor does not.

This is now an explicit design decision, not an accidental omission.

Required action:

- Decide whether factor research should consume sparse clean returns or a regularized business-day panel built after cleaning.
- If added, keep this as a separate post-cleaning step rather than mixing it into raw series cleaning.

Reason:

- This choice affects momentum, volatility, gap handling, and comparability across ETFs.

### 5. Snapshot feature semantics review

Several holdings tables appear to mix overlapping or non-exclusive exposure views. Totals do not consistently sum to `1.0`.

Required action:

- Review holdings tables one by one.
- Define which tables are expected to sum to `1.0`.
- Define which tables can exceed `1.0`.
- Define whether any tables need deduplication, rescaling, or aggregation before factor use.

This is especially important for:

- `holdings_asset_type`
- `holdings_debtor_quality`
- `holdings_maturity`
- `holdings_currency`
- `holdings_investor_country`
- `holdings_debt_type`
- `holdings_industry`

### 6. Factor-definition hardening

The current factor construction works, but the factor-definition layer is still implicit in `pystocks/analysis.py`.

Required action:

- Add an internal factor registry or equivalent explicit definition layer.
- Store factor metadata such as factor id, family, sleeve applicability, sign convention, source columns, transform, and weighting rule.

Reason:

- This makes factor definitions reproducible and easier to review.
- It also makes clustering and persistence outputs easier to interpret.

## Modeling Defaults

- Rebalance frequency is monthly.
- Snapshot data is treated as piecewise-constant state carried forward until the next valid snapshot.
- Price-derived features remain daily.
- Factor construction remains sleeve-specific.
- Sentiment stays a sidecar research track until coverage and value are better established.
- Ownership stays optional and event-style.
- ESG remains out of scope until coverage exists.

## Outputs

The analysis pipeline should continue to materialize:

- `analysis_snapshot_panel`
- `analysis_daily_returns`
- `analysis_price_eligibility`
- `analysis_factor_returns`
- `analysis_factor_clusters`
- `analysis_model_results`
- `analysis_factor_persistence`
- `analysis_current_betas`

SQLite remains the durable queryable store. Parquet outputs under `data/analysis/` remain cache artifacts for analysis workflows.

## Required Validation

Keep these scenarios covered:

- snapshot as-of joining never uses future data
- a missing later snapshot correctly carries the prior snapshot forward
- price cleaning removes invalid rows, stale runs, extreme spikes, and bridge anomalies without deleting normal step changes
- bond baseline selection only uses eligible short-duration sovereign bond ETFs
- factor clustering is deterministic
- current beta computation uses the persistent factor set, not all raw factors
- dividend event ingestion remains idempotent when snapshots are reprocessed

Add these scenarios next:

- real SQLite anomaly windows as regression tests
- instrument-level quarantine behavior
- snapshot preprocessing rules for holdings tables with ambiguous scale semantics
- persisted `debug_mismatch` propagation from ingestion into preprocessing

## Recommended Order

1. Add snapshot preprocessing as a separate module.
2. Persist raw price anomaly signals from ingestion.
3. Add instrument-level quarantine rules.
4. Decide on sparse versus regularized business-day return panels.
5. Harden factor definitions into an explicit registry or equivalent layer.
6. Expand tests around real anomaly windows and snapshot semantics.

## Notes

- Do not treat old DB coverage counts as durable planning inputs. They drift quickly and should be regenerated when needed.
- Do not reintroduce CSV-centric analysis flows.
- Keep `pystocks/` as the source of truth for production analysis logic.
