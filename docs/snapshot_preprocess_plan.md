# Snapshot Feature Preprocessing Plan

## Goal

Add a dedicated preprocessing layer for dated feature tables used at rebalance dates.

This is separate from:

- raw `*_snapshots` metadata and payload blobs
- series preprocessors for prices, dividends, and sentiment
- downstream factor construction and regression

## Scope

Implement a first snapshot preprocessor for the current core factor inputs:

- `profile_and_fees`
- holdings families
- ratio families
- `holdings_top10`

Keep these families in passthrough form for now so analysis behavior does not regress:

- `performance`
- `dividends_industry_metrics`
- `morningstar_summary`
- `lipper_ratings`

## Deliverables

- `pystocks/preprocess/snapshots.py`
- `SnapshotPreprocessConfig`
- loaders for raw snapshot feature tables
- preprocessing for merged feature output
- holdings diagnostics
- ratio diagnostics
- saved analysis artifacts under `data/analysis/`
- integration in `pystocks/analysis.py`

## Rules

- Keep one feature row per `(conid, effective_at)`.
- Keep preprocessing point-in-time safe.
- Do not silently rescale ambiguous holdings tables.
- Prefer explicit diagnostics over hidden normalization.
- Keep factor-construction behavior as close to current output as possible.

## Validation

- add direct tests for snapshot preprocessing
- confirm deterministic ratio pivoting
- confirm holdings diagnostics for near-1 totals, >1 totals, and sparse coverage
- confirm analysis panel still uses latest snapshot at or before rebalance date
