# Preprocessing And Package Reorganization Plan

## Goal

Prepare `pystocks/` for multiple preprocessing pipelines without letting the package root turn into a flat collection of unrelated scripts.

The immediate driver is dividend-aware preprocessing. Price preprocessing already exists. Snapshot preprocessing is now clearly needed. Sentiment series likely needs similar treatment later.

## Why Reorganization Is Justified Now

The SQLite review showed that preprocessing is no longer a single-series concern.

### Confirmed from the current database

- `dividends_events_series` contains usable event data:
  - 5,835 rows
  - 276 conids
  - full currency coverage
  - event dates, payment dates, declaration dates, and amounts
- Dividend currency is not always the same as product currency:
  - 1,127 event rows have a different dividend currency than the product trading currency
- Holdings and other snapshot tables are not scale-consistent and need table-specific preprocessing rules
- No normalized split or broader corporate-action table exists, so dividend events are currently the only structured corporate-action-like series available downstream

### Current package state

The package root already mixes several different responsibilities:

- ingestion
- normalization and storage
- one series preprocessor
- analysis assembly
- factor research
- comparison utilities

This is still manageable with one preprocessor, but it will become noisy once dividend and snapshot preprocessing are added.

## Decision

Reorganize now, but keep the change minimal and tied to real module boundaries.

Do not do a large abstract refactor first.

Do:

- introduce a `pystocks/preprocess/` package
- move current price preprocessing there
- add dividend preprocessing there
- reserve clear locations for snapshot and sentiment preprocessing

Defer:

- deeper analysis subpackage splits unless the implementation work immediately needs them

## Target Package Layout

Planned near-term layout:

```text
pystocks/
  preprocess/
    __init__.py
    price.py
    dividends.py
    snapshots.py
    sentiment.py
  analysis/
    __init__.py
    panel.py
    factors.py
    models.py
  cli.py
  config.py
  ingest/
    fundamentals.py
    product_scraper.py
    session.py
  fundamentals_normalizers.py
  storage/
    fundamentals_store.py
    ops_state.py
```

Minimal first step:

```text
pystocks/
  preprocess/
    __init__.py
    price.py
    dividends.py
```

## Planned Work Order

### Phase 1. Move price preprocessing into a package

- Move the current logic from the package root into `pystocks/preprocess/price.py`
- Keep behavior unchanged except for import paths

Acceptance:

- current tests continue to pass
- CLI still works through the existing entrypoint

### Phase 2. Add dividend preprocessing

Create a dedicated dividend preprocessing module that reads from:

- `dividends_events_series`
- `products`
- optionally `dividends_industry_metrics`
- optionally `ratios_dividend`
- optionally `profile_and_fees`

Primary outputs:

- cleaned dividend events table
- row-level anomaly flags
- per-conid dividend coverage and quality summary

Minimum row-level flags:

- missing amount
- nonpositive amount
- missing currency
- dividend currency differs from product currency
- duplicate event signature
- suspicious event size relative to recent clean price
- date mismatch signal when available in stored data

Derived fields to add if feasible:

- implied_yield_vs_previous_price
- rolling_trailing_dividend_sum
- usable_for_total_return_adjustment

### Phase 3. Integrate dividends with price preprocessing

After dividend preprocessing exists:

- join only high-quality dividend events into the price preprocessing flow
- keep raw price returns and total returns separate
- do not silently replace price returns with total returns

Planned output fields:

- `cash_dividend`
- `raw_price_return`
- `clean_price_return`
- `raw_total_return`
- `clean_total_return`

Rules:

- same-currency dividends can be used directly
- cross-currency dividends must be skipped or separately converted
- special handling is required if later data exposes non-regular events

### Phase 4. Add snapshot preprocessing

Create a separate snapshot preprocessing module rather than extending price preprocessing.

This module should:

- define table-specific scaling rules
- define point-in-time feature hygiene rules
- prepare analysis-ready snapshot features before panel assembly

This is needed because the holdings tables do not share one safe normalization rule.

## Dividend Preprocessing Principles

### 1. Treat dividends as a separate quality problem

Do not assume dividend events are cleaner than price series.

Use their own anomaly checks.

### 2. Keep price return and total return paths separate

Downstream analysis should be able to choose explicitly between:

- price-only returns
- dividend-adjusted total returns

### 3. Prefer conservative adjustment rules

If event currency is missing or mismatched and no FX rule exists:

- do not adjust returns
- flag the event as unusable for total-return construction

### 4. Use support tables only as validation aids

`dividends_industry_metrics` and `ratios_dividend` should help validate event plausibility, but not replace event data.

## Risks

- Cross-currency dividend events are common enough that naive total-return adjustment will be wrong
- Dividend event dates may not be perfectly aligned to price dates
- Some legitimate corporate-action-like moves may still be confused with data errors because split data is not normalized separately
- A package split can create churn if import compatibility is not preserved during the transition

## Validation Requirements

Add tests for:

- dividend event loading and cleaning
- same-currency dividend adjustment
- cross-currency dividend flagging without adjustment
- implied-yield outlier detection
- duplicate event handling
- preservation of separate price-return and total-return outputs
- CLI compatibility after the price preprocessor move

## Immediate Next Step

Implement Phase 1 and Phase 2 together:

1. create `pystocks/preprocess/`
2. move price preprocessing there with compatibility imports
3. add dividend preprocessing module and tests

This gives the repo a real reason for the new structure and avoids a refactor with no functional gain.
