# Point-In-Time Contract

This document is the short-form timestamp contract for the current `pystocks` pipeline.

## Core Meanings

- `observed_at`: when `pystocks` fetched or observed the source payload.
- endpoint `as_of_date`: the source-defined date the payload describes, when the endpoint exposes one.
- storage `effective_at`: the canonical row date chosen for downstream joins.
- analysis join / rebalance date: the downstream date at which preprocess outputs are joined in panel construction.

These are distinct concepts. New code should not collapse them for convenience.

## Fundamentals Endpoints

The authoritative storage-level endpoint contract lives in [pystocks/storage/time_contract.py](/home/alex/Documents/pystocks/pystocks/storage/time_contract.py).

- holdings, ratios, Morningstar, ESG:
  use the endpoint's own top-level `as_of_date` when present, otherwise the observation date
- profile and fees, dividends, ownership:
  use a top-level endpoint `as_of_date` only if the payload actually exposes one, otherwise the observation date
- price chart, sentiment search:
  use the latest point date present in the payload, otherwise the observation date

This is a storage contract, not an analysis freshness policy.

## Supplementary Datasets

The authoritative code contract lives in [pystocks/supplementary_contract.py](/home/alex/Documents/pystocks/pystocks/supplementary_contract.py).

- `supplementary_risk_free_sources`
  `observed_at`: `fetched_at`
  `source_as_of`: `trade_date`
  `effective_at`: `trade_date`
- `supplementary_world_bank_raw`
  `observed_at`: `fetched_at`
  `source_as_of`: `year`
  `effective_at`: chosen in preprocess as the year-end date for that source year
- `supplementary_risk_free_daily`
  `observed_at`: max fetch time across contributing source rows
  `source_as_of`: `trade_date`
  `effective_at`: `trade_date`
- `supplementary_world_bank_country_features`
  `observed_at`: max fetch time across contributing raw rows
  `source_as_of`: `feature_year`
  `effective_at`: December 31 of `feature_year`

## Snapshot Features

The authoritative feature-contract metadata lives in [pystocks/preprocess/snapshot_contract.py](/home/alex/Documents/pystocks/pystocks/preprocess/snapshot_contract.py).

- preprocess owns snapshot feature namespaces and pivoting
- snapshot preprocess composes endpoint tables with backward-looking as-of joins
- snapshot preprocess emits feature rows on the union of available endpoint dates per `conid`
- analysis joins snapshot features using `effective_at <= rebalance_date`
- analysis should not reach back into storage tables to redefine snapshot feature names
