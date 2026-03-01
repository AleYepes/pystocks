# Series Preprocessing Guide

Use this guide when adding or refactoring storage for any endpoint that returns a historical series (for example: `price_chart`, `sentiment_search`, ownership trade log, dividends events).

## Goal

Standardize how series data is:
- Parsed from payloads
- Date-normalized
- Validated
- Upserted into SQLite
- Compared across sources

This keeps series endpoints consistent and makes downstream analysis predictable.

## Canonical Pattern

For each series endpoint, use this two-layer model:

1. Snapshot table (`*_snapshots`)
- One row per `(conid, effective_at)` snapshot event.
- Stores metadata only: `observed_at`, `payload_hash`, `inserted_at`, `updated_at`, `points_count`, `min_trade_date`, `max_trade_date`.
- Do not duplicate full series rows here.

2. Series table (`*_series_raw`)
- One row per `(conid, effective_at)` series data point.
- Keep only analysis-ready values.
- Upsert by natural key (usually `(conid, effective_at)`).

For price-like series, the minimal row shape is:
- `conid`
- `effective_at`
- `price`, `open`, `high`, `low`, `close`

## Date Policy

Always define one canonical date per point (`effective_at`) and keep rules explicit.

Recommended precedence:
1. Primary timestamp/date field (`x`, `datetime`, etc.)
2. Debug/fallback date field (if primary missing/unparseable)

Example (`price_chart`):
- `x` is primary
- `debugY` is fallback only

Do not silently switch precedence without tests.

## Validation Policy

Debug fields are validation aids, not business keys.

If both primary and debug dates exist:
- Record mismatch only when difference is materially large.
- For price chart, `abs(primary_date - debug_date) > 1 day` is treated as mismatch.

Use warnings/metrics, not hard failures, so ingestion is resilient.

## What To Store vs Drop

Store:
- Canonical date key (`effective_at`)
- Analysis values (OHLC, scores, events)
- Snapshot metadata (`points_count`, date range, hash lineage)

Drop unless explicitly needed:
- Redundant unix timestamps when date is already stored
- Endpoint debug fields after validation
- Ingestion metadata duplicated at row level (`observed_at`, `inserted_at`, per-row hash) if not needed for analysis
- Formatted string columns

## Endpoint Implementation Checklist

1. Add/verify extractor function
- Parse payload structure
- Return normalized row dicts
- Produce `effective_at` as ISO date

2. Add/verify snapshot writer
- Compute `points_count`, `min_trade_date`, `max_trade_date` from extracted rows only
- Upsert snapshot row

3. Add/verify series writer
- Upsert rows into `*_series_raw` by natural key
- Keep writes idempotent

4. Add validation logging
- Add warning when mismatch threshold is exceeded
- Avoid pipeline-breaking assertions for vendor inconsistencies

5. Add tests
- Date precedence test
- Mismatch-threshold test
- Upsert/idempotency test
- Snapshot metrics test

## Comparison Across Sources (Optional But Recommended)

When comparing the web-app endpoint vs official API source:

1. Store official source in separate SQLite file.
2. Mirror core schema (`price_chart_snapshots`, `price_chart_series_raw`) for easy joins.
3. Compare on `(conid, effective_at)` and compute:
- Absolute differences
- Percent differences
- Per-conid summary stats (mean/median/max abs diff)

Do not assume values should match exactly unless source definition is identical.

## Common Pitfalls

1. Treating debug fields as canonical dates.
2. Mixing snapshot-level effective dates with point-level series dates.
3. Keeping row-level ingestion metadata that bloats tables with no analysis value.
4. Computing snapshot min/max from existing DB rows instead of current payload rows.
5. Assuming different IBKR endpoints represent identical price basis.

## Quick Template

When implementing a new series endpoint, follow this order:

1. Add extractor `def _extract_<endpoint>_rows(payload): ...`
2. Add snapshot upsert `def _upsert_<endpoint>_snapshot(...): ...`
3. Add series write `def _write_<endpoint>_series(...): ...`
4. Wire into `_write_series(...)` dispatcher
5. Add tests in `pystocks/tests/`
6. Run `./venv/bin/python -m pytest -q`

