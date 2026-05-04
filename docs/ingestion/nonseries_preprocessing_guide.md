# Endpoint Refactor Guide (Pattern From `profile_and_fees`)

This document describes a reusable pattern for refactoring any endpoint from semi-structured row storage into explicit, endpoint-specific relational tables.

## Goal

For each endpoint, reshape storage from generic key/value rows into explicit typed columns, while preserving raw payloads for future remapping.

Core principles:
- Keep schemas explicit (no dynamic table/column creation at ingest time).
- Preserve data unless there is a strong reason to exclude it.
- Make inclusion/exclusion rules explicit in code and tests.
- Avoid migrations/backfill logic during early-stage refactors when full re-fetch is acceptable.
- Keep date fields unless there is a strong justification to remove them.

## What Was Done (Generalizable Pattern)

Using `profile_and_fees` as the model, the refactor followed this structure:

1. Defined final table set for the endpoint.
- Main/core table for normalized scalar business fields.
- Snapshot table for `(conid, effective_at, observed_at, payload_hash, audit timestamps)`.
- Additional child tables for structured subdomains (for example, reports, stylebox).
- Dropped legacy tables only when data was either moved elsewhere or intentionally out of scope.

2. Converted key/value sections into explicit typed columns.
- Created mapping dictionaries from source field names to target columns and types.
- Parsed values into proper types (`TEXT`, `REAL`, `INTEGER`, ISO dates).
- Kept specialized parsing where needed (for example, split mixed text/date values).

3. Moved endpoint-level fields to correct table ownership.
- Business attributes belong in the core table.
- Snapshot metadata belongs in snapshot tables.
- Example transfer pattern: move semantic fields (`objective`, flags, themes) into core table.

4. Added dedicated tables for nested structures that do not fit core rows.
- Example pattern: `mstar.hist` -> standalone stylebox table with fixed boolean columns.

5. Explicitly excluded fields/tables with justification.
- Removed columns/tables that were redundant, low-value, or superseded.
- Never silently drop data paths; either map to a table or intentionally declare them out of scope.

6. Updated write path atomically per `(conid, effective_at)`.
- Upsert snapshot row.
- Delete endpoint child rows for same key.
- Reinsert/upsert normalized child/core rows.

7. Updated tests to match target behavior.
- Schema tests (table/column presence/absence).
- Transformation tests (parsing, typing, pivoting, overrides/fallback behavior).
- Inclusion/exclusion tests (data moved to expected table, removed from old one).

## Date Handling Policy (Critical)

Dates appear in multiple places and may conflict. Do not remove date fields without explicit justification.

Use the following canonical policy:

1. Resolve one canonical `effective_at` per scraped snapshot batch:
- Use the endpoint's own source/payload date hierarchy.
- Do not use `observed_at`, collection date, current date, or another endpoint's date as a fallback.
- If the endpoint date hierarchy cannot resolve a valid source/payload date, skip or quarantine that endpoint's canonical snapshot write.

2. Keep at least these date classes:
- `effective_at`: endpoint-level partition key derived from source payload contents.
- `observed_at`: ingestion observation timestamp.
- source business dates from payload (for example `as_of_date`, publish dates, embedded dates).

3. If multiple dates disagree, keep both when feasible.
- Example: keep `effective_at` in snapshots and `report_as_of_date` in reports.
- Do not collapse fields unless semantics are proven equivalent.

4. Add tests that lock date behavior.
- Positive case for preferred source date.
- Fallback case for the next valid source/payload date in the endpoint hierarchy.
- Edge case for malformed date formats.
- Negative case proving `observed_at` is not accepted as an `effective_at` fallback.

## Holdings Mapping Note

- Keep high-cardinality sections in long tables (`industry`, `currency`, `investor_country`, instrument-level `debt_type`).
- Keep low-cardinality fixed buckets in wide tables (`asset_type`, credit-quality buckets from `debtor`, `maturity`).
- IBKR naming is confusing for fixed-income:
  - Payload `debt_type` is instrument/issuer taxonomy and maps to long table `holdings_debt_type`.
  - Payload `debtor` carries `% Quality/...` buckets and maps to wide table `holdings_debtor_quality`.

## Inclusion/Exclusion Rules Template

Before editing code, create a short endpoint contract:

- Include:
  - All high-value business fields needed for analysis.
  - All payload hashes/audit fields needed for traceability.
  - Dates needed for chronology and debugging.

- Exclude (only with reason):
  - Redundant identifiers already normalized elsewhere.
  - Near-empty/unused fields with no current downstream consumer.
  - Tables that only duplicate data now represented in core/child tables.

- Preserve:
  - Raw payload blob linkage (`payload_hash`) so remapping is always possible later.

## Implementation Checklist (Per Endpoint)

1. Inspect existing SQLite tables for endpoint and sample payload docs.
2. Enumerate unique source field names and value shapes.
3. Propose final table set and ownership boundaries.
4. Define explicit source->column mapping + datatype rules.
5. Implement schema changes in `pystocks/storage/fundamentals_store.py`.
6. Implement endpoint upsert transform logic.
7. Remove legacy writes for dropped tables/columns.
8. Update table/index/view references.
9. Add/adjust tests under `pystocks/tests/`.
10. Run validation:
- `./venv/bin/python -m pytest -q`

## Why Explicit Schemas (vs Dynamic Creation)

Recommendation: keep explicit declarations in `storage/fundamentals_store`.

Reasons:
- Predictable schema and query contracts.
- Easier code review and CI validation.
- Lower risk of accidental schema drift from noisy payload variants.
- Better control over datatype decisions and naming standards.
- Raw compressed payloads remain available, so future remodels do not require dynamic schema now.

## Notes For Future Endpoint Refactors

- Refactor one endpoint at a time.
- Keep changes local to that endpoint’s schema and writer path.
- Defer cross-endpoint canonicalization questions (for example universal date policy) until per-endpoint mappings are stable.
- If uncertain whether to drop a field, keep it first and revisit after observing actual usage.
