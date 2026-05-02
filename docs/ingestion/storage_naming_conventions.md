# Storage Naming Conventions

These conventions apply to normalized `pystocks_next` storage tables. The goal is to keep canonical storage faithful enough to the source payload while making downstream `feature_inputs` and panel construction predictable.

## General Rules

- Use `snake_case` for all table and column names.
- Prefer endpoint/domain-specific table names, such as `holdings_currency` or `profile_report_fields`.
- Keep canonical storage tall for variable-cardinality payload sections.
- Do not pivot source facts into analysis-wide columns in storage. Pivoting belongs in `feature_inputs`.
- Keep source fields when they are not trivially reconstructable from another stored field.
- Drop source display variants when they are only formatting differences of the same value.

## Time And Capture Columns

- `observed_at`: when the payload or source record was observed by collection.
- `effective_at`: the date the normalized fact should be effective for point-in-time joins.
- `as_of_date`: endpoint-provided reporting date when the source gives one.
- `source_as_of_date`: raw-capture level source date when it should remain distinct from storage `effective_at`.
- `payload_hash`: reference to the raw payload bytes in `raw_payload_blobs`.
- `capture_batch_id`: optional identifier for the collection batch that produced the observation.

Keep these concepts separate. Do not reuse `effective_at` for source observation time, and do not force unrelated endpoint data onto another endpoint's date.

## Identifier Columns

- `conid`: product/instrument identifier and primary cross-table entity key.
- `*_id`: lower-snake technical identifiers used for stable feature keys, such as `field_id`, `report_id`, `metric_id`, `bucket_id`, `industry_id`, `region_id`, and `debt_type_id`.
- `code`: compact source/domain code when the code has meaning beyond string formatting, such as `USD` or `US`.
- `ticker`: security ticker. Do not rename this to `code`; ticker is the domain term.
- `name`: human/source label worth preserving, such as `US Dollar`, `United States`, or `NVIDIA CORPORATION`.

Use table context plus the column name to preserve meaning. Avoid renaming every key-like value to generic `name`, because `name` often means a source display label.

## Value Columns

- `value_num`: canonical numeric measurement for a row.
- `value_text`: canonical textual measurement for a row.
- `value_date`: canonical date-valued measurement for a row.
- `value_bool`: only use when the source can actually produce boolean values.
- `ratio`: use for explicit ratios in tables where the row is already ratio-specific, such as expense allocations.
- `holding_weight_num`: top-holdings weight value; more specific than `value_num` because the table also carries holding identity fields.

Avoid storing formatted display values like `"7.83%"` when the parsed numeric value is stored and the raw payload is retained.

## Comparison Columns

- `vs_peers`: peer comparison value from source payloads.
- Use this exact name across endpoints instead of endpoint-specific variants like `vs`, `vs_num`, or `peer_value`.
- Store it only when the source field is semantically a peer comparison.

## Ordering And Source Presentation

- Drop `source_order` unless ordering itself is analytically meaningful or cannot be recomputed.
- Drop rank-like fields in category-weight tables when they are just a sort order implied by weights.
- Keep `rank` in tables where rank is part of the source concept, such as `holdings_top10`.

## Current Endpoint Examples

- `profile_fields`: `field_id`, `value_text`, `value_num`, `value_date`.
- `profile_reports`: `report_id`, `report_as_of_date`.
- `profile_expense_allocations`: `expense_id`, `value_text`, `ratio`.
- `holdings_industry`: `industry_id`, `value_num`, `vs_peers`.
- `holdings_currency`: `code`, `name`, `value_num`, `vs_peers`.
- `holdings_investor_country`: `code`, `name`, `value_num`, `vs_peers`.
- `holdings_top10`: `name`, `ticker`, `rank`, `holding_weight_num`, `vs_peers`, `conids_json`.
- Ratio-like tables: `metric_id`, `value_num`, `vs_peers`.

## Downstream Shape

Storage tables may use table-specific key names. If a generic processor needs one shape, readers or `feature_inputs` should adapt rows into an internal frame like:

- `conid`
- `effective_at`
- `feature_key`
- `value_num`
- optional diagnostics fields

That keeps canonical storage meaningful while still allowing shared pivot and factor-construction logic downstream.
