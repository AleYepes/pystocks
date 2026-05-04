# Storage Naming Conventions

These conventions apply to normalized `pystocks_next` storage tables. The goal is to keep canonical storage faithful enough to the source payload while keeping downstream `feature_inputs` and `panel` simple, consistent, and predictable.

The following is a WIP and may need to be updated if a new value or format is found in a payload.

## General Rules

- Use `snake_case` for all table and column names.
- Keep canonical storage tall for variable-cardinality payload sections.
- Do not pivot source facts into analysis-wide columns in storage. Pivoting belongs in `feature_inputs`.
- Drop source display variants when they are only formatting differences of the same value.
- Keep source fields when they are not trivially reconstructable from another stored field.
- Drop rank-like fields when sort order is implied by weights.

## Identifier Columns

- `conid`: product/instrument identifier and primary cross-table entity key.
- `name_id`: technical identifiers used as the stable feature key for future pivots. Typically stores lower-snakecase values, but it can also store codes such as `USD`, `US`, or even tickers like `NVDA`.
- `name`: verbose identifier kept when a value is not derivable from the name_id column without special dependencies. For example, `US Dollar`, `United States`, or `NVIDIA CORPORATION`.

## Value Columns

- `value_text`: text value variant.
- `value_num`: numeric value variant.
- `value_date`: date or time value variant.
- `value_bool`: boolean value variant.

Many payload objects include multiple k/v variants for one value. Pick the most appropriate variant as the table's canon column. If a table requires multiple value variant columns, several value columns may be included.

## Additinal Value Columns (WIP)

- `vs_peers`: peer comparison value included in some source payloads (Use this exact col name across endpoints instead of endpoint-specific variants like `vs`, `vs_num`, or `peer_value`.)
- 

## Time And Capture Columns (WIP)

- `observed_at`: when the payload or source record was observed by collection. This is audit metadata, not a source for `effective_at`.
- `effective_at`: the source/payload-derived business date the normalized fact should be effective for point-in-time joins. It must never be the current date, collection date, or `observed_at`.
- `as_of_date`: endpoint-provided reporting date when the source gives one.
- `source_as_of_date`: raw-capture level source date when it should remain distinct from storage `effective_at`.
- `payload_hash`: reference to the raw payload bytes in `raw_payload_blobs`.
- `capture_batch_id`: optional identifier for the collection batch that produced the observation.

Keep these concepts separate. Do not reuse `effective_at` for source observation time, and do not force unrelated endpoint data onto another endpoint's date. If an endpoint payload does not contain a trustworthy date for canonical storage, skip or quarantine the canonical write instead of substituting `observed_at`.
