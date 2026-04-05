# Anomaly Review Report

## Scope

This review focused on the refactored analysis path in `pystocks/`, starting with `pystocks/preprocess/price.py` and comparing it to the legacy factor workflow in `src/analysis.py`.

The findings below are based on direct inspection of `data/pystocks.sqlite` on March 31, 2026.

## Price Series Findings

### 1. Decimal and scale glitches inside otherwise valid rows

These are the most important anomalies for the current factor pipeline.

Examples:

| Conid | Dates | Pattern |
| :--- | :--- | :--- |
| `110706696` | November 25, 2014 to November 28, 2014 | `27.303717 -> 0.027067 -> 27.354467` |
| `101674642` | August 18, 2016 to August 29, 2016 | `9.413479 -> 0.009265 -> 9.135541 -> 9.168629` |
| `105500788` | June 27, 2024 to July 2, 2024 | `507.90 -> 5.15 -> 519.30 -> 532.00` |
| `105951695` | August 16, 2012 to August 22, 2012 | `56.057815 -> 0.561496 -> 0.558796 -> 55.966032 -> 56.214386` |

Problem:

- The original refactor already removed the extreme down move and rebound rows when they showed up as return outliers.
- It did not remove the bad middle row when that row itself had only a small day-over-day return relative to another bad row.
- This left some corrupted prices marked as clean and allowed them to leak into downstream returns.

Solution implemented:

- Added `is_price_level_anomaly` to `pystocks/preprocess/price.py`.
- Added a bounded bridge-price check that looks for a short run of clean-looking rows trapped between two sign-reversing outlier returns.
- The bridge rows are compared to clean anchors outside the outlier span.
- This now catches the hidden middle bad row in the `105951695` case without stripping normal prices around it.

Result:

- The August 20, 2012 row for `105951695` is no longer treated as clean.
- Similar decimal-shift pockets are now covered by preprocessing.

### 2. Structurally broken rows with zero or negative prices

Examples:

| Conid | Dates | Pattern |
| :--- | :--- | :--- |
| `109961218` | From September 3, 2012 onward | tiny negative scientific-notation values and many `high < low` rows |
| `114850599` | March 9, 2010 onward | repeated zero close/low values alternating with valid-looking prices |
| `114853146` | March 9, 2010 onward | repeated zero close/low values alternating with valid-looking prices |
| `107113294` | October 2, 2019 onward | repeated zero rows inside an active series |

Observed counts across `price_chart_series`:

- 1,049,418 rows
- 338 conids
- 1,426 nonpositive rows
- 561 rows with `high < low`
- 27,929 stale-run rows beyond the 5-day cutoff

Problem:

- These rows are not isolated outliers. Some instruments are broken for long stretches or from inception.
- Row-level cleaning alone is not enough if the remaining history is sparse or fragmented.

Solution in current pipeline:

- Keep invalid-price filtering.
- Keep stale-run filtering.
- Keep eligibility gating by minimum history, missing ratio, and internal gap size.

Result:

- Obvious broken rows are excluded from `clean_price`.
- Many broken instruments are still excluded later by eligibility, not just row filtering.

Additional recommendation:

- Add an instrument-level quarantine rule for series with extreme invalid-row density or repeated zero/nonzero toggling.
- Prefer to flag these during ingestion rather than relying only on downstream eligibility.

### 3. Long stale runs and fragmented histories

Examples with large stale-run counts:

| Conid | Stale rows beyond 5-day cutoff | Max stale run |
| :--- | ---: | ---: |
| `105925326` | 1,935 | 79 |
| `103476642` | 1,909 | 122 |
| `105925323` | 1,902 | 79 |
| `110095665` | 1,799 | 85 |
| `113228457` | 1,706 | 103 |

Problem:

- Some ETFs have large flat stretches that make daily return features unreliable.
- The refactored pipeline does not regularize prices onto a business-day panel the way the legacy path did before factor work.

Solution in current pipeline:

- Continue dropping stale interior rows.
- Use eligibility rules to reject instruments with high missing ratios or large internal gaps.

Result:

- Several stale-heavy series become ineligible.
- This is acceptable for the current analysis build, but it is still a hard filter rather than a modeled treatment.

Additional recommendation:

- If analysis needs broader coverage, add a separate regularized return panel step after cleaning and before factor construction.
- Keep that separate from raw series cleaning.

### 4. Extreme return spikes

Examples:

| Conid | Date window | Max absolute return |
| :--- | :--- | ---: |
| `110706696` | November 2014 | `1009.625` |
| `101674642` | August 2016 | `985.071429` |
| `110408325` | November 2014 to January 2015 | `904.333333` |
| `105500788` | June 2024 to July 2024 | `99.834951` |
| `105951695` | August 2012 | `99.154589` |

Problem:

- These spikes are large enough to dominate factor returns if they survive cleaning.
- Positive-but-corrupted rows can bypass simple invalid-row checks.

Solution implemented:

- Keep robust return outlier detection.
- Add the bridge-price anomaly pass described above.

Result:

- The major observed spike pockets are now handled more safely.

## Snapshot and Holdings Findings

### 5. Holdings tables are not scale-consistent

The holdings tables do not share one safe interpretation of “weights sum to 1”.

Observed examples:

| Table | Median total per `(conid, effective_at)` | Max total |
| :--- | ---: | ---: |
| `holdings_geographic_weights` | `1.0000` | `1.0002` |
| `holdings_asset_type` | `1.0937` | `2.3770` |
| `holdings_debtor_quality` | `1.0364` | `2.8823` |
| `holdings_maturity` | `0.8210` | `2.5088` |
| `holdings_currency` | `1.0079` | `3.5884` |
| `holdings_investor_country` | `1.0005` | `5.2302` |
| `holdings_debt_type` | `1.4711` | `4.9487` |
| `holdings_industry` | `0.9992` | `2.9148` |

Problem:

- A generic “normalize all holdings tables” preprocessing step would be wrong.
- Some fields appear additive, some appear overlapping, and some likely contain duplicated exposure views.

Solution proposed:

- Do not fold this into `preprocess/price.py`.
- Add a separate snapshot preprocessing layer with explicit per-table rules.
- Treat `preprocess/price.py` as series-only preprocessing.

Reason:

- Price cleaning and snapshot denormalization have different failure modes and different correctness rules.
- Mixing them would make the pipeline harder to reason about.

### 6. Raw price anomaly signal is not persisted

In ingestion, `_extract_price_chart_rows` computes `debug_mismatch` from `x` vs `debugY`, but `price_chart_series` does not store that field.

Problem:

- Preprocessing cannot use this raw signal even though ingestion already detects it.
- The mismatch is only logged at ingest time.

Solution proposed:

- Persist `debug_mismatch` or an aggregated mismatch count into SQLite.
- Join it into series preprocessing as another anomaly flag.

Current workaround:

- The new bridge-price anomaly logic covers some of the same downstream symptoms, but it is still inferential.

## Important Changes Made During Review

### Implemented in preprocessing

- Added `local_price_ratio_threshold` and `bridge_outlier_span_max_rows` to `PricePreprocessConfig`.
- Added `is_price_level_anomaly` as an explicit row-level output.
- Tightened the clean-price definition to exclude bridge anomalies, not just invalid rows, stale rows, and direct return outliers.

### Implemented in analysis compatibility

- Normalized monthly rebalance frequency from `"M"` to `"ME"` for current pandas behavior.
- Fixed sort order before `merge_asof` in the price-feature join.

These were small but necessary to keep the current analysis tests and CLI path stable.

### Added tests

- Added a regression test for the hidden bridge anomaly case.
- Added a regression test to avoid stripping a split-like step change that should remain valid.

## Validation

Validation run:

- `./venv/bin/python -m pytest -q`

Result:

- `45 passed in 7.22s`

## Possible Errors

- The bridge-price anomaly rule is heuristic. It is designed for short decimal-shift pockets, not for every corporate action pattern.
- Some valid split-like or share-class adjustment events may still need explicit corporate-action handling later.
- Snapshot exposure tables likely mix multiple semantics. Totals above `1.0` are not automatically wrong, but they are unsafe to use without table-specific interpretation.
- The current eligibility rules may be too strict for thinly traded instruments and too loose for series with repeated structural faults.

## Suggested Next Steps

1. Add a separate snapshot preprocessing module for holdings, ratios, and point-in-time feature hygiene.
2. Persist `debug_mismatch` from price ingestion into SQLite.
3. Add instrument-level quarantine rules for series with repeated zero/nonzero toggling or extreme invalid-row density.
4. Decide whether analysis needs a regularized business-day return panel after cleaning.
5. Add tests for real anomaly windows from SQLite, not only synthetic fixtures.
6. Review holdings tables one by one and define which ones should sum to `1.0`, which should be left as-is, and which need deduplication or aggregation before factor use.
