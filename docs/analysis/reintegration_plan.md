# Re-integrate Legacy Research Features Into the Module Pipeline

## Summary
- Target the research pipeline first. Do not reintroduce the notebook’s portfolio optimization stage into the main implementation in this pass, but shape outputs so an optimizer module can consume them later without schema changes.
- Keep the current module split and preserve the new code’s strengths: normalized storage, point-in-time snapshot selection, explicit preprocess artifacts, and testable analysis stages.
- Reintroduce the legacy advantages in four ordered layers: supplementary data ingestion, point-in-time feature expansion, stricter walk-forward research, and richer diagnostics/outputs.
- Treat supplementary external data as a standard dependency. Fetch, cache, normalize, and preprocess it in the ingestion/storage pipeline rather than inside analysis ad hoc. Analysis should read only stored/preprocessed supplementary tables.

## Implementation Changes
- Add a supplementary-data ingestion module under `pystocks/ingest/` for risk-free inputs and World Bank indicators. It must fetch raw source data, record fetch metadata, and write normalized storage tables so analysis never calls external APIs directly.
- Add normalized storage for daily risk-free source series, a derived daily portfolio risk-free series, raw World Bank yearly indicator values, and a preprocessed country-year feature table. The preprocessed World Bank table must include at minimum `population`, `gdp_pcap`, `economic_output_gdp`, `foreign_direct_investment`, and `share_trade_volume`, each with level and growth features.
- Move the notebook’s World Bank preprocessing into a dedicated preprocess step. Keep these exact transformations: GDP-per-capita backfill from GDP/population, GDP converted to share of global GDP, trade-share feature derived from imports+exports, interpolation/extrapolation over yearly panels, and deterministic fill rules for remaining gaps.
- Extend snapshot-to-analysis feature construction so country holdings are collapsed into macro exposure features using stored country weights and the preprocessed World Bank table. The result must be point-in-time safe: for each rebalance date, only macro values available as of that date may be used.
- Reintroduce dynamic feature engineering that the current panel lacks. Add trailing return-stat features at the training cutoff (`momentum_3mo`, `momentum_6mo`, `momentum_1y`, `rs_3mo`, `rs_6mo`, `rs_1y`) and add rate-of-change fundamental features for the legacy profitability/growth families, including second-derivative features where the notebook used them.
- Add the notebook’s higher-level industry collapse back as explicit aggregate features. Implement `supersector_defensive`, `supersector_cyclical`, and `supersector_commodities` from holdings/industry columns using a fixed, versioned mapping in code.
- Keep the current price preprocess as the authoritative cleaner. Do not reintroduce notebook-style interpolation into stored clean prices. If aligned feature windows need denser coverage, implement bounded analysis-stage alignment on copies of return panels only, with explicit max-gap controls and no mutation of preprocess outputs.
- Replace the current “baseline-only excess return” framing with dual benchmarks. Research regressions must use risk-free excess returns as the primary target, while the existing bond baseline remains a secondary benchmarking artifact for bond-relative reporting.
- Replace the expanding-history train/test logic with fixed rolling walk-forward windows. Use configurable training window lengths with defaults matching the legacy research intent: 3-year and 4-year trailing train windows, stepped annually, with the next rebalance period as test.
- Keep the current factor clustering and persistence machinery, but feed it richer factor sets. Factor construction must include explicit benchmark factors first (`market_excess`, `SMB`, `HML/value-growth`) and then generic sleeve-specific long/short sort factors from raw/composite features.
- Tighten feature screening before factor construction. Use diagnostics already emitted by snapshot preprocess to drop structurally empty, constant, dummy-trap, and obviously redundant categories before factor generation rather than relying only on downstream clustering.
- Expand model outputs from the current thin summary to full research telemetry. Store Elastic Net hyperparameters chosen, nonzero-count, train/test MSE, train/test R², CV error summaries, and unscaled coefficients; optionally compute OLS/VIF diagnostics in a separate diagnostics table, not in the main selection path.
- Keep current factor persistence and current-beta estimation, but compute them from the reduced factor set after risk-free-based walk-forward research. Current beta fits should use the same factor naming scheme as research outputs so a future optimizer can consume them directly.

## Public Interfaces
- Extend `AnalysisConfig` with explicit controls for `training_window_years`, `walk_forward_step_months`, `use_risk_free_excess`, `require_supplementary_data`, `include_macro_features`, `include_dynamic_fundamental_trends`, and bounded return-panel alignment settings.
- Add CLI commands for supplementary refresh and preprocessing, for example a single `refresh_supplementary_data` entrypoint plus a `run_walk_forward_research` entrypoint if you want to keep it separate from the current `run_factor_research`.
- Keep existing parquet/SQLite outputs, and add new outputs for supplementary preprocess artifacts, macro-enriched panel features, factor diagnostics, and walk-forward model results. Name them as stable analysis tables rather than notebook-style one-off CSVs.
- Reserve optimizer-facing outputs now even though optimization is out of scope for this pass. The analysis layer should emit stable `asset_expected_returns`, `asset_factor_betas`, and factor covariance-compatible research outputs so the optimizer can be added without refactoring research tables.

## Test Plan
- Add ingestion/storage tests that verify supplementary fetch normalization, idempotent refresh behavior, point-in-time persistence, and World Bank preprocessing transforms including GDP-share and trade-share derivations.
- Add analysis tests proving that regressions use risk-free excess returns, not only the bond baseline, and that missing supplementary data fails fast when `require_supplementary_data=True`.
- Add feature tests for macro exposure collapse from country weights, trailing return-stat generation at a training cutoff, dynamic fundamental slope/second-derivative generation, and supersector aggregates.
- Add walk-forward tests for fixed 3-year and 4-year trailing windows, annual step behavior, correct train/test slicing, and absence of lookahead across rebalance dates.
- Add factor tests that verify canonical factors are emitted, generic factors remain sleeve-scoped, clustering still prefers composites where intended, and richer model diagnostics land in output tables with stable schemas.

## Assumptions And Defaults
- Research-first scope is the default. Portfolio optimization remains a follow-on phase that consumes the new research outputs instead of being rebuilt inside `pystocks.analysis` immediately.
- Supplementary external data is a required part of the standard pipeline, but it is fetched in ingestion and read from local storage during analysis.
- The current modular preprocess outputs remain the source of truth. Legacy behavior should be reintroduced by adding new modules and analysis-stage joins, not by collapsing back into a notebook-style monolith.
- The bond baseline stays in the system as a secondary benchmark because it is useful for sleeve-relative reporting, but it no longer defines the main regression target.
