# Operational Checklist

Use this as the default runbook for ingestion, view refresh, and tail analysis.

## 1) Preflight
1. Ensure dependencies are available:
```bash
./venv/bin/python -m pytest -q
```
2. Confirm auth state file exists:
```bash
test -f data/auth_state.json && echo "auth_state.json present"
```
3. Confirm product universe exists in DuckDB:
```bash
./venv/bin/python - <<'PY'
import duckdb
con = duckdb.connect("data/fundamentals/fundamentals.duckdb", read_only=True)
print("instruments:", con.execute("SELECT COUNT(*) FROM instruments").fetchone()[0])
con.close()
PY
```

## 2) Pick the Right Workflow
1. Refresh product universe only:
```bash
./venv/bin/python -m pystocks.cli scrape_products
```
2. Run fundamentals ingest only:
```bash
./venv/bin/python -m pystocks.cli scrape_fundamentals --limit 100 --verbose
```
3. Refresh DuckDB views only:
```bash
./venv/bin/python -m pystocks.cli refresh_fundamentals_views
```
4. Rebuild tail outputs only:
```bash
./venv/bin/python -m pystocks.cli preprocess_prices
./venv/bin/python -m pystocks.cli run_analysis
```
5. Full pipeline:
```bash
./venv/bin/python -m pystocks.cli run_pipeline --limit 100
```

Important:
- `run_pipeline` is not full-universe by default; default `limit=100`.
- For broad coverage, increase `--limit` or run `scrape_fundamentals` directly without a limit.

## 3) Post-Run Validation
1. Check telemetry and research artifacts:
```bash
jq '{run_started_at, run_finished_at, run_stats}' data/research/fundamentals_run_telemetry_latest.json
jq '{generated_at, summary}' data/research/price_quality_report_latest.json
jq '{generated_at, regressed_assets, factors_built}' data/research/analysis_summary_latest.json
```
2. Check core table/view counts:
```bash
./venv/bin/python - <<'PY'
import duckdb
con = duckdb.connect("data/fundamentals/fundamentals.duckdb", read_only=True)
checks = {
    "endpoint_events_all": "SELECT COUNT(*) FROM endpoint_events_all",
    "factor_features_all": "SELECT COUNT(*) FROM factor_features_all",
    "price_chart_series_all": "SELECT COUNT(*) FROM price_chart_series_all",
    "price_chart_series_clean_all": "SELECT COUNT(*) FROM price_chart_series_clean_all",
    "returns_daily_clean": "SELECT COUNT(*) FROM returns_daily_clean",
    "eligible_conids": "SELECT COUNT(*) FROM price_quality_catalog WHERE eligible = TRUE",
}
for name, sql in checks.items():
    print(name, con.execute(sql).fetchone()[0])
con.close()
PY
```
3. Integrity checks with clear pass criteria:
```bash
./venv/bin/python - <<'PY'
import duckdb
con = duckdb.connect("data/fundamentals/fundamentals.duckdb", read_only=True)
print("no_change_rows", con.execute(\"\"\"
SELECT COUNT(*) FROM ownership_trade_log_series_all
WHERE upper(action) = 'NO CHANGE'
\"\"\").fetchone()[0])
print("lineage_missing_or_mismatch", con.execute(\"\"\"
SELECT
  SUM(CASE WHEN e.event_id IS NULL THEN 1 ELSE 0 END) +
  SUM(CASE WHEN e.payload_hash IS DISTINCT FROM f.payload_hash THEN 1 ELSE 0 END)
FROM factor_features_all f
LEFT JOIN endpoint_events_all e
  ON f.endpoint_event_id = e.event_id
\"\"\").fetchone()[0])
print("forbidden_sentiment_cols", con.execute(\"\"\"
SELECT COUNT(*)
FROM information_schema.columns
WHERE table_name = 'sentiment_search_series_all'
  AND column_name IN ('price','open','high','low','close','price_change','price_change_p')
\"\"\").fetchone()[0])
con.close()
PY
```
Expected:
- `no_change_rows = 0`
- `lineage_missing_or_mismatch = 0`
- `forbidden_sentiment_cols = 0`

Interpret with caution:
- `weekend_rows` and timestamp/date mismatches in raw price series can be non-zero in current data; track trend, do not hard-fail solely on these counts.

## 4) Common Recovery Paths
1. `No products found in DuckDB instruments table`:
- Run `scrape_products` first.
2. Missing views (`factor_panel_long_daily`, `returns_daily_clean`, etc.):
- Run `refresh_fundamentals_views`.
3. Empty analysis output:
- Check `price_quality_catalog` eligible count and rerun `preprocess_prices` then `run_analysis`.
4. Auth failures:
- Rerun fundamentals scrape and complete interactive reauthentication when prompted.

## 5) Rebuild Guidance
Prefer non-destructive rebuilds first:
1. Views only: `refresh_fundamentals_views`
2. Tail only: `preprocess_prices` + `run_analysis`
3. Ingestion rerun: `scrape_products` + `scrape_fundamentals`

Only delete data directories when a full reset is explicitly required.
