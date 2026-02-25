# Operational Checklist

Use this as the default runbook for SQLite-first ingestion.

## 1) Preflight
1. Ensure dependencies are available:
```bash
./venv/bin/python -m pytest -q
```
2. Confirm auth state file exists:
```bash
test -f data/auth_state.json && echo "auth_state.json present"
```
3. Confirm SQLite DB exists and has products table:
```bash
./venv/bin/python - <<'PY'
import sqlite3
con = sqlite3.connect('data/pystocks.sqlite')
con.execute('SELECT 1 FROM products LIMIT 1')
print('products table ready')
con.close()
PY
```

## 2) Workflows
1. Refresh product universe only:
```bash
./venv/bin/python -m pystocks.cli scrape_products
```
2. Run fundamentals ingest only:
```bash
./venv/bin/python -m pystocks.cli scrape_fundamentals --limit 100 --verbose
```
3. Run SQLite maintenance checkpoint/vacuum:
```bash
./venv/bin/python -m pystocks.cli refresh_fundamentals_views
```
4. Full ingestion pipeline:
```bash
./venv/bin/python -m pystocks.cli run_pipeline --limit 100
```

Important:
- `run_pipeline` currently runs only products + fundamentals.
- `preprocess_prices` and `run_analysis` are deferred in this refactor.

## 3) Post-run validation
1. Check telemetry artifacts:
```bash
jq '{run_started_at, run_finished_at, run_stats}' data/research/fundamentals_run_telemetry_latest.json
```
2. Check core SQLite table counts:
```bash
./venv/bin/python - <<'PY'
import sqlite3
con = sqlite3.connect('data/pystocks.sqlite')
checks = [
    'products',
    'raw_payload_blobs',
    'landing_snapshots',
    'profile_fees_snapshots',
    'holdings_snapshots',
    'ratios_snapshots',
    'lipper_ratings_snapshots',
    'dividends_snapshots',
    'morningstar_snapshots',
    'performance_snapshots',
    'ownership_snapshots',
    'esg_snapshots',
    'price_chart_snapshots',
    'sentiment_search_snapshots',
    'price_chart_series_raw',
    'price_chart_series_latest',
    'sentiment_search_series_raw',
    'sentiment_search_series_latest',
    'ownership_trade_log_series_raw',
    'ownership_trade_log_series_latest',
    'dividends_events_series_raw',
    'dividends_events_series_latest',
]
for t in checks:
    n = con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
    print(t, n)
con.close()
PY
```
3. Integrity checks:
```bash
./venv/bin/python - <<'PY'
import sqlite3
con = sqlite3.connect('data/pystocks.sqlite')
print('ownership_no_change_latest', con.execute("""
SELECT COUNT(*) FROM ownership_trade_log_series_latest
WHERE upper(action) = 'NO CHANGE'
""").fetchone()[0])
print('price_raw_minus_latest', con.execute("""
SELECT
  (SELECT COUNT(*) FROM price_chart_series_raw)
  -
  (SELECT COUNT(*) FROM price_chart_series_latest)
""").fetchone()[0])
con.close()
PY
```

Expected:
- `ownership_no_change_latest = 0`
- `price_raw_minus_latest >= 0`

## 4) Recovery paths
1. `No products found in SQLite products table`:
- Run `scrape_products` first.
2. Empty endpoint snapshot tables:
- Rerun `scrape_fundamentals` with a limit and `--verbose`.
3. Auth failures:
- Rerun fundamentals scrape and complete interactive reauthentication.

## 5) Legacy stores
Old DuckDB/parquet artifacts may still exist under `data/` but are legacy and not read by current ingestion runtime.
