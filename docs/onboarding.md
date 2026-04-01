# Agent Onboarding

## What This Repo Is

`pystocks/` is the current production codebase for an ETF ingestion, preprocessing, and factor-research pipeline.

Use `pystocks/` as the source of truth.

Treat these as reference only unless the task explicitly asks for them:

- `src/`
- `notebooks/`

## Current Pipeline

1. Session/auth: `pystocks/session.py`
2. Product universe scrape: `pystocks/product_scraper.py`
3. Fundamentals and series scrape: `pystocks/fundamentals.py`
4. SQLite materialization: `pystocks/fundamentals_store.py`
5. Series preprocessing:
   - prices: `pystocks/preprocess/price.py`
   - dividends: `pystocks/preprocess/dividends.py`
6. Snapshot-feature preprocessing:
   - dated feature tables: `pystocks/preprocess/snapshots.py`
7. Analysis and factor research: `pystocks/analysis.py`

Compatibility wrappers still exist where needed, for example `pystocks/price_preprocess.py`.

## Data Model

Canonical store:

- `data/pystocks.sqlite`

Analysis artifacts:

- `data/analysis/`

Important table families in SQLite:

- `products`
- raw payload/blob linkage
- endpoint snapshot metadata: `*_snapshots`
- normalized endpoint tables such as:
  - `profile_and_fees`
  - `holdings_*`
  - `ratios_*`
  - `performance`
  - `dividends_industry_metrics`
  - `morningstar_summary`
  - `lipper_ratings`
- series tables such as:
  - `price_chart_series`
  - `dividends_events_series`
  - `sentiment_series`

Important distinction:

- Raw snapshot metadata: the `*_snapshots` tables and payload blobs
- Snapshot features: dated analytical state keyed by `(conid, effective_at)`
- Series features: full vectors such as prices, dividends, and sentiment

“Snapshot” in current analysis work means dated analytical feature state, not the raw JSON blobs.

## Current Analysis Model

The pipeline is building intermediate inputs for factor research, not end-user prediction models for every dataset.

Primary roles:

- Prices:
  - clean return infrastructure
  - eligibility
  - price-derived features
- Snapshot features:
  - rebalance-date state for cross-sectional sorting
  - examples: P/E, holdings exposures, fees, AUM
- Dividends:
  - supplementary support for total-return adjustment
  - not yet integrated into price preprocessing output
- Sentiment:
  - supplementary series feature family
  - not yet preprocessed

The factor workflow is:

1. take the latest valid snapshot feature row at or before each rebalance date
2. rank the sleeve cross section on a feature
3. build long/short factor portfolios from eligible ETFs
4. use clean return series to produce factor return series
5. cluster, score persistence, and estimate current betas

## Preprocessing Status

Implemented:

- `pystocks/preprocess/price.py`
  - invalid-price filtering
  - stale-run detection
  - robust return outliers
  - short bridge-price anomaly detection
  - eligibility outputs
- `pystocks/preprocess/dividends.py`
  - event loading
  - currency mismatch flags
  - duplicate detection
  - implied-yield checks
  - usable-for-total-return flags
- `pystocks/preprocess/snapshots.py`
  - merged dated feature output
  - holdings diagnostics
  - ratio diagnostics
  - passthrough handling for deferred snapshot families

Still pending or incomplete:

- dividend integration into total-return price preprocessing
- sentiment preprocessing
- persisted `debug_mismatch` from ingestion into price preprocessing
- instrument-level quarantine rules
- decision on regularized business-day return panels
- deeper review of holdings-table semantics
- explicit factor registry

## Where To Look First

If the task is about:

- ingestion/storage:
  - `pystocks/fundamentals_store.py`
  - endpoint storage tests under `pystocks/tests/`
- price anomalies or return prep:
  - `pystocks/preprocess/price.py`
  - `pystocks/tests/test_price_preprocess.py`
- dividend event quality:
  - `pystocks/preprocess/dividends.py`
  - `pystocks/tests/test_dividend_preprocess.py`
- snapshot feature assembly:
  - `pystocks/preprocess/snapshots.py`
  - `pystocks/tests/test_snapshot_preprocess.py`
- factor research behavior:
  - `pystocks/analysis.py`
  - `pystocks/tests/test_analysis_pipeline.py`

## Key Docs

- [analysis_plan.md](/home/alex/Documents/pystocks/docs/analysis_plan.md)
- [preprocess_reorg_plan.md](/home/alex/Documents/pystocks/docs/preprocess_reorg_plan.md)
- [snapshot_preprocess_plan.md](/home/alex/Documents/pystocks/docs/snapshot_preprocess_plan.md)
- [anomaly_review_report.md](/home/alex/Documents/pystocks/docs/anomaly_review_report.md)

Read those before making architectural changes.

## Common Commands

```bash
./venv/bin/python -m pystocks.cli scrape_products
./venv/bin/python -m pystocks.cli scrape_fundamentals --limit 100 --verbose
./venv/bin/python -m pystocks.cli preprocess_prices
./venv/bin/python -m pystocks.cli preprocess_dividends
./venv/bin/python -m pystocks.cli preprocess_snapshots
./venv/bin/python -m pystocks.cli build_analysis_panel
./venv/bin/python -m pystocks.cli run_factor_research
./venv/bin/python -m pystocks.cli run_analysis
./venv/bin/python -m pystocks.cli run_pipeline --limit 100
```

`run_pipeline` currently runs:

1. products
2. fundamentals
3. price preprocessing
4. analysis

It does not currently run dividend or snapshot preprocessing as standalone steps.

## Working Rules

- Do not add migrations or backfill logic unless explicitly asked.
- Prefer extending `pystocks/preprocess/` over adding more root-level scripts.
- Keep price, dividend, sentiment, and snapshot preprocessing concerns separate.
- Prefer diagnostics and explicit flags over silent normalization when table semantics are ambiguous.
- Do not rewrite user changes you did not make.

## Validation Before Handoff

Always run:

```bash
./venv/bin/python -m pytest -q
```

If storage/view logic changed, also run:

```bash
./venv/bin/python -m pystocks.cli refresh_fundamentals_views
```
