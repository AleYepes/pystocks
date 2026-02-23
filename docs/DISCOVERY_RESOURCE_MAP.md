# Discovered API Resources Map (Operational)

This map reflects current empirical behavior in the ETF universe and the default fetch strategy in `pystocks/fundamentals.py`.

## Tier A: Default Daily Snapshot (High Value / High Yield)

| Endpoint | Default | Purpose |
| --- | --- | --- |
| `landing` | Yes | Primary teaser probe and baseline metadata |
| `mf_profile_and_fees` | Yes | Fund profile, objective, fees, reports |
| `mf_holdings` | Yes | Sector/country/currency allocations |
| `mf_ratios_fundamentals` | Yes | Valuation/growth ratios and z-scores |
| `mf_lip_ratings` | Yes | Lipper analyst-style ratings |
| `dividends` | Yes | Income history or no-dividend summary |
| `mstar/fund/detail` | Yes | Medalist/pillar ratings and commentary |
| `mf_performance_chart` | Yes | Long-range daily price series (`chart_period=MAX`) |
| `mf_performance` | Yes + fallback | Returns metrics across available horizon |
| `sma/request?type=search` | Yes | Historical sentiment bars (1Y window) |
| `ownership` | Yes | Ownership breakdown and trade log |
| `impact/esg` | Yes | ESG payload when available |

## Tier B: Optional / Targeted (Low Yield in Current Universe)

| Endpoint | Default | Typical Issue |
| --- | --- | --- |
| `sma/request?type=tick` | No | Frequent 404, low useful payload |
| `sma/request?type=high_low` | No | Low-to-moderate yield, not essential for baseline |
| `mf_risks_stats` | No | Removed; overlaps with performance and derivable from price series |

## Storage Policy (Current)

All fetched endpoints use the same two-layer policy:

1. raw response in CAS blobs (`data/fundamentals/blobs`)
2. analytics parquet representation (`data/fundamentals/parquet`)

Complex endpoint analytics behavior:

- `dividends`: canonical endpoint snapshot + normalized datasets `dividends_events`, `dividends_industry_metrics`; embedded price series is excluded from normalized analytics rows.
- `ownership`: canonical endpoint snapshot + normalized dataset `ownership_trade_log`; `NO CHANGE` actions are excluded; embedded ownership price series is excluded from normalized analytics rows.

## Fallback Strategy

For `mf_performance`, periods are tried in descending order:

`10Y -> 5Y -> 3Y -> 1Y -> 6M`

First useful payload is kept; selected period is saved.

## Policy Source of Truth

Current runtime policy is defined directly in `pystocks/fundamentals.py` and validated through DuckDB + telemetry inspection.

## Discovery Reconciliation (Feb 21, 2026)

Manual discovery logs (`166` JSON captures, ~`4.6 MB`) were reviewed against:

- `pystocks/fundamentals.py`
- `docs/3.IBKR_PORTAL_API.md`
- `docs/ibkr_json_examples.md`

### Accounted and Already Covered

- Core fundamentals endpoints (`landing`, `mf_profile_and_fees`, `mf_holdings`, `mf_ratios_fundamentals`, `mf_lip_ratings`, `dividends`, `mf_performance`).
- Price chart endpoint (`mf_performance_chart?chart_period=MAX`) now included and stored under `data/prices/ibkr_mf_performance_chart/`.
- Morningstar endpoint (`mstar/fund/detail` via captured `detail?conid=...` shape).
- Sentiment endpoints (`sma/request` variants: `search`, `tick`, `high_low`), with `tick/high_low` excluded from default runtime.
- `ownership` and `impact/esg` are now fetched in default runtime and persisted with CAS + parquet policy.

### Accounted but Excluded from Daily Fundamentals Snapshot

- UI/session plumbing: `handshake`, `cp`, `categories`, `labels`, `layouts_templates`, `list`, `metadata`, `authAdd`.
- Account/position UX: `position`, `canTradeRecurringInvestment`, `recurringInvestment`, quote lookup (`?field_names=...`), `assetClasses`, `companies`.
- Lending/short-sale specialized data: `lending`, `widget`, `studyLine`, `lastLine`.
- Non-fundamentals UI chart variants (`chart_period=3M` and other UI-only variants) are excluded from current fundamentals factor pipeline.

Short-selling endpoint family (`hmds/lastLine`, `hmds/studyLine?source=FeeRate`, `hmds/studyLine?source=Inventory`) remains deferred in this phase.

No additional endpoint family from discovery was promoted into default daily fundamentals fetch policy in this pass.
