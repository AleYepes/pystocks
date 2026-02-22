# Discovered API Resources Map (Operational)

This map reflects current empirical behavior in the ETF universe and the default fetch strategy in `pystocks/fundamentals.py`.

## Tier A: Default Daily Snapshot (High Value / High Yield)

| Endpoint | Default | Purpose |
| --- | --- | --- |
| `landing` | Yes | Primary teaser probe and baseline metadata |
| `mf_profile_and_fees` | Yes | Fund profile, objective, fees, reports |
| `mf_holdings` | Yes | Sector/country/currency allocations |
| `dividends` | Conditional | Income history or no-dividend summary |
| `mstar/fund/detail` | Yes | Medalist/pillar ratings and commentary |
| `mf_ratios_fundamentals` | Conditional | Valuation/growth ratios and z-scores |
| `mf_lip_ratings` | Conditional | Lipper analyst-style ratings |
| `mf_performance` | Conditional + fallback | Returns metrics across available horizon |
| `mf_risks_stats` | Conditional + fallback | Risk/statistics metrics across available horizon |
| `sma/request?type=search` | Conditional | Historical sentiment bars (1Y window) |

## Tier B: Optional / Targeted (Low Yield in Current Universe)

| Endpoint | Default | Typical Issue |
| --- | --- | --- |
| `impact/esg` | No | Frequent 400, near-zero useful payload |
| `ownership` | No | Frequent 404 |
| `sma/request?type=tick` | No | Frequent 404, low useful payload |
| `sma/request?type=high_low` | No | Low-to-moderate yield, not essential for baseline |

## Fallback Strategy

For `mf_performance` and `mf_risks_stats`, periods are tried in descending order:

`10Y -> 5Y -> 3Y -> 1Y -> 6M`

First useful payload is kept; selected period is saved.

## Policy Source of Truth

Current policy was calibrated from `research_correlations.py` outputs:

- `data/research/research_yields.csv`
- `data/research/research_correlations_summary.csv`

Most recent large run used `sample_size=500`.

## Discovery Reconciliation (Feb 21, 2026)

Manual discovery logs (`166` JSON captures, ~`4.6 MB`) were reviewed against:

- `pystocks/fundamentals.py`
- `docs/3.IBKR_PORTAL_API.md`
- `docs/ibkr_json_examples.md`

### Accounted and Already Covered

- Core fundamentals endpoints (`landing`, `mf_profile_and_fees`, `mf_holdings`, `mf_ratios_fundamentals`, `mf_lip_ratings`, `dividends`, `mf_performance`, `mf_risks_stats`).
- Morningstar endpoint (`mstar/fund/detail` via captured `detail?conid=...` shape).
- Sentiment endpoints (`sma/request` variants: `search`, `tick`, `high_low`).
- Optional low-yield endpoints (`impact/esg`, `ownership`) already classified as opt-in/off-by-default.

### Accounted but Excluded from Daily Fundamentals Snapshot

- UI/session plumbing: `handshake`, `cp`, `categories`, `labels`, `layouts_templates`, `list`, `metadata`, `authAdd`.
- Account/position UX: `position`, `canTradeRecurringInvestment`, `recurringInvestment`, quote lookup (`?field_names=...`), `assetClasses`, `companies`.
- Lending/short-sale specialized data: `lending`, `widget`, `studyLine`, `lastLine`.
- Large chart series payloads (`{conid}?chart_period=MAX/3M`) not required for current fundamentals factor pipeline.

No additional endpoint family from discovery was promoted into default daily fundamentals fetch policy in this pass.
