# Discovered API Resources Map (Operational)

This map reflects current empirical behavior in the ETF universe and the default fetch strategy in `pystocks/fundamentals.py`.

## Tier A: Default Daily Snapshot (High Value / High Yield)

| Endpoint | Default | Purpose |
| --- | --- | --- |
| `landing` | Yes | Primary teaser probe and baseline metadata |
| `mf_profile_and_fees` | Yes | Fund profile, objective, fees, reports |
| `mf_holdings` | Yes | Sector/country/currency allocations |
| `dividends` | Yes | Income history or no-dividend summary |
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

- `research_yields.csv`
- `research_correlations_summary.csv`

Most recent large run used `sample_size=500`.
