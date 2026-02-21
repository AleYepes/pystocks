# Discovered API Resources Map: Tiered Taxonomy

This document maps IBKR Portal API endpoints to their functional roles in the quantitative pipeline and their operational handling requirements.

## Tier 1: Daily Factor Snapshots
**Handling**: Fetch daily. Store as JSON snapshots.
**Operational Cost**: Low (~10-50KB per instrument).
**Analysis Value**: Provides "Cross-Sectional" features for the current trading day.

| Functional Group | Endpoint | Key Quantitative Features |
| --- | --- | --- |
| **Valuation** | `mf_ratios_fundamentals` | P/E, P/S, P/B, Dividend Yield, Z-Scores. |
| **Composition** | `mf_holdings` | Sector (Industry) weights, Country exposure, Currency exposure. |
| **Profile** | `mf_profile_and_fees` | Expense Ratio (TER), Management Fees, Domicile (Tax factor). |
| **ESG** | `impact/esg` | Refinitiv ESG scores, Carbon Risk, Social/Governance pillars. |
| **Identity** | `landing` | Benchmark mapping, Fund Objective, Asset Class. |

## Tier 2: Periodic Historical Metrics
**Handling**: Fetch monthly/quarterly. Store as compressed time-series.
**Operational Cost**: Medium (~100-500KB per instrument).
**Analysis Value**: Provides "Longitudinal" features (Momentum, Stability, Trends).

| Functional Group | Endpoint | Key Quantitative Features |
| --- | --- | --- |
| **Momentum** | `mf_performance` | 1Y/3Y/5Y Cumulative & Annualized returns vs. Benchmark. |
| **Risk** | `mf_risks_stats` | Sharpe Ratio, Std Dev (Volatility), VaR, Max Drawdown. |
| **Analyst Sent.** | `mf_lip_ratings` | Lipper ratings (Total Return, Consistency, Preservation). |
| **Analyst Sent.** | `mstar/fund/detail`| Morningstar Medalist rating, Pillar scores (People, Process). |
| **Income** | `dividends` | Historical dividend growth rates, payout frequency. |
| **Flows** | `ownership` | Institutional vs. Insider ownership trends, Trade logs. |

## Tier 3: High-Resolution Market Data
**Handling**: Fetch on-demand/One-time. Store as partitioned Parquet.
**Operational Cost**: High (1MB+ per instrument).
**Analysis Value**: Provides raw inputs for alpha generation and model training.

| Functional Group | Endpoint | Usage Strategy |
| --- | --- | --- |
| **Price/Volume** | `chart?period=MAX` | Primary input for all price-based regression factors. |
| **Market Dynamics**| `lending` | Short interest proxy (Utilization, Lender Depth). |
| **Cost-of-Trade** | `studyLine?source=FeeRate`| Borrow fee history (Cost of shorting/Carry factor). |
| **Social Sent.** | `sma/request` | Sentiment buzz/volume (High-frequency alpha signal). |

---

## Analysis Workflow: Mapping Features to Models

| Factor Category | Source Tier | Pipeline Implementation |
| --- | --- | --- |
| **Value** | Tier 1 | PIT join of `mf_ratios_fundamentals` to Training Window. |
| **Quality** | Tier 1 & 2 | Combine `zscores` with Lipper `Consistency` ratings. |
| **Momentum** | Tier 2 & 3 | Extract periodic returns from `mf_performance` or `chart`. |
| **Sentiment** | Tier 2 & 3 | Join `sma/request` scores with Morningstar analyst pillars. |
| **Risk-Adjusted** | Tier 2 | Use `Sharpe` and `VaR` from `mf_risks_stats` for portfolio optimization. |
