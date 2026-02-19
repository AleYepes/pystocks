# Project Overview: PyStocks

PyStocks is a quantitative investment pipeline designed to fetch fundamental data for a vast universe of ETFs, perform multi-factor regression, and optimize portfolios using ElasticNet and walk-forward analysis.

## Key Goals
1. **Universe Management:** Maintain a registry of 17,000+ IBKR-listed ETF instruments.
2. **Fundamental Data Extraction:** Historically performed via fragile OCR on Trader Workstation (TWS). Now transitioning to a high-speed JSON interception strategy using the IBKR Portal Proxy.
3. **TimeSeries Management:** Consolidate historical price data from the IBKR API.
4. **Quantitative Factor Analysis:** Construct "synthetic" factors (Value, Momentum, Quality, etc.) and macro-factors (GDP, Population, Risk-Free rates) to model ETF returns.
5. **Portfolio Optimization:** Identify consistently influential factors across rolling training windows to predict future returns.

## Core Transformation
The project is currently transitioning from a **monolithic, OCR-heavy** approach (`/src`) to a **modular, API-driven** architecture (`/pystocks`). The discovery of direct JSON endpoints for fundamental data (Holdings, Ratios, Profiles) is the primary catalyst for this shift.
