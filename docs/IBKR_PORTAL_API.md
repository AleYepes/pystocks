# IBKR Portal API Discovery

The project has moved from fragile OCR to direct JSON interception using the IBKR Portal Proxy. These endpoints are available within a logged-in IBKR web session.

## Core Endpoints
Base URL: `https://www.interactivebrokers.ie/tws.proxy/fundamentals/`

### 1. `mf_holdings/{conId}`
**Returns:** Detailed holdings data.
*   `allocation_self`: Asset class weights (Equity, Cash, Bond).
*   `top10`: Top 10 holdings with `conIds` and `assets_pct`.
*   `industry`: Sector weights (e.g., Technology, Healthcare).
*   `investor_country`: Geographic exposure weights (e.g., US, IE, GB).

### 2. `mf_profile_and_fees/{conId}`
**Returns:** Fund profile and expense details.
*   `fund_and_profile`: Key metadata (Domicile, TER, Benchmark).
*   `reports`: Annual and Prospectus report summaries.
*   `expenses_allocation`: Management vs. Non-management expenses.

### 3. `landing/{conId}?widgets={...}`
**Returns:** A comprehensive overview "widget" containing multiple sections.
*   Useful for a single "bulk fetch" of fundamental data.
*   Widgets include: `objective`, `mstar`, `lipper_ratings`, `mf_key_ratios`, `risk_and_statistics`, `holdings`, `performance_and_peers`, `keyProfile`, `ownership`, `dividends`.

## Authentication Strategy
1.  **Playwright Session:** Perform a manual or automated login to the IBKR portal.
2.  **Cookie Persistence:** Save the authentication state (`auth_state.json`).
3.  **Httpx Client:** Use a persistent `httpx.AsyncClient` with the saved cookies to fetch JSON directly.
4.  **Error Handling:** Handle session timeouts (401/403) by re-triggering the Playwright login flow.
