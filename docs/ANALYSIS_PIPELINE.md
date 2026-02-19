# Analysis Pipeline: Multi-Factor Regression

The analysis layer is a sophisticated quantitative pipeline designed to model ETF returns through "Synthetic" and "External" factors.

## Core Components

### 1. Factor Construction
Factors are generated from the fundamental snapshots:
*   **Synthetic Factors:** Created by scaling fundamental metrics (e.g., `Price/Book`, `LTDebt/Shareholders`) and creating Long/Short portfolios.
*   **Classic Factors:** Market (Beta), SMB (Small Minus Big), HML (High Minus Low).
*   **Macro Factors:** GDP growth, Population change (World Bank), and Risk-Free rates (FRED).

### 2. ElasticNet Regression
Used for factor selection and regularization:
*   **Objective:** Identify the sparse set of factors that consistently explain returns across multiple ETF portfolios.
*   **Hyperparameters:** Rolling `alpha` and `l1_ratio` to balance Lasso (L1) and Ridge (L2) penalties.
*   **Pre-screening:** High-correlation factors (e.g., >0.95) are pruned to avoid multicollinearity.

### 3. Walk-Forward Analysis
The pipeline uses a rolling window approach:
*   **Training Period:** Typically 1–3 years of aligned fundamental and price data.
*   **Test Period:** Predict the following year's returns to validate factor stability.
*   **Goal:** Ensure factor exposures are persistent and not skewed by temporary market regimes (Stationarity check).

## Data Alignment (Point-in-Time)
To eliminate look-ahead bias, the new pipeline uses a "Point-in-Time" (PIT) join:
1.  Filter for the most recent fundamental snapshot where `as_of_date < current_training_window_start`.
2.  Align this snapshot with the corresponding historical price series.
3.  This ensures that the model only "knows" what was publicly available at the time of the training window.
