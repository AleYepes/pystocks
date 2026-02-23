import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

from .config import RESEARCH_DIR, FUNDAMENTALS_DUCKDB_PATH

logger = logging.getLogger(__name__)

def _get_duckdb_con():
    return duckdb.connect(str(FUNDAMENTALS_DUCKDB_PATH), read_only=True)

def _load_returns(con):
    try:
        return con.execute("SELECT * FROM returns_daily_clean").fetch_df()
    except Exception:
        logger.warning("returns_daily_clean view not found.")
        return pd.DataFrame()

def _load_factor_panel(con):
    try:
        # Load only necessary features for factor construction to save memory
        query = """
        SELECT conid, trade_date, feature_name, feature_value 
        FROM factor_panel_long_daily
        WHERE feature_name IN ('marketcap_small', 'marketcap_large', 'style_value', 'style_growth')
        """
        return con.execute(query).fetch_df()
    except Exception:
        logger.warning("factor_panel_long_daily view not found.")
        return pd.DataFrame()

def _identify_bond_universe(con):
    try:
        query = """
        SELECT DISTINCT conid 
        FROM factor_features_latest 
        WHERE feature_name = 'asset_bond' AND feature_value >= 0.9
        """
        df = con.execute(query).fetch_df()
        return df['conid'].tolist()
    except Exception:
        return []

def _compute_risk_free_rate(returns_df, bond_conids):
    dates = returns_df['trade_date'].unique()
    dates.sort()
    
    if not bond_conids:
        logger.warning("No Bond ETFs found for risk-free rate. Using 0.0.")
        return pd.DataFrame({'trade_date': dates, 'daily_nominal_rate': 0.0})

    bond_returns = returns_df[returns_df['conid'].isin(bond_conids)]
    if bond_returns.empty:
        logger.warning("No returns found for Bond ETFs. Using 0.0.")
        return pd.DataFrame({'trade_date': dates, 'daily_nominal_rate': 0.0})
        
    # Average daily return of bond universe
    rf_daily = bond_returns.groupby('trade_date')['pct_change'].mean().reset_index()
    rf_daily.rename(columns={'pct_change': 'daily_nominal_rate'}, inplace=True)
    
    # Fill missing dates with 0.0
    rf_daily = rf_daily.set_index('trade_date').reindex(dates, fill_value=0.0).reset_index()
    
    return rf_daily

def _construct_factor_returns(returns_df, panel_df, rf_df, bond_conids):
    if returns_df.empty:
        return pd.DataFrame()

    pivoted_returns = returns_df.pivot(index='trade_date', columns='conid', values='pct_change')
    all_conids = pivoted_returns.columns.tolist()
    equity_conids = [c for c in all_conids if c not in bond_conids] 
    
    # Market Factor (Mkt-RF)
    if equity_conids:
        market_returns = pivoted_returns[equity_conids].mean(axis=1)
    else:
        market_returns = pd.Series(0.0, index=pivoted_returns.index)
        
    factors = pd.DataFrame(index=pivoted_returns.index)
    
    rf_df_indexed = rf_df.set_index('trade_date')
    factors = factors.join(rf_df_indexed)
    factors['daily_nominal_rate'] = factors['daily_nominal_rate'].fillna(0.0)
    
    factors['Mkt-RF'] = market_returns - factors['daily_nominal_rate']
    
    if panel_df.empty:
        return factors.fillna(0.0)

    # Prepare features for SMB/HML
    # We assume panel_df covers the trade dates in returns_df
    # We need to construct Small/Large and Value/Growth portfolios day-by-day.
    # Joining panel_df with returns_df is memory intensive if done fully.
    # But since we filtered panel_df features, it might be okay.
    
    # Let's pivot panel_df to: index=[trade_date, conid], columns=feature_name
    feature_wide = panel_df.pivot_table(
        index=['trade_date', 'conid'], 
        columns='feature_name', 
        values='feature_value'
    )
    
    # Melt returns to align with features
    returns_long = returns_df.set_index(['trade_date', 'conid'])[['pct_change']]
    
    # Join
    data = feature_wide.join(returns_long, how='inner')
    
    if data.empty:
        factors['SMB'] = 0.0
        factors['HML'] = 0.0
        return factors.fillna(0.0)

    # Group by date to calculate factor returns
    grouped = data.groupby(level='trade_date')
    
    # Use aggregation to speed up
    def calc_spread(grp, long_col, short_col):
        long_mask = grp[long_col] > 0 if long_col in grp else pd.Series(False, index=grp.index)
        short_mask = grp[short_col] > 0 if short_col in grp else pd.Series(False, index=grp.index)
        
        r_long = grp.loc[long_mask, 'pct_change'].mean() if long_mask.any() else 0.0
        r_short = grp.loc[short_mask, 'pct_change'].mean() if short_mask.any() else 0.0
        return r_long - r_short

    smb_series = {}
    hml_series = {}

    # It's faster to iterate if we can't vectorize the masks easily
    # But groupby apply is slow.
    # Let's try to do it via conditional means.
    
    # Actually, iterative is fine for daily data (~5000 days max).
    for date, group in grouped:
        smb = calc_spread(group, 'marketcap_small', 'marketcap_large')
        hml = calc_spread(group, 'style_value', 'style_growth')
        smb_series[date] = smb
        hml_series[date] = hml
        
    factors['SMB'] = pd.Series(smb_series)
    factors['HML'] = pd.Series(hml_series)
    
    return factors.fillna(0.0)

def _screen_factors(factors_df):
    df = factors_df.drop(columns=['daily_nominal_rate', 'trade_date'], errors='ignore')
    if df.empty:
        return factors_df
        
    corr_matrix = df.corr().abs()
    drop_cols = set()
    cols = df.columns.tolist()
    
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1 = cols[i]
            c2 = cols[j]
            if c1 in drop_cols or c2 in drop_cols:
                continue
            
            if corr_matrix.loc[c1, c2] > 0.95:
                # Deterministic drop: drop the second one
                drop_cols.add(c2)
                
    return factors_df.drop(columns=list(drop_cols))

def _run_regression(returns_df, factors_df):
    if returns_df.empty or factors_df.empty:
        return pd.DataFrame()
        
    pivoted = returns_df.pivot(index='trade_date', columns='conid', values='pct_change')
    
    # Align dates
    common_dates = pivoted.index.intersection(factors_df.index)
    if len(common_dates) < 60:
        logger.warning("Insufficient common history for regression.")
        return pd.DataFrame()
        
    Y = pivoted.loc[common_dates]
    
    # Factors
    X_raw = factors_df.loc[common_dates].drop(columns=['daily_nominal_rate', 'trade_date'], errors='ignore')
    if X_raw.empty:
         return pd.DataFrame()
         
    rf = factors_df.loc[common_dates, 'daily_nominal_rate']
    
    # Y excess
    Y_excess = Y.sub(rf, axis=0)
    
    # Fill NaNs in Y (missing prices)
    # We should probably drop rows where Y is NaN for a specific asset during fit,
    # or fill with 0 if we assume no return? No, fill with 0 biases beta to 0.
    # Better to drop NaNs per asset.
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_scaled_df = pd.DataFrame(X_scaled, index=X_raw.index, columns=X_raw.columns)
    
    results = []
    
    for conid in Y_excess.columns:
        y_series = Y_excess[conid].dropna()
        if len(y_series) < 60:
            continue
            
        # Align X to this asset's valid dates
        valid_dates = y_series.index
        X_sub = X_scaled_df.loc[valid_dates]
        
        try:
            # ElasticNetCV
            model = ElasticNetCV(cv=3, random_state=42, max_iter=2000)
            model.fit(X_sub, y_series)
            
            res = {
                "conid": conid,
                "alpha": model.intercept_,
                "r2": model.score(X_sub, y_series),
                "n_obs": len(y_series)
            }
            
            for i, col in enumerate(X_raw.columns):
                res[f"beta_{col}"] = model.coef_[i]
                
            results.append(res)
        except Exception:
            pass
            
    return pd.DataFrame(results)

def run():
    con = _get_duckdb_con()
    try:
        returns_df = _load_returns(con)
        panel_df = _load_factor_panel(con)
        bond_conids = _identify_bond_universe(con)
    finally:
        con.close()
        
    if returns_df.empty:
        logger.warning("No returns data available.")
        return {"status": "empty"}
        
    rf_df = _compute_risk_free_rate(returns_df, bond_conids)
    
    factors_df = _construct_factor_returns(returns_df, panel_df, rf_df, bond_conids)
    
    screened_factors = _screen_factors(factors_df)
    
    betas_df = _run_regression(returns_df, screened_factors)
    
    # Save artifacts
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    
    factors_path = RESEARCH_DIR / "factor_returns_daily_latest.parquet"
    betas_path = RESEARCH_DIR / "asset_factor_betas_latest.parquet"
    summary_path = RESEARCH_DIR / "analysis_summary_latest.json"
    
    screened_factors.to_parquet(factors_path)
    betas_df.to_parquet(betas_path)
    
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe_size": len(betas_df),
        "factors": list(screened_factors.columns),
        "bond_universe_size": len(bond_conids),
        "rf_mean": rf_df['daily_nominal_rate'].mean() if not rf_df.empty else 0.0
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    return {
        "status": "success",
        "artifacts": [str(factors_path), str(betas_path), str(summary_path)]
    }

if __name__ == "__main__":
    run()
