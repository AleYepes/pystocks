import json
import logging
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

from .config import FUNDAMENTALS_DUCKDB_PATH, RESEARCH_DIR

logger = logging.getLogger(__name__)

MIN_REGRESSION_DAYS = 60
MIN_SIDE_MEMBERS = 3


def _get_duckdb_con():
    return duckdb.connect(str(FUNDAMENTALS_DUCKDB_PATH), read_only=True)


def _load_returns(con):
    try:
        df = con.execute("SELECT conid, trade_date, pct_change FROM returns_daily_clean").fetch_df()
    except Exception:
        logger.warning("returns_daily_clean view not found.")
        return pd.DataFrame()

    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df["pct_change"] = pd.to_numeric(df["pct_change"], errors="coerce")
    df["conid"] = df["conid"].astype(str)
    return df.dropna(subset=["trade_date"])


def _load_factor_panel(con):
    try:
        query = """
        SELECT conid, trade_date, feature_name, feature_value
        FROM factor_panel_long_daily
        WHERE feature_name LIKE 'marketcap_%'
           OR feature_name LIKE 'style_%'
        """
        df = con.execute(query).fetch_df()
    except Exception:
        logger.warning("factor_panel_long_daily view not found.")
        return pd.DataFrame()

    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df["feature_value"] = pd.to_numeric(df["feature_value"], errors="coerce")
    df["conid"] = df["conid"].astype(str)
    return df.dropna(subset=["trade_date"])


def _load_eligible_conids(con):
    try:
        df = con.execute(
            """
            SELECT DISTINCT conid
            FROM price_quality_catalog
            WHERE eligible = TRUE
            """
        ).fetch_df()
        return set(df["conid"].astype(str))
    except Exception:
        return set()


def _identify_bond_universe(con):
    try:
        df = con.execute(
            """
            SELECT DISTINCT conid
            FROM factor_features_latest
            WHERE feature_name = 'asset_bond'
              AND feature_value >= 0.9
            """
        ).fetch_df()
        return set(df["conid"].astype(str))
    except Exception:
        return set()


def _compute_risk_free_rate(returns_df, bond_conids):
    dates = pd.Index(sorted(returns_df["trade_date"].unique()), name="trade_date")

    if not bond_conids:
        logger.warning("No bond universe found, using 0.0 risk-free fallback.")
        return pd.DataFrame({"trade_date": dates, "daily_nominal_rate": 0.0})

    bond_returns = returns_df[returns_df["conid"].isin(bond_conids)].copy()
    if bond_returns.empty:
        logger.warning("No bond returns found, using 0.0 risk-free fallback.")
        return pd.DataFrame({"trade_date": dates, "daily_nominal_rate": 0.0})

    rf = (
        bond_returns.groupby("trade_date", as_index=False)["pct_change"]
        .mean()
        .rename(columns={"pct_change": "daily_nominal_rate"})
    )
    rf = rf.set_index("trade_date").reindex(dates, fill_value=0.0).reset_index()
    return rf


def _spread_return(group, long_mask, short_mask):
    if int(long_mask.sum()) < MIN_SIDE_MEMBERS or int(short_mask.sum()) < MIN_SIDE_MEMBERS:
        return np.nan
    long_ret = group.loc[long_mask, "pct_change"].mean()
    short_ret = group.loc[short_mask, "pct_change"].mean()
    return long_ret - short_ret


def _build_smb_from_group(group):
    cols = list(group.columns)
    small_cols = [c for c in cols if "marketcap_small" in c]
    large_cols = [c for c in cols if "marketcap_large" in c]

    if small_cols and large_cols:
        long_mask = group[small_cols].fillna(0).max(axis=1) > 0
        short_mask = group[large_cols].fillna(0).max(axis=1) > 0
        return _spread_return(group, long_mask, short_mask)

    if large_cols:
        large_mask = group[large_cols].fillna(0).max(axis=1) > 0
        known_large_signal = group[large_cols].notnull().any(axis=1)
        long_mask = known_large_signal & (~large_mask)
        short_mask = large_mask
        return _spread_return(group, long_mask, short_mask)

    return np.nan


def _build_hml_from_group(group):
    cols = list(group.columns)
    value_cols = [c for c in cols if c.startswith("style_") and "value" in c]
    growth_cols = [c for c in cols if c.startswith("style_") and "growth" in c]

    if not value_cols or not growth_cols:
        return np.nan

    long_mask = group[value_cols].fillna(0).max(axis=1) > 0
    short_mask = group[growth_cols].fillna(0).max(axis=1) > 0
    return _spread_return(group, long_mask, short_mask)


def _construct_factor_returns(returns_df, panel_df, rf_df, bond_conids):
    pivoted_returns = returns_df.pivot(index="trade_date", columns="conid", values="pct_change")
    pivoted_returns = pivoted_returns.sort_index()

    all_conids = list(pivoted_returns.columns)
    equity_conids = [c for c in all_conids if c not in bond_conids]

    factors = pd.DataFrame(index=pivoted_returns.index)
    factors.index.name = "trade_date"
    factors = factors.join(rf_df.set_index("trade_date"), how="left")
    factors["daily_nominal_rate"] = factors["daily_nominal_rate"].fillna(0.0)

    if equity_conids:
        market_returns = pivoted_returns[equity_conids].mean(axis=1)
    else:
        market_returns = pd.Series(0.0, index=pivoted_returns.index)
    factors["Mkt-RF"] = market_returns - factors["daily_nominal_rate"]

    if panel_df.empty:
        return factors

    feature_wide = panel_df.pivot_table(
        index=["trade_date", "conid"],
        columns="feature_name",
        values="feature_value",
        aggfunc="last",
    )
    returns_long = returns_df.set_index(["trade_date", "conid"])[["pct_change"]]
    joined = feature_wide.join(returns_long, how="inner")
    if joined.empty:
        return factors

    smb_values = {}
    hml_values = {}
    for date, group in joined.groupby(level="trade_date"):
        g = group.reset_index(level=0, drop=True)
        smb_values[date] = _build_smb_from_group(g)
        hml_values[date] = _build_hml_from_group(g)

    smb_series = pd.Series(smb_values).reindex(factors.index)
    hml_series = pd.Series(hml_values).reindex(factors.index)

    if smb_series.notna().any():
        factors["SMB"] = smb_series.fillna(0.0)
    if hml_series.notna().any():
        factors["HML"] = hml_series.fillna(0.0)

    return factors


def _screen_factors(factors_df):
    out = factors_df.copy()
    feature_cols = [c for c in out.columns if c != "daily_nominal_rate"]
    if not feature_cols:
        return out

    # Drop constants first.
    constant_cols = [c for c in feature_cols if out[c].nunique(dropna=True) <= 1]
    out = out.drop(columns=constant_cols, errors="ignore")

    feature_cols = [c for c in out.columns if c != "daily_nominal_rate"]
    if len(feature_cols) <= 1:
        return out

    corr = out[feature_cols].corr().abs()
    drop_cols = set()
    ordered = list(feature_cols)
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            c1 = ordered[i]
            c2 = ordered[j]
            if c1 in drop_cols or c2 in drop_cols:
                continue
            if corr.loc[c1, c2] > 0.95:
                drop_cols.add(c2)

    return out.drop(columns=list(drop_cols), errors="ignore")


def _run_regression(returns_df, factors_df):
    feature_cols = [c for c in factors_df.columns if c != "daily_nominal_rate"]
    if not feature_cols:
        return pd.DataFrame()

    returns_wide = returns_df.pivot(index="trade_date", columns="conid", values="pct_change")
    common_dates = returns_wide.index.intersection(factors_df.index)
    if len(common_dates) < MIN_REGRESSION_DAYS:
        logger.warning("Insufficient common history for regression.")
        return pd.DataFrame()

    returns_wide = returns_wide.loc[common_dates]
    rf = factors_df.loc[common_dates, "daily_nominal_rate"]
    x_raw_all = factors_df.loc[common_dates, feature_cols]

    results = []
    for conid in returns_wide.columns:
        y = returns_wide[conid].dropna()
        if len(y) < MIN_REGRESSION_DAYS:
            continue

        valid_dates = y.index
        x_raw = x_raw_all.loc[valid_dates]
        y_excess = y - rf.loc[valid_dates]

        try:
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x_raw)
            model = ElasticNetCV(cv=3, random_state=42, max_iter=3000)
            model.fit(x_scaled, y_excess)

            coef_raw = model.coef_ / scaler.scale_
            intercept_raw = model.intercept_ - float(np.dot(coef_raw, scaler.mean_))

            row = {
                "conid": conid,
                "alpha": float(intercept_raw),
                "r2": float(model.score(x_scaled, y_excess)),
                "n_obs": int(len(y_excess)),
            }
            for idx, col in enumerate(feature_cols):
                row[f"beta_{col}"] = float(coef_raw[idx])
            results.append(row)
        except Exception:
            continue

    return pd.DataFrame(results)


def run():
    con = _get_duckdb_con()
    try:
        returns_df = _load_returns(con)
        panel_df = _load_factor_panel(con)
        eligible_conids = _load_eligible_conids(con)
        bond_conids = _identify_bond_universe(con)
    finally:
        con.close()

    if returns_df.empty:
        logger.warning("No returns data available.")
        return {"status": "empty_returns"}

    total_conids_before = int(returns_df["conid"].nunique())
    if eligible_conids:
        returns_df = returns_df[returns_df["conid"].isin(eligible_conids)].copy()
        panel_df = panel_df[panel_df["conid"].isin(eligible_conids)].copy()
    if returns_df.empty:
        return {"status": "empty_after_eligibility_filter"}

    rf_df = _compute_risk_free_rate(returns_df, bond_conids)
    factors_df = _construct_factor_returns(returns_df, panel_df, rf_df, bond_conids)
    factors_df = _screen_factors(factors_df)
    betas_df = _run_regression(returns_df, factors_df)

    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    factors_path = RESEARCH_DIR / "factor_returns_daily_latest.parquet"
    betas_path = RESEARCH_DIR / "asset_factor_betas_latest.parquet"
    summary_path = RESEARCH_DIR / "analysis_summary_latest.json"

    factor_out = factors_df.reset_index().rename(columns={"index": "trade_date"})
    factor_out.to_parquet(factors_path, index=False)
    betas_df.to_parquet(betas_path, index=False)

    built_factor_cols = [c for c in factors_df.columns if c != "daily_nominal_rate"]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "returns_conids_before_filter": total_conids_before,
        "returns_conids_after_filter": int(returns_df["conid"].nunique()),
        "eligible_conids_available": int(len(eligible_conids)),
        "regressed_assets": int(len(betas_df)),
        "factors_built": built_factor_cols,
        "bond_universe_size": int(len(bond_conids)),
        "rf_mean": float(rf_df["daily_nominal_rate"].mean()) if not rf_df.empty else 0.0,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "status": "success",
        "artifacts": [str(factors_path), str(betas_path), str(summary_path)],
        "factors_built": built_factor_cols,
    }


if __name__ == "__main__":
    run()
