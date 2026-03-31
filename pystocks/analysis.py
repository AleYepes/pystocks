from dataclasses import dataclass
from pathlib import Path
import sqlite3

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DATA_DIR, SQLITE_DB_PATH
from .price_preprocess import (
    PricePreprocessConfig,
    load_price_history,
    preprocess_price_history,
    save_price_preprocess_results,
)


@dataclass
class AnalysisConfig:
    sqlite_path: Path = SQLITE_DB_PATH
    output_dir: Path = DATA_DIR / "analysis"
    rebalance_freq: str = "ME"
    min_assets_per_factor: int = 12
    quantile: float = 0.20
    factor_corr_threshold: float = 0.90
    min_factor_coverage: float = 0.60
    min_train_days: int = 126
    min_test_days: int = 21
    trailing_beta_days: int = 252
    selection_frequency_threshold: float = 0.15
    min_selection_count: int = 2
    outlier_z_threshold: float = 50.0


def _sanitize_segment(value):
    text = str(value or "").strip().lower()
    out = []
    last_was_sep = False
    for ch in text:
        if ch.isalnum():
            out.append(ch)
            last_was_sep = False
        elif not last_was_sep:
            out.append("_")
            last_was_sep = True
    return "".join(out).strip("_") or "field"


def _parse_scaled_number(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip()
    if not text:
        return np.nan
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()").replace(",", "")
    multiplier = 1.0
    if text[-1:].upper() in {"K", "M", "B", "T"}:
        multiplier = {
            "K": 1e3,
            "M": 1e6,
            "B": 1e9,
            "T": 1e12,
        }[text[-1].upper()]
        text = text[:-1]
    text = text.replace("$", "").replace("€", "").replace("£", "").replace("¥", "")
    try:
        parsed = float(text) * multiplier
    except ValueError:
        return np.nan
    return -parsed if negative else parsed


def _write_output(name, df, output_dir, sqlite_path, long_sql_df=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / f"{name}.parquet"
    parquet_df = df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df.copy()
    parquet_df.to_parquet(parquet_path, index=False)

    sql_df = long_sql_df if long_sql_df is not None else parquet_df
    with sqlite3.connect(str(sqlite_path)) as conn:
        sql_df.to_sql(name, conn, if_exists="replace", index=False)

    return str(parquet_path)


def _prefix_frame(df, prefix, keep=("conid", "effective_at")):
    if df.empty:
        return df
    renamed = {}
    for col in df.columns:
        if col in keep:
            continue
        renamed[col] = f"{prefix}__{_sanitize_segment(col)}"
    return df.rename(columns=renamed)


def _series_or_zero(df, column):
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)


def _text_series(df, column):
    if column in df.columns:
        return df[column].fillna("").astype(str)
    return pd.Series("", index=df.index, dtype=object)


def _pivot_series_frame(df, key_col, value_col, prefix):
    if df.empty:
        return pd.DataFrame(columns=["conid", "effective_at"])
    work = df.copy()
    work["pivot_key"] = work[key_col].map(_sanitize_segment)
    pivoted = (
        work.pivot_table(
            index=["conid", "effective_at"],
            columns="pivot_key",
            values=value_col,
            aggfunc="first",
        )
        .reset_index()
    )
    pivoted.columns = [
        col if col in {"conid", "effective_at"} else f"{prefix}__{col}"
        for col in pivoted.columns
    ]
    return pivoted


def _pivot_metric_frame(df, prefix, key_cols):
    if df.empty:
        return pd.DataFrame(columns=["conid", "effective_at"])
    work = df.copy()
    work["pivot_key"] = work[key_cols].astype(str).agg("__".join, axis=1).map(_sanitize_segment)
    value_pivot = _pivot_series_frame(work[["conid", "effective_at", "pivot_key", "value_num"]], "pivot_key", "value_num", prefix)
    if "vs_num" in work.columns:
        vs_work = work[["conid", "effective_at", "pivot_key", "vs_num"]].rename(columns={"vs_num": "value_num"})
        vs_pivot = _pivot_series_frame(vs_work, "pivot_key", "value_num", f"{prefix}_vs")
        value_pivot = value_pivot.merge(vs_pivot, on=["conid", "effective_at"], how="outer")
    return value_pivot


def _load_sql_frame(conn, query):
    df = pd.read_sql_query(query, conn)
    if "conid" in df.columns:
        df["conid"] = df["conid"].astype(str)
    if "effective_at" in df.columns:
        df["effective_at"] = pd.to_datetime(df["effective_at"])
    return df


def load_snapshot_features(sqlite_path=SQLITE_DB_PATH):
    with sqlite3.connect(str(sqlite_path)) as conn:
        profile = _load_sql_frame(conn, "SELECT * FROM profile_and_fees")
        holdings_asset = _prefix_frame(_load_sql_frame(conn, "SELECT * FROM holdings_asset_type"), "holding_asset")
        holdings_quality = _prefix_frame(_load_sql_frame(conn, "SELECT * FROM holdings_debtor_quality"), "holding_quality")
        holdings_maturity = _prefix_frame(_load_sql_frame(conn, "SELECT * FROM holdings_maturity"), "holding_maturity")

        holdings_industry = _pivot_series_frame(
            _load_sql_frame(conn, "SELECT conid, effective_at, industry, value_num FROM holdings_industry"),
            "industry",
            "value_num",
            "industry",
        )
        holdings_currency = _pivot_series_frame(
            _load_sql_frame(
                conn,
                """
                SELECT conid, effective_at, COALESCE(code, currency) AS currency_key, value_num
                FROM holdings_currency
                """,
            ),
            "currency_key",
            "value_num",
            "currency",
        )
        holdings_country = _pivot_series_frame(
            _load_sql_frame(
                conn,
                """
                SELECT conid, effective_at, COALESCE(country_code, country) AS country_key, value_num
                FROM holdings_investor_country
                """,
            ),
            "country_key",
            "value_num",
            "country",
        )
        holdings_region = _pivot_series_frame(
            _load_sql_frame(conn, "SELECT conid, effective_at, region, value_num FROM holdings_geographic_weights"),
            "region",
            "value_num",
            "region",
        )
        holdings_debt_type = _pivot_series_frame(
            _load_sql_frame(conn, "SELECT conid, effective_at, debt_type, value_num FROM holdings_debt_type"),
            "debt_type",
            "value_num",
            "debt_type",
        )
        top10 = _load_sql_frame(conn, "SELECT * FROM holdings_top10")
        top10_agg = pd.DataFrame(columns=["conid", "effective_at"])
        if not top10.empty:
            top10_agg = (
                top10.groupby(["conid", "effective_at"], as_index=False)
                .agg(
                    top10_count=("name", "nunique"),
                    top10_weight_sum=("holding_weight_num", "sum"),
                    top10_weight_max=("holding_weight_num", "max"),
                )
            )
            top10_agg = _prefix_frame(top10_agg, "top10")

        ratio_key = _pivot_metric_frame(_load_sql_frame(conn, "SELECT * FROM ratios_key_ratios"), "ratio_key", ["metric_id"])
        ratio_financial = _pivot_metric_frame(_load_sql_frame(conn, "SELECT * FROM ratios_financials"), "ratio_financial", ["metric_id"])
        ratio_fixed_income = _pivot_metric_frame(_load_sql_frame(conn, "SELECT * FROM ratios_fixed_income"), "ratio_fixed_income", ["metric_id"])
        ratio_dividend = _pivot_metric_frame(_load_sql_frame(conn, "SELECT * FROM ratios_dividend"), "ratio_dividend", ["metric_id"])
        ratio_zscore = _pivot_metric_frame(_load_sql_frame(conn, "SELECT * FROM ratios_zscore"), "ratio_zscore", ["metric_id"])
        performance = _pivot_metric_frame(_load_sql_frame(conn, "SELECT * FROM performance"), "performance", ["section", "metric_id"])

        dividend_metrics = _prefix_frame(_load_sql_frame(conn, "SELECT * FROM dividends_industry_metrics"), "dividend_metric")
        morningstar = _prefix_frame(_load_sql_frame(conn, "SELECT * FROM morningstar_summary"), "morningstar")
        lipper = _pivot_metric_frame(_load_sql_frame(conn, "SELECT conid, effective_at, period, metric_id, rating_value AS value_num FROM lipper_ratings"), "lipper", ["period", "metric_id"])

    if not profile.empty:
        profile["total_net_assets_num"] = profile["total_net_assets_value"].map(_parse_scaled_number)
        profile = _prefix_frame(profile, "profile")

    frames = [
        profile,
        holdings_asset,
        holdings_quality,
        holdings_maturity,
        holdings_industry,
        holdings_currency,
        holdings_country,
        holdings_region,
        holdings_debt_type,
        top10_agg,
        ratio_key,
        ratio_financial,
        ratio_fixed_income,
        ratio_dividend,
        ratio_zscore,
        performance,
        dividend_metrics,
        morningstar,
        lipper,
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["conid", "effective_at"])

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["conid", "effective_at"], how="outer")

    merged["conid"] = merged["conid"].astype(str)
    merged["effective_at"] = pd.to_datetime(merged["effective_at"])
    merged["sleeve"] = merged.apply(_assign_sleeve, axis=1)
    return merged.sort_values(["conid", "effective_at"]).reset_index(drop=True)


def _assign_sleeve(row):
    asset_type = str(row.get("profile__asset_type") or "").strip().lower()
    fixed_income = float(row.get("holding_asset__fixed_income") or 0.0)
    equity = float(row.get("holding_asset__equity") or 0.0)

    if "bond" in asset_type or fixed_income >= max(0.50, equity):
        return "bond"
    if "commodity" in asset_type:
        return "commodity"
    if "mixed" in asset_type or "alternative" in asset_type or "money" in asset_type:
        return "other"
    return "equity"


def _build_price_features(prices):
    clean = prices.loc[prices["is_clean_price"]].copy()
    if clean.empty:
        return pd.DataFrame(columns=["conid", "trade_date"])

    clean = clean.sort_values(["conid", "trade_date"])
    frames = []
    for conid, group in clean.groupby("conid"):
        g = group.copy()
        g["price_feature__momentum_21"] = g["clean_price"].pct_change(21)
        g["price_feature__momentum_63"] = g["clean_price"].pct_change(63)
        g["price_feature__momentum_126"] = g["clean_price"].pct_change(126)
        g["price_feature__momentum_252"] = g["clean_price"].pct_change(252)
        g["price_feature__volatility_21"] = g["clean_return"].rolling(21, min_periods=10).std() * np.sqrt(252.0)
        g["price_feature__volatility_63"] = g["clean_return"].rolling(63, min_periods=21).std() * np.sqrt(252.0)
        g["price_feature__downside_volatility_63"] = (
            g["clean_return"]
            .where(g["clean_return"] < 0.0)
            .rolling(63, min_periods=21)
            .std()
            * np.sqrt(252.0)
        )
        rolling_peak = g["clean_price"].rolling(126, min_periods=21).max()
        drawdown = g["clean_price"] / rolling_peak - 1.0
        g["price_feature__max_drawdown_126"] = drawdown.rolling(126, min_periods=21).min()
        frames.append(
            g[
                [
                    "conid",
                    "trade_date",
                    "price_feature__momentum_21",
                    "price_feature__momentum_63",
                    "price_feature__momentum_126",
                    "price_feature__momentum_252",
                    "price_feature__volatility_21",
                    "price_feature__volatility_63",
                    "price_feature__downside_volatility_63",
                    "price_feature__max_drawdown_126",
                ]
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _build_rebalance_dates(snapshot_features, prices, freq):
    if snapshot_features.empty or prices.empty:
        return pd.DatetimeIndex([])
    if freq == "M":
        freq = "ME"
    start = snapshot_features["effective_at"].min().normalize()
    end = prices["trade_date"].max().normalize()
    dates = pd.date_range(start=start, end=end, freq=freq)
    if len(dates) == 0 or dates[-1] != end:
        dates = dates.append(pd.DatetimeIndex([end]))
    return dates.unique().sort_values()


def _merge_price_features(latest, price_features):
    if latest.empty or price_features.empty:
        latest["feature_trade_date"] = pd.NaT
        return latest

    left = latest.sort_values(["rebalance_date", "conid"]).copy()
    right = price_features.sort_values(["trade_date", "conid"]).copy()
    merged = pd.merge_asof(
        left,
        right,
        by="conid",
        left_on="rebalance_date",
        right_on="trade_date",
        direction="backward",
    )
    merged = merged.rename(columns={"trade_date": "feature_trade_date"})
    return merged


def build_analysis_panel_data(snapshot_features, price_result, config):
    prices = price_result["prices"]
    eligibility = price_result["eligibility"]
    price_features = _build_price_features(prices)
    rebalance_dates = _build_rebalance_dates(snapshot_features, prices, config.rebalance_freq)
    eligible_conids = set(eligibility.loc[eligibility["eligible"], "conid"].astype(str))

    panels = []
    for rebalance_date in rebalance_dates:
        eligible_snapshots = snapshot_features.loc[snapshot_features["effective_at"] <= rebalance_date]
        if eligible_snapshots.empty:
            continue
        latest = (
            eligible_snapshots.sort_values(["conid", "effective_at"])
            .groupby("conid", as_index=False)
            .tail(1)
            .copy()
        )
        latest = latest.loc[latest["conid"].isin(eligible_conids)].copy()
        if latest.empty:
            continue
        latest["rebalance_date"] = rebalance_date
        latest["snapshot_age_days"] = (rebalance_date - latest["effective_at"]).dt.days
        latest = latest.merge(eligibility, on="conid", how="left")
        latest = _merge_price_features(latest, price_features)
        panels.append(latest)

    if not panels:
        return pd.DataFrame()

    panel = pd.concat(panels, ignore_index=True)
    panel = _add_composite_features(panel)
    return panel.sort_values(["rebalance_date", "conid"]).reset_index(drop=True)


def _mean_if_present(df, columns):
    present = [col for col in columns if col in df.columns]
    if not present:
        return pd.Series(np.nan, index=df.index)
    return df[present].mean(axis=1, skipna=True)


def _sum_if_present(df, columns):
    present = [col for col in columns if col in df.columns]
    if not present:
        return pd.Series(np.nan, index=df.index)
    return df[present].sum(axis=1, skipna=True)


def _add_composite_features(panel):
    df = panel.copy()
    df["composite__value"] = -_mean_if_present(
        df,
        [
            "ratio_key__price_book",
            "ratio_key__price_cash",
            "ratio_key__price_earnings",
            "ratio_key__price_sales",
        ],
    )
    df["composite__profitability"] = _mean_if_present(
        df,
        [
            "ratio_key__return_on_assets_1yr",
            "ratio_key__return_on_assets_3yr",
            "ratio_key__return_on_capital",
            "ratio_key__return_on_capital_3yr",
            "ratio_key__return_on_equity_1yr",
            "ratio_key__return_on_equity_3yr",
            "ratio_key__return_on_investment_1yr",
            "ratio_key__return_on_investment_3yr",
        ],
    )
    df["composite__leverage"] = _mean_if_present(
        df,
        [
            "ratio_key__lt_debt_shareholders_equity",
            "ratio_key__total_debt_total_capital",
            "ratio_key__total_debt_total_equity",
            "ratio_key__total_assets_total_equity",
        ],
    )
    df["composite__momentum"] = _mean_if_present(
        df,
        [
            "price_feature__momentum_63",
            "price_feature__momentum_126",
            "price_feature__momentum_252",
        ],
    )
    df["composite__income"] = _mean_if_present(
        df,
        [
            "dividend_metric__dividend_yield",
            "dividend_metric__dividend_yield_ttm",
            "ratio_dividend__dividend_yield",
        ],
    )
    df["composite__duration"] = (
        _sum_if_present(
            df,
            [
                "holding_maturity__maturity_10_to_20_years",
                "holding_maturity__maturity_20_to_30_years",
                "holding_maturity__maturity_greater_than_30_years",
            ],
        )
        - _sum_if_present(
            df,
            [
                "holding_maturity__maturity_less_than_1_year",
                "holding_maturity__maturity_1_to_3_years",
                "holding_maturity__maturity_3_to_5_years",
            ],
        )
    )
    df["composite__credit"] = (
        _sum_if_present(
            df,
            [
                "holding_quality__quality_aaa",
                "holding_quality__quality_aa",
                "holding_quality__quality_a",
                "holding_quality__quality_bbb",
            ],
        )
        - _sum_if_present(
            df,
            [
                "holding_quality__quality_bb",
                "holding_quality__quality_b",
                "holding_quality__quality_ccc",
                "holding_quality__quality_cc",
                "holding_quality__quality_c",
                "holding_quality__quality_d",
            ],
        )
    )
    industry_cols = [col for col in df.columns if col.startswith("industry__")]
    country_cols = [col for col in df.columns if col.startswith("country__")]
    df["composite__concentration"] = _mean_if_present(
        df,
        ["top10__top10_weight_sum", "top10__top10_weight_max"]
        + ([industry_cols[0]] if industry_cols else [])
        + ([country_cols[0]] if country_cols else []),
    )
    return df


def _build_returns_wide(prices):
    clean = prices.loc[prices["is_clean_price"], ["conid", "trade_date", "clean_return"]].copy()
    if clean.empty:
        return pd.DataFrame()
    return (
        clean.pivot(index="trade_date", columns="conid", values="clean_return")
        .sort_index()
        .sort_index(axis=1)
    )


def _normalized_weights(values):
    series = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan)
    series = series.where(series > 0.0)
    if series.notna().sum() == 0:
        series = pd.Series(1.0, index=series.index)
    series = series.fillna(0.0)
    total = float(series.sum())
    if total <= 0:
        return pd.Series(1.0 / len(series), index=series.index)
    return series / total


def _select_factor_columns(panel_slice, sleeve):
    numeric_cols = panel_slice.select_dtypes(include=[np.number, bool]).columns.tolist()
    excluded = {
        "valid_rows",
        "total_rows",
        "expected_business_days",
        "eligible",
    }
    factor_cols = []
    for col in numeric_cols:
        if col in excluded:
            continue
        if col.startswith("holding_maturity__") or col.startswith("holding_quality__") or col.startswith("debt_type__") or col.startswith("ratio_fixed_income__"):
            if sleeve != "bond":
                continue
        if col.startswith("composite__") and col.endswith("concentration"):
            factor_cols.append(col)
            continue
        factor_cols.append(col)
    return factor_cols


def _factor_direction(column):
    inverse_tokens = [
        "expense",
        "price_book",
        "price_cash",
        "price_earnings",
        "price_sales",
        "leverage",
        "volatility",
        "drawdown",
        "missing_ratio",
        "snapshot_age",
    ]
    return -1.0 if any(token in column for token in inverse_tokens) else 1.0


def _factor_family(column):
    return column.split("__", 1)[0]


def _factor_kind(column):
    return "composite" if column.startswith("composite__") else "raw"


def _build_long_short_series(values, returns_frame, size_weights, direction, quantile, min_assets):
    valid = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < min_assets or valid.nunique() < 4:
        return None

    scores = valid * direction
    bucket_size = max(2, int(np.floor(len(scores) * quantile)))
    if bucket_size * 2 > len(scores):
        return None

    ranked = scores.sort_values()
    short_ids = ranked.index[:bucket_size]
    long_ids = ranked.index[-bucket_size:]
    if len(set(long_ids) & set(short_ids)) > 0:
        return None

    available_ids = [conid for conid in returns_frame.columns if conid in valid.index]
    if len(available_ids) < min_assets:
        return None

    long_weights = _normalized_weights(size_weights.reindex(long_ids))
    short_weights = _normalized_weights(size_weights.reindex(short_ids))
    long_returns = returns_frame.reindex(columns=list(long_weights.index)).fillna(0.0).dot(long_weights)
    short_returns = returns_frame.reindex(columns=list(short_weights.index)).fillna(0.0).dot(short_weights)
    return long_returns - short_returns


def _select_baseline_bond_members(panel_slice):
    bond_slice = panel_slice.loc[panel_slice["sleeve"] == "bond"].copy()
    if bond_slice.empty:
        return bond_slice

    short_score = (
        _series_or_zero(bond_slice, "holding_maturity__maturity_less_than_1_year")
        + 0.75 * _series_or_zero(bond_slice, "holding_maturity__maturity_1_to_3_years")
        - _series_or_zero(bond_slice, "holding_maturity__maturity_10_to_20_years")
        - _series_or_zero(bond_slice, "holding_maturity__maturity_20_to_30_years")
        - _series_or_zero(bond_slice, "holding_maturity__maturity_greater_than_30_years")
    )
    quality_score = (
        _series_or_zero(bond_slice, "holding_quality__quality_aaa")
        + _series_or_zero(bond_slice, "holding_quality__quality_aa")
        + _series_or_zero(bond_slice, "holding_quality__quality_a")
        - _series_or_zero(bond_slice, "holding_quality__quality_bb")
        - _series_or_zero(bond_slice, "holding_quality__quality_b")
        - _series_or_zero(bond_slice, "holding_quality__quality_ccc")
    )
    sovereign_cols = [col for col in bond_slice.columns if col.startswith("debt_type__") and any(token in col for token in ["sovereign", "government", "treasury"])]
    sovereign_score = bond_slice[sovereign_cols].sum(axis=1) if sovereign_cols else pd.Series(0.0, index=bond_slice.index)
    text_bonus = (
        _text_series(bond_slice, "profile__classification").str.contains("treasury|government|short", case=False, regex=True).astype(float)
        + _text_series(bond_slice, "profile__objective").str.contains("treasury|government|short", case=False, regex=True).astype(float)
    )
    bond_slice["baseline_score"] = short_score + quality_score + sovereign_score + text_bonus
    bond_slice = bond_slice.sort_values(
        ["baseline_score", "profile__total_net_assets_num"],
        ascending=[False, False],
        na_position="last",
    )
    target_count = min(10, max(3, int(np.ceil(len(bond_slice) * 0.10))))
    return bond_slice.head(target_count)


def _build_baseline_returns(panel, returns_wide):
    if panel.empty or returns_wide.empty:
        return pd.Series(dtype=float), pd.DataFrame()

    rebalance_dates = sorted(pd.to_datetime(panel["rebalance_date"]).unique())
    baseline = pd.Series(0.0, index=returns_wide.index, name="bond_baseline_return")
    membership_rows = []

    for start, end in zip(rebalance_dates[:-1], rebalance_dates[1:]):
        panel_slice = panel.loc[panel["rebalance_date"] == start]
        selected = _select_baseline_bond_members(panel_slice)
        if selected.empty:
            continue
        interval_returns = returns_wide.loc[(returns_wide.index > start) & (returns_wide.index <= end)]
        if interval_returns.empty:
            continue
        conids = selected["conid"].astype(str).tolist()
        weights = _normalized_weights(selected.set_index("conid")["profile__total_net_assets_num"].reindex(conids))
        portfolio = interval_returns.reindex(columns=conids).fillna(0.0).dot(weights)
        baseline.loc[portfolio.index] = portfolio

        for conid, weight in weights.items():
            membership_rows.append(
                {
                    "rebalance_date": pd.Timestamp(start),
                    "conid": str(conid),
                    "weight": float(weight),
                }
            )

    return baseline, pd.DataFrame(membership_rows)


def build_factor_returns(panel, prices, config):
    returns_wide = _build_returns_wide(prices)
    baseline_returns, baseline_members = _build_baseline_returns(panel, returns_wide)
    if panel.empty or returns_wide.empty:
        return pd.DataFrame(), pd.DataFrame(), baseline_returns, baseline_members

    factor_map = {}
    metadata = {}
    rebalance_dates = sorted(pd.to_datetime(panel["rebalance_date"]).unique())

    for start, end in zip(rebalance_dates[:-1], rebalance_dates[1:]):
        interval_returns = returns_wide.loc[(returns_wide.index > start) & (returns_wide.index <= end)]
        if interval_returns.empty:
            continue

        panel_slice = panel.loc[panel["rebalance_date"] == start].copy()
        for sleeve in sorted(panel_slice["sleeve"].dropna().unique()):
            sleeve_slice = panel_slice.loc[panel_slice["sleeve"] == sleeve].copy()
            if len(sleeve_slice) < config.min_assets_per_factor:
                continue

            size_weights = sleeve_slice.set_index("conid")["profile__total_net_assets_num"]
            market_weights = _normalized_weights(size_weights.reindex(sleeve_slice["conid"]))
            market_series = interval_returns.reindex(columns=list(market_weights.index)).fillna(0.0).dot(market_weights)
            market_factor_id = f"{sleeve}__market"
            factor_map.setdefault(market_factor_id, []).append(market_series.rename(market_factor_id))
            metadata[market_factor_id] = {
                "factor_id": market_factor_id,
                "sleeve": sleeve,
                "family": "market",
                "kind": "market",
                "source_column": None,
            }

            for column in _select_factor_columns(sleeve_slice, sleeve):
                coverage = sleeve_slice[column].notna().mean()
                if coverage < config.min_factor_coverage:
                    continue

                factor_id = f"{sleeve}__{_factor_kind(column)}__{column}"
                series = _build_long_short_series(
                    sleeve_slice.set_index("conid")[column],
                    interval_returns,
                    size_weights,
                    _factor_direction(column),
                    config.quantile,
                    config.min_assets_per_factor,
                )
                if series is None or series.abs().sum() == 0.0:
                    continue

                factor_map.setdefault(factor_id, []).append(series.rename(factor_id))
                metadata[factor_id] = {
                    "factor_id": factor_id,
                    "sleeve": sleeve,
                    "family": _factor_family(column),
                    "kind": _factor_kind(column),
                    "source_column": column,
                }

    if not factor_map:
        return pd.DataFrame(), pd.DataFrame(), baseline_returns, baseline_members

    factor_returns = pd.concat(
        [pd.concat(parts).groupby(level=0).sum().rename(factor_id) for factor_id, parts in factor_map.items()],
        axis=1,
    ).sort_index()
    factor_meta = pd.DataFrame(sorted(metadata.values(), key=lambda row: row["factor_id"]))
    return factor_returns, factor_meta, baseline_returns, baseline_members


def cluster_factor_returns(factor_returns, factor_meta, config):
    if factor_returns.empty or factor_meta.empty:
        return pd.DataFrame(columns=["factor_id", "cluster_id", "cluster_representative", "cluster_size", "keep_factor"]), pd.DataFrame()

    cluster_rows = []
    keepers = []

    for sleeve, meta_slice in factor_meta.groupby("sleeve"):
        factor_ids = meta_slice["factor_id"].tolist()
        sleeve_returns = factor_returns.reindex(columns=factor_ids)
        sleeve_returns = sleeve_returns.loc[:, sleeve_returns.notna().sum() >= config.min_train_days]
        if sleeve_returns.empty:
            continue

        corr = sleeve_returns.corr().abs().fillna(0.0)
        graph = nx.Graph()
        graph.add_nodes_from(corr.columns.tolist())
        for left in corr.columns:
            for right in corr.columns:
                if left >= right:
                    continue
                if corr.loc[left, right] >= config.factor_corr_threshold:
                    graph.add_edge(left, right)

        for cluster_id, component in enumerate(nx.connected_components(graph), start=1):
            members = sorted(component)
            member_meta = meta_slice.set_index("factor_id").loc[members].reset_index()
            coverage = sleeve_returns[members].notna().sum().rename("coverage")
            member_meta = member_meta.merge(coverage, left_on="factor_id", right_index=True, how="left")
            member_meta["kind_priority"] = member_meta["kind"].map({"composite": 0, "market": 1, "raw": 2}).fillna(3)
            representative = (
                member_meta.sort_values(
                    ["kind_priority", "coverage", "factor_id"],
                    ascending=[True, False, True],
                )
                .iloc[0]["factor_id"]
            )
            keepers.append(representative)
            for factor_id in members:
                cluster_rows.append(
                    {
                        "factor_id": factor_id,
                        "sleeve": sleeve,
                        "cluster_id": f"{sleeve}_{cluster_id}",
                        "cluster_representative": representative,
                        "cluster_size": len(members),
                        "keep_factor": bool(factor_id == representative),
                    }
                )

    cluster_df = pd.DataFrame(cluster_rows).sort_values(["sleeve", "cluster_id", "factor_id"])
    reduced = factor_returns.reindex(columns=sorted(set(keepers)))
    return cluster_df, reduced


def _fit_elastic_net(X_train, y_train):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "enet",
                ElasticNetCV(
                    alphas=np.logspace(-4, 0, 20),
                    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                    cv=5,
                    random_state=42,
                    max_iter=20000,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["enet"]
    coefs = model.coef_ / scaler.scale_
    intercept = model.intercept_ - np.dot(coefs, scaler.mean_)
    return pipeline, intercept, coefs


def run_factor_research_data(panel, prices, config):
    factor_returns, factor_meta, baseline_returns, baseline_members = build_factor_returns(panel, prices, config)
    cluster_df, reduced_factors = cluster_factor_returns(factor_returns, factor_meta, config)
    returns_wide = _build_returns_wide(prices)

    if factor_returns.empty or reduced_factors.empty or returns_wide.empty:
        empty = pd.DataFrame()
        return {
            "factor_returns": factor_returns,
            "factor_meta": factor_meta,
            "factor_clusters": cluster_df,
            "baseline_returns": baseline_returns,
            "baseline_members": baseline_members,
            "model_results": empty,
            "factor_persistence": empty,
            "current_betas": empty,
        }

    unique_snapshots = sorted(pd.to_datetime(panel["effective_at"].dropna().unique()))
    research_rows = []
    selection_rows = []

    for sleeve in sorted(panel["sleeve"].dropna().unique()):
        sleeve_conids = sorted(panel.loc[panel["sleeve"] == sleeve, "conid"].astype(str).unique())
        sleeve_factors = sorted([col for col in reduced_factors.columns if col.startswith(f"{sleeve}__")])
        if not sleeve_conids or not sleeve_factors:
            continue

        for train_end, test_end in zip(unique_snapshots[2:-1], unique_snapshots[3:]):
            X_train = reduced_factors.loc[reduced_factors.index <= train_end, sleeve_factors].dropna(how="all")
            X_test = reduced_factors.loc[(reduced_factors.index > train_end) & (reduced_factors.index <= test_end), sleeve_factors].dropna(how="all")
            if len(X_train) < config.min_train_days or len(X_test) < config.min_test_days:
                continue

            for conid in sleeve_conids:
                y = returns_wide.get(conid)
                if y is None:
                    continue
                y_excess = y.subtract(baseline_returns, fill_value=0.0)
                train = pd.concat([X_train, y_excess.rename("target")], axis=1).dropna()
                test = pd.concat([X_test, y_excess.rename("target")], axis=1).dropna()
                if len(train) < config.min_train_days or len(test) < config.min_test_days:
                    continue

                X_train_fit = train[sleeve_factors].values
                y_train_fit = train["target"].values
                X_test_fit = test[sleeve_factors].values
                y_test_fit = test["target"].values

                try:
                    pipeline, intercept, coefs = _fit_elastic_net(X_train_fit, y_train_fit)
                except ValueError:
                    continue

                preds = pipeline.predict(X_test_fit)
                denom = float(np.sum((y_test_fit - y_test_fit.mean()) ** 2))
                r2_test = float(1.0 - np.sum((y_test_fit - preds) ** 2) / denom) if denom > 0 else np.nan

                selected = []
                for factor_id, beta in zip(sleeve_factors, coefs):
                    if abs(beta) <= 1e-8:
                        continue
                    selected.append(factor_id)
                    selection_rows.append(
                        {
                            "sleeve": sleeve,
                            "conid": conid,
                            "train_end": pd.Timestamp(train_end),
                            "test_end": pd.Timestamp(test_end),
                            "factor_id": factor_id,
                            "beta": float(beta),
                            "abs_beta": float(abs(beta)),
                            "sign": float(np.sign(beta)),
                        }
                    )

                research_rows.append(
                    {
                        "sleeve": sleeve,
                        "conid": conid,
                        "train_end": pd.Timestamp(train_end),
                        "test_end": pd.Timestamp(test_end),
                        "alpha": float(intercept),
                        "selected_factor_count": int(len(selected)),
                        "selected_factors": "|".join(sorted(selected)),
                        "test_r2": r2_test,
                    }
                )

    model_results = pd.DataFrame(research_rows)
    selections = pd.DataFrame(selection_rows)
    persistence = pd.DataFrame()
    if not selections.empty and not model_results.empty:
        fit_counts = model_results.groupby("sleeve").size().rename("model_fit_count")
        persistence = (
            selections.groupby(["sleeve", "factor_id"], as_index=False)
            .agg(
                selection_count=("factor_id", "size"),
                median_abs_beta=("abs_beta", "median"),
                sign_consistency=("sign", lambda s: float(abs(np.nanmean(s)))),
            )
        )
        persistence = persistence.merge(fit_counts, on="sleeve", how="left")
        persistence["selection_frequency"] = persistence["selection_count"] / persistence["model_fit_count"]
        persistence["is_persistent"] = (
            (persistence["selection_count"] >= config.min_selection_count)
            & (persistence["selection_frequency"] >= config.selection_frequency_threshold)
        )

    current_betas = compute_current_betas_data(
        panel=panel,
        prices=prices,
        reduced_factors=reduced_factors,
        baseline_returns=baseline_returns,
        persistence=persistence,
        config=config,
    )

    return {
        "factor_returns": factor_returns,
        "factor_meta": factor_meta,
        "factor_clusters": cluster_df,
        "baseline_returns": baseline_returns,
        "baseline_members": baseline_members,
        "model_results": model_results,
        "factor_persistence": persistence,
        "current_betas": current_betas,
    }


def compute_current_betas_data(panel, prices, reduced_factors, baseline_returns, persistence, config):
    returns_wide = _build_returns_wide(prices)
    if returns_wide.empty or reduced_factors.empty or persistence.empty:
        return pd.DataFrame()

    latest_rebalance = pd.to_datetime(panel["rebalance_date"].max())
    latest_panel = panel.loc[panel["rebalance_date"] == latest_rebalance, ["conid", "sleeve"]].drop_duplicates()
    start_date = returns_wide.index.max() - pd.Timedelta(days=int(config.trailing_beta_days * 1.5))

    rows = []
    for sleeve, sleeve_panel in latest_panel.groupby("sleeve"):
        persistent_factors = persistence.loc[
            (persistence["sleeve"] == sleeve) & (persistence["is_persistent"]),
            "factor_id",
        ].tolist()
        if not persistent_factors:
            continue

        X = reduced_factors.loc[reduced_factors.index >= start_date, persistent_factors].dropna()
        if len(X) < config.min_test_days:
            continue

        for conid in sleeve_panel["conid"].astype(str):
            y = returns_wide.get(conid)
            if y is None:
                continue
            y_excess = y.subtract(baseline_returns, fill_value=0.0)
            data = pd.concat([X, y_excess.rename("target")], axis=1).dropna()
            if len(data) < config.min_test_days:
                continue

            X_fit = data[persistent_factors].values
            y_fit = data["target"].values
            model = LinearRegression()
            model.fit(X_fit, y_fit)

            rows.append(
                {
                    "conid": conid,
                    "sleeve": sleeve,
                    "window_start": pd.Timestamp(data.index.min()),
                    "window_end": pd.Timestamp(data.index.max()),
                    "n_obs": int(len(data)),
                    "alpha": float(model.intercept_),
                    "r2": float(model.score(X_fit, y_fit)),
                    **{f"beta__{factor_id}": float(beta) for factor_id, beta in zip(persistent_factors, model.coef_)},
                }
            )

    return pd.DataFrame(rows)


def build_analysis_panel(sqlite_path=SQLITE_DB_PATH, output_dir=None, **config_kwargs):
    config = AnalysisConfig(sqlite_path=Path(sqlite_path), output_dir=Path(output_dir or (DATA_DIR / "analysis")), **config_kwargs)
    price_config = PricePreprocessConfig(outlier_z_threshold=config.outlier_z_threshold)
    price_result = preprocess_price_history(load_price_history(config.sqlite_path), config=price_config)
    save_price_preprocess_results(price_result, output_dir=config.output_dir)
    snapshot_features = load_snapshot_features(config.sqlite_path)
    panel = build_analysis_panel_data(snapshot_features, price_result, config)

    panel_path = _write_output("analysis_snapshot_panel", panel, config.output_dir, config.sqlite_path)
    return {
        "status": "ok",
        "rows": int(len(panel)),
        "rebalance_dates": int(panel["rebalance_date"].nunique()) if not panel.empty else 0,
        "snapshot_panel_path": panel_path,
    }


def run_factor_research(sqlite_path=SQLITE_DB_PATH, output_dir=None, **config_kwargs):
    config = AnalysisConfig(sqlite_path=Path(sqlite_path), output_dir=Path(output_dir or (DATA_DIR / "analysis")), **config_kwargs)
    price_config = PricePreprocessConfig(outlier_z_threshold=config.outlier_z_threshold)
    price_result = preprocess_price_history(load_price_history(config.sqlite_path), config=price_config)
    save_price_preprocess_results(price_result, output_dir=config.output_dir)
    snapshot_features = load_snapshot_features(config.sqlite_path)
    panel = build_analysis_panel_data(snapshot_features, price_result, config)
    research = run_factor_research_data(panel, price_result["prices"], config)

    factor_returns_wide = research["factor_returns"].copy()
    factor_returns_long = pd.DataFrame()
    if not factor_returns_wide.empty:
        factor_returns_long = (
            factor_returns_wide.reset_index()
            .rename(columns={"index": "trade_date"})
            .melt(id_vars=["trade_date"], var_name="factor_id", value_name="factor_return")
            .dropna(subset=["factor_return"])
        )

    paths = {
        "snapshot_panel_path": _write_output("analysis_snapshot_panel", panel, config.output_dir, config.sqlite_path),
        "factor_returns_path": _write_output("analysis_factor_returns", factor_returns_wide, config.output_dir, config.sqlite_path, long_sql_df=factor_returns_long) if not factor_returns_wide.empty else None,
        "factor_clusters_path": _write_output("analysis_factor_clusters", research["factor_clusters"], config.output_dir, config.sqlite_path),
        "factor_persistence_path": _write_output("analysis_factor_persistence", research["factor_persistence"], config.output_dir, config.sqlite_path),
        "model_results_path": _write_output("analysis_model_results", research["model_results"], config.output_dir, config.sqlite_path),
        "current_betas_path": _write_output("analysis_current_betas", research["current_betas"], config.output_dir, config.sqlite_path),
        "baseline_members_path": _write_output("analysis_bond_baseline_members", research["baseline_members"], config.output_dir, config.sqlite_path),
    }

    persistent = research["factor_persistence"]
    return {
        "status": "ok",
        "snapshot_rows": int(len(panel)),
        "factor_count": int(factor_returns_wide.shape[1]) if not factor_returns_wide.empty else 0,
        "persistent_factor_count": int(persistent["is_persistent"].sum()) if not persistent.empty else 0,
        **paths,
    }


def compute_current_betas(sqlite_path=SQLITE_DB_PATH, output_dir=None, **config_kwargs):
    result = run_factor_research(sqlite_path=sqlite_path, output_dir=output_dir, **config_kwargs)
    return {
        "status": result["status"],
        "current_betas_path": result["current_betas_path"],
        "persistent_factor_count": result["persistent_factor_count"],
    }


def run_analysis_pipeline(sqlite_path=SQLITE_DB_PATH, output_dir=None, **config_kwargs):
    panel_result = build_analysis_panel(sqlite_path=sqlite_path, output_dir=output_dir, **config_kwargs)
    research_result = run_factor_research(sqlite_path=sqlite_path, output_dir=output_dir, **config_kwargs)
    return {
        "status": "ok",
        "panel": panel_result,
        "research": research_result,
    }
