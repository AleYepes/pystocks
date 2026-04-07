# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false, reportGeneralTypeIssues=false
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import DATA_DIR, SQLITE_DB_PATH
from ..preprocess.price import (
    PricePreprocessConfig,
    load_price_history,
    preprocess_price_history,
    save_price_preprocess_results,
)
from ..preprocess.snapshots import (
    load_snapshot_features as load_preprocessed_snapshot_features,
)
from ..preprocess.supplementary import (
    load_risk_free_daily as load_preprocessed_risk_free_daily,
)
from ..preprocess.supplementary import (
    load_world_bank_country_features as load_preprocessed_world_bank_country_features,
)
from ..progress import make_progress_bar, track_progress
from ..storage.txn import transaction


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
    training_window_years: tuple[int, ...] = (3, 4)
    walk_forward_step_months: int = 12
    use_risk_free_excess: bool = True
    require_supplementary_data: bool = True
    include_macro_features: bool = True
    include_dynamic_fundamental_trends: bool = True
    return_alignment_max_gap_days: int = 0
    sparse_feature_max_ratio: float = 0.995


SUPERSECTOR_MAP = {
    "defensive": [
        "industry__consumer_non_cyclicals",
        "industry__utilities",
        "industry__healthcare",
        "industry__telecommunication_services",
        "industry__academic_educational_services",
    ],
    "cyclical": [
        "industry__technology",
        "industry__consumer_cyclicals",
        "industry__industrials",
        "industry__financials",
        "industry__real_estate",
    ],
    "commodities": [
        "industry__basic_materials",
        "industry__energy",
    ],
}

MACRO_FEATURE_COLUMNS = [
    "population_level",
    "population_growth",
    "gdp_pcap_level",
    "gdp_pcap_growth",
    "economic_output_gdp_level",
    "economic_output_gdp_growth",
    "foreign_direct_investment_level",
    "foreign_direct_investment_growth",
    "share_trade_volume_level",
    "share_trade_volume_growth",
]


def _empty_frame(columns):
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


def _to_timestamp(value):
    return pd.Timestamp(value)


def _write_output(name, df, output_dir, sqlite_path, long_sql_df=None, tx=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / f"{name}.parquet"
    parquet_df = (
        df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df.copy()
    )
    parquet_df.to_parquet(parquet_path, index=False)

    sql_df = long_sql_df if long_sql_df is not None else parquet_df
    if tx is not None:
        if len(sql_df.columns) == 0:
            tx.execute(f"DROP TABLE IF EXISTS {name}")
        else:
            tx.write_frame(name, sql_df, if_exists="replace", index=False)
    else:
        with transaction(sqlite_path) as managed_tx:
            if len(sql_df.columns) == 0:
                managed_tx.execute(f"DROP TABLE IF EXISTS {name}")
            else:
                managed_tx.write_frame(name, sql_df, if_exists="replace", index=False)

    return str(parquet_path)


def _series_or_zero(df, column):
    if column in df.columns:
        return pd.Series(
            pd.to_numeric(df[column], errors="coerce"), index=df.index
        ).fillna(0.0)
    return pd.Series(0.0, index=df.index, dtype=float)


def _text_series(df, column):
    if column in df.columns:
        return pd.Series(df[column], index=df.index).fillna("").astype(str)
    return pd.Series("", index=df.index, dtype=object)


def _safe_numeric(df, column):
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _mean_if_present(df, columns):
    present = [col for col in columns if col in df.columns]
    if not present:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return df[present].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)


def _sum_if_present(df, columns):
    present = [col for col in columns if col in df.columns]
    if not present:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return df[present].apply(pd.to_numeric, errors="coerce").sum(axis=1, skipna=True)


def _rolling_compound(values: pd.Series, window: int, min_periods: int) -> pd.Series:
    return (1.0 + values.fillna(0.0)).rolling(window, min_periods=min_periods).apply(
        np.prod, raw=True
    ) - 1.0


def _bounded_align_return_frame(returns_wide, max_gap_days):
    if returns_wide.empty or int(max_gap_days) <= 0:
        return returns_wide
    aligned = returns_wide.copy()
    for column in aligned.columns:
        aligned[column] = aligned[column].interpolate(
            method="linear",
            limit=int(max_gap_days),
            limit_area="inside",
        )
    return aligned


def load_snapshot_features(sqlite_path=SQLITE_DB_PATH):
    return load_preprocessed_snapshot_features(sqlite_path=sqlite_path)


def load_risk_free_daily(sqlite_path=SQLITE_DB_PATH):
    return load_preprocessed_risk_free_daily(sqlite_path=sqlite_path)


def load_world_bank_country_features(sqlite_path=SQLITE_DB_PATH):
    return load_preprocessed_world_bank_country_features(sqlite_path=sqlite_path)


def _prepare_analysis_inputs(config, show_progress=False):
    price_config = PricePreprocessConfig(outlier_z_threshold=config.outlier_z_threshold)
    price_result = preprocess_price_history(
        load_price_history(config.sqlite_path),
        config=price_config,
        show_progress=show_progress,
    )
    save_price_preprocess_results(price_result, output_dir=config.output_dir)
    snapshot_features = load_snapshot_features(config.sqlite_path)
    risk_free_daily = load_risk_free_daily(config.sqlite_path)
    world_bank_country_features = load_world_bank_country_features(config.sqlite_path)

    if config.require_supplementary_data:
        if config.use_risk_free_excess and risk_free_daily.empty:
            raise RuntimeError(
                "Missing supplementary risk-free data. Run refresh_supplementary_data first."
            )
        if config.include_macro_features and world_bank_country_features.empty:
            raise RuntimeError(
                "Missing supplementary World Bank features. Run refresh_supplementary_data first."
            )

    return snapshot_features, price_result, risk_free_daily, world_bank_country_features


def _empty_cluster_frame():
    return _empty_frame(
        [
            "factor_id",
            "sleeve",
            "cluster_id",
            "cluster_representative",
            "cluster_size",
            "keep_factor",
        ]
    )


def _build_price_features(prices, show_progress=False):
    clean = prices.loc[prices["is_clean_price"]].copy()
    if clean.empty:
        return _empty_frame(["conid", "trade_date"])

    clean = clean.sort_values(["conid", "trade_date"])
    frames = []
    group_count = int(clean["conid"].nunique())
    for _, group in track_progress(
        clean.groupby("conid"),
        show_progress=show_progress,
        total=group_count,
        desc="Price features",
        unit="conid",
        leave=False,
    ):
        g = group.copy()
        returns = pd.to_numeric(g["clean_return"], errors="coerce")
        g["price_feature__momentum_21"] = g["clean_price"].pct_change(21)
        g["price_feature__momentum_63"] = g["clean_price"].pct_change(63)
        g["price_feature__momentum_126"] = g["clean_price"].pct_change(126)
        g["price_feature__momentum_252"] = g["clean_price"].pct_change(252)
        g["price_feature__momentum_3mo"] = returns.rolling(63, min_periods=21).mean()
        g["price_feature__momentum_6mo"] = returns.rolling(126, min_periods=42).mean()
        g["price_feature__momentum_1y"] = returns.rolling(252, min_periods=84).mean()
        g["price_feature__rs_3mo"] = _rolling_compound(returns, 63, 21)
        g["price_feature__rs_6mo"] = _rolling_compound(returns, 126, 42)
        g["price_feature__rs_1y"] = _rolling_compound(returns, 252, 84)
        g["price_feature__volatility_21"] = returns.rolling(
            21, min_periods=10
        ).std() * np.sqrt(252.0)
        g["price_feature__volatility_63"] = returns.rolling(
            63, min_periods=21
        ).std() * np.sqrt(252.0)
        g["price_feature__downside_volatility_63"] = returns.where(
            returns < 0.0
        ).rolling(63, min_periods=21).std() * np.sqrt(252.0)
        rolling_peak = g["clean_price"].rolling(126, min_periods=21).max()
        drawdown = g["clean_price"] / rolling_peak - 1.0
        g["price_feature__max_drawdown_126"] = drawdown.rolling(
            126, min_periods=21
        ).min()
        frames.append(g)

    price_feature_columns = [
        "conid",
        "trade_date",
        "price_feature__momentum_21",
        "price_feature__momentum_63",
        "price_feature__momentum_126",
        "price_feature__momentum_252",
        "price_feature__momentum_3mo",
        "price_feature__momentum_6mo",
        "price_feature__momentum_1y",
        "price_feature__rs_3mo",
        "price_feature__rs_6mo",
        "price_feature__rs_1y",
        "price_feature__volatility_21",
        "price_feature__volatility_63",
        "price_feature__downside_volatility_63",
        "price_feature__max_drawdown_126",
    ]
    return pd.concat(frames, ignore_index=True)[price_feature_columns]


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
    return merged.rename(columns={"trade_date": "feature_trade_date"})


def _add_dynamic_fundamental_features(df):
    out = df.copy()

    def _slope(col_a, col_b, time_a, time_b):
        if col_a not in out.columns or col_b not in out.columns:
            return pd.Series(np.nan, index=out.index, dtype=float)
        return (
            pd.to_numeric(out[col_a], errors="coerce")
            - pd.to_numeric(out[col_b], errors="coerce")
        ) / float(time_a - time_b)

    slope_specs = [
        (
            "trend__eps_growth_slope",
            "ratio_key__eps_growth_1yr",
            "ratio_key__eps_growth_5yr",
            1,
            5,
        ),
        (
            "trend__return_on_assets_slope",
            "ratio_key__return_on_assets_1yr",
            "ratio_key__return_on_assets_3yr",
            1,
            3,
        ),
        (
            "trend__return_on_capital_slope",
            "ratio_key__return_on_capital",
            "ratio_key__return_on_capital_3yr",
            1,
            3,
        ),
        (
            "trend__return_on_equity_slope",
            "ratio_key__return_on_equity_1yr",
            "ratio_key__return_on_equity_3yr",
            1,
            3,
        ),
        (
            "trend__return_on_investment_slope",
            "ratio_key__return_on_investment_1yr",
            "ratio_key__return_on_investment_3yr",
            1,
            3,
        ),
    ]
    for new_col, col_a, col_b, time_a, time_b in slope_specs:
        out[new_col] = _slope(col_a, col_b, time_a, time_b)

    if {
        "ratio_key__eps_growth_1yr",
        "ratio_key__eps_growth_3yr",
        "ratio_key__eps_growth_5yr",
    }.issubset(out.columns):
        slope_1_3 = _slope(
            "ratio_key__eps_growth_1yr", "ratio_key__eps_growth_3yr", 1, 3
        )
        slope_3_5 = _slope(
            "ratio_key__eps_growth_3yr", "ratio_key__eps_growth_5yr", 3, 5
        )
        out["trend__eps_growth_second_derivative"] = (slope_1_3 - slope_3_5) / -2.0

    return out


def _add_supersector_features(df):
    out = df.copy()
    for name, columns in SUPERSECTOR_MAP.items():
        present = [column for column in columns if column in out.columns]
        if present:
            out[f"supersector__{name}"] = (
                out[present].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            )
    return out


def _add_macro_features(panel, world_bank_country_features):
    if panel.empty:
        return panel.copy()
    if world_bank_country_features is None or world_bank_country_features.empty:
        return panel.copy()

    out = panel.copy()
    country_columns = [col for col in out.columns if col.startswith("country__")]
    if not country_columns:
        return out

    world_bank = world_bank_country_features.copy()
    world_bank["economy_code"] = world_bank["economy_code"].astype(str).str.upper()
    world_bank["effective_at"] = pd.to_datetime(world_bank["effective_at"])

    enriched_parts = []
    for rebalance_date, panel_slice in out.groupby("rebalance_date", sort=True):
        latest = (
            world_bank.loc[world_bank["effective_at"] <= rebalance_date]
            .sort_values(["economy_code", "effective_at"])
            .groupby("economy_code", as_index=False)
            .tail(1)
        )
        if latest.empty:
            enriched_parts.append(panel_slice)
            continue

        latest = latest.set_index("economy_code")
        work = panel_slice.copy()
        for macro_column in MACRO_FEATURE_COLUMNS:
            weighted = pd.Series(0.0, index=work.index, dtype=float)
            for country_column in country_columns:
                code = country_column.replace("country__", "").upper()
                if code not in latest.index or macro_column not in latest.columns:
                    continue
                weight = pd.to_numeric(work[country_column], errors="coerce").fillna(
                    0.0
                )
                value = pd.to_numeric(latest.loc[code, macro_column], errors="coerce")
                if np.isscalar(value):
                    weighted = weighted.add(weight * float(value), fill_value=0.0)
            work[f"macro__{macro_column}"] = weighted
        enriched_parts.append(work)

    return pd.concat(enriched_parts, ignore_index=True)


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
            "price_feature__momentum_3mo",
            "price_feature__momentum_6mo",
            "price_feature__momentum_1y",
            "price_feature__rs_3mo",
            "price_feature__rs_6mo",
            "price_feature__rs_1y",
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
    df["composite__duration"] = _sum_if_present(
        df,
        [
            "holding_maturity__maturity_10_to_20_years",
            "holding_maturity__maturity_20_to_30_years",
            "holding_maturity__maturity_greater_than_30_years",
        ],
    ) - _sum_if_present(
        df,
        [
            "holding_maturity__maturity_less_than_1_year",
            "holding_maturity__maturity_1_to_3_years",
            "holding_maturity__maturity_3_to_5_years",
        ],
    )
    df["composite__credit"] = _sum_if_present(
        df,
        [
            "holding_quality__quality_aaa",
            "holding_quality__quality_aa",
            "holding_quality__quality_a",
            "holding_quality__quality_bbb",
        ],
    ) - _sum_if_present(
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
    industry_cols = [col for col in df.columns if col.startswith("industry__")]
    country_cols = [col for col in df.columns if col.startswith("country__")]
    df["composite__concentration"] = _mean_if_present(
        df,
        ["top10__top10_weight_sum", "top10__top10_weight_max"]
        + ([industry_cols[0]] if industry_cols else [])
        + ([country_cols[0]] if country_cols else []),
    )
    return df


def build_analysis_panel_data(
    snapshot_features,
    price_result,
    config,
    world_bank_country_features=None,
    show_progress=False,
):
    prices = price_result["prices"]
    eligibility = price_result["eligibility"]
    price_features = _build_price_features(prices, show_progress=show_progress)
    rebalance_dates = _build_rebalance_dates(
        snapshot_features, prices, config.rebalance_freq
    )
    eligible_conids = set(eligibility.loc[eligibility["eligible"], "conid"].astype(str))

    if (
        config.require_supplementary_data
        and config.include_macro_features
        and (world_bank_country_features is None or world_bank_country_features.empty)
    ):
        raise RuntimeError(
            "Missing supplementary World Bank features. Run refresh_supplementary_data first."
        )

    panels = []
    for rebalance_date in track_progress(
        rebalance_dates,
        show_progress=show_progress,
        total=len(rebalance_dates),
        desc="Analysis rebalance dates",
        unit="date",
        leave=False,
    ):
        eligible_snapshots = snapshot_features.loc[
            snapshot_features["effective_at"] <= rebalance_date
        ]
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
    if config.include_dynamic_fundamental_trends:
        panel = _add_dynamic_fundamental_features(panel)
    panel = _add_supersector_features(panel)
    if config.include_macro_features:
        panel = _add_macro_features(panel, world_bank_country_features)
    return panel.sort_values(["rebalance_date", "conid"]).reset_index(drop=True)


def _build_returns_wide(prices, max_gap_days=0):
    clean = prices.loc[
        prices["is_clean_price"], ["conid", "trade_date", "clean_return"]
    ].copy()
    if clean.empty:
        return pd.DataFrame()
    wide = (
        clean.pivot(index="trade_date", columns="conid", values="clean_return")
        .sort_index()
        .sort_index(axis=1)
    )
    return _bounded_align_return_frame(wide, max_gap_days)


def _build_risk_free_series(risk_free_daily):
    if risk_free_daily is None or risk_free_daily.empty:
        return pd.Series(dtype=float, name="daily_nominal_rate")
    rf = risk_free_daily.copy()
    rf["trade_date"] = pd.to_datetime(rf["trade_date"])
    rf["daily_nominal_rate"] = pd.to_numeric(rf["daily_nominal_rate"], errors="coerce")
    return rf.set_index("trade_date")["daily_nominal_rate"].sort_index()


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


def _drop_uninformative_factor_columns(panel_slice, candidate_columns, config):
    keep = []
    for column in candidate_columns:
        values = pd.to_numeric(panel_slice[column], errors="coerce")
        if values.notna().sum() == 0:
            continue
        if values.nunique(dropna=True) <= 1:
            continue
        nonzero_ratio = float(values.fillna(0.0).eq(0.0).mean())
        if nonzero_ratio >= config.sparse_feature_max_ratio:
            continue
        keep.append(column)
    return keep


def _select_factor_columns(panel_slice, sleeve, config):
    numeric_cols = panel_slice.select_dtypes(include=[np.number, bool]).columns.tolist()
    excluded = {
        "valid_rows",
        "total_rows",
        "expected_business_days",
        "eligible",
        "snapshot_age_days",
        "max_internal_gap_days",
        "feature_year",
        "source_count",
    }
    candidates = []
    for col in numeric_cols:
        if col in excluded or col.startswith("beta__"):
            continue
        if (
            col.startswith("holding_maturity__")
            or col.startswith("holding_quality__")
            or col.startswith("debt_type__")
            or col.startswith("ratio_fixed_income__")
        ) and sleeve != "bond":
            continue
        candidates.append(col)
    return _drop_uninformative_factor_columns(panel_slice, candidates, config)


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
    if column.startswith("benchmark__"):
        return "benchmark"
    return "composite" if column.startswith("composite__") else "raw"


def _build_long_short_series(
    values, returns_frame, size_weights, direction, quantile, min_assets
):
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

    long_weights = _normalized_weights(size_weights.reindex(long_ids))
    short_weights = _normalized_weights(size_weights.reindex(short_ids))
    long_returns = (
        returns_frame.reindex(columns=list(long_weights.index))
        .fillna(0.0)
        .dot(long_weights)
    )
    short_returns = (
        returns_frame.reindex(columns=list(short_weights.index))
        .fillna(0.0)
        .dot(short_weights)
    )
    return long_returns - short_returns


def _build_benchmark_factors(
    sleeve_slice,
    interval_returns,
    risk_free_series,
    config,
):
    size_weights = sleeve_slice.set_index("conid")["profile__total_net_assets_num"]
    benchmark_map = {}

    market_weights = _normalized_weights(size_weights.reindex(sleeve_slice["conid"]))
    market = (
        interval_returns.reindex(columns=list(market_weights.index))
        .fillna(0.0)
        .dot(market_weights)
    )
    if not risk_free_series.empty and config.use_risk_free_excess:
        market = market.subtract(risk_free_series.reindex(market.index).fillna(0.0))
    benchmark_map["benchmark__market_excess"] = market

    size_signal = -pd.to_numeric(
        sleeve_slice.set_index("conid")["profile__total_net_assets_num"],
        errors="coerce",
    )
    smb = _build_long_short_series(
        size_signal,
        interval_returns,
        size_weights,
        1.0,
        config.quantile,
        config.min_assets_per_factor,
    )
    if smb is not None:
        benchmark_map["benchmark__smb"] = smb

    if "composite__value" in sleeve_slice.columns:
        hml = _build_long_short_series(
            pd.to_numeric(
                sleeve_slice.set_index("conid")["composite__value"], errors="coerce"
            ),
            interval_returns,
            size_weights,
            1.0,
            config.quantile,
            config.min_assets_per_factor,
        )
        if hml is not None:
            benchmark_map["benchmark__hml"] = hml

    return benchmark_map


def _select_baseline_bond_members(panel_slice):
    bond_slice = panel_slice.loc[panel_slice["sleeve"] == "bond"].copy()
    if bond_slice.empty:
        return bond_slice

    short_score = (
        _series_or_zero(bond_slice, "holding_maturity__maturity_less_than_1_year")
        + 0.75 * _series_or_zero(bond_slice, "holding_maturity__maturity_1_to_3_years")
        - _series_or_zero(bond_slice, "holding_maturity__maturity_10_to_20_years")
        - _series_or_zero(bond_slice, "holding_maturity__maturity_20_to_30_years")
        - _series_or_zero(
            bond_slice, "holding_maturity__maturity_greater_than_30_years"
        )
    )
    quality_score = (
        _series_or_zero(bond_slice, "holding_quality__quality_aaa")
        + _series_or_zero(bond_slice, "holding_quality__quality_aa")
        + _series_or_zero(bond_slice, "holding_quality__quality_a")
        - _series_or_zero(bond_slice, "holding_quality__quality_bb")
        - _series_or_zero(bond_slice, "holding_quality__quality_b")
        - _series_or_zero(bond_slice, "holding_quality__quality_ccc")
    )
    sovereign_cols = [
        col
        for col in bond_slice.columns
        if col.startswith("debt_type__")
        and any(token in col for token in ["sovereign", "government", "treasury"])
    ]
    sovereign_score = (
        bond_slice[sovereign_cols].sum(axis=1)
        if sovereign_cols
        else pd.Series(0.0, index=bond_slice.index)
    )
    text_bonus = _text_series(bond_slice, "profile__classification").str.contains(
        "treasury|government|short", case=False, regex=True
    ).astype(float) + _text_series(bond_slice, "profile__objective").str.contains(
        "treasury|government|short", case=False, regex=True
    ).astype(float)
    bond_slice["baseline_score"] = (
        short_score + quality_score + sovereign_score + text_bonus
    )
    bond_slice = bond_slice.sort_values(
        ["baseline_score", "profile__total_net_assets_num"],
        ascending=[False, False],
        na_position="last",
    )
    target_count = min(10, max(3, int(np.ceil(len(bond_slice) * 0.10))))
    return bond_slice.head(target_count)


def _build_baseline_returns(panel, returns_wide, show_progress=False):
    if panel.empty or returns_wide.empty:
        return pd.Series(dtype=float), pd.DataFrame()

    rebalance_dates = sorted(pd.to_datetime(panel["rebalance_date"]).unique())
    baseline = pd.Series(0.0, index=returns_wide.index, name="bond_baseline_return")
    membership_rows = []

    intervals = list(zip(rebalance_dates[:-1], rebalance_dates[1:]))
    for start, end in track_progress(
        intervals,
        show_progress=show_progress,
        total=len(intervals),
        desc="Baseline windows",
        unit="window",
        leave=False,
    ):
        panel_slice = panel.loc[panel["rebalance_date"] == start]
        selected = _select_baseline_bond_members(panel_slice)
        if selected.empty:
            continue
        interval_returns = returns_wide.loc[
            (returns_wide.index > start) & (returns_wide.index <= end)
        ]
        if interval_returns.empty:
            continue
        conids = selected["conid"].astype(str).tolist()
        weights = _normalized_weights(
            selected.set_index("conid")["profile__total_net_assets_num"].reindex(conids)
        )
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


def build_factor_returns(
    panel,
    prices,
    risk_free_daily,
    config,
    show_progress=False,
):
    returns_wide = _build_returns_wide(prices, config.return_alignment_max_gap_days)
    risk_free_series = _build_risk_free_series(risk_free_daily)
    baseline_returns, baseline_members = _build_baseline_returns(
        panel, returns_wide, show_progress=show_progress
    )
    if panel.empty or returns_wide.empty:
        return pd.DataFrame(), pd.DataFrame(), baseline_returns, baseline_members

    factor_map: dict[str, list[pd.Series]] = {}
    metadata = {}
    rebalance_dates = sorted(pd.to_datetime(panel["rebalance_date"]).unique())

    intervals = list(zip(rebalance_dates[:-1], rebalance_dates[1:]))
    for start, end in track_progress(
        intervals,
        show_progress=show_progress,
        total=len(intervals),
        desc="Factor return windows",
        unit="window",
        leave=False,
    ):
        interval_returns = returns_wide.loc[
            (returns_wide.index > start) & (returns_wide.index <= end)
        ]
        if interval_returns.empty:
            continue

        interval_risk_free = risk_free_series.loc[
            (risk_free_series.index > start) & (risk_free_series.index <= end)
        ]
        panel_slice = panel.loc[panel["rebalance_date"] == start].copy()
        for sleeve in sorted(panel_slice["sleeve"].dropna().unique()):
            sleeve_slice = panel_slice.loc[panel_slice["sleeve"] == sleeve].copy()
            if len(sleeve_slice) < config.min_assets_per_factor:
                continue

            benchmark_factors = _build_benchmark_factors(
                sleeve_slice=sleeve_slice,
                interval_returns=interval_returns,
                risk_free_series=interval_risk_free,
                config=config,
            )
            for factor_key, series in benchmark_factors.items():
                factor_id = f"{sleeve}__{factor_key}"
                factor_map.setdefault(factor_id, []).append(series.rename(factor_id))
                metadata[factor_id] = {
                    "factor_id": factor_id,
                    "sleeve": sleeve,
                    "family": factor_key.split("__", 1)[-1],
                    "kind": "benchmark",
                    "source_column": None,
                }

            size_weights = sleeve_slice.set_index("conid")[
                "profile__total_net_assets_num"
            ]
            for column in _select_factor_columns(sleeve_slice, sleeve, config):
                coverage = sleeve_slice[column].notna().mean()
                if coverage < config.min_factor_coverage:
                    continue

                factor_id = f"{sleeve}__{_factor_kind(column)}__{column}"
                series = _build_long_short_series(
                    pd.to_numeric(
                        sleeve_slice.set_index("conid")[column], errors="coerce"
                    ),
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
        [
            pd.concat(parts).groupby(level=0).sum().rename(factor_id)
            for factor_id, parts in factor_map.items()
        ],
        axis=1,
    ).sort_index()
    factor_meta = pd.DataFrame(
        sorted(metadata.values(), key=lambda row: row["factor_id"])
    )
    return factor_returns, factor_meta, baseline_returns, baseline_members


def cluster_factor_returns(factor_returns, factor_meta, config, show_progress=False):
    if factor_returns.empty or factor_meta.empty:
        return _empty_cluster_frame(), pd.DataFrame()

    cluster_rows = []
    keepers = []

    sleeve_count = int(factor_meta["sleeve"].nunique())
    for sleeve, meta_slice in track_progress(
        factor_meta.groupby("sleeve"),
        show_progress=show_progress,
        total=sleeve_count,
        desc="Factor clustering",
        unit="sleeve",
        leave=False,
    ):
        factor_ids = meta_slice["factor_id"].tolist()
        sleeve_returns = factor_returns.reindex(columns=factor_ids)
        sleeve_returns = sleeve_returns.loc[
            :, sleeve_returns.notna().sum() >= config.min_train_days
        ]
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
            member_meta = member_meta.merge(
                coverage, left_on="factor_id", right_index=True, how="left"
            )
            member_meta["kind_priority"] = (
                member_meta["kind"]
                .map({"composite": 0, "benchmark": 1, "raw": 2})
                .fillna(3)
            )
            representative = member_meta.sort_values(
                ["kind_priority", "coverage", "factor_id"],
                ascending=[True, False, True],
            ).iloc[0]["factor_id"]
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

    if not cluster_rows:
        return _empty_cluster_frame(), pd.DataFrame()

    cluster_df = pd.DataFrame(cluster_rows).sort_values(
        ["sleeve", "cluster_id", "factor_id"]
    )
    reduced = factor_returns.reindex(columns=sorted(set(keepers)))
    return cluster_df, reduced


def _fit_elastic_net(X_train, y_train):
    elastic_net_params: Any = {
        "alphas": [float(alpha) for alpha in np.logspace(-4, 0, 20)],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        "cv": 5,
        "random_state": 42,
        "max_iter": 20000,
    }
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("enet", ElasticNetCV(**elastic_net_params)),
        ]
    )
    pipeline.fit(X_train, y_train)
    scaler: Any = pipeline.named_steps["scaler"]
    model: Any = pipeline.named_steps["enet"]
    coefs = model.coef_ / scaler.scale_
    intercept = model.intercept_ - np.dot(coefs, scaler.mean_)
    return pipeline, model, intercept, coefs


def _build_research_windows(
    panel, reduced_factors, config
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, int]]:
    if panel.empty or reduced_factors.empty:
        return []

    rebalance_dates = sorted(pd.to_datetime(panel["rebalance_date"].dropna().unique()))
    if len(rebalance_dates) < 2:
        return []
    first_factor_date = pd.to_datetime(reduced_factors.index.min())

    windows = []
    last_step_end = None
    for train_end, test_end in zip(rebalance_dates[:-1], rebalance_dates[1:]):
        if test_end <= first_factor_date:
            continue
        if last_step_end is not None:
            month_gap = (test_end.year - last_step_end.year) * 12 + (
                test_end.month - last_step_end.month
            )
            if month_gap < config.walk_forward_step_months:
                continue
        for years in sorted(set(int(year) for year in config.training_window_years)):
            train_start = pd.Timestamp(train_end) - pd.DateOffset(years=years)
            windows.append(
                (
                    pd.Timestamp(train_start),
                    pd.Timestamp(train_end),
                    pd.Timestamp(test_end),
                    int(years),
                )
            )
        last_step_end = pd.Timestamp(test_end)
    return windows


def _factor_diagnostics(factor_returns, reduced_factors, factor_meta):
    if factor_returns.empty:
        return pd.DataFrame()
    diagnostics = pd.DataFrame(
        {
            "factor_id": factor_returns.columns.astype(str),
            "coverage_ratio": factor_returns.notna().mean().to_numpy(dtype=float),
            "mean_return": factor_returns.mean().to_numpy(dtype=float),
            "volatility": factor_returns.std().to_numpy(dtype=float),
            "selected_for_model": factor_returns.columns.isin(reduced_factors.columns),
        }
    )
    if not factor_meta.empty:
        diagnostics = diagnostics.merge(factor_meta, on="factor_id", how="left")
    return diagnostics.sort_values("factor_id").reset_index(drop=True)


def _build_expected_return_outputs(current_betas, reduced_factors):
    if current_betas.empty or reduced_factors.empty:
        return pd.DataFrame(), pd.DataFrame()

    factor_premia = reduced_factors.mean()
    beta_columns = [col for col in current_betas.columns if col.startswith("beta__")]
    long_rows = []
    expected_rows = []
    for _, row in current_betas.iterrows():
        expected_return = float(row.get("alpha", 0.0))
        for beta_column in beta_columns:
            factor_id = beta_column.replace("beta__", "", 1)
            beta_value = pd.to_numeric(row.get(beta_column), errors="coerce")
            if not np.isfinite(beta_value):
                continue
            premium = float(
                pd.to_numeric(factor_premia.get(factor_id), errors="coerce")
            )
            if np.isfinite(premium):
                expected_return += float(beta_value) * premium
            long_rows.append(
                {
                    "conid": row["conid"],
                    "sleeve": row["sleeve"],
                    "factor_id": factor_id,
                    "beta": float(beta_value),
                }
            )
        expected_rows.append(
            {
                "conid": row["conid"],
                "sleeve": row["sleeve"],
                "alpha": float(pd.to_numeric(row.get("alpha"), errors="coerce")),
                "expected_return": float(expected_return),
            }
        )
    return pd.DataFrame(expected_rows), pd.DataFrame(long_rows)


def run_factor_research_data(
    panel,
    prices,
    risk_free_daily,
    config,
    show_progress=False,
):
    factor_returns, factor_meta, baseline_returns, baseline_members = (
        build_factor_returns(
            panel,
            prices,
            risk_free_daily,
            config,
            show_progress=show_progress,
        )
    )
    cluster_df, reduced_factors = cluster_factor_returns(
        factor_returns, factor_meta, config, show_progress=show_progress
    )
    factor_diagnostics = _factor_diagnostics(
        factor_returns, reduced_factors, factor_meta
    )
    returns_wide = _build_returns_wide(prices, config.return_alignment_max_gap_days)
    risk_free_series = _build_risk_free_series(risk_free_daily)

    if factor_returns.empty or reduced_factors.empty or returns_wide.empty:
        empty = pd.DataFrame()
        return {
            "factor_returns": factor_returns,
            "factor_meta": factor_meta,
            "factor_clusters": cluster_df,
            "factor_diagnostics": factor_diagnostics,
            "baseline_returns": baseline_returns,
            "baseline_members": baseline_members,
            "model_results": empty,
            "factor_persistence": empty,
            "current_betas": empty,
            "asset_expected_returns": empty,
            "asset_factor_betas": empty,
        }

    research_rows = []
    selection_rows = []
    train_test_windows = _build_research_windows(panel, reduced_factors, config)
    sleeve_specs = []
    for sleeve in sorted(panel["sleeve"].dropna().unique()):
        sleeve_conids = sorted(
            panel.loc[panel["sleeve"] == sleeve, "conid"].astype(str).unique()
        )
        sleeve_factors = sorted(
            [col for col in reduced_factors.columns if col.startswith(f"{sleeve}__")]
        )
        if sleeve_conids and sleeve_factors:
            sleeve_specs.append((sleeve, sleeve_conids, sleeve_factors))

    total_model_fits = sum(
        len(sleeve_conids) * len(train_test_windows)
        for _, sleeve_conids, _ in sleeve_specs
    )
    with make_progress_bar(
        show_progress=show_progress,
        total=total_model_fits,
        desc="Research model fits",
        unit="fit",
        leave=False,
    ) as progress_bar:
        for sleeve, sleeve_conids, sleeve_factors in sleeve_specs:
            progress_bar.set_postfix_str(str(sleeve), refresh=False)
            for train_start, train_end, test_end, window_years in train_test_windows:
                X_train = reduced_factors.loc[
                    (reduced_factors.index > train_start)
                    & (reduced_factors.index <= train_end),
                    sleeve_factors,
                ].dropna(how="all")
                X_test = reduced_factors.loc[
                    (reduced_factors.index > train_end)
                    & (reduced_factors.index <= test_end),
                    sleeve_factors,
                ].dropna(how="all")
                if (
                    len(X_train) < config.min_train_days
                    or len(X_test) < config.min_test_days
                ):
                    progress_bar.update(len(sleeve_conids))
                    continue

                for conid in sleeve_conids:
                    y = returns_wide.get(conid)
                    if y is None:
                        progress_bar.update(1)
                        continue
                    if config.use_risk_free_excess:
                        y_target = y.subtract(risk_free_series, fill_value=0.0)
                    else:
                        y_target = y.copy()

                    train = pd.concat(
                        [X_train, y_target.rename("target")], axis=1
                    ).dropna()
                    test = pd.concat(
                        [X_test, y_target.rename("target")], axis=1
                    ).dropna()
                    if (
                        len(train) < config.min_train_days
                        or len(test) < config.min_test_days
                    ):
                        progress_bar.update(1)
                        continue

                    X_train_fit = np.asarray(
                        train[sleeve_factors].to_numpy(), dtype=float
                    )
                    y_train_fit = np.asarray(train["target"].to_numpy(), dtype=float)
                    X_test_fit = np.asarray(
                        test[sleeve_factors].to_numpy(), dtype=float
                    )
                    y_test_fit = np.asarray(test["target"].to_numpy(), dtype=float)

                    try:
                        pipeline, model, intercept, coefs = _fit_elastic_net(
                            X_train_fit, y_train_fit
                        )
                    except ValueError:
                        progress_bar.update(1)
                        continue

                    pred_train = pipeline.predict(X_train_fit)
                    pred_test = pipeline.predict(X_test_fit)
                    selected = []
                    nonzero = int(np.sum(np.abs(coefs) > 1e-8))
                    for factor_id, beta in zip(sleeve_factors, coefs):
                        if abs(beta) <= 1e-8:
                            continue
                        selected.append(factor_id)
                        selection_rows.append(
                            {
                                "sleeve": sleeve,
                                "conid": conid,
                                "train_start": train_start,
                                "train_end": train_end,
                                "test_end": test_end,
                                "training_window_years": window_years,
                                "factor_id": factor_id,
                                "beta": float(beta),
                                "abs_beta": float(abs(beta)),
                                "sign": float(np.sign(beta)),
                            }
                        )

                    research_row = {
                        "sleeve": sleeve,
                        "conid": conid,
                        "train_start": train_start,
                        "train_end": train_end,
                        "test_end": test_end,
                        "training_window_years": window_years,
                        "alpha": float(intercept),
                        "enet_alpha": float(model.alpha_),
                        "l1_ratio": float(model.l1_ratio_),
                        "n_iter": int(model.n_iter_),
                        "dual_gap": float(model.dual_gap_),
                        "n_nonzero": nonzero,
                        "selected_factor_count": int(len(selected)),
                        "selected_factors": "|".join(sorted(selected)),
                        "mse_train": float(mean_squared_error(y_train_fit, pred_train)),
                        "mse_test": float(mean_squared_error(y_test_fit, pred_test)),
                        "r2_train": float(r2_score(y_train_fit, pred_train)),
                        "r2_test": float(r2_score(y_test_fit, pred_test)),
                        "cv_mse_best": float(np.min(model.mse_path_.mean(axis=1))),
                        "cv_mse_average": float(np.mean(model.mse_path_.mean(axis=1))),
                        "cv_mse_worst": float(np.max(model.mse_path_.mean(axis=1))),
                    }
                    for factor_id, beta in zip(sleeve_factors, coefs):
                        research_row[f"beta__{factor_id}"] = float(beta)
                    research_rows.append(research_row)
                    progress_bar.update(1)

    model_results = pd.DataFrame(research_rows)
    selections = pd.DataFrame(selection_rows)
    persistence = pd.DataFrame()
    if not selections.empty and not model_results.empty:
        sleeve_fit_counts = model_results.groupby("sleeve").size()
        fit_counts = pd.DataFrame(
            {
                "sleeve": sleeve_fit_counts.index.astype(str),
                "model_fit_count": sleeve_fit_counts.to_numpy(),
            }
        )
        persistence = selections.groupby(["sleeve", "factor_id"], as_index=False).agg(
            selection_count=("factor_id", "size"),
            median_abs_beta=("abs_beta", "median"),
            sign_consistency=("sign", lambda s: float(abs(np.nanmean(s)))),
        )
        persistence = persistence.merge(fit_counts, on="sleeve", how="left")
        persistence["selection_frequency"] = (
            persistence["selection_count"] / persistence["model_fit_count"]
        )
        persistence["is_persistent"] = (
            persistence["selection_count"] >= config.min_selection_count
        ) & (persistence["selection_frequency"] >= config.selection_frequency_threshold)

    current_betas = compute_current_betas_data(
        panel=panel,
        prices=prices,
        reduced_factors=reduced_factors,
        risk_free_daily=risk_free_daily,
        persistence=persistence,
        config=config,
        show_progress=show_progress,
    )
    asset_expected_returns, asset_factor_betas = _build_expected_return_outputs(
        current_betas, reduced_factors
    )

    return {
        "factor_returns": factor_returns,
        "factor_meta": factor_meta,
        "factor_clusters": cluster_df,
        "factor_diagnostics": factor_diagnostics,
        "baseline_returns": baseline_returns,
        "baseline_members": baseline_members,
        "model_results": model_results,
        "factor_persistence": persistence,
        "current_betas": current_betas,
        "asset_expected_returns": asset_expected_returns,
        "asset_factor_betas": asset_factor_betas,
    }


def compute_current_betas_data(
    panel,
    prices,
    reduced_factors,
    risk_free_daily,
    persistence,
    config,
    show_progress=False,
):
    returns_wide = _build_returns_wide(prices, config.return_alignment_max_gap_days)
    risk_free_series = _build_risk_free_series(risk_free_daily)
    if returns_wide.empty or reduced_factors.empty or persistence.empty:
        return pd.DataFrame()

    latest_rebalance = _to_timestamp(panel["rebalance_date"].max())
    latest_panel = panel.loc[
        panel["rebalance_date"] == latest_rebalance, ["conid", "sleeve"]
    ].drop_duplicates()
    last_return_date = _to_timestamp(returns_wide.index.max())
    start_date = last_return_date - pd.Timedelta(
        days=int(config.trailing_beta_days * 1.5)
    )

    rows = []
    beta_fit_total = sum(
        len(sleeve_panel)
        for sleeve, sleeve_panel in latest_panel.groupby("sleeve")
        if not persistence.loc[
            (persistence["sleeve"] == sleeve) & (persistence["is_persistent"]),
            "factor_id",
        ].empty
    )
    with make_progress_bar(
        show_progress=show_progress,
        total=beta_fit_total,
        desc="Current beta fits",
        unit="fit",
        leave=False,
    ) as progress_bar:
        for sleeve, sleeve_panel in latest_panel.groupby("sleeve"):
            progress_bar.set_postfix_str(str(sleeve), refresh=False)
            persistent_factors = persistence.loc[
                (persistence["sleeve"] == sleeve) & (persistence["is_persistent"]),
                "factor_id",
            ].tolist()
            if not persistent_factors:
                continue

            X = reduced_factors.loc[
                reduced_factors.index >= start_date, persistent_factors
            ].dropna()
            if len(X) < config.min_test_days:
                progress_bar.update(len(sleeve_panel))
                continue

            for conid in sleeve_panel["conid"].astype(str).tolist():
                y = returns_wide.get(conid)
                if y is None:
                    progress_bar.update(1)
                    continue
                if config.use_risk_free_excess:
                    y_target = y.subtract(risk_free_series, fill_value=0.0)
                else:
                    y_target = y.copy()
                data = pd.concat([X, y_target.rename("target")], axis=1).dropna()
                if len(data) < config.min_test_days:
                    progress_bar.update(1)
                    continue

                model = LinearRegression()
                X_fit = data[persistent_factors].to_numpy(dtype=float)
                y_fit = data["target"].to_numpy(dtype=float)
                model.fit(X_fit, y_fit)
                rows.append(
                    {
                        "conid": conid,
                        "sleeve": sleeve,
                        "window_start": _to_timestamp(data.index.min()),
                        "window_end": _to_timestamp(data.index.max()),
                        "n_obs": int(len(data)),
                        "alpha": float(model.intercept_),
                        "r2": float(model.score(X_fit, y_fit)),
                        **{
                            f"beta__{factor_id}": float(beta)
                            for factor_id, beta in zip(persistent_factors, model.coef_)
                        },
                    }
                )
                progress_bar.update(1)

    return pd.DataFrame(rows)


def build_analysis_panel(
    sqlite_path=SQLITE_DB_PATH,
    output_dir=None,
    show_progress=False,
    **config_kwargs,
):
    config = AnalysisConfig(
        sqlite_path=Path(sqlite_path),
        output_dir=Path(output_dir or (DATA_DIR / "analysis")),
        **config_kwargs,
    )
    (
        snapshot_features,
        price_result,
        _risk_free_daily,
        world_bank_country_features,
    ) = _prepare_analysis_inputs(config, show_progress=show_progress)
    panel = build_analysis_panel_data(
        snapshot_features,
        price_result,
        config,
        world_bank_country_features=world_bank_country_features,
        show_progress=show_progress,
    )

    with transaction(config.sqlite_path) as tx:
        panel_path = _write_output(
            "analysis_snapshot_panel",
            panel,
            config.output_dir,
            config.sqlite_path,
            tx=tx,
        )
    return {
        "status": "ok",
        "rows": int(len(panel)),
        "rebalance_dates": int(panel["rebalance_date"].nunique())
        if not panel.empty
        else 0,
        "snapshot_panel_path": panel_path,
    }


def run_factor_research(
    sqlite_path=SQLITE_DB_PATH,
    output_dir=None,
    show_progress=False,
    **config_kwargs,
):
    config = AnalysisConfig(
        sqlite_path=Path(sqlite_path),
        output_dir=Path(output_dir or (DATA_DIR / "analysis")),
        **config_kwargs,
    )
    (
        snapshot_features,
        price_result,
        risk_free_daily,
        world_bank_country_features,
    ) = _prepare_analysis_inputs(config, show_progress=show_progress)
    panel = build_analysis_panel_data(
        snapshot_features,
        price_result,
        config,
        world_bank_country_features=world_bank_country_features,
        show_progress=show_progress,
    )
    research = run_factor_research_data(
        panel,
        price_result["prices"],
        risk_free_daily,
        config,
        show_progress=show_progress,
    )

    factor_returns_wide = research["factor_returns"].copy()
    factor_returns_long = pd.DataFrame()
    if not factor_returns_wide.empty:
        factor_returns_long = (
            factor_returns_wide.reset_index()
            .rename(columns={"index": "trade_date"})
            .melt(
                id_vars=["trade_date"], var_name="factor_id", value_name="factor_return"
            )
            .dropna(subset=["factor_return"])
        )

    with transaction(config.sqlite_path) as tx:
        paths = {
            "snapshot_panel_path": _write_output(
                "analysis_snapshot_panel",
                panel,
                config.output_dir,
                config.sqlite_path,
                tx=tx,
            ),
            "factor_returns_path": _write_output(
                "analysis_factor_returns",
                factor_returns_wide,
                config.output_dir,
                config.sqlite_path,
                long_sql_df=factor_returns_long,
                tx=tx,
            )
            if not factor_returns_wide.empty
            else None,
            "factor_clusters_path": _write_output(
                "analysis_factor_clusters",
                research["factor_clusters"],
                config.output_dir,
                config.sqlite_path,
                tx=tx,
            ),
            "factor_diagnostics_path": _write_output(
                "analysis_factor_diagnostics",
                research["factor_diagnostics"],
                config.output_dir,
                config.sqlite_path,
                tx=tx,
            ),
            "factor_persistence_path": _write_output(
                "analysis_factor_persistence",
                research["factor_persistence"],
                config.output_dir,
                config.sqlite_path,
                tx=tx,
            ),
            "model_results_path": _write_output(
                "analysis_model_results",
                research["model_results"],
                config.output_dir,
                config.sqlite_path,
                tx=tx,
            ),
            "current_betas_path": _write_output(
                "analysis_current_betas",
                research["current_betas"],
                config.output_dir,
                config.sqlite_path,
                tx=tx,
            ),
            "asset_expected_returns_path": _write_output(
                "analysis_asset_expected_returns",
                research["asset_expected_returns"],
                config.output_dir,
                config.sqlite_path,
                tx=tx,
            ),
            "asset_factor_betas_path": _write_output(
                "analysis_asset_factor_betas",
                research["asset_factor_betas"],
                config.output_dir,
                config.sqlite_path,
                tx=tx,
            ),
            "baseline_members_path": _write_output(
                "analysis_bond_baseline_members",
                research["baseline_members"],
                config.output_dir,
                config.sqlite_path,
                tx=tx,
            ),
        }

    persistent = research["factor_persistence"]
    return {
        "status": "ok",
        "snapshot_rows": int(len(panel)),
        "factor_count": int(factor_returns_wide.shape[1])
        if not factor_returns_wide.empty
        else 0,
        "persistent_factor_count": int(persistent["is_persistent"].sum())
        if not persistent.empty
        else 0,
        **paths,
    }


def compute_current_betas(
    sqlite_path=SQLITE_DB_PATH,
    output_dir=None,
    show_progress=False,
    **config_kwargs,
):
    result = run_factor_research(
        sqlite_path=sqlite_path,
        output_dir=output_dir,
        show_progress=show_progress,
        **config_kwargs,
    )
    return {
        "status": result["status"],
        "current_betas_path": result["current_betas_path"],
        "persistent_factor_count": result["persistent_factor_count"],
    }


def run_analysis_pipeline(
    sqlite_path=SQLITE_DB_PATH,
    output_dir=None,
    show_progress=False,
    **config_kwargs,
):
    return run_factor_research(
        sqlite_path=sqlite_path,
        output_dir=output_dir,
        show_progress=show_progress,
        **config_kwargs,
    )
