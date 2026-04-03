from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LinearRegression
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
from ..storage import replace_table, transaction


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
    replace_table(name, sql_df, sqlite_path=sqlite_path, tx=tx, index=False)

    return str(parquet_path)


def _series_or_zero(df, column):
    if column in df.columns:
        return pd.Series(
            pd.to_numeric(df[column], errors="coerce"), index=df.index
        ).fillna(0.0)
    return pd.Series(0.0, index=df.index)


def _text_series(df, column):
    if column in df.columns:
        return pd.Series(df[column], index=df.index).fillna("").astype(str)
    return pd.Series("", index=df.index, dtype=object)


def load_snapshot_features(sqlite_path=SQLITE_DB_PATH):
    return load_preprocessed_snapshot_features(sqlite_path=sqlite_path)


def _build_price_features(prices):
    clean = prices.loc[prices["is_clean_price"]].copy()
    if clean.empty:
        return _empty_frame(["conid", "trade_date"])

    clean = clean.sort_values(["conid", "trade_date"])
    frames = []
    for conid, group in clean.groupby("conid"):
        g = group.copy()
        g["price_feature__momentum_21"] = g["clean_price"].pct_change(21)
        g["price_feature__momentum_63"] = g["clean_price"].pct_change(63)
        g["price_feature__momentum_126"] = g["clean_price"].pct_change(126)
        g["price_feature__momentum_252"] = g["clean_price"].pct_change(252)
        g["price_feature__volatility_21"] = g["clean_return"].rolling(
            21, min_periods=10
        ).std() * np.sqrt(252.0)
        g["price_feature__volatility_63"] = g["clean_return"].rolling(
            63, min_periods=21
        ).std() * np.sqrt(252.0)
        g["price_feature__downside_volatility_63"] = g["clean_return"].where(
            g["clean_return"] < 0.0
        ).rolling(63, min_periods=21).std() * np.sqrt(252.0)
        rolling_peak = g["clean_price"].rolling(126, min_periods=21).max()
        drawdown = g["clean_price"] / rolling_peak - 1.0
        g["price_feature__max_drawdown_126"] = drawdown.rolling(
            126, min_periods=21
        ).min()
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
    rebalance_dates = _build_rebalance_dates(
        snapshot_features, prices, config.rebalance_freq
    )
    eligible_conids = set(eligibility.loc[eligibility["eligible"], "conid"].astype(str))

    panels = []
    for rebalance_date in rebalance_dates:
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


def _build_returns_wide(prices):
    clean = prices.loc[
        prices["is_clean_price"], ["conid", "trade_date", "clean_return"]
    ].copy()
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
        if (
            col.startswith("holding_maturity__")
            or col.startswith("holding_quality__")
            or col.startswith("debt_type__")
            or col.startswith("ratio_fixed_income__")
        ):
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

    available_ids = [conid for conid in returns_frame.columns if conid in valid.index]
    if len(available_ids) < min_assets:
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


def build_factor_returns(panel, prices, config):
    returns_wide = _build_returns_wide(prices)
    baseline_returns, baseline_members = _build_baseline_returns(panel, returns_wide)
    if panel.empty or returns_wide.empty:
        return pd.DataFrame(), pd.DataFrame(), baseline_returns, baseline_members

    factor_map = {}
    metadata = {}
    rebalance_dates = sorted(pd.to_datetime(panel["rebalance_date"]).unique())

    for start, end in zip(rebalance_dates[:-1], rebalance_dates[1:]):
        interval_returns = returns_wide.loc[
            (returns_wide.index > start) & (returns_wide.index <= end)
        ]
        if interval_returns.empty:
            continue

        panel_slice = panel.loc[panel["rebalance_date"] == start].copy()
        for sleeve in sorted(panel_slice["sleeve"].dropna().unique()):
            sleeve_slice = panel_slice.loc[panel_slice["sleeve"] == sleeve].copy()
            if len(sleeve_slice) < config.min_assets_per_factor:
                continue

            size_weights = sleeve_slice.set_index("conid")[
                "profile__total_net_assets_num"
            ]
            market_weights = _normalized_weights(
                size_weights.reindex(sleeve_slice["conid"])
            )
            market_series = (
                interval_returns.reindex(columns=list(market_weights.index))
                .fillna(0.0)
                .dot(market_weights)
            )
            market_factor_id = f"{sleeve}__market"
            factor_map.setdefault(market_factor_id, []).append(
                market_series.rename(market_factor_id)
            )
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


def cluster_factor_returns(factor_returns, factor_meta, config):
    if factor_returns.empty or factor_meta.empty:
        return _empty_frame(
            [
                "factor_id",
                "cluster_id",
                "cluster_representative",
                "cluster_size",
                "keep_factor",
            ]
        ), pd.DataFrame()

    cluster_rows = []
    keepers = []

    for sleeve, meta_slice in factor_meta.groupby("sleeve"):
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
                .map({"composite": 0, "market": 1, "raw": 2})
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
            (
                "enet",
                ElasticNetCV(**elastic_net_params),
            ),
        ]
    )
    typed_pipeline = pipeline
    typed_pipeline.fit(X_train, y_train)
    scaler: Any = typed_pipeline.named_steps["scaler"]
    model: Any = typed_pipeline.named_steps["enet"]
    coefs = model.coef_ / scaler.scale_
    intercept = model.intercept_ - np.dot(coefs, scaler.mean_)
    return typed_pipeline, intercept, coefs


def run_factor_research_data(panel, prices, config):
    factor_returns, factor_meta, baseline_returns, baseline_members = (
        build_factor_returns(panel, prices, config)
    )
    cluster_df, reduced_factors = cluster_factor_returns(
        factor_returns, factor_meta, config
    )
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
        sleeve_conids = sorted(
            panel.loc[panel["sleeve"] == sleeve, "conid"].astype(str).unique()
        )
        sleeve_factors = sorted(
            [col for col in reduced_factors.columns if col.startswith(f"{sleeve}__")]
        )
        if not sleeve_conids or not sleeve_factors:
            continue

        for train_end, test_end in zip(unique_snapshots[2:-1], unique_snapshots[3:]):
            X_train = reduced_factors.loc[
                reduced_factors.index <= train_end, sleeve_factors
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
                continue

            for conid in sleeve_conids:
                y = returns_wide.get(conid)
                if y is None:
                    continue
                y_excess = y.subtract(baseline_returns, fill_value=0.0)
                train = pd.concat([X_train, y_excess.rename("target")], axis=1).dropna()
                test = pd.concat([X_test, y_excess.rename("target")], axis=1).dropna()
                if (
                    len(train) < config.min_train_days
                    or len(test) < config.min_test_days
                ):
                    continue

                X_train_fit = np.asarray(train[sleeve_factors].to_numpy(), dtype=float)
                y_train_fit = np.asarray(train["target"].to_numpy(), dtype=float)
                X_test_fit = np.asarray(test[sleeve_factors].to_numpy(), dtype=float)
                y_test_fit = np.asarray(test["target"].to_numpy(), dtype=float)

                try:
                    pipeline, intercept, coefs = _fit_elastic_net(
                        X_train_fit, y_train_fit
                    )
                except ValueError:
                    continue

                preds = pipeline.predict(X_test_fit)
                denom = float(np.sum((y_test_fit - y_test_fit.mean()) ** 2))
                r2_test = (
                    float(1.0 - np.sum((y_test_fit - preds) ** 2) / denom)
                    if denom > 0
                    else np.nan
                )

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


def compute_current_betas_data(
    panel, prices, reduced_factors, baseline_returns, persistence, config
):
    returns_wide = _build_returns_wide(prices)
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
    for sleeve, sleeve_panel in latest_panel.groupby("sleeve"):
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

    return pd.DataFrame(rows)


def build_analysis_panel(sqlite_path=SQLITE_DB_PATH, output_dir=None, **config_kwargs):
    config = AnalysisConfig(
        sqlite_path=Path(sqlite_path),
        output_dir=Path(output_dir or (DATA_DIR / "analysis")),
        **config_kwargs,
    )
    price_config = PricePreprocessConfig(outlier_z_threshold=config.outlier_z_threshold)
    price_result = preprocess_price_history(
        load_price_history(config.sqlite_path), config=price_config
    )
    save_price_preprocess_results(price_result, output_dir=config.output_dir)
    snapshot_features = load_snapshot_features(config.sqlite_path)
    panel = build_analysis_panel_data(snapshot_features, price_result, config)

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


def run_factor_research(sqlite_path=SQLITE_DB_PATH, output_dir=None, **config_kwargs):
    config = AnalysisConfig(
        sqlite_path=Path(sqlite_path),
        output_dir=Path(output_dir or (DATA_DIR / "analysis")),
        **config_kwargs,
    )
    price_config = PricePreprocessConfig(outlier_z_threshold=config.outlier_z_threshold)
    price_result = preprocess_price_history(
        load_price_history(config.sqlite_path), config=price_config
    )
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


def compute_current_betas(sqlite_path=SQLITE_DB_PATH, output_dir=None, **config_kwargs):
    result = run_factor_research(
        sqlite_path=sqlite_path, output_dir=output_dir, **config_kwargs
    )
    return {
        "status": result["status"],
        "current_betas_path": result["current_betas_path"],
        "persistent_factor_count": result["persistent_factor_count"],
    }


def run_analysis_pipeline(sqlite_path=SQLITE_DB_PATH, output_dir=None, **config_kwargs):
    panel_result = build_analysis_panel(
        sqlite_path=sqlite_path, output_dir=output_dir, **config_kwargs
    )
    research_result = run_factor_research(
        sqlite_path=sqlite_path, output_dir=output_dir, **config_kwargs
    )
    return {
        "status": "ok",
        "panel": panel_result,
        "research": research_result,
    }
