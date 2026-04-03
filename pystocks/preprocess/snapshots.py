from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DATA_DIR, SQLITE_DB_PATH
from ..storage import load_snapshot_feature_tables as _load_snapshot_feature_tables


@dataclass
class SnapshotPreprocessConfig:
    holdings_sum_tolerance: float = 0.05
    sparse_category_threshold: int = 1
    include_deferred_families: bool = True


SNAPSHOT_TABLE_COLUMNS = {
    "profile_and_fees": ["conid", "effective_at"],
    "holdings_asset_type": ["conid", "effective_at"],
    "holdings_debtor_quality": ["conid", "effective_at"],
    "holdings_maturity": ["conid", "effective_at"],
    "holdings_industry": ["conid", "effective_at", "industry", "value_num"],
    "holdings_currency": ["conid", "effective_at", "code", "currency", "value_num"],
    "holdings_investor_country": [
        "conid",
        "effective_at",
        "country_code",
        "country",
        "value_num",
    ],
    "holdings_geographic_weights": ["conid", "effective_at", "region", "value_num"],
    "holdings_debt_type": ["conid", "effective_at", "debt_type", "value_num"],
    "holdings_top10": ["conid", "effective_at", "name", "holding_weight_num"],
    "ratios_key_ratios": ["conid", "effective_at", "metric_id", "value_num", "vs_num"],
    "ratios_financials": ["conid", "effective_at", "metric_id", "value_num", "vs_num"],
    "ratios_fixed_income": [
        "conid",
        "effective_at",
        "metric_id",
        "value_num",
        "vs_num",
    ],
    "ratios_dividend": ["conid", "effective_at", "metric_id", "value_num", "vs_num"],
    "ratios_zscore": ["conid", "effective_at", "metric_id", "value_num", "vs_num"],
    "performance": [
        "conid",
        "effective_at",
        "section",
        "metric_id",
        "value_num",
        "vs_num",
    ],
    "dividends_industry_metrics": ["conid", "effective_at"],
    "morningstar_summary": ["conid", "effective_at"],
    "lipper_ratings": ["conid", "effective_at", "period", "metric_id", "rating_value"],
}


def _empty_frame(columns):
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


def _empty_table(name):
    return _empty_frame(SNAPSHOT_TABLE_COLUMNS.get(name, ["conid", "effective_at"]))


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


def _normalize_snapshot_frame(df):
    if df.empty:
        return df.copy()
    out = df.copy()
    if "conid" in out.columns:
        out["conid"] = out["conid"].astype(str)
    if "effective_at" in out.columns:
        out["effective_at"] = pd.to_datetime(out["effective_at"])
    return out


def _normalize_snapshot_tables(tables):
    normalized = {}
    for name in SNAPSHOT_TABLE_COLUMNS:
        normalized[name] = _normalize_snapshot_frame(
            tables.get(name, _empty_table(name))
        )
    return normalized


def _prefix_frame(df, prefix, keep=("conid", "effective_at")):
    if df.empty:
        return df.copy()
    renamed = {}
    for col in df.columns:
        if col in keep:
            continue
        renamed[col] = f"{prefix}__{_sanitize_segment(col)}"
    return df.rename(columns=renamed)


def _pivot_series_frame(df, key_col, value_col, prefix):
    if df.empty:
        return _empty_frame(["conid", "effective_at"])
    work = df.copy()
    work["pivot_key"] = work[key_col].map(_sanitize_segment)
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    sort_cols = ["conid", "effective_at", "pivot_key", value_col]
    work = work.sort_values(sort_cols, na_position="last")
    pivoted = work.pivot_table(
        index=["conid", "effective_at"],
        columns="pivot_key",
        values=value_col,
        aggfunc="first",
    ).reset_index()
    pivoted.columns = [
        col if col in {"conid", "effective_at"} else f"{prefix}__{col}"
        for col in pivoted.columns
    ]
    return pivoted.sort_values(["conid", "effective_at"]).reset_index(drop=True)


def _pivot_metric_frame(df, prefix, key_cols):
    if df.empty:
        return _empty_frame(["conid", "effective_at"])
    work = df.copy()
    work["pivot_key"] = (
        work[key_cols].astype(str).agg("__".join, axis=1).map(_sanitize_segment)
    )
    value_pivot = _pivot_series_frame(
        work[["conid", "effective_at", "pivot_key", "value_num"]],
        "pivot_key",
        "value_num",
        prefix,
    )
    if "vs_num" in work.columns:
        vs_work = work[["conid", "effective_at", "pivot_key", "vs_num"]].rename(
            columns={"vs_num": "value_num"}
        )
        vs_pivot = _pivot_series_frame(
            vs_work, "pivot_key", "value_num", f"{prefix}_vs"
        )
        value_pivot = value_pivot.merge(
            vs_pivot, on=["conid", "effective_at"], how="outer"
        )
    return value_pivot.sort_values(["conid", "effective_at"]).reset_index(drop=True)


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


def _summarize_source_table(df, table_name):
    if df.empty:
        return pd.DataFrame(
            [
                {
                    "table_name": table_name,
                    "row_count": 0,
                    "key_count": 0,
                    "conid_count": 0,
                    "min_effective_at": pd.NaT,
                    "max_effective_at": pd.NaT,
                }
            ]
        )
    key_count = (
        int(df[["conid", "effective_at"]].drop_duplicates().shape[0])
        if {"conid", "effective_at"}.issubset(df.columns)
        else 0
    )
    return pd.DataFrame(
        [
            {
                "table_name": table_name,
                "row_count": int(len(df)),
                "key_count": key_count,
                "conid_count": int(df["conid"].nunique())
                if "conid" in df.columns
                else 0,
                "min_effective_at": df["effective_at"].min()
                if "effective_at" in df.columns
                else pd.NaT,
                "max_effective_at": df["effective_at"].max()
                if "effective_at" in df.columns
                else pd.NaT,
            }
        ]
    )


def _apply_holdings_flags(df, config):
    if df.empty:
        return df.copy()
    out = df.copy()
    tolerance = float(config.holdings_sum_tolerance)
    out["is_sum_near_one"] = out["value_sum"].notna() & out["value_sum"].sub(
        1.0
    ).abs().le(tolerance)
    out["is_sum_over_one"] = out["value_sum"].notna() & out["value_sum"].gt(
        1.0 + tolerance
    )
    out["is_sparse_category_coverage"] = (
        out["category_count"].fillna(0).le(config.sparse_category_threshold)
    )
    return out


def _wide_holdings_diagnostics(df, table_name, value_columns, config):
    if df.empty:
        return _empty_frame(
            [
                "conid",
                "effective_at",
                "table_name",
                "value_sum",
                "category_count",
                "max_value",
                "is_sum_near_one",
                "is_sum_over_one",
                "is_sparse_category_coverage",
            ]
        )
    numeric = df[value_columns].apply(pd.to_numeric, errors="coerce")
    diagnostics = pd.DataFrame(
        {
            "conid": df["conid"].astype(str),
            "effective_at": pd.to_datetime(df["effective_at"]),
            "table_name": table_name,
            "value_sum": numeric.sum(axis=1, min_count=1),
            "category_count": numeric.fillna(0.0).abs().gt(0.0).sum(axis=1),
            "max_value": numeric.max(axis=1, skipna=True),
        }
    )
    return (
        _apply_holdings_flags(diagnostics, config)
        .sort_values(["conid", "effective_at"])
        .reset_index(drop=True)
    )


def _long_holdings_diagnostics(df, table_name, key_col, value_col, config):
    if df.empty:
        return _empty_frame(
            [
                "conid",
                "effective_at",
                "table_name",
                "value_sum",
                "category_count",
                "max_value",
                "is_sum_near_one",
                "is_sum_over_one",
                "is_sparse_category_coverage",
            ]
        )
    work = df.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    diagnostics = (
        work.groupby(["conid", "effective_at"], as_index=False)
        .agg(
            value_sum=(value_col, "sum"),
            category_count=(key_col, "nunique"),
            max_value=(value_col, "max"),
        )
        .assign(table_name=table_name)
    )
    return (
        _apply_holdings_flags(diagnostics, config)
        .sort_values(["conid", "effective_at"])
        .reset_index(drop=True)
    )


def _top10_features_and_diagnostics(df, config):
    if df.empty:
        empty_features = _empty_frame(["conid", "effective_at"])
        empty_diag = _empty_frame(
            [
                "conid",
                "effective_at",
                "table_name",
                "value_sum",
                "category_count",
                "max_value",
                "is_sum_near_one",
                "is_sum_over_one",
                "is_sparse_category_coverage",
            ]
        )
        return empty_features, empty_diag

    work = df.copy()
    work["holding_weight_num"] = pd.to_numeric(
        work["holding_weight_num"], errors="coerce"
    )
    aggregated = work.groupby(["conid", "effective_at"], as_index=False).agg(
        top10_count=("name", "nunique"),
        top10_weight_sum=("holding_weight_num", "sum"),
        top10_weight_max=("holding_weight_num", "max"),
    )
    diagnostics = aggregated.rename(
        columns={
            "top10_weight_sum": "value_sum",
            "top10_count": "category_count",
            "top10_weight_max": "max_value",
        }
    )
    diagnostics["table_name"] = "holdings_top10"
    diagnostics = _apply_holdings_flags(diagnostics, config)
    return (
        _prefix_frame(aggregated, "top10")
        .sort_values(["conid", "effective_at"])
        .reset_index(drop=True),
        diagnostics.sort_values(["conid", "effective_at"]).reset_index(drop=True),
    )


def _ratio_diagnostics(df, table_name, key_cols):
    columns = [
        "conid",
        "effective_at",
        "table_name",
        "metric_rows",
        "distinct_metric_keys",
        "duplicate_metric_keys",
        "duplicate_row_count",
        "nonnull_value_rows",
        "null_value_rows",
        "all_values_null",
    ]
    if df.empty:
        return _empty_frame(columns)

    work = df.copy()
    work["value_num"] = pd.to_numeric(work.get("value_num"), errors="coerce")
    work["metric_key"] = (
        work[key_cols].astype(str).agg("__".join, axis=1).map(_sanitize_segment)
    )
    duplicate_rows = (
        work.groupby(["conid", "effective_at", "metric_key"])
        .size()
        .reset_index(name="metric_row_count")
    )
    duplicate_rows = duplicate_rows.loc[duplicate_rows["metric_row_count"] > 1].copy()

    rows = []
    for (conid, effective_at), group in work.groupby(
        ["conid", "effective_at"], sort=True
    ):
        dup_group = duplicate_rows.loc[
            (duplicate_rows["conid"] == conid)
            & (duplicate_rows["effective_at"] == effective_at)
        ]
        metric_rows = int(len(group))
        distinct_metric_keys = int(group["metric_key"].nunique())
        rows.append(
            {
                "conid": str(conid),
                "effective_at": pd.Timestamp(effective_at),
                "table_name": table_name,
                "metric_rows": metric_rows,
                "distinct_metric_keys": distinct_metric_keys,
                "duplicate_metric_keys": int(len(dup_group)),
                "duplicate_row_count": int(metric_rows - distinct_metric_keys),
                "nonnull_value_rows": int(group["value_num"].notna().sum()),
                "null_value_rows": int(group["value_num"].isna().sum()),
                "all_values_null": bool(group["value_num"].notna().sum() == 0),
            }
        )
    return (
        pd.DataFrame(rows)
        .loc[:, columns]
        .sort_values(["table_name", "conid", "effective_at"])
        .reset_index(drop=True)
    )


def load_snapshot_feature_tables(sqlite_path=SQLITE_DB_PATH):
    return _load_snapshot_feature_tables(sqlite_path=sqlite_path)


def preprocess_snapshot_features(tables=None, config=None, sqlite_path=SQLITE_DB_PATH):
    config = config or SnapshotPreprocessConfig()
    tables = (
        load_snapshot_feature_tables(sqlite_path)
        if tables is None
        else _normalize_snapshot_tables(tables)
    )

    frames = []
    holdings_diagnostics = []
    ratio_diagnostics = []
    table_summary = []

    for table_name, df in tables.items():
        table_summary.append(_summarize_source_table(df, table_name))

    profile = tables["profile_and_fees"].copy()
    if not profile.empty:
        if "total_net_assets_value" in profile.columns:
            profile["total_net_assets_num"] = profile["total_net_assets_value"].map(
                _parse_scaled_number
            )
        frames.append(_prefix_frame(profile, "profile"))

    wide_holdings = [
        ("holdings_asset_type", "holding_asset"),
        ("holdings_debtor_quality", "holding_quality"),
        ("holdings_maturity", "holding_maturity"),
    ]
    for table_name, prefix in wide_holdings:
        df = tables[table_name].copy()
        if df.empty:
            continue
        value_columns = [
            col for col in df.columns if col not in {"conid", "effective_at"}
        ]
        holdings_diagnostics.append(
            _wide_holdings_diagnostics(df, table_name, value_columns, config)
        )
        frames.append(_prefix_frame(df, prefix))

    industry = tables["holdings_industry"].copy()
    if not industry.empty:
        holdings_diagnostics.append(
            _long_holdings_diagnostics(
                industry, "holdings_industry", "industry", "value_num", config
            )
        )
        frames.append(
            _pivot_series_frame(industry, "industry", "value_num", "industry")
        )

    currency = tables["holdings_currency"].copy()
    if not currency.empty:
        holdings_diagnostics.append(
            _long_holdings_diagnostics(
                currency, "holdings_currency", "code", "value_num", config
            )
        )
        currency["currency_key"] = currency["code"].where(
            currency["code"].notna(), currency["currency"]
        )
        frames.append(
            _pivot_series_frame(currency, "currency_key", "value_num", "currency")
        )

    country = tables["holdings_investor_country"].copy()
    if not country.empty:
        holdings_diagnostics.append(
            _long_holdings_diagnostics(
                country,
                "holdings_investor_country",
                "country_code",
                "value_num",
                config,
            )
        )
        country["country_key"] = country["country_code"].where(
            country["country_code"].notna(), country["country"]
        )
        frames.append(
            _pivot_series_frame(country, "country_key", "value_num", "country")
        )

    region = tables["holdings_geographic_weights"].copy()
    if not region.empty:
        holdings_diagnostics.append(
            _long_holdings_diagnostics(
                region, "holdings_geographic_weights", "region", "value_num", config
            )
        )
        frames.append(_pivot_series_frame(region, "region", "value_num", "region"))

    debt_type = tables["holdings_debt_type"].copy()
    if not debt_type.empty:
        holdings_diagnostics.append(
            _long_holdings_diagnostics(
                debt_type, "holdings_debt_type", "debt_type", "value_num", config
            )
        )
        frames.append(
            _pivot_series_frame(debt_type, "debt_type", "value_num", "debt_type")
        )

    top10_features, top10_diagnostics = _top10_features_and_diagnostics(
        tables["holdings_top10"], config
    )
    if not top10_features.empty:
        frames.append(top10_features)
    if not top10_diagnostics.empty:
        holdings_diagnostics.append(top10_diagnostics)

    ratio_tables = [
        ("ratios_key_ratios", "ratio_key", ["metric_id"]),
        ("ratios_financials", "ratio_financial", ["metric_id"]),
        ("ratios_fixed_income", "ratio_fixed_income", ["metric_id"]),
        ("ratios_dividend", "ratio_dividend", ["metric_id"]),
        ("ratios_zscore", "ratio_zscore", ["metric_id"]),
    ]
    for table_name, prefix, key_cols in ratio_tables:
        df = tables[table_name].copy()
        if df.empty:
            continue
        ratio_diagnostics.append(_ratio_diagnostics(df, table_name, key_cols))
        frames.append(_pivot_metric_frame(df, prefix, key_cols))

    if config.include_deferred_families:
        performance = tables["performance"].copy()
        if not performance.empty:
            ratio_diagnostics.append(
                _ratio_diagnostics(performance, "performance", ["section", "metric_id"])
            )
            frames.append(
                _pivot_metric_frame(
                    performance, "performance", ["section", "metric_id"]
                )
            )

        dividend_metrics = tables["dividends_industry_metrics"].copy()
        if not dividend_metrics.empty:
            frames.append(_prefix_frame(dividend_metrics, "dividend_metric"))

        morningstar = tables["morningstar_summary"].copy()
        if not morningstar.empty:
            frames.append(_prefix_frame(morningstar, "morningstar"))

        lipper = tables["lipper_ratings"].copy()
        if not lipper.empty:
            ratio_diagnostics.append(
                _ratio_diagnostics(lipper, "lipper_ratings", ["period", "metric_id"])
            )
            frames.append(
                _pivot_metric_frame(lipper, "lipper", ["period", "metric_id"])
            )

    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        features = _empty_frame(["conid", "effective_at", "sleeve"])
    else:
        features = frames[0]
        for frame in frames[1:]:
            features = features.merge(frame, on=["conid", "effective_at"], how="outer")
        features["conid"] = features["conid"].astype(str)
        features["effective_at"] = pd.to_datetime(features["effective_at"])
        features["sleeve"] = features.apply(_assign_sleeve, axis=1)
        features = features.sort_values(["conid", "effective_at"]).reset_index(
            drop=True
        )

    holdings_diag = (
        pd.concat(holdings_diagnostics, ignore_index=True)
        if holdings_diagnostics
        else _empty_frame(
            [
                "conid",
                "effective_at",
                "table_name",
                "value_sum",
                "category_count",
                "max_value",
                "is_sum_near_one",
                "is_sum_over_one",
                "is_sparse_category_coverage",
            ]
        )
    )
    ratio_diag = (
        pd.concat(ratio_diagnostics, ignore_index=True)
        if ratio_diagnostics
        else _empty_frame(
            [
                "conid",
                "effective_at",
                "table_name",
                "metric_rows",
                "distinct_metric_keys",
                "duplicate_metric_keys",
                "duplicate_row_count",
                "nonnull_value_rows",
                "null_value_rows",
                "all_values_null",
            ]
        )
    )
    summary = (
        pd.concat(table_summary, ignore_index=True)
        .sort_values("table_name")
        .reset_index(drop=True)
    )

    return {
        "features": features,
        "holdings_diagnostics": holdings_diag.sort_values(
            ["table_name", "conid", "effective_at"]
        ).reset_index(drop=True),
        "ratio_diagnostics": ratio_diag.sort_values(
            ["table_name", "conid", "effective_at"]
        ).reset_index(drop=True),
        "table_summary": summary,
        "config": config,
    }


def save_snapshot_preprocess_results(result, output_dir=None):
    output_dir = Path(output_dir or (DATA_DIR / "analysis"))
    output_dir.mkdir(parents=True, exist_ok=True)

    features_path = output_dir / "analysis_snapshot_features.parquet"
    holdings_path = output_dir / "analysis_snapshot_holdings_diagnostics.parquet"
    ratio_path = output_dir / "analysis_snapshot_ratio_diagnostics.parquet"
    summary_path = output_dir / "analysis_snapshot_table_summary.parquet"

    result["features"].to_parquet(features_path, index=False)
    result["holdings_diagnostics"].to_parquet(holdings_path, index=False)
    result["ratio_diagnostics"].to_parquet(ratio_path, index=False)
    result["table_summary"].to_parquet(summary_path, index=False)

    return {
        "features_path": str(features_path),
        "holdings_diagnostics_path": str(holdings_path),
        "ratio_diagnostics_path": str(ratio_path),
        "table_summary_path": str(summary_path),
    }


def run_snapshot_preprocess(
    sqlite_path=SQLITE_DB_PATH, output_dir=None, **config_kwargs
):
    config = SnapshotPreprocessConfig(**config_kwargs)
    result = preprocess_snapshot_features(config=config, sqlite_path=sqlite_path)
    paths = save_snapshot_preprocess_results(result, output_dir=output_dir)

    features = result["features"]
    return {
        "status": "ok",
        "rows": int(len(features)),
        "conids": int(features["conid"].nunique()) if not features.empty else 0,
        "holdings_diagnostic_rows": int(len(result["holdings_diagnostics"])),
        "ratio_diagnostic_rows": int(len(result["ratio_diagnostics"])),
        **paths,
    }


def load_snapshot_features(sqlite_path=SQLITE_DB_PATH):
    return preprocess_snapshot_features(sqlite_path=sqlite_path)["features"]
