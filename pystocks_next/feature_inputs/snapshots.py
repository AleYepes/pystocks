from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import pandas as pd

from ..storage.reads import SnapshotFeatureTablesRead, load_snapshot_feature_tables
from .bundle import (
    SNAPSHOT_HOLDINGS_DIAGNOSTIC_COLUMNS,
    SNAPSHOT_RATIO_DIAGNOSTIC_COLUMNS,
    SNAPSHOT_TABLE_SUMMARY_COLUMNS,
    AnalysisInputBundle,
)


@dataclass(frozen=True, slots=True)
class SnapshotInputConfig:
    holdings_sum_tolerance: float = 0.05
    sparse_category_threshold: int = 1


def _empty_frame(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        {column: pd.Series(dtype="object") for column in columns}
    ).reindex(columns=pd.Index(columns))


def _select_frame(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(frame.loc[:, list(columns)].copy())


def _sanitize_segment(value: object) -> str:
    text = str(value or "").strip().lower()
    out: list[str] = []
    last_was_sep = False
    for ch in text:
        if ch.isalnum():
            out.append(ch)
            last_was_sep = False
        elif not last_was_sep:
            out.append("_")
            last_was_sep = True
    return "".join(out).strip("_") or "field"


def _parse_scaled_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None

    negative = text.startswith("(") and text.endswith(")")
    clean = text.strip("()").replace(",", "")
    multiplier = 1.0
    suffix = clean[-1:].upper()
    if suffix in {"K", "M", "B", "T"}:
        multiplier = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[suffix]
        clean = clean[:-1]
    clean = clean.replace("$", "").replace("€", "").replace("£", "").replace("¥", "")
    try:
        parsed = float(clean) * multiplier
    except ValueError:
        return None
    return -parsed if negative else parsed


def _is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, pd.Series | pd.DataFrame):
        return False
    return bool(pd.isna(value))


def _factor_values(frame: pd.DataFrame) -> pd.Series:
    out = pd.Series(index=frame.index, dtype="object")
    if "value_num" in frame.columns:
        numeric_values = pd.Series(
            pd.to_numeric(frame["value_num"], errors="coerce"),
            index=frame.index,
            dtype="object",
        )
        out.loc[:] = numeric_values

    if "value_text" in frame.columns:
        text_values = pd.Series(frame["value_text"], index=frame.index, dtype="object")
        text_mask = out.isna() & text_values.notna()
        out.loc[text_mask] = text_values.loc[text_mask]

    if "value_date" in frame.columns:
        date_values = pd.Series(frame["value_date"], index=frame.index, dtype="object")
        date_mask = out.isna() & date_values.notna()
        out.loc[date_mask] = date_values.loc[date_mask]

    if "value_bool" in frame.columns:
        bool_values = pd.Series(
            pd.to_numeric(frame["value_bool"], errors="coerce"),
            index=frame.index,
            dtype="object",
        )
        bool_mask = out.isna() & bool_values.notna()
        out.loc[bool_mask] = bool_values.loc[bool_mask]
    return out


def _profile_value_rows(
    frame: pd.DataFrame,
    *,
    id_column: str,
    prefix: str = "profile",
) -> pd.DataFrame:
    if frame.empty:
        return _empty_frame(("conid", "effective_at"))
    work = frame.copy()
    work["value"] = _factor_values(work)
    return _pivot_keyed_values(
        _select_frame(work, ("conid", "effective_at", id_column, "value")),
        key_column=id_column,
        value_column="value",
        prefix=prefix,
    )


def _build_profile_features(
    snapshot_tables: dict[str, pd.DataFrame],
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []

    overview = snapshot_tables["profile_overview"].copy()
    if not overview.empty:
        overview_rows: list[dict[str, object]] = []
        for _, row in overview.iterrows():
            for field_id in ("symbol", "objective", "jap_fund_warning"):
                value = row.get(field_id)
                if _is_missing_scalar(value):
                    continue
                overview_rows.append(
                    {
                        "conid": row["conid"],
                        "effective_at": row["effective_at"],
                        "field_id": field_id,
                        "value": value,
                    }
                )
        if overview_rows:
            frames.append(
                _pivot_keyed_values(
                    pd.DataFrame(overview_rows),
                    key_column="field_id",
                    value_column="value",
                    prefix="profile",
                )
            )

    profile_fields = snapshot_tables["profile_fields"].copy()
    if not profile_fields.empty:
        field_rows = profile_fields.copy()
        field_rows["feature_id"] = field_rows["field_id"].replace(
            {
                "launch_opening_price": "inception_date",
                "total_net_assets_month_end": "total_net_assets_value",
            }
        )
        frames.append(_profile_value_rows(field_rows, id_column="feature_id"))

        net_asset_dates = field_rows.loc[
            field_rows["field_id"].eq("total_net_assets_month_end")
            & field_rows["value_date"].notna()
        ].copy()
        if not net_asset_dates.empty:
            net_asset_dates["feature_id"] = "total_net_assets_date"
            net_asset_dates["value_text"] = None
            net_asset_dates["value_num"] = None
            frames.append(_profile_value_rows(net_asset_dates, id_column="feature_id"))

    for report_type in ("annual", "prospectus"):
        table_name = f"profile_{report_type}_report"
        report_data = snapshot_tables[table_name].copy()
        if not report_data.empty:
            report_data["feature_id"] = (
                f"report_{report_type}_" + report_data["field_id"]
            )
            frames.append(_profile_value_rows(report_data, id_column="feature_id"))

            report_dates = (
                report_data.loc[:, ["conid", "effective_at"]].drop_duplicates().copy()
            )
            report_dates["feature_id"] = f"report_{report_type}_as_of_date"
            report_dates["value_date"] = report_dates["effective_at"]
            frames.append(_profile_value_rows(report_dates, id_column="feature_id"))

    themes = snapshot_tables["profile_themes"].copy()
    if not themes.empty:
        theme_flags = themes.loc[:, ["conid", "effective_at", "theme_id"]].copy()
        theme_flags["value_num"] = 1.0
        frames.append(
            _pivot_keyed_values(
                theme_flags,
                key_column="theme_id",
                value_column="value_num",
                prefix="profile_theme",
            )
        )

    expenses = snapshot_tables["profile_expense_allocations"].copy()
    if not expenses.empty:
        frames.append(
            _pivot_keyed_values(
                _select_frame(
                    expenses,
                    ("conid", "effective_at", "expense_id", "ratio"),
                ),
                key_column="expense_id",
                value_column="ratio",
                prefix="profile_expense",
            )
        )

    stylebox = snapshot_tables["profile_stylebox"].copy()
    if not stylebox.empty:
        style_flags = stylebox.loc[:, ["conid", "effective_at", "stylebox_id"]].copy()
        style_flags["value_num"] = 1.0
        frames.append(
            _pivot_keyed_values(
                style_flags,
                key_column="stylebox_id",
                value_column="value_num",
                prefix="profile_stylebox",
            )
        )
        style_rows: list[dict[str, object]] = []
        for _, row in stylebox.iterrows():
            for field_id, source_column in (
                ("morningstar_stylebox", "stylebox_id"),
                ("morningstar_stylebox_x", "x_label"),
                ("morningstar_stylebox_y", "y_label"),
                ("morningstar_stylebox_x_index", "x_index"),
                ("morningstar_stylebox_y_index", "y_index"),
            ):
                value = row.get(source_column)
                if _is_missing_scalar(value):
                    continue
                style_rows.append(
                    {
                        "conid": row["conid"],
                        "effective_at": row["effective_at"],
                        "field_id": field_id,
                        "value": value,
                    }
                )
        if style_rows:
            frames.append(
                _pivot_keyed_values(
                    pd.DataFrame(style_rows),
                    key_column="field_id",
                    value_column="value",
                    prefix="profile",
                )
            )

    for frame in frames:
        if "profile__total_net_assets_value" in frame.columns:
            frame["profile__total_net_assets_num"] = frame[
                "profile__total_net_assets_value"
            ].map(_parse_scaled_number)
    return frames


def _pivot_keyed_values(
    frame: pd.DataFrame,
    *,
    key_column: str,
    value_column: str,
    prefix: str,
) -> pd.DataFrame:
    if frame.empty:
        return _empty_frame(("conid", "effective_at"))

    work = frame.copy()
    work["pivot_key"] = work[key_column].map(_sanitize_segment)
    sort_columns = ["conid", "effective_at", "pivot_key", value_column]
    work = work.sort_values(sort_columns, na_position="last")
    pivoted = pd.DataFrame(
        work.pivot_table(
            index=["conid", "effective_at"],
            columns="pivot_key",
            values=value_column,
            aggfunc="first",
        ).reset_index()
    )
    pivoted.columns = [
        column if column in {"conid", "effective_at"} else f"{prefix}__{column}"
        for column in pivoted.columns
    ]
    return pivoted.sort_values(["conid", "effective_at"]).reset_index(drop=True)


def _pivot_metric_values(
    frame: pd.DataFrame,
    *,
    prefix: str,
    key_columns: tuple[str, ...],
) -> pd.DataFrame:
    if frame.empty:
        return _empty_frame(("conid", "effective_at"))

    work = frame.copy()
    work["pivot_key"] = (
        work.loc[:, list(key_columns)]
        .astype(str)
        .agg("__".join, axis=1)
        .map(_sanitize_segment)
    )
    value_pivot = _pivot_keyed_values(
        _select_frame(work, ("conid", "effective_at", "pivot_key", "value_num")),
        key_column="pivot_key",
        value_column="value_num",
        prefix=prefix,
    )
    if "vs_peers" in work.columns:
        vs_work = _select_frame(
            work, ("conid", "effective_at", "pivot_key", "vs_peers")
        )
        vs_work.columns = ["conid", "effective_at", "pivot_key", "value_num"]
        vs_pivot = _pivot_keyed_values(
            vs_work,
            key_column="pivot_key",
            value_column="value_num",
            prefix=f"{prefix}_vs",
        )
        value_pivot = value_pivot.merge(
            vs_pivot,
            on=["conid", "effective_at"],
            how="outer",
        )
    return value_pivot.sort_values(["conid", "effective_at"]).reset_index(drop=True)


def _assign_sleeve(row: pd.Series) -> str:
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


def _summarize_source_table(frame: pd.DataFrame, table_name: str) -> pd.DataFrame:
    if frame.empty:
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

    key_count = int(frame[["conid", "effective_at"]].drop_duplicates().shape[0])
    return pd.DataFrame(
        [
            {
                "table_name": table_name,
                "row_count": int(len(frame)),
                "key_count": key_count,
                "conid_count": int(frame["conid"].nunique()),
                "min_effective_at": frame["effective_at"].min(),
                "max_effective_at": frame["effective_at"].max(),
            }
        ]
    )


def _apply_holdings_flags(
    frame: pd.DataFrame,
    *,
    config: SnapshotInputConfig,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
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


def _long_holdings_diagnostics(
    frame: pd.DataFrame,
    *,
    table_name: str,
    key_column: str,
    value_column: str,
    config: SnapshotInputConfig,
) -> pd.DataFrame:
    if frame.empty:
        return _empty_frame(SNAPSHOT_HOLDINGS_DIAGNOSTIC_COLUMNS)

    work = frame.copy()
    work[value_column] = pd.to_numeric(work[value_column], errors="coerce")
    diagnostics = (
        work.groupby(["conid", "effective_at"], as_index=False)
        .agg(
            value_sum=(value_column, "sum"),
            category_count=(key_column, "nunique"),
            max_value=(value_column, "max"),
        )
        .assign(table_name=table_name)
    )
    return (
        _apply_holdings_flags(diagnostics, config=config)
        .loc[:, list(SNAPSHOT_HOLDINGS_DIAGNOSTIC_COLUMNS)]
        .sort_values(["table_name", "conid", "effective_at"])
        .reset_index(drop=True)
    )


def _top10_features_and_diagnostics(
    frame: pd.DataFrame,
    *,
    config: SnapshotInputConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return _empty_frame(("conid", "effective_at")), _empty_frame(
            SNAPSHOT_HOLDINGS_DIAGNOSTIC_COLUMNS
        )

    work = frame.copy()
    work["holding_weight_num"] = pd.to_numeric(
        work["holding_weight_num"], errors="coerce"
    )
    aggregated = pd.DataFrame(
        work.groupby(["conid", "effective_at"], as_index=False).agg(
            top10_count=("name", "nunique"),
            top10_weight_sum=("holding_weight_num", "sum"),
            top10_weight_max=("holding_weight_num", "max"),
        )
    )
    diagnostics = _select_frame(aggregated, ("conid", "effective_at"))
    diagnostics["value_sum"] = aggregated["top10_weight_sum"]
    diagnostics["category_count"] = aggregated["top10_count"]
    diagnostics["max_value"] = aggregated["top10_weight_max"]
    diagnostics["table_name"] = "holdings_top10"
    diagnostics = _apply_holdings_flags(diagnostics, config=config)

    features = _select_frame(aggregated, ("conid", "effective_at"))
    features["top10__top10_count"] = aggregated["top10_count"]
    features["top10__top10_weight_sum"] = aggregated["top10_weight_sum"]
    features["top10__top10_weight_max"] = aggregated["top10_weight_max"]
    return (
        features.sort_values(["conid", "effective_at"]).reset_index(drop=True),
        diagnostics.loc[:, list(SNAPSHOT_HOLDINGS_DIAGNOSTIC_COLUMNS)]
        .sort_values(["table_name", "conid", "effective_at"])
        .reset_index(drop=True),
    )


def _ratio_diagnostics(
    frame: pd.DataFrame,
    *,
    table_name: str,
    key_columns: tuple[str, ...],
) -> pd.DataFrame:
    if frame.empty:
        return _empty_frame(SNAPSHOT_RATIO_DIAGNOSTIC_COLUMNS)

    work = frame.copy()
    work["value_num"] = pd.to_numeric(work["value_num"], errors="coerce")
    work["metric_key"] = (
        work.loc[:, list(key_columns)]
        .astype(str)
        .agg("__".join, axis=1)
        .map(_sanitize_segment)
    )
    duplicate_rows = pd.DataFrame(
        work.groupby(["conid", "effective_at", "metric_key"])
        .size()
        .to_frame("metric_row_count")
        .reset_index()
    )
    duplicate_rows = duplicate_rows.loc[duplicate_rows["metric_row_count"] > 1].copy()

    rows: list[dict[str, object]] = []
    for group_key, group in work.groupby(["conid", "effective_at"], sort=True):
        if isinstance(group_key, tuple):
            conid = str(group_key[0])
            effective_at = group_key[1]
        else:
            conid = str(group_key)
            effective_at = pd.NaT
        effective_at_text = str(effective_at)
        effective_at_value = (
            pd.NaT if effective_at_text == "NaT" else pd.Timestamp(effective_at_text)
        )
        duplicate_group = duplicate_rows.loc[
            (duplicate_rows["conid"] == conid)
            & (duplicate_rows["effective_at"] == effective_at)
        ]
        metric_rows = int(len(group))
        distinct_metric_keys = int(group["metric_key"].nunique())
        rows.append(
            {
                "conid": str(conid),
                "effective_at": effective_at_value,
                "table_name": table_name,
                "metric_rows": metric_rows,
                "distinct_metric_keys": distinct_metric_keys,
                "duplicate_metric_keys": int(len(duplicate_group)),
                "duplicate_row_count": int(metric_rows - distinct_metric_keys),
                "nonnull_value_rows": int(group["value_num"].notna().sum()),
                "null_value_rows": int(group["value_num"].isna().sum()),
                "all_values_null": bool(group["value_num"].notna().sum() == 0),
            }
        )
    return (
        pd.DataFrame(rows)
        .loc[:, list(SNAPSHOT_RATIO_DIAGNOSTIC_COLUMNS)]
        .sort_values(["table_name", "conid", "effective_at"])
        .reset_index(drop=True)
    )


def build_snapshot_input_bundle(
    *,
    conn: sqlite3.Connection | None = None,
    tables: SnapshotFeatureTablesRead | dict[str, pd.DataFrame] | None = None,
    config: SnapshotInputConfig | None = None,
) -> AnalysisInputBundle:
    config = config or SnapshotInputConfig()
    if tables is None:
        if conn is None:
            raise ValueError("conn or tables is required")
        snapshot_tables = load_snapshot_feature_tables(conn).tables
    elif isinstance(tables, SnapshotFeatureTablesRead):
        snapshot_tables = tables.tables
    else:
        snapshot_tables = SnapshotFeatureTablesRead.from_tables(tables).tables

    frames: list[pd.DataFrame] = []
    holdings_diagnostics: list[pd.DataFrame] = []
    ratio_diagnostics: list[pd.DataFrame] = []
    table_summary = [
        _summarize_source_table(frame, table_name)
        for table_name, frame in snapshot_tables.items()
    ]

    frames.extend(_build_profile_features(snapshot_tables))

    holdings_specs = [
        ("holdings_asset_type", "bucket_id", "value_num", "holding_asset"),
        ("holdings_debtor_quality", "bucket_id", "value_num", "holding_quality"),
        ("holdings_maturity", "bucket_id", "value_num", "holding_maturity"),
        ("holdings_industry", "industry_id", "value_num", "industry"),
        ("holdings_geographic_weights", "region_id", "value_num", "region"),
        ("holdings_debt_type", "debt_type_id", "value_num", "debt_type"),
    ]
    for table_name, key_column, value_column, prefix in holdings_specs:
        frame = snapshot_tables[table_name].copy()
        if frame.empty:
            continue
        holdings_diagnostics.append(
            _long_holdings_diagnostics(
                frame,
                table_name=table_name,
                key_column=key_column,
                value_column=value_column,
                config=config,
            )
        )
        frames.append(
            _pivot_keyed_values(
                _select_frame(
                    frame, ("conid", "effective_at", key_column, value_column)
                ),
                key_column=key_column,
                value_column=value_column,
                prefix=prefix,
            )
        )

    currency = snapshot_tables["holdings_currency"].copy()
    if not currency.empty:
        holdings_diagnostics.append(
            _long_holdings_diagnostics(
                currency,
                table_name="holdings_currency",
                key_column="code",
                value_column="value_num",
                config=config,
            )
        )
        currency["currency_key"] = currency["code"].where(
            currency["code"].notna(), currency["name"]
        )
        frames.append(
            _pivot_keyed_values(
                _select_frame(
                    currency,
                    ("conid", "effective_at", "currency_key", "value_num"),
                ),
                key_column="currency_key",
                value_column="value_num",
                prefix="currency",
            )
        )

    country = snapshot_tables["holdings_investor_country"].copy()
    if not country.empty:
        holdings_diagnostics.append(
            _long_holdings_diagnostics(
                country,
                table_name="holdings_investor_country",
                key_column="code",
                value_column="value_num",
                config=config,
            )
        )
        country["country_key"] = country["code"].where(
            country["code"].notna(), country["name"]
        )
        frames.append(
            _pivot_keyed_values(
                _select_frame(
                    country,
                    ("conid", "effective_at", "country_key", "value_num"),
                ),
                key_column="country_key",
                value_column="value_num",
                prefix="country",
            )
        )

    top10_features, top10_diagnostics = _top10_features_and_diagnostics(
        snapshot_tables["holdings_top10"],
        config=config,
    )
    if not top10_features.empty:
        frames.append(top10_features)
    if not top10_diagnostics.empty:
        holdings_diagnostics.append(top10_diagnostics)

    ratio_specs = [
        ("ratios_key_ratios", "ratio_key", ("metric_id",)),
        ("ratios_financials", "ratio_financial", ("metric_id",)),
        ("ratios_fixed_income", "ratio_fixed_income", ("metric_id",)),
        ("ratios_dividend", "ratio_dividend", ("metric_id",)),
        ("ratios_zscore", "ratio_zscore", ("metric_id",)),
    ]
    for table_name, prefix, key_columns in ratio_specs:
        frame = snapshot_tables[table_name].copy()
        if frame.empty:
            continue
        ratio_diagnostics.append(
            _ratio_diagnostics(frame, table_name=table_name, key_columns=key_columns)
        )
        frames.append(
            _pivot_metric_values(frame, prefix=prefix, key_columns=key_columns)
        )

    dividend_metrics = snapshot_tables["dividends_industry_metrics"].copy()
    if not dividend_metrics.empty:
        frames.append(
            _pivot_keyed_values(
                _select_frame(
                    dividend_metrics,
                    ("conid", "effective_at", "metric_id", "value_num"),
                ),
                key_column="metric_id",
                value_column="value_num",
                prefix="dividend_metric",
            )
        )

    morningstar = snapshot_tables["morningstar_summary"].copy()
    if not morningstar.empty:
        morningstar["value"] = _factor_values(morningstar)
        frames.append(
            _pivot_keyed_values(
                _select_frame(
                    morningstar,
                    ("conid", "effective_at", "metric_id", "value"),
                ),
                key_column="metric_id",
                value_column="value",
                prefix="morningstar",
            )
        )

    lipper = snapshot_tables["lipper_ratings"].copy()
    if not lipper.empty:
        ratio_diagnostics.append(
            _ratio_diagnostics(
                lipper,
                table_name="lipper_ratings",
                key_columns=("period", "metric_id"),
            )
        )
        frames.append(
            _pivot_metric_values(
                lipper,
                prefix="lipper",
                key_columns=("period", "metric_id"),
            )
        )

    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        snapshot_features = _empty_frame(("conid", "effective_at", "sleeve"))
    else:
        snapshot_features = frames[0]
        for frame in frames[1:]:
            snapshot_features = snapshot_features.merge(
                frame,
                on=["conid", "effective_at"],
                how="outer",
            )
        snapshot_features["conid"] = snapshot_features["conid"].astype(str)
        snapshot_features["effective_at"] = pd.to_datetime(
            snapshot_features["effective_at"]
        )
        snapshot_features["sleeve"] = snapshot_features.apply(_assign_sleeve, axis=1)
        snapshot_features = snapshot_features.sort_values(
            ["conid", "effective_at"]
        ).reset_index(drop=True)

    holdings_diag = (
        pd.concat(holdings_diagnostics, ignore_index=True)
        if holdings_diagnostics
        else _empty_frame(SNAPSHOT_HOLDINGS_DIAGNOSTIC_COLUMNS)
    )
    ratio_diag = (
        pd.concat(ratio_diagnostics, ignore_index=True)
        if ratio_diagnostics
        else _empty_frame(SNAPSHOT_RATIO_DIAGNOSTIC_COLUMNS)
    )
    summary = (
        pd.concat(table_summary, ignore_index=True)
        .loc[:, list(SNAPSHOT_TABLE_SUMMARY_COLUMNS)]
        .sort_values("table_name")
        .reset_index(drop=True)
    )

    return AnalysisInputBundle.from_frames(
        snapshot_features=snapshot_features,
        snapshot_holdings_diagnostics=holdings_diag,
        snapshot_ratio_diagnostics=ratio_diag,
        snapshot_table_summary=summary,
    )
