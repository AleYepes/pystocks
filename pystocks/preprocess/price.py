import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DATA_DIR, SQLITE_DB_PATH


@dataclass
class PricePreprocessConfig:
    min_history_days: int = 252
    max_missing_ratio: float = 0.30
    max_internal_gap_days: int = 20
    stale_run_max_days: int = 5
    outlier_z_threshold: float = 50.0
    local_price_ratio_threshold: float = 5.0
    bridge_outlier_span_max_rows: int = 5


def _empty_frame(columns):
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


def _numeric_series(df, column):
    return pd.Series(pd.to_numeric(df[column], errors="coerce"), index=df.index)


def _compute_internal_gap_days(dates):
    if len(dates) <= 1:
        return 0
    date_arr = np.array(pd.to_datetime(dates).date, dtype="datetime64[D]")
    prev_dates = date_arr[:-1]
    next_dates = date_arr[1:]
    gap_days = np.busday_count(prev_dates, next_dates) - 1
    if gap_days.size == 0:
        return 0
    gap_days = np.maximum(gap_days, 0)
    return int(gap_days.max())


def _compute_eligibility(df, config: PricePreprocessConfig):
    rows = []
    for conid, group in df.groupby("conid"):
        valid = group[group["is_clean_price"]].copy()

        total_rows = int(len(group))
        valid_rows = int(len(valid))

        if valid.empty:
            min_date = None
            max_date = None
            expected_days = 0
            missing_ratio = 1.0
            max_internal_gap_days = 0
        else:
            valid_dates = sorted(pd.to_datetime(valid["trade_date"]).dt.date.unique())
            min_date = str(valid_dates[0]) if valid_dates else None
            max_date = str(valid_dates[-1]) if valid_dates else None
            expected_days = (
                int(len(pd.bdate_range(valid_dates[0], valid_dates[-1])))
                if len(valid_dates) > 0
                else 0
            )
            missing_ratio = (
                float(max(0.0, 1.0 - (valid_rows / expected_days)))
                if expected_days > 0
                else 1.0
            )
            max_internal_gap_days = _compute_internal_gap_days(valid_dates)

        reasons = []
        if valid_rows < config.min_history_days:
            reasons.append("Insufficient history")
        if missing_ratio > config.max_missing_ratio:
            reasons.append("Excessive missing ratio")
        if max_internal_gap_days > config.max_internal_gap_days:
            reasons.append("Large internal gap")

        eligible = len(reasons) == 0
        reason = "OK" if eligible else "; ".join(reasons)

        rows.append(
            {
                "conid": str(conid),
                "total_rows": total_rows,
                "valid_rows": valid_rows,
                "min_date": min_date,
                "max_date": max_date,
                "expected_business_days": expected_days,
                "missing_ratio": missing_ratio,
                "max_internal_gap_days": max_internal_gap_days,
                "eligible": bool(eligible),
                "eligibility_reason": reason,
            }
        )

    return pd.DataFrame(rows).sort_values("conid").reset_index(drop=True)


def load_price_history(sqlite_path=SQLITE_DB_PATH):
    with sqlite3.connect(str(sqlite_path)) as conn:
        df = pd.read_sql_query(
            """
            SELECT
                conid,
                effective_at AS trade_date,
                price,
                open,
                high,
                low,
                close
            FROM price_chart_series
            ORDER BY conid, effective_at
            """,
            conn,
        )
    if df.empty:
        return df

    df["conid"] = df["conid"].astype(str)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    for col in ["price", "open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _robust_outlier_mask(values, threshold):
    valid = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 5:
        return pd.Series(False, index=values.index)

    median = valid.median()
    mad = (valid - median).abs().median()
    if mad == 0 or np.isnan(mad):
        return pd.Series(False, index=values.index)

    scores = (values - median).abs() / mad
    return scores > threshold


def _mark_stale_rows(group, stale_run_max_days):
    series = group["price_value"]
    change_groups = series.ne(series.shift()).cumsum()
    group_sizes = series.groupby(change_groups).transform("size")
    in_run = change_groups.duplicated(keep="first") & change_groups.duplicated(
        keep="last"
    )
    return (group_sizes > stale_run_max_days) & in_run


def _mark_price_level_anomalies(
    group, base_clean_mask, outlier_mask, ratio_threshold, max_bridge_span
):
    flagged = pd.Series(False, index=group.index)
    outlier_positions = np.flatnonzero(outlier_mask.to_numpy())
    if len(outlier_positions) < 2:
        return flagged

    prices = group["price_value"].to_numpy(dtype=float)
    returns = group["raw_return"].to_numpy(dtype=float)
    clean_positions = np.flatnonzero(base_clean_mask.to_numpy())
    if len(clean_positions) == 0:
        return flagged

    for left_outlier, right_outlier in zip(
        outlier_positions[:-1], outlier_positions[1:]
    ):
        if right_outlier - left_outlier <= 1:
            continue
        if right_outlier - left_outlier > max_bridge_span:
            continue
        left_return = returns[left_outlier]
        right_return = returns[right_outlier]
        if not np.isfinite(left_return) or not np.isfinite(right_return):
            continue
        if np.sign(left_return) == np.sign(right_return):
            continue

        left_anchors = clean_positions[clean_positions < left_outlier]
        right_anchors = clean_positions[clean_positions > right_outlier]
        if len(left_anchors) == 0 or len(right_anchors) == 0:
            continue

        reference = np.sqrt(prices[left_anchors[-1]] * prices[right_anchors[0]])
        if not np.isfinite(reference) or reference <= 0:
            continue

        bridge_positions = np.arange(left_outlier + 1, right_outlier)
        bridge_positions = bridge_positions[
            base_clean_mask.to_numpy()[bridge_positions]
        ]
        if len(bridge_positions) == 0:
            continue

        bridge_ratios = prices[bridge_positions] / reference
        flagged_positions = bridge_positions[
            (bridge_ratios < (1.0 / ratio_threshold))
            | (bridge_ratios > ratio_threshold)
        ]
        if len(flagged_positions) > 0:
            flagged.iloc[flagged_positions] = True

    return flagged


def preprocess_price_history(price_df=None, config=None):
    config = config or PricePreprocessConfig()
    price_df = load_price_history() if price_df is None else price_df.copy()

    if price_df.empty:
        empty = _empty_frame(
            [
                "conid",
                "trade_date",
                "price_value",
                "clean_price",
                "raw_return",
                "clean_return",
                "is_valid_price",
                "is_stale_price",
                "is_outlier_return",
                "is_price_level_anomaly",
                "is_clean_price",
            ]
        )
        return {
            "prices": empty,
            "eligibility": _compute_eligibility(empty, config),
            "config": config,
        }

    df = price_df.copy()
    df["conid"] = df["conid"].astype(str)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["conid", "trade_date"]).reset_index(drop=True)

    df["price_value"] = (
        df["close"]
        .where(df["close"].notna(), df["price"])
        .where(df["close"].notna() | df["price"].notna(), df["open"])
    )
    high_values = _numeric_series(df, "high")
    low_values = _numeric_series(df, "low")
    df["is_valid_price"] = (
        df["price_value"].notna()
        & np.isfinite(df["price_value"])
        & (df["price_value"] > 0)
        & ~(high_values.notna() & low_values.notna() & (low_values > high_values))
    )

    df["raw_return"] = df.groupby("conid")["price_value"].pct_change(fill_method=None)
    stale_mask = pd.Series(False, index=df.index)
    for _, group in df.groupby("conid"):
        stale_mask.loc[group.index] = _mark_stale_rows(
            group, config.stale_run_max_days
        ).to_numpy()
    df["is_stale_price"] = stale_mask.fillna(False)

    candidate_returns = df["raw_return"].where(
        df["is_valid_price"] & ~df["is_stale_price"]
    )
    outlier_mask = pd.Series(False, index=df.index)
    for _, group in df.assign(candidate_return=candidate_returns).groupby("conid"):
        outlier_mask.loc[group.index] = _robust_outlier_mask(
            group["candidate_return"],
            config.outlier_z_threshold,
        ).to_numpy()
    df["is_outlier_return"] = outlier_mask.fillna(False)
    base_clean_mask = (
        df["is_valid_price"] & ~df["is_stale_price"] & ~df["is_outlier_return"]
    )
    price_level_anomaly_mask = pd.Series(False, index=df.index)
    for _, group in df.groupby("conid"):
        price_level_anomaly_mask.loc[group.index] = _mark_price_level_anomalies(
            group,
            base_clean_mask.loc[group.index],
            df.loc[group.index, "is_outlier_return"],
            config.local_price_ratio_threshold,
            config.bridge_outlier_span_max_rows,
        ).to_numpy()
    df["is_price_level_anomaly"] = price_level_anomaly_mask.fillna(False)
    df["is_clean_price"] = base_clean_mask & ~df["is_price_level_anomaly"]
    df["clean_price"] = df["price_value"].where(df["is_clean_price"])
    df["clean_return"] = df.groupby("conid")["clean_price"].pct_change(fill_method=None)

    price_cols = [
        "conid",
        "trade_date",
        "price",
        "open",
        "high",
        "low",
        "close",
        "price_value",
        "clean_price",
        "raw_return",
        "clean_return",
        "is_valid_price",
        "is_stale_price",
        "is_outlier_return",
        "is_price_level_anomaly",
        "is_clean_price",
    ]
    prices = df[price_cols].copy()
    eligibility = _compute_eligibility(prices, config)

    return {
        "prices": prices,
        "eligibility": eligibility,
        "config": config,
    }


def save_price_preprocess_results(result, output_dir=None):
    output_dir = Path(output_dir or (DATA_DIR / "analysis"))
    output_dir.mkdir(parents=True, exist_ok=True)

    prices_path = output_dir / "analysis_daily_returns.parquet"
    eligibility_path = output_dir / "analysis_price_eligibility.parquet"
    result["prices"].to_parquet(prices_path, index=False)
    result["eligibility"].to_parquet(eligibility_path, index=False)

    return {
        "prices_path": str(prices_path),
        "eligibility_path": str(eligibility_path),
    }


def run_price_preprocess(sqlite_path=SQLITE_DB_PATH, output_dir=None, **config_kwargs):
    config = PricePreprocessConfig(**config_kwargs)
    result = preprocess_price_history(load_price_history(sqlite_path), config=config)
    paths = save_price_preprocess_results(result, output_dir=output_dir)

    eligibility = result["eligibility"]
    return {
        "status": "ok",
        "rows": int(len(result["prices"])),
        "eligible_conids": int(eligibility["eligible"].sum())
        if not eligibility.empty
        else 0,
        "ineligible_conids": int((~eligibility["eligible"]).sum())
        if not eligibility.empty
        else 0,
        **paths,
    }
