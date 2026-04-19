from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from ..storage.reads import PriceHistoryRead, load_price_history
from .bundle import (
    PRICE_ELIGIBILITY_COLUMNS,
    PRICE_INPUT_COLUMNS,
    AnalysisInputBundle,
)


@dataclass(frozen=True, slots=True)
class PriceInputConfig:
    min_history_days: int = 252
    max_missing_ratio: float = 0.30
    max_internal_gap_days: int = 20
    stale_run_max_days: int = 5
    outlier_z_threshold: float = 50.0
    local_price_ratio_threshold: float = 5.0
    bridge_outlier_span_max_rows: int = 5


def _empty_frame(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        {column: pd.Series(dtype="object") for column in columns}
    ).reindex(columns=pd.Index(columns))


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.Series(pd.to_numeric(frame[column], errors="coerce"), index=frame.index)


def _compute_internal_gap_days(dates: list[pd.Timestamp]) -> int:
    if len(dates) <= 1:
        return 0
    date_arr = np.array([ts.date() for ts in dates], dtype="datetime64[D]")
    prev_dates = date_arr[:-1]
    next_dates = date_arr[1:]
    gap_days = np.busday_count(prev_dates, next_dates) - 1
    if gap_days.size == 0:
        return 0
    gap_days = np.maximum(gap_days, 0)
    return int(gap_days.max())


def _compute_eligibility(frame: pd.DataFrame, config: PriceInputConfig) -> pd.DataFrame:
    if frame.empty:
        return _empty_frame(PRICE_ELIGIBILITY_COLUMNS)

    rows: list[dict[str, object]] = []
    for conid, group in frame.groupby("conid", sort=True):
        valid = group.loc[group["is_clean_price"]].copy()
        total_rows = int(len(group))
        valid_rows = int(len(valid))

        if valid.empty:
            min_date = pd.NaT
            max_date = pd.NaT
            expected_days = 0
            missing_ratio = 1.0
            max_internal_gap_days = 0
        else:
            valid_date_values: list[object] = [
                value
                for value in pd.to_datetime(valid["trade_date"]).dt.normalize().tolist()
                if str(value) != "NaT"
            ]
            valid_dates: list[pd.Timestamp] = []
            for value in valid_date_values:
                timestamp = pd.Timestamp(str(value))
                if str(timestamp) == "NaT":
                    continue
                valid_dates.append(cast(pd.Timestamp, timestamp))
            valid_dates.sort()
            min_date = pd.Timestamp(valid_dates[0])
            max_date = pd.Timestamp(valid_dates[-1])
            expected_days = int(len(pd.bdate_range(min_date, max_date)))
            missing_ratio = (
                float(max(0.0, 1.0 - (valid_rows / expected_days)))
                if expected_days > 0
                else 1.0
            )
            max_internal_gap_days = _compute_internal_gap_days(valid_dates)

        reasons: list[str] = []
        if valid_rows < config.min_history_days:
            reasons.append("Insufficient history")
        if missing_ratio > config.max_missing_ratio:
            reasons.append("Excessive missing ratio")
        if max_internal_gap_days > config.max_internal_gap_days:
            reasons.append("Large internal gap")

        eligible = len(reasons) == 0
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
                "eligibility_reason": "OK" if eligible else "; ".join(reasons),
            }
        )
    return (
        pd.DataFrame(rows)
        .reindex(columns=pd.Index(PRICE_ELIGIBILITY_COLUMNS))
        .sort_values("conid")
        .reset_index(drop=True)
    )


def _robust_outlier_mask(values: pd.Series, threshold: float) -> pd.Series:
    numeric = pd.Series(pd.to_numeric(values, errors="coerce"), index=values.index)
    valid = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 5:
        return pd.Series(False, index=values.index, dtype=bool)

    median = float(valid.median())
    mad = float((valid - median).abs().median())
    if mad == 0.0 or np.isnan(mad):
        return pd.Series(False, index=values.index, dtype=bool)

    scores = (numeric - median).abs() / mad
    return pd.Series(scores.gt(threshold).fillna(False), index=values.index, dtype=bool)


def _mark_stale_rows(group: pd.DataFrame, stale_run_max_days: int) -> pd.Series:
    series = pd.Series(
        pd.to_numeric(group["price_value"], errors="coerce"), index=group.index
    )
    change_groups = series.ne(series.shift()).cumsum()
    group_sizes = series.groupby(change_groups).transform("size")
    in_run = change_groups.duplicated(keep="first") & change_groups.duplicated(
        keep="last"
    )
    return pd.Series(
        ((group_sizes > stale_run_max_days) & in_run).fillna(False),
        index=group.index,
        dtype=bool,
    )


def _mark_price_level_anomalies(
    group: pd.DataFrame,
    base_clean_mask: pd.Series,
    outlier_mask: pd.Series,
    ratio_threshold: float,
    max_bridge_span: int,
) -> pd.Series:
    flagged = pd.Series(False, index=group.index, dtype=bool)
    outlier_positions = np.flatnonzero(np.asarray(outlier_mask.to_numpy(), dtype=bool))
    if len(outlier_positions) < 2:
        return flagged

    prices = np.asarray(
        pd.to_numeric(group["price_value"], errors="coerce"), dtype=float
    )
    returns = np.asarray(
        pd.to_numeric(group["raw_return"], errors="coerce"), dtype=float
    )
    base_clean_arr = np.asarray(base_clean_mask.to_numpy(), dtype=bool)
    clean_positions = np.flatnonzero(base_clean_arr)
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
        if not np.isfinite(reference) or reference <= 0.0:
            continue

        bridge_positions = np.arange(left_outlier + 1, right_outlier)
        bridge_positions = bridge_positions[base_clean_arr[bridge_positions]]
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


def _resolve_price_frame(
    *,
    conn: sqlite3.Connection | None,
    prices: pd.DataFrame | PriceHistoryRead | None,
) -> pd.DataFrame:
    if prices is None:
        if conn is None:
            raise ValueError("conn or prices is required")
        return load_price_history(conn).frame.copy()
    if isinstance(prices, PriceHistoryRead):
        return prices.frame.copy()
    return PriceHistoryRead.from_frame(prices).frame.copy()


def build_price_input_bundle(
    *,
    conn: sqlite3.Connection | None = None,
    prices: pd.DataFrame | PriceHistoryRead | None = None,
    config: PriceInputConfig | None = None,
) -> AnalysisInputBundle:
    config = config or PriceInputConfig()
    frame = _resolve_price_frame(conn=conn, prices=prices)
    if frame.empty:
        return AnalysisInputBundle.from_frames(
            prices=_empty_frame(PRICE_INPUT_COLUMNS),
            price_eligibility=_empty_frame(PRICE_ELIGIBILITY_COLUMNS),
        )

    df = frame.copy()
    df["conid"] = df["conid"].astype(str)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["conid", "trade_date"]).reset_index(drop=True)

    price_values = _numeric_series(df, "price")
    open_values = _numeric_series(df, "open")
    high_values = _numeric_series(df, "high")
    low_values = _numeric_series(df, "low")
    close_values = _numeric_series(df, "close")

    df["price_value"] = close_values.where(close_values.notna(), price_values).where(
        close_values.notna() | price_values.notna(), open_values
    )
    finite_price = pd.Series(np.isfinite(df["price_value"]), index=df.index, dtype=bool)
    df["is_valid_price"] = (
        df["price_value"].notna()
        & finite_price
        & df["price_value"].gt(0.0)
        & ~(high_values.notna() & low_values.notna() & low_values.gt(high_values))
    )
    df["raw_return"] = df.groupby("conid")["price_value"].pct_change(fill_method=None)

    stale_mask = pd.Series(False, index=df.index, dtype=bool)
    for _, group in df.groupby("conid", sort=True):
        stale_mask.loc[group.index] = _mark_stale_rows(
            group, config.stale_run_max_days
        ).to_numpy()
    df["is_stale_price"] = stale_mask

    candidate_returns = df["raw_return"].where(
        df["is_valid_price"] & ~df["is_stale_price"]
    )
    outlier_mask = pd.Series(False, index=df.index, dtype=bool)
    grouped_candidate = df.assign(candidate_return=candidate_returns)
    for _, group in grouped_candidate.groupby("conid", sort=True):
        candidate_series = pd.Series(group["candidate_return"], index=group.index)
        outlier_mask.loc[group.index] = _robust_outlier_mask(
            candidate_series,
            config.outlier_z_threshold,
        ).to_numpy()
    df["is_outlier_return"] = outlier_mask

    base_clean_mask = (
        df["is_valid_price"] & ~df["is_stale_price"] & ~df["is_outlier_return"]
    )
    price_level_anomaly_mask = pd.Series(False, index=df.index, dtype=bool)
    for _, group in df.groupby("conid", sort=True):
        price_level_anomaly_mask.loc[group.index] = _mark_price_level_anomalies(
            group,
            base_clean_mask.loc[group.index],
            df.loc[group.index, "is_outlier_return"],
            config.local_price_ratio_threshold,
            config.bridge_outlier_span_max_rows,
        ).to_numpy()
    df["is_price_level_anomaly"] = price_level_anomaly_mask
    df["is_clean_price"] = base_clean_mask & ~df["is_price_level_anomaly"]
    df["clean_price"] = df["price_value"].where(df["is_clean_price"])
    df["clean_return"] = df.groupby("conid")["clean_price"].pct_change(fill_method=None)

    price_output = (
        df.loc[:, list(PRICE_INPUT_COLUMNS)]
        .sort_values(["conid", "trade_date"])
        .reset_index(drop=True)
    )
    eligibility = _compute_eligibility(price_output, config)
    return AnalysisInputBundle.from_frames(
        prices=price_output,
        price_eligibility=eligibility,
    )
