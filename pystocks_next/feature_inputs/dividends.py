from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..storage.reads import DividendEventsRead, load_dividend_events
from .bundle import (
    DIVIDEND_EVENT_COLUMNS,
    DIVIDEND_SUMMARY_COLUMNS,
    AnalysisInputBundle,
)
from .prices import build_price_input_bundle


@dataclass(frozen=True, slots=True)
class DividendInputConfig:
    max_implied_yield: float = 0.25
    max_price_reference_age_days: int = 10


def _empty_frame(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        {column: pd.Series(dtype="object") for column in columns}
    ).reindex(columns=pd.Index(columns))


def _resolve_dividend_frame(
    *,
    conn: sqlite3.Connection | None,
    dividends: pd.DataFrame | DividendEventsRead | None,
) -> pd.DataFrame:
    if dividends is None:
        if conn is None:
            raise ValueError("conn or dividends is required")
        return load_dividend_events(conn).frame.copy()
    if isinstance(dividends, DividendEventsRead):
        return dividends.frame.copy()
    return DividendEventsRead.from_frame(dividends).frame.copy()


def _build_clean_price_reference(
    *,
    conn: sqlite3.Connection | None,
    price_reference: pd.DataFrame | AnalysisInputBundle | None,
) -> pd.DataFrame:
    if price_reference is None:
        if conn is None:
            raise ValueError("conn or price_reference is required")
        reference = (
            build_price_input_bundle(conn=conn)
            .prices[["conid", "trade_date", "clean_price"]]
            .copy()
        )
    elif isinstance(price_reference, AnalysisInputBundle):
        reference = price_reference.prices[
            ["conid", "trade_date", "clean_price"]
        ].copy()
    else:
        reference = price_reference.copy()

    required_columns = {"conid", "trade_date", "clean_price"}
    missing_columns = sorted(required_columns - set(reference.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"price_reference is missing required columns: {missing}")

    reference["conid"] = reference["conid"].astype(str)
    reference["trade_date"] = pd.to_datetime(reference["trade_date"])
    reference["clean_price"] = pd.to_numeric(reference["clean_price"], errors="coerce")
    clean_mask = pd.Series(reference["clean_price"], index=reference.index).notna()
    reference = reference.loc[clean_mask].copy()
    return reference.sort_values(["trade_date", "conid"]).reset_index(drop=True)


def _compute_trailing_dividend_sum(frame: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    for _, group in frame.groupby("conid", sort=True):
        indexed = group.set_index("event_date")["amount"].fillna(0.0)
        trailing = indexed.rolling("365D", min_periods=1).sum()
        out.loc[group.index] = np.asarray(trailing, dtype=float)
    return out


def _summarize_dividend_events(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_frame(DIVIDEND_SUMMARY_COLUMNS)

    rows: list[dict[str, object]] = []
    for conid, group in frame.groupby("conid", sort=True):
        event_rows = int(len(group))
        usable_rows = int(group["usable_for_total_return_adjustment"].sum())
        rows.append(
            {
                "conid": str(conid),
                "symbol": group["symbol"].dropna().iloc[0]
                if bool(group["symbol"].notna().any())
                else pd.NA,
                "product_currency": group["product_currency"].dropna().iloc[0]
                if bool(group["product_currency"].notna().any())
                else pd.NA,
                "event_rows": event_rows,
                "usable_rows": usable_rows,
                "duplicate_rows": int(group["is_duplicate_event_signature"].sum()),
                "currency_mismatch_rows": int(group["is_currency_mismatch"].sum()),
                "missing_currency_rows": int(group["is_missing_currency"].sum()),
                "suspicious_yield_rows": int(
                    group["is_suspicious_implied_yield"].sum()
                ),
                "missing_price_reference_rows": int(
                    group["is_missing_price_reference"].sum()
                ),
                "min_event_date": group["event_date"].min(),
                "max_event_date": group["event_date"].max(),
                "usable_ratio": (
                    float(usable_rows / event_rows) if event_rows > 0 else pd.NA
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .reindex(columns=pd.Index(DIVIDEND_SUMMARY_COLUMNS))
        .sort_values("conid")
        .reset_index(drop=True)
    )


def build_dividend_input_bundle(
    *,
    conn: sqlite3.Connection | None = None,
    dividends: pd.DataFrame | DividendEventsRead | None = None,
    price_reference: pd.DataFrame | AnalysisInputBundle | None = None,
    config: DividendInputConfig | None = None,
) -> AnalysisInputBundle:
    config = config or DividendInputConfig()
    dividend_frame = _resolve_dividend_frame(conn=conn, dividends=dividends)
    if dividend_frame.empty:
        return AnalysisInputBundle.from_frames(
            dividends=_empty_frame(DIVIDEND_EVENT_COLUMNS),
            dividend_summary=_empty_frame(DIVIDEND_SUMMARY_COLUMNS),
        )

    reference = _build_clean_price_reference(conn=conn, price_reference=price_reference)
    left = dividend_frame.sort_values(["event_date", "conid"]).reset_index(drop=True)
    right = reference.rename(
        columns={
            "trade_date": "previous_price_date",
            "clean_price": "previous_clean_price",
        }
    )

    merged = pd.merge_asof(
        left,
        right,
        by="conid",
        left_on="event_date",
        right_on="previous_price_date",
        direction="backward",
        allow_exact_matches=False,
    )
    merged["price_reference_age_days"] = (
        merged["event_date"] - merged["previous_price_date"]
    ).dt.days
    merged["implied_yield_vs_previous_price"] = (
        merged["amount"] / merged["previous_clean_price"]
    )

    trailing_input = merged.sort_values(["conid", "event_date"]).copy()
    merged["trailing_dividend_sum_365d"] = _compute_trailing_dividend_sum(
        trailing_input
    ).reindex(merged.index)
    merged["is_missing_amount"] = merged["amount"].isna()
    merged["is_nonpositive_amount"] = merged["amount"].notna() & merged["amount"].le(
        0.0
    )
    merged["is_missing_currency"] = (
        merged["dividend_currency"].fillna("").astype(str).str.strip().eq("")
    )
    merged["is_currency_mismatch"] = (
        ~merged["is_missing_currency"]
        & merged["product_currency"].fillna("").astype(str).str.strip().ne("")
        & merged["product_currency"]
        .fillna("")
        .astype(str)
        .ne(merged["dividend_currency"].fillna("").astype(str))
    )
    merged["is_duplicate_event_signature"] = merged.duplicated(
        [
            "conid",
            "event_date",
            "amount",
            "dividend_currency",
            "description",
            "event_type",
            "payment_date",
        ],
        keep=False,
    )
    merged["is_missing_price_reference"] = merged["previous_clean_price"].isna()
    merged["is_stale_price_reference"] = merged["price_reference_age_days"].notna() & (
        merged["price_reference_age_days"] > config.max_price_reference_age_days
    )
    merged["is_suspicious_implied_yield"] = merged[
        "implied_yield_vs_previous_price"
    ].notna() & (
        merged["implied_yield_vs_previous_price"].le(0.0)
        | merged["implied_yield_vs_previous_price"].gt(config.max_implied_yield)
    )
    merged["usable_for_total_return_adjustment"] = ~(
        merged["is_missing_amount"]
        | merged["is_nonpositive_amount"]
        | merged["is_missing_currency"]
        | merged["is_currency_mismatch"]
        | merged["is_duplicate_event_signature"]
        | merged["is_missing_price_reference"]
        | merged["is_stale_price_reference"]
        | merged["is_suspicious_implied_yield"]
    )

    events = (
        merged.loc[:, list(DIVIDEND_EVENT_COLUMNS)]
        .sort_values(["conid", "event_date"])
        .reset_index(drop=True)
    )
    summary = _summarize_dividend_events(events)
    return AnalysisInputBundle.from_frames(
        dividends=events,
        dividend_summary=summary,
    )
