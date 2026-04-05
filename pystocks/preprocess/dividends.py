from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DATA_DIR, SQLITE_DB_PATH
from ..storage.readers import load_dividend_events as _load_dividend_events
from .price import load_price_history, preprocess_price_history


@dataclass
class DividendPreprocessConfig:
    max_implied_yield: float = 0.25
    max_price_reference_age_days: int = 10


def _empty_frame(columns):
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


def load_dividend_events(sqlite_path=SQLITE_DB_PATH):
    return _load_dividend_events(sqlite_path=sqlite_path)


def _normalize_dividend_frame(dividend_df):
    df = dividend_df.copy()
    rename_map = {}
    if "effective_at" in df.columns and "event_date" not in df.columns:
        rename_map["effective_at"] = "event_date"
    if "currency" in df.columns and "dividend_currency" not in df.columns:
        rename_map["currency"] = "dividend_currency"
    if rename_map:
        df = df.rename(columns=rename_map)

    if "conid" in df.columns:
        df["conid"] = df["conid"].astype(str)
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"])
    for col in ["declaration_date", "record_date", "payment_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    for col in [
        "symbol",
        "product_currency",
        "description",
        "event_type",
        "declaration_date",
        "record_date",
        "payment_date",
        "dividend_currency",
    ]:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def _build_clean_price_reference(price_reference=None, sqlite_path=SQLITE_DB_PATH):
    if price_reference is None:
        price_result = preprocess_price_history(load_price_history(sqlite_path))
        reference = price_result["prices"][
            ["conid", "trade_date", "clean_price"]
        ].copy()
    else:
        reference = price_reference.copy()

    if "trade_date" not in reference.columns:
        raise ValueError("price_reference must contain trade_date")
    if "clean_price" not in reference.columns:
        raise ValueError("price_reference must contain clean_price")

    reference["conid"] = reference["conid"].astype(str)
    reference["trade_date"] = pd.to_datetime(reference["trade_date"])
    reference["clean_price"] = pd.to_numeric(reference["clean_price"], errors="coerce")
    reference = reference.loc[reference["clean_price"].notna()].copy()
    return reference.sort_values(["trade_date", "conid"]).reset_index(drop=True)


def _compute_trailing_dividend_sum(df):
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, group in df.groupby("conid"):
        indexed = group.set_index("event_date")["amount"].fillna(0.0)
        trailing = indexed.rolling("365D", min_periods=1).sum()
        out.loc[group.index] = trailing.to_numpy()
    return out


def _summarize_dividend_events(df):
    if df.empty:
        return _empty_frame(
            [
                "conid",
                "symbol",
                "product_currency",
                "event_rows",
                "usable_rows",
                "duplicate_rows",
                "currency_mismatch_rows",
                "missing_currency_rows",
                "suspicious_yield_rows",
                "missing_price_reference_rows",
                "min_event_date",
                "max_event_date",
                "usable_ratio",
            ]
        )

    rows = []
    for conid, group in df.groupby("conid"):
        event_rows = int(len(group))
        usable_rows = int(group["usable_for_total_return_adjustment"].sum())
        rows.append(
            {
                "conid": str(conid),
                "symbol": group["symbol"].dropna().iloc[0]
                if group["symbol"].notna().any()
                else None,
                "product_currency": group["product_currency"].dropna().iloc[0]
                if group["product_currency"].notna().any()
                else None,
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
                "usable_ratio": float(usable_rows / event_rows)
                if event_rows > 0
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("conid").reset_index(drop=True)


def preprocess_dividend_events(
    dividend_df=None, price_reference=None, config=None, sqlite_path=SQLITE_DB_PATH
):
    config = config or DividendPreprocessConfig()
    dividend_df = (
        load_dividend_events(sqlite_path)
        if dividend_df is None
        else _normalize_dividend_frame(dividend_df)
    )

    if dividend_df.empty:
        empty_events = _empty_frame(
            [
                "conid",
                "symbol",
                "event_date",
                "amount",
                "dividend_currency",
                "product_currency",
                "description",
                "event_type",
                "declaration_date",
                "record_date",
                "payment_date",
                "previous_price_date",
                "previous_clean_price",
                "price_reference_age_days",
                "implied_yield_vs_previous_price",
                "trailing_dividend_sum_365d",
                "is_missing_amount",
                "is_nonpositive_amount",
                "is_missing_currency",
                "is_currency_mismatch",
                "is_duplicate_event_signature",
                "is_missing_price_reference",
                "is_stale_price_reference",
                "is_suspicious_implied_yield",
                "usable_for_total_return_adjustment",
            ]
        )
        return {
            "events": empty_events,
            "summary": _summarize_dividend_events(empty_events),
            "config": config,
        }

    reference = _build_clean_price_reference(
        price_reference=price_reference, sqlite_path=sqlite_path
    )
    left = dividend_df.sort_values(["event_date", "conid"]).reset_index(drop=True)
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
    trailing_values = _compute_trailing_dividend_sum(trailing_input)
    merged["trailing_dividend_sum_365d"] = trailing_values.reindex(merged.index)
    merged["is_missing_amount"] = merged["amount"].isna()
    merged["is_nonpositive_amount"] = merged["amount"].notna() & (
        merged["amount"] <= 0.0
    )
    merged["is_missing_currency"] = (
        merged["dividend_currency"].fillna("").str.strip().eq("")
    )
    merged["is_currency_mismatch"] = (
        ~merged["is_missing_currency"]
        & merged["product_currency"].fillna("").str.strip().ne("")
        & merged["product_currency"]
        .fillna("")
        .ne(merged["dividend_currency"].fillna(""))
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
        (merged["implied_yield_vs_previous_price"] <= 0.0)
        | (merged["implied_yield_vs_previous_price"] > config.max_implied_yield)
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

    merged = merged.sort_values(["conid", "event_date"]).reset_index(drop=True)
    summary = _summarize_dividend_events(merged)
    return {
        "events": merged,
        "summary": summary,
        "config": config,
    }


def save_dividend_preprocess_results(result, output_dir=None):
    output_dir = Path(output_dir or (DATA_DIR / "analysis"))
    output_dir.mkdir(parents=True, exist_ok=True)

    events_path = output_dir / "analysis_dividend_events.parquet"
    summary_path = output_dir / "analysis_dividend_summary.parquet"
    result["events"].to_parquet(events_path, index=False)
    result["summary"].to_parquet(summary_path, index=False)

    return {
        "events_path": str(events_path),
        "summary_path": str(summary_path),
    }


def run_dividend_preprocess(
    sqlite_path=SQLITE_DB_PATH, output_dir=None, **config_kwargs
):
    config = DividendPreprocessConfig(**config_kwargs)
    result = preprocess_dividend_events(config=config, sqlite_path=sqlite_path)
    paths = save_dividend_preprocess_results(result, output_dir=output_dir)

    summary = result["summary"]
    return {
        "status": "ok",
        "rows": int(len(result["events"])),
        "conids": int(summary["conid"].nunique()) if not summary.empty else 0,
        "usable_rows": int(result["events"]["usable_for_total_return_adjustment"].sum())
        if not result["events"].empty
        else 0,
        **paths,
    }
