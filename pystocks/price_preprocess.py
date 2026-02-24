import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .config import (
    FUNDAMENTALS_DUCKDB_PATH,
    PRICE_CHART_CLEAN_PARQUET_DIR,
    RESEARCH_DIR,
)
from .fundamentals_store import FundamentalsStore

logger = logging.getLogger(__name__)


@dataclass
class PricePreprocessConfig:
    min_history_days: int = 252
    max_missing_ratio: float = 0.30
    max_internal_gap_days: int = 20
    stale_run_max_days: int = 5
    outlier_z_threshold: float = 50.0


def _get_price_series(con):
    query = """
    SELECT
        conid,
        trade_date,
        timestamp_ms,
        price,
        open,
        high,
        low,
        close,
        effective_at,
        observed_at,
        endpoint_event_id
    FROM price_chart_series_all
    WHERE trade_date IS NOT NULL
    """
    return con.execute(query).fetch_df()


def _deduplicate_series(df):
    df = df.copy()
    df["effective_at"] = pd.to_datetime(df["effective_at"], errors="coerce")
    df["observed_at"] = pd.to_datetime(df["observed_at"], errors="coerce")
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date

    df = df.sort_values(
        by=["conid", "trade_date", "effective_at", "observed_at", "endpoint_event_id"],
        ascending=[True, True, True, True, True],
    )
    return df.drop_duplicates(subset=["conid", "trade_date"], keep="last").copy()


def _flag_quality(df):
    df = df.copy()
    df["close_price"] = pd.to_numeric(df["close"].combine_first(df["price"]), errors="coerce")

    df["is_non_positive"] = df["close_price"] <= 0

    has_ohlc = df[["open", "high", "low", "close"]].notnull().all(axis=1)
    inconsistent = (
        (df["low"] > df["high"])
        | (df["low"] > df["close"])
        | (df["low"] > df["open"])
        | (df["high"] < df["close"])
        | (df["high"] < df["open"])
    )
    df["is_ohlc_inconsistent"] = has_ohlc & inconsistent
    df["is_valid_row"] = (~df["is_non_positive"]) & (~df["is_ohlc_inconsistent"])
    return df


def _compute_returns_and_outliers(df, config: PricePreprocessConfig):
    df = df.copy()
    df = df.sort_values(by=["conid", "trade_date"])

    df["prev_close"] = df.groupby("conid")["close_price"].shift(1)
    df["pct_change"] = (df["close_price"] / df["prev_close"]) - 1.0

    valid_returns = df.loc[df["is_valid_row"] & df["pct_change"].notnull(), "pct_change"]
    if not valid_returns.empty:
        median = float(valid_returns.median())
        mad = float((valid_returns - median).abs().median())
        if mad == 0:
            mad = 1e-9
        df["modified_z_score"] = 0.6745 * (df["pct_change"] - median).abs() / mad
        df["is_outlier"] = df["modified_z_score"] > config.outlier_z_threshold
    else:
        df["modified_z_score"] = 0.0
        df["is_outlier"] = False

    df["price_diff"] = df.groupby("conid")["close_price"].diff()
    df["is_stale_day"] = df["price_diff"] == 0
    df["is_stale_run"] = False

    for _, idx in df.groupby("conid").groups.items():
        group = df.loc[idx]
        stale = group["is_stale_day"].fillna(False)
        block_id = stale.ne(stale.shift(fill_value=False)).cumsum()
        run_len = stale.groupby(block_id).transform("sum")
        df.loc[idx, "is_stale_run"] = stale & (run_len > config.stale_run_max_days)

    df["is_clean_price"] = df["is_valid_row"] & (~df["is_outlier"]) & (~df["is_stale_run"])
    return df


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
            valid_dates = sorted(valid["trade_date"].dropna().unique())
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


def run(config: PricePreprocessConfig = None):
    if config is None:
        config = PricePreprocessConfig()

    con = duckdb.connect(str(FUNDAMENTALS_DUCKDB_PATH), read_only=True)
    try:
        raw_df = _get_price_series(con)
    finally:
        con.close()

    if raw_df.empty:
        logger.warning("No price data found.")
        return {"status": "empty"}

    deduped_df = _deduplicate_series(raw_df)
    quality_df = _flag_quality(deduped_df)
    processed_df = _compute_returns_and_outliers(quality_df, config)
    eligibility_df = _compute_eligibility(processed_df, config)

    clean_df = processed_df[processed_df["is_clean_price"]].copy()

    clean_output_dir = PRICE_CHART_CLEAN_PARQUET_DIR
    if clean_output_dir.exists():
        shutil.rmtree(clean_output_dir)
    clean_output_dir.mkdir(parents=True, exist_ok=True)

    files_written = 0
    for conid, subset in clean_df.groupby("conid"):
        if subset.empty:
            continue
        conid_dir = clean_output_dir / f"conid={conid}"
        conid_dir.mkdir(parents=True, exist_ok=True)
        subset.to_parquet(conid_dir / "prices.parquet", index=False)
        files_written += 1

    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESEARCH_DIR / "price_quality_report_latest.json"
    catalog_path = RESEARCH_DIR / "price_quality_catalog.parquet"
    eligibility_df.to_parquet(catalog_path, index=False)

    report_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "summary": {
            "total_conids": int(len(eligibility_df)),
            "eligible_conids": int(eligibility_df["eligible"].sum()),
            "total_rows_raw": int(len(raw_df)),
            "total_rows_clean": int(len(clean_df)),
        },
        "conid_details": eligibility_df.to_dict(orient="records"),
    }
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)

    _register_clean_views(FundamentalsStore(), clean_output_dir, catalog_path)

    return {
        "status": "success",
        "files_written": files_written,
        "report_path": str(report_path),
        "catalog_path": str(catalog_path),
    }


def _create_empty_clean_views(con):
    con.execute(
        """
        CREATE OR REPLACE VIEW price_chart_series_clean_all AS
        SELECT
            CAST(NULL AS VARCHAR) AS conid,
            CAST(NULL AS VARCHAR) AS trade_date,
            CAST(NULL AS DOUBLE) AS close_price,
            CAST(NULL AS DOUBLE) AS pct_change,
            CAST(NULL AS BOOLEAN) AS is_clean_price
        WHERE FALSE
        """
    )
    con.execute(
        """
        CREATE OR REPLACE VIEW returns_daily_clean AS
        SELECT conid, trade_date, close_price, pct_change
        FROM price_chart_series_clean_all
        """
    )


def _register_clean_views(store, clean_dir: Path, catalog_path: Path):
    con = duckdb.connect(str(store.duckdb_path))
    try:
        clean_files = list(clean_dir.glob("conid=*/*.parquet"))
        if clean_files:
            pattern = f"{clean_dir.as_posix()}/conid=*/*.parquet"
            con.execute(
                f"""
                CREATE OR REPLACE VIEW price_chart_series_clean_all AS
                SELECT * FROM read_parquet('{pattern}', union_by_name=true)
                """
            )
            con.execute(
                """
                CREATE OR REPLACE VIEW returns_daily_clean AS
                SELECT conid, trade_date, close_price, pct_change
                FROM price_chart_series_clean_all
                """
            )
        else:
            _create_empty_clean_views(con)

        if catalog_path.exists():
            con.execute(
                f"""
                CREATE OR REPLACE VIEW price_quality_catalog AS
                SELECT * FROM read_parquet('{catalog_path.as_posix()}')
                """
            )
        else:
            con.execute(
                """
                CREATE OR REPLACE VIEW price_quality_catalog AS
                SELECT
                    CAST(NULL AS VARCHAR) AS conid,
                    CAST(NULL AS BOOLEAN) AS eligible,
                    CAST(NULL AS VARCHAR) AS eligibility_reason
                WHERE FALSE
                """
            )
    finally:
        con.close()


if __name__ == "__main__":
    run()
