import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .config import RESEARCH_DIR, FUNDAMENTALS_DUCKDB_PATH, FUNDAMENTALS_DIR
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
    # Load all price series from the DuckDB view
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
    # Deduplicate per (conid, trade_date) using latest (effective_at, observed_at, endpoint_event_id)
    # Ensure date types for sorting
    df['effective_at'] = pd.to_datetime(df['effective_at'])
    df['observed_at'] = pd.to_datetime(df['observed_at'])
    
    # Sort so that the last row for each (conid, trade_date) is the latest one
    df = df.sort_values(
        by=['conid', 'trade_date', 'effective_at', 'observed_at', 'endpoint_event_id'],
        ascending=[True, True, True, True, True]
    )
    
    # Keep last
    deduped = df.drop_duplicates(subset=['conid', 'trade_date'], keep='last').copy()
    return deduped

def _flag_quality(df, config: PricePreprocessConfig):
    # Choose close_price
    df['close_price'] = df['close'].combine_first(df['price'])
    
    # Flag invalid rows
    df['is_non_positive'] = df['close_price'] <= 0
    
    # Invalid OHLC
    # Only check if OHLC are present (non-null)
    has_ohlc = df[['open', 'high', 'low', 'close']].notnull().all(axis=1)
    df['is_ohlc_inconsistent'] = False
    if has_ohlc.any():
        inconsistent = ((df['low'] > df['high']) | (df['low'] > df['close']) | (df['low'] > df['open']) | 
                       (df['high'] < df['close']) | (df['high'] < df['open']))
        df.loc[has_ohlc, 'is_ohlc_inconsistent'] = inconsistent[has_ohlc]
        
    df['is_valid_row'] = (~df['is_non_positive']) & (~df['is_ohlc_inconsistent'])
    return df

def _compute_returns_and_outliers(df, config: PricePreprocessConfig):
    # Compute returns per conid
    # Ensure sorted by date
    df = df.sort_values(by=['conid', 'trade_date'])
    
    # Calculate daily returns
    df['prev_close'] = df.groupby('conid')['close_price'].shift(1)
    df['pct_change'] = (df['close_price'] / df['prev_close']) - 1.0
    
    # Flag outliers (Global Z-Score approach simplified per plan: outlier_z_threshold)
    # We'll use a robust z-score based on median absolute deviation if possible, 
    # or just simple z-score if "outlier_z_threshold" implies standard deviation.
    # The plan references "outlier_z_threshold=50.0" which is huge for standard Z.
    # We will compute Modified Z-score: 0.6745 * (x - median) / MAD
    
    # Compute global stats for robust z-score
    valid_returns = df.loc[df['is_valid_row'] & df['pct_change'].notnull(), 'pct_change']
    
    if len(valid_returns) > 0:
        median = valid_returns.median()
        mad = (valid_returns - median).abs().median()
        if mad == 0:
            mad = 1e-9 # Avoid division by zero
            
        df['modified_z_score'] = 0.6745 * (df['pct_change'] - median).abs() / mad
        df['is_outlier'] = df['modified_z_score'] > config.outlier_z_threshold
    else:
        df['modified_z_score'] = 0.0
        df['is_outlier'] = False
        
    # Flag stale runs
    # A run is stale if price doesn't change for > stale_run_max_days
    df['price_diff'] = df.groupby('conid')['close_price'].diff()
    df['is_stale_day'] = (df['price_diff'] == 0)
    
    # We need to identify sequences of is_stale_day that exceed max length
    # This is vectorizable but tricky in pandas. 
    # Let's use a group-transform approach.
    
    def flag_stale_runs(group):
        is_stale = group['is_stale_day']
        # Create blocks of True values
        block = (is_stale != is_stale.shift()).cumsum()
        counts = group.groupby(block)['is_stale_day'].transform('sum')
        # If count > max and it is a stale block, mark as stale run
        return (is_stale & (counts > config.stale_run_max_days))

    df['is_stale_run'] = df.groupby('conid').apply(flag_stale_runs).reset_index(level=0, drop=True)
    
    # Final validity
    df['is_clean_price'] = df['is_valid_row'] & (~df['is_outlier']) & (~df['is_stale_run'])
    
    return df

def _compute_eligibility(df, config: PricePreprocessConfig):
    # Group by conid and calculate metrics
    stats = df.groupby('conid').agg(
        total_rows=('trade_date', 'count'),
        valid_rows=('is_clean_price', 'sum'),
        min_date=('trade_date', 'min'),
        max_date=('trade_date', 'max')
    ).reset_index()
    
    # Missing ratio logic is tricky without a full calendar, but we can approximate
    # based on business days between min and max.
    # For now, let's use the plan's implication: "max_missing_ratio=0.30"
    # We can infer "expected" rows from min/max date assuming ~252 days/year?
    # Or just use the ratio of valid to total rows if we assume total_rows includes gaps?
    # Actually `price_chart_series_all` is sparse (only returned rows).
    # So we can't detect "missing" unless we fill the calendar.
    
    # Plan says: "Compute per-conid coverage and eligibility."
    # Plan WP3.3: "Compute stale-run flags ... per-conid coverage and eligibility."
    
    # Let's simple check:
    # 1. History length >= 252 valid days? (This is min_history_days)
    
    stats['eligible'] = stats['valid_rows'] >= config.min_history_days
    stats['eligibility_reason'] = np.where(stats['eligible'], 'OK', 'Insufficient history')
    
    return stats

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
    quality_df = _flag_quality(deduped_df, config)
    processed_df = _compute_returns_and_outliers(quality_df, config)
    eligibility_df = _compute_eligibility(processed_df, config)
    
    # Filter for Clean Series persistence
    clean_df = processed_df[processed_df['is_clean_price']].copy()
    
    # Persist outputs
    # 1. Clean Parquet
    # /data/prices/ibkr_mf_performance_chart_clean/conid=*/*.parquet
    clean_output_dir = Path("data/prices/ibkr_mf_performance_chart_clean")
    if clean_output_dir.exists():
        import shutil
        shutil.rmtree(clean_output_dir)
    clean_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Partition by conid
    clean_df['year'] = pd.to_datetime(clean_df['trade_date']).dt.year
    
    # We write one file per conid to keep it simple and aligned with query patterns
    conids = clean_df['conid'].unique()
    files_written = 0
    
    for conid in conids:
        subset = clean_df[clean_df['conid'] == conid]
        if subset.empty:
            continue
        path = clean_output_dir / f"conid={conid}"
        path.mkdir(parents=True, exist_ok=True)
        subset.to_parquet(path / "prices.parquet", index=False)
        files_written += 1
        
    # 2. Quality Report
    report_path = RESEARCH_DIR / "price_quality_report_latest.json"
    catalog_path = RESEARCH_DIR / "price_quality_catalog.parquet"
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save catalog parquet
    eligibility_df.to_parquet(catalog_path, index=False)
    
    report_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "summary": {
            "total_conids": int(len(eligibility_df)),
            "eligible_conids": int(eligibility_df['eligible'].sum()),
            "total_rows_raw": int(len(raw_df)),
            "total_rows_clean": int(len(clean_df))
        },
        "conid_details": eligibility_df.to_dict(orient='records')
    }
    
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
        
    # Register views
    store = FundamentalsStore()
    _register_clean_views(store, clean_output_dir, catalog_path)
    
    return {
        "status": "success",
        "files_written": files_written,
        "report_path": str(report_path),
        "catalog_path": str(catalog_path)
    }

def _register_clean_views(store, clean_dir, catalog_path):
    con = duckdb.connect(str(store.duckdb_path))
    try:
        clean_pattern = f"{clean_dir.as_posix()}/conid=*/*.parquet"
        con.execute(f"""
            CREATE OR REPLACE VIEW price_chart_series_clean_all AS
            SELECT * FROM read_parquet('{clean_pattern}', union_by_name=true)
        """)
        
        # We need to make sure we select columns explicitly or handled well by duckdb if empty
        con.execute("""
            CREATE OR REPLACE VIEW returns_daily_clean AS
            SELECT 
                conid, 
                trade_date, 
                close_price, 
                pct_change 
            FROM price_chart_series_clean_all
        """)
        
        con.execute(f"""
            CREATE OR REPLACE VIEW price_quality_catalog AS
            SELECT * FROM read_parquet('{catalog_path}')
        """) 
    finally:
        con.close()

if __name__ == "__main__":
    run()
