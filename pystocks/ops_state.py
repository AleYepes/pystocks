from datetime import datetime, timezone

import duckdb
import pandas as pd

from .config import FUNDAMENTALS_DUCKDB_PATH


DB_PATH = FUNDAMENTALS_DUCKDB_PATH


def get_connection():
    return duckdb.connect(str(DB_PATH))


def init_db():
    """Initializes operational tables inside DuckDB."""
    con = get_connection()
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS instruments (
                conid VARCHAR PRIMARY KEY,
                symbol VARCHAR,
                exchange VARCHAR,
                isin VARCHAR,
                currency VARCHAR,
                name VARCHAR,
                last_scraped_fundamentals DATE,
                last_status_fundamentals VARCHAR,
                updated_at TIMESTAMP
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS scraper_logs (
                logged_at TIMESTAMP,
                conid VARCHAR,
                endpoint VARCHAR,
                status_code INTEGER,
                error_message VARCHAR
            )
            """
        )
    finally:
        con.close()


def _coalesce_series(series, default=None):
    if series is None:
        return pd.Series(default)
    return series


def upsert_instruments_from_products(products_df):
    """
    Upserts product metadata into instruments from the IB products scrape payload.
    Returns number of unique conids processed.
    """
    if products_df is None or products_df.empty:
        return 0

    df = products_df.copy()
    if "conid" not in df.columns:
        raise ValueError("products_df must include a 'conid' column")

    normalized = pd.DataFrame(
        {
            "conid": df["conid"].astype(str),
            "symbol": _coalesce_series(df.get("symbol"), "").astype(str),
            "exchange": _coalesce_series(df.get("exchangeId"), "").astype(str),
            "isin": _coalesce_series(df.get("isin"), "").astype(str),
            "currency": _coalesce_series(df.get("currency"), "").astype(str),
            "name": _coalesce_series(df.get("name"), _coalesce_series(df.get("description"), "")).astype(str),
        }
    ).drop_duplicates(subset=["conid"], keep="last")

    now_iso = datetime.now(timezone.utc).isoformat()
    con = get_connection()
    try:
        init_db()
        con.register("products_df", normalized)
        con.execute(
            """
            MERGE INTO instruments AS t
            USING products_df AS s
              ON t.conid = s.conid
            WHEN MATCHED THEN UPDATE SET
                symbol = s.symbol,
                exchange = s.exchange,
                isin = s.isin,
                currency = s.currency,
                name = s.name,
                updated_at = ?
            WHEN NOT MATCHED THEN INSERT (
                conid,
                symbol,
                exchange,
                isin,
                currency,
                name,
                updated_at
            ) VALUES (
                s.conid,
                s.symbol,
                s.exchange,
                s.isin,
                s.currency,
                s.name,
                ?
            )
            """,
            [now_iso, now_iso],
        )
        return int(len(normalized))
    finally:
        con.close()


def get_all_instrument_conids():
    init_db()
    con = get_connection()
    try:
        rows = con.execute(
            """
            SELECT conid
            FROM instruments
            WHERE conid IS NOT NULL
              AND TRIM(conid) <> ''
            ORDER BY conid
            """
        ).fetchall()
        return [str(r[0]) for r in rows]
    finally:
        con.close()


def get_scraped_conids(today=None):
    init_db()
    if today is None:
        today = datetime.now().strftime("%Y-%m-%d")
    con = get_connection()
    try:
        rows = con.execute(
            """
            SELECT conid
            FROM instruments
            WHERE last_scraped_fundamentals = ?
              AND conid IS NOT NULL
            """,
            [today],
        ).fetchall()
        return [str(r[0]) for r in rows]
    finally:
        con.close()


def log_scrape(conid, endpoint, status_code, error_message=None):
    init_db()
    con = get_connection()
    try:
        con.execute(
            """
            INSERT INTO scraper_logs (logged_at, conid, endpoint, status_code, error_message)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                datetime.now(timezone.utc).isoformat(),
                str(conid),
                str(endpoint),
                int(status_code),
                error_message,
            ],
        )
    finally:
        con.close()


def update_instrument_fundamentals_status(conid, status, mark_scraped=False):
    """
    Updates per-conid fundamentals scrape status.
    `mark_scraped=True` marks the conid as successfully scraped today.
    """
    init_db()
    con = get_connection()
    try:
        if mark_scraped:
            con.execute(
                """
                UPDATE instruments
                SET last_scraped_fundamentals = ?, last_status_fundamentals = ?, updated_at = ?
                WHERE conid = ?
                """,
                [
                    datetime.now().strftime("%Y-%m-%d"),
                    str(status),
                    datetime.now(timezone.utc).isoformat(),
                    str(conid),
                ],
            )
        else:
            con.execute(
                """
                UPDATE instruments
                SET last_status_fundamentals = ?, updated_at = ?
                WHERE conid = ?
                """,
                [str(status), datetime.now(timezone.utc).isoformat(), str(conid)],
            )
    finally:
        con.close()
