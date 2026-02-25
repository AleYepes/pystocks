from datetime import datetime, timezone
import sqlite3

import pandas as pd

from .config import SQLITE_DB_PATH


DB_PATH = SQLITE_DB_PATH
_INITIALIZED_DB_PATH = None


def _connect():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


def get_connection():
    return _connect()


def init_db():
    global _INITIALIZED_DB_PATH
    db_path = str(DB_PATH)
    if _INITIALIZED_DB_PATH == db_path:
        return

    conn = _connect()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS products (
                conid TEXT PRIMARY KEY,
                symbol TEXT,
                exchange TEXT,
                isin TEXT,
                currency TEXT,
                name TEXT,
                last_scraped_fundamentals TEXT,
                last_status_fundamentals TEXT,
                updated_at TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()
    _INITIALIZED_DB_PATH = db_path


def _series_or_default(df, column, default=""):
    if column in df.columns:
        series = df[column]
    else:
        series = pd.Series([default] * len(df), index=df.index)
    return series.fillna(default)


def upsert_instruments_from_products(products_df):
    if products_df is None or products_df.empty:
        return 0

    df = products_df.copy()
    if "conid" not in df.columns:
        raise ValueError("products_df must include a 'conid' column")

    conid_series = _series_or_default(df, "conid", "").astype(str).str.strip()
    name_series = _series_or_default(df, "name", "").astype(str)
    description_series = _series_or_default(df, "description", "").astype(str)
    merged_name = name_series.where(name_series.str.strip() != "", description_series)

    normalized = pd.DataFrame(
        {
            "conid": conid_series,
            "symbol": _series_or_default(df, "symbol", "").astype(str),
            "exchange": _series_or_default(df, "exchangeId", "").astype(str),
            "isin": _series_or_default(df, "isin", "").astype(str),
            "currency": _series_or_default(df, "currency", "").astype(str),
            "name": merged_name,
        }
    )
    normalized = normalized[
        normalized["conid"].notna()
        & normalized["conid"].ne("")
        & (normalized["conid"].str.lower() != "nan")
    ].drop_duplicates(subset=["conid"], keep="last")

    now_iso = datetime.now(timezone.utc).isoformat()
    init_db()
    conn = _connect()
    try:
        conn.executemany(
            """
            INSERT INTO products (
                conid,
                symbol,
                exchange,
                isin,
                currency,
                name,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(conid) DO UPDATE SET
                symbol=excluded.symbol,
                exchange=excluded.exchange,
                isin=excluded.isin,
                currency=excluded.currency,
                name=excluded.name,
                updated_at=excluded.updated_at
            """,
            [
                (
                    str(row.conid),
                    str(row.symbol),
                    str(row.exchange),
                    str(row.isin),
                    str(row.currency),
                    str(row.name),
                    now_iso,
                )
                for row in normalized.itertuples(index=False)
            ],
        )
        conn.commit()
        return int(len(normalized))
    finally:
        conn.close()


def get_all_instrument_conids():
    init_db()
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT conid
            FROM products
            WHERE conid IS NOT NULL
              AND TRIM(conid) <> ''
            ORDER BY conid
            """
        ).fetchall()
        return [str(r[0]) for r in rows]
    finally:
        conn.close()


def get_scraped_conids(today=None):
    init_db()
    if today is None:
        today = datetime.now().strftime("%Y-%m-%d")
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT conid
            FROM products
            WHERE last_scraped_fundamentals = ?
              AND conid IS NOT NULL
            """,
            [today],
        ).fetchall()
        return [str(r[0]) for r in rows]
    finally:
        conn.close()


def log_scrape(conid, endpoint, status_code, error_message=None):
    return None


def update_instrument_fundamentals_status(conid, status, mark_scraped=False):
    init_db()
    conid = str(conid)
    now_iso = datetime.now(timezone.utc).isoformat()
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO products (conid, updated_at)
            VALUES (?, ?)
            ON CONFLICT(conid) DO NOTHING
            """,
            [conid, now_iso],
        )
        if mark_scraped:
            conn.execute(
                """
                UPDATE products
                SET last_scraped_fundamentals = ?,
                    last_status_fundamentals = ?,
                    updated_at = ?
                WHERE conid = ?
                """,
                [
                    datetime.now().strftime("%Y-%m-%d"),
                    str(status),
                    now_iso,
                    conid,
                ],
            )
        else:
            conn.execute(
                """
                UPDATE products
                SET last_status_fundamentals = ?,
                    updated_at = ?
                WHERE conid = ?
                """,
                [str(status), now_iso, conid],
            )
        conn.commit()
    finally:
        conn.close()
