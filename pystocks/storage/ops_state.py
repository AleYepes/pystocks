import sqlite3
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from ..config import SQLITE_DB_PATH
from ._sqlite import open_connection
from .schema import init_storage

DB_PATH = SQLITE_DB_PATH
_INITIALIZED_DB_PATH = None


def _connect() -> sqlite3.Connection:
    return open_connection(DB_PATH)


def get_connection() -> sqlite3.Connection:
    return _connect()


def init_db() -> None:
    global _INITIALIZED_DB_PATH
    db_path = str(DB_PATH)
    if _INITIALIZED_DB_PATH == db_path:
        return

    init_storage(DB_PATH)
    _INITIALIZED_DB_PATH = db_path


def _series_or_default(
    df: pd.DataFrame, column: str, default: object = ""
) -> pd.Series:
    if column in df.columns:
        series = pd.Series(df[column], index=df.index)
    else:
        series = pd.Series([default] * len(df), index=df.index)
    return series.fillna(default)


def upsert_instruments_from_products(products_df: pd.DataFrame | None) -> int:
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
    valid_mask = (
        normalized["conid"].notna()
        & normalized["conid"].ne("")
        & (normalized["conid"].str.lower() != "nan")
    )
    normalized = normalized.loc[valid_mask].copy()
    normalized = normalized.drop_duplicates(subset="conid", keep="last")
    records = normalized.to_dict(orient="records")

    now_iso = datetime.now(UTC).isoformat()
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
                    str(record.get("conid", "")),
                    str(record.get("symbol", "")),
                    str(record.get("exchange", "")),
                    str(record.get("isin", "")),
                    str(record.get("currency", "")),
                    str(record.get("name", "")),
                    now_iso,
                )
                for record in records
            ],
        )
        conn.commit()
        return int(len(normalized))
    finally:
        conn.close()


def get_all_instrument_conids() -> list[str]:
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


def get_scraped_conids(today: str | date | datetime | None = None) -> list[str]:
    init_db()
    if today is None:
        current_day = datetime.now().date()
    elif isinstance(today, str):
        current_day = datetime.fromisoformat(today).date()
    elif isinstance(today, datetime):
        current_day = today.date()
    else:
        current_day = today

    cutoff = (current_day - timedelta(days=6)).isoformat()
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT conid
            FROM products
            WHERE last_scraped_fundamentals >= ?
              AND conid IS NOT NULL
            """,
            [cutoff],
        ).fetchall()
        return [str(r[0]) for r in rows]
    finally:
        conn.close()


def update_instrument_fundamentals_status(
    conid: object, status: str, mark_scraped: bool = False
) -> None:
    init_db()
    conid = str(conid)
    now_iso = datetime.now(UTC).isoformat()
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
