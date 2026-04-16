from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True, slots=True)
class UniverseInstrument:
    conid: str
    symbol: str | None = None
    name: str | None = None
    exchange: str | None = None
    isin: str | None = None
    currency: str | None = None
    product_type: str | None = None
    is_active: bool = True
    updated_at: str | None = None


def upsert_instruments(
    conn: sqlite3.Connection,
    instruments: list[UniverseInstrument],
) -> int:
    if not instruments:
        return 0

    normalized_rows = []
    for instrument in instruments:
        normalized_rows.append(
            (
                instrument.conid,
                instrument.symbol,
                instrument.name,
                instrument.exchange,
                instrument.isin,
                instrument.currency,
                instrument.product_type,
                int(instrument.is_active),
                instrument.updated_at or datetime.now(tz=UTC).isoformat(),
            )
        )

    conn.executemany(
        """
        INSERT INTO universe_instruments (
            conid,
            symbol,
            name,
            exchange,
            isin,
            currency,
            product_type,
            is_active,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(conid) DO UPDATE SET
            symbol = excluded.symbol,
            name = excluded.name,
            exchange = excluded.exchange,
            isin = excluded.isin,
            currency = excluded.currency,
            product_type = excluded.product_type,
            is_active = excluded.is_active,
            updated_at = excluded.updated_at
        """,
        normalized_rows,
    )
    return len(normalized_rows)


def list_instruments(
    conn: sqlite3.Connection,
    *,
    active_only: bool = False,
) -> list[UniverseInstrument]:
    query = """
        SELECT
            conid,
            symbol,
            name,
            exchange,
            isin,
            currency,
            product_type,
            is_active,
            updated_at
        FROM universe_instruments
    """
    params: tuple[int, ...] = ()
    if active_only:
        query += " WHERE is_active = ?"
        params = (1,)
    query += " ORDER BY conid"

    rows = conn.execute(query, params).fetchall()
    return [
        UniverseInstrument(
            conid=str(row["conid"]),
            symbol=row["symbol"],
            name=row["name"],
            exchange=row["exchange"],
            isin=row["isin"],
            currency=row["currency"],
            product_type=row["product_type"],
            is_active=bool(row["is_active"]),
            updated_at=str(row["updated_at"]),
        )
        for row in rows
    ]
