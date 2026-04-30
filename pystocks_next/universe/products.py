from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True, slots=True)
class UniverseInstrument:
    conid: str
    symbol: str | None = None
    local_symbol: str | None = None
    name: str | None = None
    exchange: str | None = None
    isin: str | None = None
    cusip: str | None = None
    currency: str | None = None
    country: str | None = None
    product_type: str | None = None
    under_conid: str | None = None
    is_prime_exch_id: str | None = None
    is_new_pdt: str | None = None
    assoc_entity_id: str | None = None
    fc_conid: str | None = None
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
                instrument.local_symbol,
                instrument.name,
                instrument.exchange,
                instrument.isin,
                instrument.cusip,
                instrument.currency,
                instrument.country,
                instrument.product_type,
                instrument.under_conid,
                instrument.is_prime_exch_id,
                instrument.is_new_pdt,
                instrument.assoc_entity_id,
                instrument.fc_conid,
                int(instrument.is_active),
                instrument.updated_at or datetime.now(tz=UTC).isoformat(),
            )
        )

    conn.executemany(
        """
        INSERT INTO universe_instruments (
            conid,
            symbol,
            local_symbol,
            name,
            exchange,
            isin,
            cusip,
            currency,
            country,
            product_type,
            under_conid,
            is_prime_exch_id,
            is_new_pdt,
            assoc_entity_id,
            fc_conid,
            is_active,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(conid) DO UPDATE SET
            symbol = excluded.symbol,
            local_symbol = excluded.local_symbol,
            name = excluded.name,
            exchange = excluded.exchange,
            isin = excluded.isin,
            cusip = excluded.cusip,
            currency = excluded.currency,
            country = excluded.country,
            product_type = excluded.product_type,
            under_conid = excluded.under_conid,
            is_prime_exch_id = excluded.is_prime_exch_id,
            is_new_pdt = excluded.is_new_pdt,
            assoc_entity_id = excluded.assoc_entity_id,
            fc_conid = excluded.fc_conid,
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
            local_symbol,
            name,
            exchange,
            isin,
            cusip,
            currency,
            country,
            product_type,
            under_conid,
            is_prime_exch_id,
            is_new_pdt,
            assoc_entity_id,
            fc_conid,
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
            local_symbol=row["local_symbol"],
            name=row["name"],
            exchange=row["exchange"],
            isin=row["isin"],
            cusip=row["cusip"],
            currency=row["currency"],
            country=row["country"],
            product_type=row["product_type"],
            under_conid=row["under_conid"],
            is_prime_exch_id=row["is_prime_exch_id"],
            is_new_pdt=row["is_new_pdt"],
            assoc_entity_id=row["assoc_entity_id"],
            fc_conid=row["fc_conid"],
            is_active=bool(row["is_active"]),
            updated_at=str(row["updated_at"]),
        )
        for row in rows
    ]
