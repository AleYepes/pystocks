from __future__ import annotations

import sqlite3


def select_explicit_targets(conids: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for conid in conids:
        if conid in seen:
            continue
        seen.add(conid)
        ordered.append(conid)
    return ordered


def select_governed_targets(
    conn: sqlite3.Connection,
    *,
    limit: int | None = None,
) -> list[str]:
    query = """
        SELECT i.conid
        FROM universe_instruments AS i
        LEFT JOIN universe_exclusions AS e
            ON e.conid = i.conid
        WHERE i.is_active = 1
          AND e.conid IS NULL
        ORDER BY i.conid
    """
    params: tuple[int, ...] = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (limit,)

    rows = conn.execute(query, params).fetchall()
    return [str(row["conid"]) for row in rows]
