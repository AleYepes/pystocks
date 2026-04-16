from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True, slots=True)
class UniverseExclusion:
    conid: str
    reason: str
    effective_at: str


def upsert_exclusion(
    conn: sqlite3.Connection,
    exclusion: UniverseExclusion,
) -> None:
    conn.execute(
        """
        INSERT INTO universe_exclusions (conid, reason, effective_at)
        VALUES (?, ?, ?)
        ON CONFLICT(conid) DO UPDATE SET
            reason = excluded.reason,
            effective_at = excluded.effective_at
        """,
        (exclusion.conid, exclusion.reason, exclusion.effective_at),
    )


def list_exclusions(conn: sqlite3.Connection) -> list[UniverseExclusion]:
    rows = conn.execute(
        """
        SELECT conid, reason, effective_at
        FROM universe_exclusions
        ORDER BY conid
        """
    ).fetchall()
    return [
        UniverseExclusion(
            conid=str(row["conid"]),
            reason=str(row["reason"]),
            effective_at=str(row["effective_at"]),
        )
        for row in rows
    ]


def exclusion_now(conid: str, reason: str) -> UniverseExclusion:
    return UniverseExclusion(
        conid=conid,
        reason=reason,
        effective_at=datetime.now(tz=UTC).isoformat(),
    )
