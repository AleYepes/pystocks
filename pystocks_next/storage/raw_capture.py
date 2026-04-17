from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from .dates import DateLike, to_iso_date

JsonValue = dict[str, Any] | list[Any] | str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class RawPayloadCapture:
    payload_hash: str
    blob_inserted: bool
    observation_inserted: bool
    capture_batch_id: str | None


def _serialize_payload(payload: JsonValue | bytes) -> bytes:
    if isinstance(payload, bytes):
        return payload
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _normalize_timestamp(value: datetime | str | None) -> str:
    if value is None:
        return datetime.now(tz=UTC).isoformat()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC).isoformat()
        return value.astimezone(UTC).isoformat()
    return value


def new_capture_batch_id() -> str:
    return uuid4().hex


def capture_raw_payload(
    conn: sqlite3.Connection,
    *,
    source_family: str,
    endpoint: str,
    payload: JsonValue | bytes,
    observed_at: datetime | str | None = None,
    conid: str | None = None,
    source_as_of_date: DateLike = None,
    capture_batch_id: str | None = None,
) -> RawPayloadCapture:
    payload_bytes = _serialize_payload(payload)
    payload_hash = hashlib.sha256(payload_bytes).hexdigest()
    normalized_observed_at = _normalize_timestamp(observed_at)
    normalized_source_as_of_date = to_iso_date(source_as_of_date)
    normalized_capture_batch_id = (
        capture_batch_id.strip() if capture_batch_id is not None else None
    )
    if normalized_capture_batch_id == "":
        normalized_capture_batch_id = None

    inserted_blob = False
    inserted_observation = False

    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO raw_payload_blobs (
            payload_hash,
            payload_bytes,
            payload_size,
            created_at
        ) VALUES (?, ?, ?, ?)
        """,
        (
            payload_hash,
            payload_bytes,
            len(payload_bytes),
            normalized_observed_at,
        ),
    )
    inserted_blob = cursor.rowcount > 0

    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO raw_payload_observations (
            payload_hash,
            source_family,
            endpoint,
            conid,
            observed_at,
            source_as_of_date,
            capture_batch_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload_hash,
            source_family,
            endpoint,
            conid,
            normalized_observed_at,
            normalized_source_as_of_date,
            normalized_capture_batch_id,
        ),
    )
    inserted_observation = cursor.rowcount > 0

    return RawPayloadCapture(
        payload_hash=payload_hash,
        blob_inserted=inserted_blob,
        observation_inserted=inserted_observation,
        capture_batch_id=normalized_capture_batch_id,
    )
