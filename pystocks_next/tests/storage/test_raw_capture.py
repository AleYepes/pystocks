from __future__ import annotations

from datetime import UTC, datetime

from pystocks_next.storage.raw_capture import capture_raw_payload


def test_capture_raw_payload_deduplicates_blobs_and_keeps_observations(
    temp_store,
    sample_raw_payload,
) -> None:
    first = capture_raw_payload(
        temp_store,
        source_family="ibkr",
        endpoint="holdings",
        conid="123",
        observed_at=datetime(2026, 1, 2, 10, 0, tzinfo=UTC),
        payload=sample_raw_payload,
    )
    second = capture_raw_payload(
        temp_store,
        source_family="ibkr",
        endpoint="holdings",
        conid="123",
        observed_at=datetime(2026, 1, 3, 10, 0, tzinfo=UTC),
        payload=sample_raw_payload,
    )

    assert first.blob_inserted is True
    assert second.blob_inserted is False
    assert first.payload_hash == second.payload_hash

    blob_count = temp_store.execute(
        "SELECT COUNT(*) FROM raw_payload_blobs"
    ).fetchone()[0]
    observation_count = temp_store.execute(
        "SELECT COUNT(*) FROM raw_payload_observations"
    ).fetchone()[0]

    assert blob_count == 1
    assert observation_count == 2


def test_capture_raw_payload_persists_batch_identity_and_normalized_source_date(
    temp_store,
    sample_raw_payload,
) -> None:
    result = capture_raw_payload(
        temp_store,
        source_family="ibkr",
        endpoint="dividends",
        conid="123",
        observed_at=datetime(2026, 1, 2, 10, 0, tzinfo=UTC),
        source_as_of_date={"y": 2025, "m": "DEC", "d": 31},
        capture_batch_id="batch-001",
        payload=sample_raw_payload,
    )

    row = temp_store.execute(
        """
        SELECT source_as_of_date, capture_batch_id
        FROM raw_payload_observations
        WHERE payload_hash = ?
        """,
        (result.payload_hash,),
    ).fetchone()

    assert result.capture_batch_id == "batch-001"
    assert row["source_as_of_date"] == "2025-12-31"
    assert row["capture_batch_id"] == "batch-001"
