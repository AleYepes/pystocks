from __future__ import annotations

from datetime import UTC, datetime

from pystocks_next.storage.raw_capture import capture_raw_payload


def test_capture_raw_payload_deduplicates_blobs_and_keeps_observations(
    temp_store,
) -> None:
    first = capture_raw_payload(
        temp_store,
        source_family="ibkr",
        endpoint="holdings",
        conid="123",
        observed_at=datetime(2026, 1, 2, 10, 0, tzinfo=UTC),
        payload={"rows": [1, 2, 3]},
    )
    second = capture_raw_payload(
        temp_store,
        source_family="ibkr",
        endpoint="holdings",
        conid="123",
        observed_at=datetime(2026, 1, 3, 10, 0, tzinfo=UTC),
        payload={"rows": [1, 2, 3]},
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
