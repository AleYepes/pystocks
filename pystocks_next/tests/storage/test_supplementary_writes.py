from __future__ import annotations

import pytest

from pystocks_next.storage.writes import (
    write_supplementary_fetch_log,
    write_supplementary_risk_free_sources,
    write_supplementary_world_bank_raw,
)


def test_write_supplementary_tables_persist_rows(
    temp_store,
    sample_risk_free_sources_frame,
    sample_world_bank_raw_frame,
) -> None:
    sources = write_supplementary_risk_free_sources(
        temp_store,
        frame=sample_risk_free_sources_frame,
        observed_at="2026-01-05T10:00:00+00:00",
    )
    raw = write_supplementary_world_bank_raw(
        temp_store,
        frame=sample_world_bank_raw_frame,
        observed_at="2026-01-05T10:00:00+00:00",
    )
    logs = write_supplementary_fetch_log(
        temp_store,
        dataset="world_bank_raw",
        observed_at="2026-01-05T10:00:00+00:00",
        status="ok",
        record_count=2,
        min_key="2025",
        max_key="2025",
        notes="sample import",
    )

    risk_free_row = temp_store.execute(
        """
        SELECT economy_code, nominal_rate, observed_at
        FROM supplementary_risk_free_sources
        WHERE series_id = 'DTB3'
        """
    ).fetchone()
    world_bank_row = temp_store.execute(
        """
        SELECT economy_code, indicator_id, observed_at
        FROM supplementary_world_bank_raw
        WHERE economy_code = 'USA' AND indicator_id = 'SP.POP.TOTL'
        """
    ).fetchone()
    log_count = temp_store.execute(
        "SELECT COUNT(*) FROM supplementary_fetch_log"
    ).fetchone()[0]

    assert sources.rows_written == 2
    assert raw.rows_written == 2
    assert logs.rows_written == 1
    assert risk_free_row["economy_code"] == "USA"
    assert risk_free_row["nominal_rate"] == pytest.approx(0.03)
    assert world_bank_row["observed_at"] == "2026-01-05T10:00:00+00:00"
    assert log_count == 1
