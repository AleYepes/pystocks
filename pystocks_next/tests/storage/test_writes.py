from __future__ import annotations

from pystocks_next.storage.writes import (
    write_dividend_events_series,
    write_price_chart_series,
)
from pystocks_next.universe.products import UniverseInstrument, upsert_instruments


def test_write_price_chart_series_persists_rows_and_raw_capture(
    temp_store,
    sample_price_chart_payload: dict[str, object],
) -> None:
    upsert_instruments(temp_store, [UniverseInstrument(conid="100", symbol="AAA")])

    result = write_price_chart_series(
        temp_store,
        conid="100",
        payload=sample_price_chart_payload,
        observed_at="2026-01-05T10:00:00+00:00",
        capture_batch_id="batch-price-001",
    )

    rows = temp_store.execute(
        """
        SELECT effective_at, close, capture_batch_id, debug_mismatch
        FROM price_chart_series
        WHERE conid = '100'
        ORDER BY effective_at
        """
    ).fetchall()

    assert result.raw_observation_inserted is True
    assert result.rows_upserted == 2
    assert [row["effective_at"] for row in rows] == ["2025-12-31", "2026-01-02"]
    assert rows[0]["capture_batch_id"] == "batch-price-001"
    assert rows[1]["debug_mismatch"] == 0


def test_write_dividend_events_series_deduplicates_identical_events(
    temp_store,
    sample_dividends_payload: dict[str, object],
) -> None:
    upsert_instruments(
        temp_store,
        [UniverseInstrument(conid="100", symbol="AAA", currency="USD")],
    )

    first = write_dividend_events_series(
        temp_store,
        conid="100",
        payload=sample_dividends_payload,
        observed_at="2026-01-05T10:00:00+00:00",
        source_as_of_date={"y": 2026, "m": 1, "d": 3},
        capture_batch_id="batch-div-001",
    )
    second = write_dividend_events_series(
        temp_store,
        conid="100",
        payload=sample_dividends_payload,
        observed_at="2026-01-06T10:00:00+00:00",
        source_as_of_date="2026-01-03",
        capture_batch_id="batch-div-002",
    )

    rows = temp_store.execute(
        """
        SELECT effective_at, currency, capture_batch_id
        FROM dividends_events_series
        WHERE conid = '100'
        ORDER BY effective_at
        """
    ).fetchall()
    observations = temp_store.execute(
        """
        SELECT COUNT(*)
        FROM raw_payload_observations
        WHERE conid = '100' AND endpoint = 'dividends'
        """
    ).fetchone()[0]

    assert first.rows_inserted == 2
    assert second.rows_inserted == 0
    assert [row["effective_at"] for row in rows] == ["2026-01-03", "2026-01-10"]
    assert rows[0]["currency"] == "USD"
    assert rows[0]["capture_batch_id"] == "batch-div-001"
    assert observations == 2
