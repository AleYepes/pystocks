from __future__ import annotations

from datetime import date

from pystocks_next.storage.dates import (
    parse_date_candidate,
    parse_ymd_text,
    to_iso_date,
)


def test_parse_date_candidate_handles_varied_ibkr_formats() -> None:
    assert parse_date_candidate("20251231") == date(2025, 12, 31)
    assert parse_date_candidate("2025/12/31") == date(2025, 12, 31)
    assert parse_date_candidate("12/31/2025") == date(2025, 12, 31)
    assert parse_date_candidate(20251231) == date(2025, 12, 31)
    assert parse_date_candidate(1767139200) == date(2025, 12, 31)
    assert parse_date_candidate(1767139200000) == date(2025, 12, 31)


def test_parse_date_candidate_handles_structured_dates_and_embedded_text() -> None:
    assert parse_date_candidate({"y": 2025, "m": "DEC", "d": 31}) == date(2025, 12, 31)
    assert to_iso_date({"t": "2025-12-31T18:45:00Z"}) == "2025-12-31"
    assert parse_ymd_text("$1.2B (2025-12-31)") == date(2025, 12, 31)


def test_parse_date_candidate_rejects_invalid_values() -> None:
    assert parse_date_candidate("not-a-date") is None
    assert parse_date_candidate(0) is None
