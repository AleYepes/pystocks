from __future__ import annotations

from datetime import date

import pytest

from pystocks_next.storage.series_rows import (
    UnresolvedSeriesRowDateError,
    resolve_series_row_date,
)


def test_price_chart_series_row_date_uses_x_then_debug_fallback() -> None:
    first = resolve_series_row_date(
        "price_chart_series",
        {"x": "20251231", "debugY": "2026-01-01"},
    )
    second = resolve_series_row_date(
        "price_chart_series",
        {"x": None, "debugY": "2026-01-02"},
    )

    assert first.row_date == date(2025, 12, 31)
    assert first.source_field == "x"
    assert second.row_date == date(2026, 1, 2)
    assert second.source_field == "debugY"


def test_dividends_events_series_row_date_uses_canonical_fallback_order() -> None:
    raw_point = resolve_series_row_date(
        "dividends_events_series",
        {"x": None, "ex_dividend_date": "2026-01-03"},
    )
    normalized_point = resolve_series_row_date(
        "dividends_events_series",
        {"trade_date": "2026-01-02", "event_date": "2026-01-03"},
    )

    assert raw_point.row_date == date(2026, 1, 3)
    assert raw_point.source_field == "ex_dividend_date"
    assert normalized_point.row_date == date(2026, 1, 2)
    assert normalized_point.source_field == "trade_date"


def test_ownership_trade_log_series_row_date_supports_raw_or_normalized_rows() -> None:
    raw_point = resolve_series_row_date(
        "ownership_trade_log_series",
        {"displayDate": "2026-01-04"},
    )
    normalized_point = resolve_series_row_date(
        "ownership_trade_log_series",
        {"trade_date": "2026-01-05"},
    )

    assert raw_point.row_date == date(2026, 1, 4)
    assert normalized_point.row_date == date(2026, 1, 5)


def test_unresolved_series_row_date_raises() -> None:
    with pytest.raises(
        UnresolvedSeriesRowDateError,
        match="row date is required",
    ):
        resolve_series_row_date("sentiment_series", {"sscore": 1.2})
