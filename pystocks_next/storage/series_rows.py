from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

from .dates import parse_date_candidate

SeriesDateSource = Literal[
    "trade_date",
    "event_date",
    "x",
    "ex_dividend_date",
    "debugY",
    "datetime",
    "displayDate",
]

SERIES_ROW_DATE_FIELDS: dict[str, tuple[SeriesDateSource, ...]] = {
    "price_chart_series": ("x", "debugY"),
    "sentiment_series": ("datetime",),
    "ownership_trade_log_series": ("trade_date", "displayDate"),
    "dividends_events_series": ("trade_date", "event_date", "x", "ex_dividend_date"),
}


@dataclass(frozen=True, slots=True)
class SeriesRowDateResolution:
    endpoint: str
    row_date: date
    source_field: SeriesDateSource


class UnresolvedSeriesRowDateError(ValueError):
    def __init__(self, endpoint: str, reason: str) -> None:
        super().__init__(f"{endpoint}: {reason}")
        self.endpoint = endpoint
        self.reason = reason


def resolve_series_row_date(
    endpoint: str,
    row: Mapping[str, object],
) -> SeriesRowDateResolution:
    try:
        candidate_fields = SERIES_ROW_DATE_FIELDS[endpoint]
    except KeyError as exc:
        raise UnresolvedSeriesRowDateError(
            endpoint, "no series row-date policy is defined"
        ) from exc

    for field in candidate_fields:
        candidate_value = row.get(field)
        if candidate_value is not None and not isinstance(
            candidate_value,
            (date, datetime, int, float, str, Mapping),
        ):
            continue
        parsed = parse_date_candidate(candidate_value)
        if parsed is not None:
            return SeriesRowDateResolution(
                endpoint=endpoint,
                row_date=parsed,
                source_field=field,
            )

    fields = ", ".join(candidate_fields)
    raise UnresolvedSeriesRowDateError(
        endpoint,
        f"row date is required from one of: {fields}",
    )
