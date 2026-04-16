from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

DateSource = Literal["source_as_of_date", "observed_at", "row_date"]
EndpointKind = Literal["snapshot", "series"]


@dataclass(frozen=True, slots=True)
class EndpointTimePolicy:
    endpoint: str
    kind: EndpointKind
    date_source: DateSource
    observed_at_fallback: bool = False


@dataclass(frozen=True, slots=True)
class EffectiveAtResolution:
    endpoint: str
    effective_at: date
    source: DateSource


class UnresolvedEffectiveAtError(ValueError):
    def __init__(self, endpoint: str, reason: str) -> None:
        super().__init__(f"{endpoint}: {reason}")
        self.endpoint = endpoint
        self.reason = reason


ENDPOINT_TIME_POLICIES: dict[str, EndpointTimePolicy] = {
    "profile_and_fees_snapshot": EndpointTimePolicy(
        endpoint="profile_and_fees_snapshot",
        kind="snapshot",
        date_source="observed_at",
        observed_at_fallback=True,
    ),
    "holdings_snapshot": EndpointTimePolicy(
        endpoint="holdings_snapshot",
        kind="snapshot",
        date_source="source_as_of_date",
    ),
    "ratios_snapshot": EndpointTimePolicy(
        endpoint="ratios_snapshot",
        kind="snapshot",
        date_source="source_as_of_date",
    ),
    "lipper_ratings_snapshot": EndpointTimePolicy(
        endpoint="lipper_ratings_snapshot",
        kind="snapshot",
        date_source="source_as_of_date",
    ),
    "dividends_snapshot": EndpointTimePolicy(
        endpoint="dividends_snapshot",
        kind="snapshot",
        date_source="source_as_of_date",
    ),
    "morningstar_snapshot": EndpointTimePolicy(
        endpoint="morningstar_snapshot",
        kind="snapshot",
        date_source="source_as_of_date",
    ),
    "ownership_snapshot": EndpointTimePolicy(
        endpoint="ownership_snapshot",
        kind="snapshot",
        date_source="source_as_of_date",
    ),
    "esg_snapshot": EndpointTimePolicy(
        endpoint="esg_snapshot",
        kind="snapshot",
        date_source="source_as_of_date",
    ),
    "price_chart_snapshot": EndpointTimePolicy(
        endpoint="price_chart_snapshot",
        kind="snapshot",
        date_source="observed_at",
        observed_at_fallback=True,
    ),
    "sentiment_snapshot": EndpointTimePolicy(
        endpoint="sentiment_snapshot",
        kind="snapshot",
        date_source="observed_at",
        observed_at_fallback=True,
    ),
    "price_chart_series": EndpointTimePolicy(
        endpoint="price_chart_series",
        kind="series",
        date_source="row_date",
    ),
    "sentiment_series": EndpointTimePolicy(
        endpoint="sentiment_series",
        kind="series",
        date_source="row_date",
    ),
    "ownership_trade_log_series": EndpointTimePolicy(
        endpoint="ownership_trade_log_series",
        kind="series",
        date_source="row_date",
    ),
    "dividends_events_series": EndpointTimePolicy(
        endpoint="dividends_events_series",
        kind="series",
        date_source="row_date",
    ),
}


def _coerce_date(value: date | datetime | str | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def resolve_effective_at(
    endpoint: str,
    *,
    observed_at: date | datetime | str | None,
    source_as_of_date: date | datetime | str | None = None,
    row_date: date | datetime | str | None = None,
) -> EffectiveAtResolution:
    try:
        policy = ENDPOINT_TIME_POLICIES[endpoint]
    except KeyError as exc:
        raise UnresolvedEffectiveAtError(endpoint, "no time policy is defined") from exc

    if policy.date_source == "row_date":
        effective_at = _coerce_date(row_date)
        if effective_at is None:
            raise UnresolvedEffectiveAtError(endpoint, "row_date is required")
        return EffectiveAtResolution(
            endpoint=endpoint, effective_at=effective_at, source="row_date"
        )

    source_date = _coerce_date(source_as_of_date)
    if source_date is not None:
        return EffectiveAtResolution(
            endpoint=endpoint,
            effective_at=source_date,
            source="source_as_of_date",
        )

    if policy.observed_at_fallback:
        observed_date = _coerce_date(observed_at)
        if observed_date is None:
            raise UnresolvedEffectiveAtError(endpoint, "observed_at is required")
        return EffectiveAtResolution(
            endpoint=endpoint,
            effective_at=observed_date,
            source="observed_at",
        )

    raise UnresolvedEffectiveAtError(endpoint, "source_as_of_date is required")
