from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from pystocks_next.storage.time import UnresolvedEffectiveAtError, resolve_effective_at


def test_snapshot_endpoint_uses_source_as_of_date() -> None:
    resolution = resolve_effective_at(
        "holdings_snapshot",
        observed_at=datetime(2026, 1, 3, 9, 0, tzinfo=UTC),
        source_as_of_date=date(2025, 12, 31),
    )

    assert resolution.effective_at == date(2025, 12, 31)
    assert resolution.source == "source_as_of_date"


def test_observed_snapshot_endpoint_falls_back_to_observed_at() -> None:
    resolution = resolve_effective_at(
        "profile_and_fees_snapshot",
        observed_at=datetime(2026, 1, 3, 9, 0, tzinfo=UTC),
    )

    assert resolution.effective_at == date(2026, 1, 3)
    assert resolution.source == "observed_at"


def test_series_endpoint_uses_row_date() -> None:
    resolution = resolve_effective_at(
        "price_chart_series",
        observed_at=datetime(2026, 1, 3, 9, 0, tzinfo=UTC),
        row_date="2025-12-30",
    )

    assert resolution.effective_at == date(2025, 12, 30)
    assert resolution.source == "row_date"


def test_missing_required_source_date_raises() -> None:
    with pytest.raises(
        UnresolvedEffectiveAtError, match="source_as_of_date is required"
    ):
        resolve_effective_at(
            "holdings_snapshot",
            observed_at=datetime(2026, 1, 3, 9, 0, tzinfo=UTC),
        )
