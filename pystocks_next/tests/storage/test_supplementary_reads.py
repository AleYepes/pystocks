from __future__ import annotations

import pandas as pd
import pytest

from pystocks_next.storage import (
    load_latest_holdings_country_weights,
    load_risk_free_sources,
    load_world_bank_raw,
)
from pystocks_next.storage.writes import (
    write_holdings_snapshot,
    write_supplementary_risk_free_sources,
    write_supplementary_world_bank_raw,
)
from pystocks_next.universe.products import UniverseInstrument, upsert_instruments


def test_load_supplementary_raw_reads_normalize_types(
    temp_store,
    sample_risk_free_sources_frame: pd.DataFrame,
    sample_world_bank_raw_frame: pd.DataFrame,
) -> None:
    observed_at = "2026-01-05T10:00:00+00:00"
    write_supplementary_risk_free_sources(
        temp_store,
        frame=sample_risk_free_sources_frame.assign(observed_at=observed_at),
        observed_at=observed_at,
    )
    write_supplementary_world_bank_raw(
        temp_store,
        frame=sample_world_bank_raw_frame.assign(observed_at=observed_at),
        observed_at=observed_at,
    )

    risk_free = load_risk_free_sources(temp_store).frame
    world_bank = load_world_bank_raw(temp_store).frame

    assert risk_free["economy_code"].tolist() == ["USA", "CAN"]
    assert pd.api.types.is_datetime64_any_dtype(risk_free["trade_date"])
    assert pd.api.types.is_datetime64_any_dtype(risk_free["observed_at"])
    assert world_bank["economy_code"].tolist() == ["USA", "USA"]
    assert str(world_bank["year"].dtype) == "Int64"
    assert pd.api.types.is_datetime64_any_dtype(world_bank["observed_at"])


def test_load_latest_holdings_country_weights_reads_latest_snapshot_only(
    temp_store,
    sample_holdings_payload: dict[str, object],
) -> None:
    upsert_instruments(
        temp_store,
        [UniverseInstrument(conid="100", symbol="AAA", currency="USD")],
    )
    older_payload = dict(sample_holdings_payload)
    older_payload["as_of_date"] = "2026-01-01"
    older_payload["investor_country"] = [
        {"name": "Canada", "weight": "100%", "country_code": "CA"}
    ]

    write_holdings_snapshot(
        temp_store,
        conid="100",
        payload=older_payload,
        observed_at="2026-01-02T10:00:00+00:00",
    )
    write_holdings_snapshot(
        temp_store,
        conid="100",
        payload=sample_holdings_payload,
        observed_at="2026-01-05T10:00:00+00:00",
    )

    result = load_latest_holdings_country_weights(temp_store).frame

    assert result["economy_code"].tolist() == ["US"]
    assert result["weight"].tolist() == pytest.approx([0.973418])
