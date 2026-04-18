from __future__ import annotations

import pytest

from pystocks_next.storage.writes import (
    write_holdings_snapshot,
    write_profile_and_fees_snapshot,
)
from pystocks_next.universe.products import UniverseInstrument, upsert_instruments


def test_write_profile_and_fees_snapshot_persists_canonical_row(
    temp_store,
    sample_profile_and_fees_payload: dict[str, object],
) -> None:
    upsert_instruments(temp_store, [UniverseInstrument(conid="100", symbol="AAA")])

    result = write_profile_and_fees_snapshot(
        temp_store,
        conid="100",
        payload=sample_profile_and_fees_payload,
        observed_at="2026-01-05T10:00:00+00:00",
        capture_batch_id="batch-profile-001",
    )

    snapshot_row = temp_store.execute(
        """
        SELECT effective_at, capture_batch_id
        FROM profile_and_fees_snapshots
        WHERE conid = '100'
        """
    ).fetchone()
    canonical_row = temp_store.execute(
        """
        SELECT asset_type, management_expenses, total_net_assets_value, total_net_assets_date
        FROM profile_and_fees
        WHERE conid = '100'
        """
    ).fetchone()

    assert result.effective_at == "2026-01-05"
    assert snapshot_row["capture_batch_id"] == "batch-profile-001"
    assert canonical_row["asset_type"] == "Equity"
    assert canonical_row["management_expenses"] == pytest.approx(0.0012)
    assert canonical_row["total_net_assets_value"] == "$1.2B"
    assert canonical_row["total_net_assets_date"] == "2026-01-02"


def test_write_holdings_snapshot_persists_asset_type_weights(
    temp_store,
    sample_holdings_payload: dict[str, object],
) -> None:
    upsert_instruments(temp_store, [UniverseInstrument(conid="100", symbol="AAA")])

    result = write_holdings_snapshot(
        temp_store,
        conid="100",
        payload=sample_holdings_payload,
        observed_at="2026-01-05T10:00:00+00:00",
        capture_batch_id="batch-holdings-001",
    )

    snapshot_row = temp_store.execute(
        """
        SELECT effective_at, as_of_date, capture_batch_id
        FROM holdings_snapshots
        WHERE conid = '100'
        """
    ).fetchone()
    canonical_row = temp_store.execute(
        """
        SELECT equity, cash, fixed_income, other
        FROM holdings_asset_type
        WHERE conid = '100'
        """
    ).fetchone()

    assert result.effective_at == "2026-01-03"
    assert snapshot_row["as_of_date"] == "2026-01-03"
    assert snapshot_row["capture_batch_id"] == "batch-holdings-001"
    assert canonical_row["equity"] == pytest.approx(0.85)
    assert canonical_row["cash"] == pytest.approx(0.10)
    assert canonical_row["fixed_income"] == pytest.approx(0.05)
    assert canonical_row["other"] is None
