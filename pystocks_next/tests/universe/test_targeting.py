from __future__ import annotations

from pystocks_next.universe.governance import UniverseExclusion, upsert_exclusion
from pystocks_next.universe.products import (
    UniverseInstrument,
    list_instruments,
    upsert_instruments,
)
from pystocks_next.universe.targeting import (
    select_explicit_targets,
    select_governed_targets,
)


def test_select_explicit_targets_deduplicates_in_order() -> None:
    assert select_explicit_targets(["3", "1", "3", "2"]) == ["3", "1", "2"]


def test_upsert_instruments_preserves_catalog_fields(temp_store) -> None:
    upsert_instruments(
        temp_store,
        [
            UniverseInstrument(
                conid="100",
                symbol="AAA",
                local_symbol="AAA.L",
                name="Alpha ETF",
                exchange="BEX",
                isin="US0000000001",
                cusip="000000001",
                currency="USD",
                country="US",
                product_type="ETF",
                under_conid="200",
                is_prime_exch_id="T",
                is_new_pdt="F",
                assoc_entity_id="300",
                fc_conid="1",
            )
        ],
    )

    instrument = list_instruments(temp_store)[0]

    assert instrument.local_symbol == "AAA.L"
    assert instrument.cusip == "000000001"
    assert instrument.country == "US"
    assert instrument.under_conid == "200"
    assert instrument.is_prime_exch_id == "T"
    assert instrument.is_new_pdt == "F"
    assert instrument.assoc_entity_id == "300"
    assert instrument.fc_conid == "1"


def test_select_governed_targets_uses_active_universe_minus_exclusions(
    temp_store,
) -> None:
    upsert_instruments(
        temp_store,
        [
            UniverseInstrument(conid="100", symbol="AAA"),
            UniverseInstrument(conid="200", symbol="BBB", is_active=False),
            UniverseInstrument(conid="300", symbol="CCC"),
        ],
    )
    upsert_exclusion(
        temp_store,
        UniverseExclusion(
            conid="300",
            reason="manual exclusion",
            effective_at="2026-01-05",
        ),
    )

    assert select_governed_targets(temp_store) == ["100"]
