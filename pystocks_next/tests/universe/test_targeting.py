from __future__ import annotations

from pystocks_next.universe.governance import UniverseExclusion, upsert_exclusion
from pystocks_next.universe.products import UniverseInstrument, upsert_instruments
from pystocks_next.universe.targeting import (
    select_explicit_targets,
    select_governed_targets,
)


def test_select_explicit_targets_deduplicates_in_order() -> None:
    assert select_explicit_targets(["3", "1", "3", "2"]) == ["3", "1", "2"]


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
