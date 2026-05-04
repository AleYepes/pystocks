from __future__ import annotations

import pytest

from pystocks_next.feature_inputs import build_snapshot_input_bundle
from pystocks_next.storage.writes import (
    write_dividends_snapshot,
    write_holdings_snapshot,
    write_lipper_ratings_snapshot,
    write_morningstar_snapshot,
    write_profile_and_fees_snapshot,
    write_ratios_snapshot,
)
from pystocks_next.universe.products import UniverseInstrument, upsert_instruments


def test_build_snapshot_input_bundle_builds_features_and_diagnostics(
    temp_store,
    sample_profile_and_fees_payload: dict[str, object],
    sample_holdings_payload: dict[str, object],
    sample_ratios_payload: dict[str, object],
    sample_dividends_snapshot_payload: dict[str, object],
    sample_morningstar_payload: dict[str, object],
    sample_lipper_payload: dict[str, object],
) -> None:
    upsert_instruments(
        temp_store,
        [UniverseInstrument(conid="100", symbol="AAA", currency="USD")],
    )
    observed_at = "2026-01-05T10:00:00+00:00"

    write_profile_and_fees_snapshot(
        temp_store,
        conid="100",
        payload=sample_profile_and_fees_payload,
        observed_at=observed_at,
    )
    write_holdings_snapshot(
        temp_store,
        conid="100",
        payload=sample_holdings_payload,
        observed_at=observed_at,
    )
    write_ratios_snapshot(
        temp_store,
        conid="100",
        payload=sample_ratios_payload,
        observed_at=observed_at,
    )
    write_dividends_snapshot(
        temp_store,
        conid="100",
        payload=sample_dividends_snapshot_payload,
        observed_at=observed_at,
    )
    write_morningstar_snapshot(
        temp_store,
        conid="100",
        payload=sample_morningstar_payload,
        observed_at=observed_at,
    )
    write_lipper_ratings_snapshot(
        temp_store,
        conid="100",
        payload=sample_lipper_payload,
        observed_at=observed_at,
    )

    bundle = build_snapshot_input_bundle(conn=temp_store)

    assert bundle.snapshot_features["conid"].tolist() == ["100", "100", "100", "100"]
    assert bundle.snapshot_features["effective_at"].dt.strftime(
        "%Y-%m-%d"
    ).tolist() == [
        "2026-01-02",
        "2026-01-03",
        "2026-01-30",
        "2026-01-31",
    ]

    row_asof = bundle.snapshot_features.loc[
        bundle.snapshot_features["effective_at"].dt.strftime("%Y-%m-%d") == "2026-01-03"
    ].iloc[0]
    row_profile = bundle.snapshot_features.loc[
        bundle.snapshot_features["effective_at"].dt.strftime("%Y-%m-%d") == "2026-01-02"
    ].iloc[0]
    row_lipper = bundle.snapshot_features.loc[
        bundle.snapshot_features["effective_at"].dt.strftime("%Y-%m-%d") == "2026-01-30"
    ].iloc[0]
    row_morningstar = bundle.snapshot_features.loc[
        bundle.snapshot_features["effective_at"].dt.strftime("%Y-%m-%d") == "2026-01-31"
    ].iloc[0]

    assert row_profile["profile__asset_type"] == "Equity"
    assert row_profile["profile__management_expenses_ratio"] == pytest.approx(1.0)
    assert row_profile["profile__total_net_assets_num"] == pytest.approx(1.2e9)
    assert row_profile["sleeve"] == "equity"

    assert row_asof["holding_asset__equity"] == pytest.approx(0.85)
    assert row_asof["holding_quality__quality_aa"] == pytest.approx(0.15)
    assert row_asof["holding_maturity__maturity_1_to_3_years"] == pytest.approx(0.125)
    assert row_asof["industry__technology"] == pytest.approx(0.448681)
    assert row_asof["currency__usd"] == pytest.approx(0.999604)
    assert row_asof["country__us"] == pytest.approx(0.973418)
    assert row_asof["region__us"] == pytest.approx(0.9734)
    assert row_asof["debt_type__sovereign_bond"] == pytest.approx(0.20)
    assert row_asof["top10__top10_count"] == pytest.approx(1.0)
    assert row_asof["ratio_key__price_sales"] == pytest.approx(3.63)
    assert row_asof["ratio_key_vs__price_sales"] == pytest.approx(0.0146)
    assert row_asof["ratio_fixed_income__current_yield"] == pytest.approx(3.17)
    assert row_asof["dividend_metric__dividend_yield"] == pytest.approx(0.0122)

    assert row_lipper["lipper__overall_total_return"] == pytest.approx(5.0)
    assert row_lipper["lipper__3_year_total_return"] == pytest.approx(4.0)

    assert row_morningstar["morningstar__medalist_rating"] == "Silver"
    assert row_morningstar["morningstar__morningstar_rating"] == pytest.approx(4.0)

    holdings_tables = set(bundle.snapshot_holdings_diagnostics["table_name"].tolist())
    ratio_tables = set(bundle.snapshot_ratio_diagnostics["table_name"].tolist())
    summary_tables = set(bundle.snapshot_table_summary["table_name"].tolist())

    assert "holdings_asset_type" in holdings_tables
    assert "holdings_top10" in holdings_tables
    assert "ratios_key_ratios" in ratio_tables
    assert "lipper_ratings" in ratio_tables
    assert "profile_fields" in summary_tables
    assert "profile_overview" in summary_tables
    assert "morningstar_summary" in summary_tables


def test_build_snapshot_input_bundle_requires_conn_or_tables() -> None:
    with pytest.raises(ValueError, match="conn or tables is required"):
        build_snapshot_input_bundle()
