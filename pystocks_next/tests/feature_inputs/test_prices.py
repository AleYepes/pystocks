from __future__ import annotations

import pandas as pd
import pytest

from pystocks_next.feature_inputs import PriceInputConfig, build_price_input_bundle
from pystocks_next.storage.writes import write_price_chart_series
from pystocks_next.universe.products import UniverseInstrument, upsert_instruments


def test_build_price_input_bundle_flags_stale_outliers_and_eligibility() -> None:
    price_frame = pd.DataFrame(
        [
            {
                "conid": "100",
                "trade_date": "2026-01-05",
                "price": 10.0,
                "open": 9.9,
                "high": 10.1,
                "low": 9.8,
                "close": 10.0,
            },
            {
                "conid": "100",
                "trade_date": "2026-01-06",
                "price": 10.0,
                "open": 9.9,
                "high": 10.1,
                "low": 9.8,
                "close": 10.0,
            },
            {
                "conid": "100",
                "trade_date": "2026-01-07",
                "price": 10.0,
                "open": 9.9,
                "high": 10.1,
                "low": 9.8,
                "close": 10.0,
            },
            {
                "conid": "100",
                "trade_date": "2026-01-08",
                "price": 10.0,
                "open": 9.9,
                "high": 10.1,
                "low": 9.8,
                "close": 10.0,
            },
            {
                "conid": "100",
                "trade_date": "2026-01-09",
                "price": 10.0,
                "open": 9.9,
                "high": 10.1,
                "low": 9.8,
                "close": 10.0,
            },
            {
                "conid": "100",
                "trade_date": "2026-01-12",
                "price": 11.0,
                "open": 10.9,
                "high": 11.1,
                "low": 10.8,
                "close": 11.0,
            },
            {
                "conid": "100",
                "trade_date": "2026-01-13",
                "price": 11.0,
                "open": 10.9,
                "high": 11.1,
                "low": 10.8,
                "close": 11.0,
            },
            {
                "conid": "200",
                "trade_date": "2026-01-05",
                "price": 10.0,
                "open": 9.9,
                "high": 10.1,
                "low": 9.8,
                "close": 10.0,
            },
            {
                "conid": "200",
                "trade_date": "2026-01-06",
                "price": 10.1,
                "open": 10.0,
                "high": 10.2,
                "low": 9.9,
                "close": 10.1,
            },
            {
                "conid": "200",
                "trade_date": "2026-01-07",
                "price": 10.2,
                "open": 10.1,
                "high": 10.3,
                "low": 10.0,
                "close": 10.2,
            },
            {
                "conid": "200",
                "trade_date": "2026-01-08",
                "price": 30.0,
                "open": 29.8,
                "high": 30.2,
                "low": 29.7,
                "close": 30.0,
            },
            {
                "conid": "200",
                "trade_date": "2026-01-09",
                "price": 10.3,
                "open": 10.2,
                "high": 10.4,
                "low": 10.1,
                "close": 10.3,
            },
            {
                "conid": "200",
                "trade_date": "2026-01-12",
                "price": 10.4,
                "open": 10.3,
                "high": 10.5,
                "low": 10.2,
                "close": 10.4,
            },
            {
                "conid": "200",
                "trade_date": "2026-01-13",
                "price": 10.5,
                "open": 10.4,
                "high": 10.6,
                "low": 10.3,
                "close": 10.5,
            },
            {
                "conid": "300",
                "trade_date": "2026-01-05",
                "price": 5.0,
                "open": 5.0,
                "high": 4.0,
                "low": 5.0,
                "close": 5.0,
            },
            {
                "conid": "300",
                "trade_date": "2026-01-06",
                "price": 5.1,
                "open": 5.1,
                "high": 4.5,
                "low": 5.2,
                "close": 5.1,
            },
        ]
    )
    config = PriceInputConfig(
        min_history_days=3,
        max_missing_ratio=0.50,
        max_internal_gap_days=5,
        stale_run_max_days=3,
        outlier_z_threshold=3.0,
    )

    bundle = build_price_input_bundle(prices=price_frame, config=config)

    prices = bundle.prices
    eligibility = bundle.price_eligibility

    stale_rows = prices.loc[(prices["conid"] == "100") & prices["is_stale_price"]]
    assert len(stale_rows) == 3
    assert stale_rows["clean_price"].isna().all()

    outlier_rows = prices.loc[(prices["conid"] == "200") & prices["is_outlier_return"]]
    assert len(outlier_rows) == 2
    assert outlier_rows["clean_price"].isna().all()

    conid100 = eligibility.loc[eligibility["conid"] == "100"].iloc[0]
    conid200 = eligibility.loc[eligibility["conid"] == "200"].iloc[0]
    conid300 = eligibility.loc[eligibility["conid"] == "300"].iloc[0]

    assert bool(conid100["eligible"]) is True
    assert bool(conid200["eligible"]) is True
    assert bool(conid300["eligible"]) is False
    assert (
        conid300["eligibility_reason"]
        == "Insufficient history; Excessive missing ratio"
    )


def test_build_price_input_bundle_reads_from_canonical_storage(
    temp_store,
    sample_price_chart_payload: dict[str, object],
) -> None:
    upsert_instruments(
        temp_store,
        [UniverseInstrument(conid="100", symbol="AAA", currency="USD")],
    )
    write_price_chart_series(
        temp_store,
        conid="100",
        payload=sample_price_chart_payload,
        observed_at="2026-01-05T10:00:00+00:00",
    )

    bundle = build_price_input_bundle(
        conn=temp_store,
        config=PriceInputConfig(min_history_days=3),
    )

    assert bundle.prices["trade_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2025-12-31",
        "2026-01-02",
    ]
    assert bundle.prices["price_value"].tolist() == pytest.approx([10.2, 10.3])
    assert bool(bundle.price_eligibility.iloc[0]["eligible"]) is False
    assert (
        bundle.price_eligibility.iloc[0]["eligibility_reason"]
        == "Insufficient history; Excessive missing ratio"
    )


def test_build_price_input_bundle_requires_conn_or_prices() -> None:
    with pytest.raises(ValueError, match="conn or prices is required"):
        build_price_input_bundle()
