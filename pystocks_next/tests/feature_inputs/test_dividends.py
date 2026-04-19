from __future__ import annotations

import pandas as pd
import pytest

from pystocks_next.feature_inputs import (
    DividendInputConfig,
    build_dividend_input_bundle,
)
from pystocks_next.storage.writes import (
    write_dividend_events_series,
    write_price_chart_series,
)
from pystocks_next.universe.products import UniverseInstrument, upsert_instruments


def test_build_dividend_input_bundle_flags_usability_and_summary() -> None:
    dividend_frame = pd.DataFrame(
        [
            {
                "conid": "100",
                "symbol": "AAA",
                "event_date": "2026-01-03",
                "amount": 0.10,
                "dividend_currency": "USD",
                "product_currency": "USD",
                "description": "Monthly",
                "event_type": "cash",
                "declaration_date": "2025-12-10",
                "record_date": "2026-01-04",
                "payment_date": "2026-01-12",
            },
            {
                "conid": "100",
                "symbol": "AAA",
                "event_date": "2026-01-10",
                "amount": 0.12,
                "dividend_currency": "USD",
                "product_currency": "USD",
                "description": "Monthly",
                "event_type": "cash",
                "declaration_date": "2025-12-20",
                "record_date": "2026-01-11",
                "payment_date": "2026-01-19",
            },
            {
                "conid": "100",
                "symbol": "AAA",
                "event_date": "2026-01-10",
                "amount": 0.12,
                "dividend_currency": "USD",
                "product_currency": "USD",
                "description": "Monthly",
                "event_type": "cash",
                "declaration_date": "2025-12-20",
                "record_date": "2026-01-11",
                "payment_date": "2026-01-19",
            },
            {
                "conid": "100",
                "symbol": "AAA",
                "event_date": "2026-01-20",
                "amount": 3.00,
                "dividend_currency": "USD",
                "product_currency": "USD",
                "description": "Special",
                "event_type": "cash",
                "declaration_date": "2026-01-10",
                "record_date": "2026-01-21",
                "payment_date": "2026-01-25",
            },
            {
                "conid": "200",
                "symbol": "BBB",
                "event_date": "2026-01-05",
                "amount": 0.05,
                "dividend_currency": "EUR",
                "product_currency": "USD",
                "description": "Quarterly",
                "event_type": "cash",
                "declaration_date": "2025-12-22",
                "record_date": "2026-01-06",
                "payment_date": "2026-01-15",
            },
        ]
    )
    price_reference = pd.DataFrame(
        [
            {"conid": "100", "trade_date": "2026-01-02", "clean_price": 10.0},
            {"conid": "100", "trade_date": "2026-01-09", "clean_price": 10.0},
            {"conid": "200", "trade_date": "2026-01-02", "clean_price": 20.0},
        ]
    )

    bundle = build_dividend_input_bundle(
        dividends=dividend_frame,
        price_reference=price_reference,
        config=DividendInputConfig(
            max_implied_yield=0.25, max_price_reference_age_days=10
        ),
    )

    events = bundle.dividends
    summary = bundle.dividend_summary

    first_row = events.loc[
        (events["conid"] == "100")
        & (events["event_date"] == pd.Timestamp("2026-01-03"))
    ].iloc[0]
    duplicate_rows = events.loc[
        (events["conid"] == "100")
        & (events["event_date"] == pd.Timestamp("2026-01-10"))
    ]
    special_row = events.loc[
        (events["conid"] == "100")
        & (events["event_date"] == pd.Timestamp("2026-01-20"))
    ].iloc[0]
    mismatch_row = events.loc[events["conid"] == "200"].iloc[0]

    assert first_row["previous_price_date"] == pd.Timestamp("2026-01-02")
    assert first_row["previous_clean_price"] == pytest.approx(10.0)
    assert first_row["implied_yield_vs_previous_price"] == pytest.approx(0.01)
    assert bool(first_row["usable_for_total_return_adjustment"]) is True

    assert duplicate_rows["is_duplicate_event_signature"].all()
    assert (~duplicate_rows["usable_for_total_return_adjustment"]).all()

    assert bool(special_row["is_stale_price_reference"]) is True
    assert bool(special_row["is_suspicious_implied_yield"]) is True
    assert bool(special_row["usable_for_total_return_adjustment"]) is False

    assert bool(mismatch_row["is_currency_mismatch"]) is True
    assert bool(mismatch_row["usable_for_total_return_adjustment"]) is False

    conid100 = summary.loc[summary["conid"] == "100"].iloc[0]
    conid200 = summary.loc[summary["conid"] == "200"].iloc[0]

    assert conid100["usable_rows"] == pytest.approx(1.0)
    assert conid100["duplicate_rows"] == pytest.approx(2.0)
    assert conid100["suspicious_yield_rows"] == pytest.approx(1.0)
    assert conid100["usable_ratio"] == pytest.approx(0.25)
    assert conid200["currency_mismatch_rows"] == pytest.approx(1.0)


def test_build_dividend_input_bundle_reads_from_canonical_storage(
    temp_store,
    sample_price_chart_payload: dict[str, object],
    sample_dividends_payload: dict[str, object],
) -> None:
    upsert_instruments(
        temp_store,
        [UniverseInstrument(conid="100", symbol="AAA", currency="USD")],
    )
    observed_at = "2026-01-05T10:00:00+00:00"
    write_price_chart_series(
        temp_store,
        conid="100",
        payload=sample_price_chart_payload,
        observed_at=observed_at,
    )
    write_dividend_events_series(
        temp_store,
        conid="100",
        payload=sample_dividends_payload,
        observed_at=observed_at,
    )

    bundle = build_dividend_input_bundle(conn=temp_store)

    assert bundle.dividends["event_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-01-03",
        "2026-01-10",
    ]
    assert bundle.dividends["symbol"].tolist() == ["AAA", "AAA"]
    assert bundle.dividends["product_currency"].tolist() == ["USD", "USD"]
    assert bundle.dividends["previous_price_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-01-02",
        "2026-01-02",
    ]
    assert bundle.dividend_summary.iloc[0]["usable_rows"] == pytest.approx(2.0)


def test_build_dividend_input_bundle_requires_price_reference_without_conn(
    sample_dividend_events_frame: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="conn or price_reference is required"):
        build_dividend_input_bundle(dividends=sample_dividend_events_frame)
