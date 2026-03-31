import pandas as pd

from pystocks.preprocess.dividends import DividendPreprocessConfig, preprocess_dividend_events


def test_preprocess_dividend_events_marks_same_currency_event_usable():
    dividend_df = pd.DataFrame(
        [
            {
                "conid": "x",
                "symbol": "X",
                "event_date": "2024-01-03",
                "amount": 1.5,
                "dividend_currency": "USD",
                "product_currency": "USD",
                "description": "Regular Dividend",
                "event_type": "ACTUAL",
            }
        ]
    )
    price_reference = pd.DataFrame(
        [
            {"conid": "x", "trade_date": "2024-01-01", "clean_price": 99.0},
            {"conid": "x", "trade_date": "2024-01-02", "clean_price": 100.0},
        ]
    )

    result = preprocess_dividend_events(
        dividend_df=dividend_df,
        price_reference=price_reference,
        config=DividendPreprocessConfig(max_implied_yield=0.25),
    )
    row = result["events"].iloc[0]

    assert row["previous_clean_price"] == 100.0
    assert row["implied_yield_vs_previous_price"] == 0.015
    assert not bool(row["is_currency_mismatch"])
    assert bool(row["usable_for_total_return_adjustment"])


def test_preprocess_dividend_events_blocks_cross_currency_adjustment():
    dividend_df = pd.DataFrame(
        [
            {
                "conid": "x",
                "symbol": "X",
                "event_date": "2024-01-03",
                "amount": 1.5,
                "dividend_currency": "USD",
                "product_currency": "MXN",
                "description": "Regular Dividend",
                "event_type": "ACTUAL",
            }
        ]
    )
    price_reference = pd.DataFrame(
        [
            {"conid": "x", "trade_date": "2024-01-02", "clean_price": 100.0},
        ]
    )

    result = preprocess_dividend_events(dividend_df=dividend_df, price_reference=price_reference)
    row = result["events"].iloc[0]

    assert bool(row["is_currency_mismatch"])
    assert not bool(row["usable_for_total_return_adjustment"])


def test_preprocess_dividend_events_flags_suspicious_implied_yield():
    dividend_df = pd.DataFrame(
        [
            {
                "conid": "x",
                "symbol": "X",
                "event_date": "2024-01-03",
                "amount": 40.0,
                "dividend_currency": "USD",
                "product_currency": "USD",
                "description": "Regular Dividend",
                "event_type": "ACTUAL",
            }
        ]
    )
    price_reference = pd.DataFrame(
        [
            {"conid": "x", "trade_date": "2024-01-02", "clean_price": 100.0},
        ]
    )

    result = preprocess_dividend_events(
        dividend_df=dividend_df,
        price_reference=price_reference,
        config=DividendPreprocessConfig(max_implied_yield=0.25),
    )
    row = result["events"].iloc[0]

    assert bool(row["is_suspicious_implied_yield"])
    assert not bool(row["usable_for_total_return_adjustment"])


def test_preprocess_dividend_events_flags_duplicate_signatures():
    dividend_df = pd.DataFrame(
        [
            {
                "conid": "x",
                "symbol": "X",
                "event_date": "2024-01-03",
                "amount": 1.5,
                "dividend_currency": "USD",
                "product_currency": "USD",
                "description": "Regular Dividend",
                "event_type": "ACTUAL",
                "payment_date": "2024-01-31",
            },
            {
                "conid": "x",
                "symbol": "X",
                "event_date": "2024-01-03",
                "amount": 1.5,
                "dividend_currency": "USD",
                "product_currency": "USD",
                "description": "Regular Dividend",
                "event_type": "ACTUAL",
                "payment_date": "2024-01-31",
            },
        ]
    )
    price_reference = pd.DataFrame(
        [
            {"conid": "x", "trade_date": "2024-01-02", "clean_price": 100.0},
        ]
    )

    result = preprocess_dividend_events(dividend_df=dividend_df, price_reference=price_reference)

    assert result["events"]["is_duplicate_event_signature"].tolist() == [True, True]
    assert result["events"]["usable_for_total_return_adjustment"].tolist() == [False, False]
