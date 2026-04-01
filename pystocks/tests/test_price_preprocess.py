import pandas as pd

from pystocks.price_preprocess import PricePreprocessConfig, preprocess_price_history


def test_preprocess_price_history_flags_bridge_price_level_anomaly():
    price_df = pd.DataFrame(
        [
            {
                "conid": "x",
                "trade_date": "2024-01-01",
                "price": 100.0,
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
            },
            {
                "conid": "x",
                "trade_date": "2024-01-02",
                "price": 101.0,
                "open": 101.0,
                "high": 101.0,
                "low": 101.0,
                "close": 101.0,
            },
            {
                "conid": "x",
                "trade_date": "2024-01-03",
                "price": 1.01,
                "open": 1.01,
                "high": 1.01,
                "low": 1.01,
                "close": 1.01,
            },
            {
                "conid": "x",
                "trade_date": "2024-01-04",
                "price": 1.00,
                "open": 1.00,
                "high": 1.00,
                "low": 1.00,
                "close": 1.00,
            },
            {
                "conid": "x",
                "trade_date": "2024-01-05",
                "price": 102.0,
                "open": 102.0,
                "high": 102.0,
                "low": 102.0,
                "close": 102.0,
            },
            {
                "conid": "x",
                "trade_date": "2024-01-08",
                "price": 103.0,
                "open": 103.0,
                "high": 103.0,
                "low": 103.0,
                "close": 103.0,
            },
        ]
    )

    result = preprocess_price_history(
        price_df=price_df,
        config=PricePreprocessConfig(
            outlier_z_threshold=5.0, local_price_ratio_threshold=5.0
        ),
    )
    prices = result["prices"].set_index("trade_date").sort_index()

    assert bool(prices.loc[pd.Timestamp("2024-01-03"), "is_outlier_return"])
    assert bool(prices.loc[pd.Timestamp("2024-01-05"), "is_outlier_return"])
    assert bool(prices.loc[pd.Timestamp("2024-01-04"), "is_price_level_anomaly"])
    assert not bool(prices.loc[pd.Timestamp("2024-01-04"), "is_clean_price"])
    assert pd.isna(prices.loc[pd.Timestamp("2024-01-04"), "clean_price"])


def test_preprocess_price_history_does_not_flag_split_like_step_as_price_level_anomaly():
    price_df = pd.DataFrame(
        [
            {
                "conid": "x",
                "trade_date": "2024-01-01",
                "price": 100.0,
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
            },
            {
                "conid": "x",
                "trade_date": "2024-01-02",
                "price": 10.0,
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
            },
            {
                "conid": "x",
                "trade_date": "2024-01-03",
                "price": 10.2,
                "open": 10.2,
                "high": 10.2,
                "low": 10.2,
                "close": 10.2,
            },
            {
                "conid": "x",
                "trade_date": "2024-01-04",
                "price": 10.3,
                "open": 10.3,
                "high": 10.3,
                "low": 10.3,
                "close": 10.3,
            },
            {
                "conid": "x",
                "trade_date": "2024-01-05",
                "price": 10.4,
                "open": 10.4,
                "high": 10.4,
                "low": 10.4,
                "close": 10.4,
            },
        ]
    )

    result = preprocess_price_history(
        price_df=price_df,
        config=PricePreprocessConfig(
            outlier_z_threshold=500.0, local_price_ratio_threshold=5.0
        ),
    )
    prices = result["prices"].set_index("trade_date").sort_index()

    assert not bool(prices.loc[pd.Timestamp("2024-01-02"), "is_price_level_anomaly"])
    assert bool(prices.loc[pd.Timestamp("2024-01-02"), "is_clean_price"])
