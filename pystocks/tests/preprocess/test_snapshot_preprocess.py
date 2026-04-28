import pandas as pd

from pystocks.analysis import AnalysisConfig, build_analysis_panel_data
from pystocks.preprocess.snapshots import preprocess_snapshot_features


def test_preprocess_snapshot_features_builds_merged_feature_rows():
    tables = {
        "profile_and_fees": pd.DataFrame(
            [
                {
                    "conid": "a",
                    "effective_at": "2026-01-31",
                    "asset_type": "Equity",
                    "total_net_assets_value": "$1.2B",
                },
                {
                    "conid": "b",
                    "effective_at": "2026-01-31",
                    "asset_type": "Bond",
                    "total_net_assets_value": "$80M",
                },
            ]
        ),
        "holdings_asset_type": pd.DataFrame(
            [
                {
                    "conid": "a",
                    "effective_at": "2026-01-31",
                    "equity": 0.97,
                    "fixed_income": 0.0,
                },
                {
                    "conid": "b",
                    "effective_at": "2026-01-31",
                    "equity": 0.0,
                    "fixed_income": 0.98,
                },
            ]
        ),
        "ratios_key_ratios": pd.DataFrame(
            [
                {
                    "conid": "a",
                    "effective_at": "2026-01-31",
                    "metric_id": "price_book",
                    "value_num": 1.5,
                },
                {
                    "conid": "b",
                    "effective_at": "2026-01-31",
                    "metric_id": "price_book",
                    "value_num": 3.0,
                },
            ]
        ),
    }

    result = preprocess_snapshot_features(tables=tables)
    features = result["features"]

    assert features[["conid", "effective_at"]].drop_duplicates().shape[0] == 2
    row_a = features.loc[features["conid"] == "a"].iloc[0]
    row_b = features.loc[features["conid"] == "b"].iloc[0]
    assert row_a["profile__total_net_assets_num"] == 1.2e9
    assert row_a["ratio_key__price_book"] == 1.5
    assert row_a["sleeve"] == "equity"
    assert row_b["sleeve"] == "bond"


def test_preprocess_snapshot_features_builds_holdings_diagnostics():
    tables = {
        "holdings_geographic_weights": pd.DataFrame(
            [
                {
                    "conid": "near",
                    "effective_at": "2026-01-31",
                    "region": "North America",
                    "value_num": 0.98,
                },
                {
                    "conid": "near",
                    "effective_at": "2026-01-31",
                    "region": "Europe",
                    "value_num": 0.01,
                },
            ]
        ),
        "holdings_currency": pd.DataFrame(
            [
                {
                    "conid": "over",
                    "effective_at": "2026-01-31",
                    "code": "USD",
                    "currency": "US Dollar",
                    "value_num": 1.15,
                },
            ]
        ),
    }

    diagnostics = preprocess_snapshot_features(tables=tables)["holdings_diagnostics"]
    near_row = diagnostics.loc[
        (diagnostics["table_name"] == "holdings_geographic_weights")
        & (diagnostics["conid"] == "near")
    ].iloc[0]
    over_row = diagnostics.loc[
        (diagnostics["table_name"] == "holdings_currency")
        & (diagnostics["conid"] == "over")
    ].iloc[0]

    assert bool(near_row["is_sum_near_one"])
    assert not bool(near_row["is_sum_over_one"])
    assert near_row["category_count"] == 2
    assert bool(over_row["is_sum_over_one"])
    assert bool(over_row["is_sparse_category_coverage"])


def test_preprocess_snapshot_features_ratio_pivot_is_deterministic_and_flags_duplicates():
    base_rows = [
        {
            "conid": "a",
            "effective_at": "2026-01-31",
            "metric_id": "price_book",
            "value_num": 3.0,
        },
        {
            "conid": "a",
            "effective_at": "2026-01-31",
            "metric_id": "price_book",
            "value_num": 2.0,
        },
        {
            "conid": "a",
            "effective_at": "2026-01-31",
            "metric_id": "price_sales",
            "value_num": 4.0,
        },
    ]

    result_a = preprocess_snapshot_features(
        tables={"ratios_key_ratios": pd.DataFrame(base_rows)}
    )
    result_b = preprocess_snapshot_features(
        tables={"ratios_key_ratios": pd.DataFrame(list(reversed(base_rows)))}
    )

    feature_cols = [
        "conid",
        "effective_at",
        "ratio_key__price_book",
        "ratio_key__price_sales",
    ]
    features_a = (
        result_a["features"][feature_cols]
        .sort_values(["conid", "effective_at"])
        .reset_index(drop=True)
    )
    features_b = (
        result_b["features"][feature_cols]
        .sort_values(["conid", "effective_at"])
        .reset_index(drop=True)
    )
    diagnostics = result_a["ratio_diagnostics"]
    row = diagnostics.loc[diagnostics["table_name"] == "ratios_key_ratios"].iloc[0]

    pd.testing.assert_frame_equal(features_a, features_b)
    assert features_a.loc[0, "ratio_key__price_book"] == 2.0
    assert row["duplicate_metric_keys"] == 1
    assert row["duplicate_row_count"] == 1


def test_preprocess_snapshot_features_drops_storage_only_columns():
    result = preprocess_snapshot_features(
        tables={
            "profile_and_fees": pd.DataFrame(
                [
                    {
                        "conid": "a",
                        "effective_at": "2026-01-31",
                        "asset_type": "Equity",
                        "total_net_assets_value": "$100M",
                        "storage_only_flag": "ignore-me",
                    }
                ]
            ),
            "holdings_asset_type": pd.DataFrame(
                [
                    {
                        "conid": "a",
                        "effective_at": "2026-01-31",
                        "equity": 1.0,
                        "storage_only_weight": 0.5,
                    }
                ]
            ),
        }
    )

    features = result["features"]

    assert "profile__storage_only_flag" not in features.columns
    assert "holding_asset__storage_only_weight" not in features.columns


def test_preprocess_snapshot_features_carries_forward_unaligned_endpoint_dates():
    result = preprocess_snapshot_features(
        tables={
            "profile_and_fees": pd.DataFrame(
                [
                    {
                        "conid": "a",
                        "effective_at": "2026-01-31",
                        "asset_type": "Equity",
                        "total_net_assets_value": "$100M",
                    }
                ]
            ),
            "holdings_asset_type": pd.DataFrame(
                [
                    {
                        "conid": "a",
                        "effective_at": "2026-02-15",
                        "equity": 0.9,
                        "fixed_income": 0.1,
                    }
                ]
            ),
            "ratios_key_ratios": pd.DataFrame(
                [
                    {
                        "conid": "a",
                        "effective_at": "2026-02-28",
                        "metric_id": "price_book",
                        "value_num": 1.25,
                    }
                ]
            ),
        }
    )

    features = (
        result["features"].sort_values(["conid", "effective_at"]).reset_index(drop=True)
    )

    assert features["effective_at"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-01-31",
        "2026-02-15",
        "2026-02-28",
    ]
    final_row = features.iloc[-1]
    assert final_row["profile__total_net_assets_num"] == 100_000_000.0
    assert final_row["holding_asset__equity"] == 0.9
    assert final_row["ratio_key__price_book"] == 1.25


def test_build_analysis_panel_uses_processed_snapshot_features():
    snapshot_result = preprocess_snapshot_features(
        tables={
            "profile_and_fees": pd.DataFrame(
                [
                    {
                        "conid": "a",
                        "effective_at": "2026-01-31",
                        "asset_type": "Equity",
                        "total_net_assets_value": "$100M",
                    },
                    {
                        "conid": "a",
                        "effective_at": "2026-02-28",
                        "asset_type": "Equity",
                        "total_net_assets_value": "$110M",
                    },
                    {
                        "conid": "b",
                        "effective_at": "2026-01-31",
                        "asset_type": "Bond",
                        "total_net_assets_value": "$80M",
                    },
                ]
            ),
            "holdings_asset_type": pd.DataFrame(
                [
                    {
                        "conid": "a",
                        "effective_at": "2026-01-31",
                        "equity": 1.0,
                        "fixed_income": 0.0,
                    },
                    {
                        "conid": "a",
                        "effective_at": "2026-02-28",
                        "equity": 1.0,
                        "fixed_income": 0.0,
                    },
                    {
                        "conid": "b",
                        "effective_at": "2026-01-31",
                        "equity": 0.0,
                        "fixed_income": 1.0,
                    },
                ]
            ),
            "ratios_key_ratios": pd.DataFrame(
                [
                    {
                        "conid": "a",
                        "effective_at": "2026-01-31",
                        "metric_id": "price_book",
                        "value_num": 1.0,
                    },
                    {
                        "conid": "a",
                        "effective_at": "2026-02-28",
                        "metric_id": "price_book",
                        "value_num": 2.0,
                    },
                    {
                        "conid": "b",
                        "effective_at": "2026-01-31",
                        "metric_id": "price_book",
                        "value_num": 3.0,
                    },
                ]
            ),
        }
    )
    prices = pd.DataFrame(
        [
            {
                "conid": "a",
                "trade_date": pd.Timestamp("2026-01-30"),
                "price": 10.0,
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "price_value": 10.0,
                "clean_price": 10.0,
                "raw_return": None,
                "clean_return": None,
                "is_valid_price": True,
                "is_stale_price": False,
                "is_outlier_return": False,
                "is_clean_price": True,
            },
            {
                "conid": "a",
                "trade_date": pd.Timestamp("2026-02-27"),
                "price": 11.0,
                "open": 11.0,
                "high": 11.0,
                "low": 11.0,
                "close": 11.0,
                "price_value": 11.0,
                "clean_price": 11.0,
                "raw_return": 0.1,
                "clean_return": 0.1,
                "is_valid_price": True,
                "is_stale_price": False,
                "is_outlier_return": False,
                "is_clean_price": True,
            },
            {
                "conid": "a",
                "trade_date": pd.Timestamp("2026-03-05"),
                "price": 12.0,
                "open": 12.0,
                "high": 12.0,
                "low": 12.0,
                "close": 12.0,
                "price_value": 12.0,
                "clean_price": 12.0,
                "raw_return": 0.09,
                "clean_return": 0.09,
                "is_valid_price": True,
                "is_stale_price": False,
                "is_outlier_return": False,
                "is_clean_price": True,
            },
            {
                "conid": "b",
                "trade_date": pd.Timestamp("2026-01-30"),
                "price": 20.0,
                "open": 20.0,
                "high": 20.0,
                "low": 20.0,
                "close": 20.0,
                "price_value": 20.0,
                "clean_price": 20.0,
                "raw_return": None,
                "clean_return": None,
                "is_valid_price": True,
                "is_stale_price": False,
                "is_outlier_return": False,
                "is_clean_price": True,
            },
            {
                "conid": "b",
                "trade_date": pd.Timestamp("2026-03-05"),
                "price": 20.2,
                "open": 20.2,
                "high": 20.2,
                "low": 20.2,
                "close": 20.2,
                "price_value": 20.2,
                "clean_price": 20.2,
                "raw_return": 0.01,
                "clean_return": 0.01,
                "is_valid_price": True,
                "is_stale_price": False,
                "is_outlier_return": False,
                "is_clean_price": True,
            },
        ]
    )
    eligibility = pd.DataFrame(
        [
            {
                "conid": "a",
                "eligible": True,
                "eligibility_reason": "OK",
                "total_rows": 3,
                "valid_rows": 3,
                "min_date": "2026-01-30",
                "max_date": "2026-03-05",
                "expected_business_days": 25,
                "missing_ratio": 0.0,
                "max_internal_gap_days": 0,
            },
            {
                "conid": "b",
                "eligible": True,
                "eligibility_reason": "OK",
                "total_rows": 2,
                "valid_rows": 2,
                "min_date": "2026-01-30",
                "max_date": "2026-03-05",
                "expected_business_days": 25,
                "missing_ratio": 0.0,
                "max_internal_gap_days": 0,
            },
        ]
    )

    panel = build_analysis_panel_data(
        snapshot_result["features"],
        {"prices": prices, "eligibility": eligibility},
        AnalysisConfig(
            rebalance_freq="M",
            include_macro_features=False,
            require_supplementary_data=False,
        ),
    )

    row_feb = panel[
        (panel["conid"] == "a")
        & (panel["rebalance_date"] == pd.Timestamp("2026-02-28"))
    ].iloc[0]
    row_mar = panel[
        (panel["conid"] == "a")
        & (panel["rebalance_date"] == pd.Timestamp("2026-03-05"))
    ].iloc[0]
    assert row_feb["ratio_key__price_book"] == 2.0
    assert row_mar["ratio_key__price_book"] == 2.0
    assert row_mar["snapshot_age_days"] == 5
