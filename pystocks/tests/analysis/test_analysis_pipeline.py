import pandas as pd

from pystocks.analysis import (
    AnalysisConfig,
    build_analysis_panel_data,
    cluster_factor_returns,
)


def test_build_analysis_panel_uses_latest_snapshot_at_or_before_rebalance_date():
    snapshot_features = pd.DataFrame(
        [
            {
                "conid": "a",
                "effective_at": pd.Timestamp("2026-01-31"),
                "profile__asset_type": "Equity",
                "profile__total_net_assets_num": 100.0,
                "ratio_key__price_book": 1.0,
            },
            {
                "conid": "a",
                "effective_at": pd.Timestamp("2026-02-28"),
                "profile__asset_type": "Equity",
                "profile__total_net_assets_num": 110.0,
                "ratio_key__price_book": 2.0,
            },
            {
                "conid": "b",
                "effective_at": pd.Timestamp("2026-01-31"),
                "profile__asset_type": "Bond",
                "profile__total_net_assets_num": 80.0,
                "ratio_key__price_book": 3.0,
            },
        ]
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
    price_result = {"prices": prices, "eligibility": eligibility}
    config = AnalysisConfig(rebalance_freq="M")

    panel = build_analysis_panel_data(snapshot_features, price_result, config)

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


def test_cluster_factor_returns_prefers_composite_representative():
    index = pd.date_range("2026-01-01", periods=150, freq="B")
    factor_returns = pd.DataFrame(
        {
            "equity__composite__value": pd.Series(range(150), index=index, dtype=float),
            "equity__raw__ratio_key__price_book": pd.Series(
                range(150), index=index, dtype=float
            ),
            "equity__raw__ratio_key__price_sales": pd.Series(
                range(149, -1, -1), index=index, dtype=float
            ),
        }
    )
    factor_meta = pd.DataFrame(
        [
            {
                "factor_id": "equity__composite__value",
                "sleeve": "equity",
                "family": "composite",
                "kind": "composite",
            },
            {
                "factor_id": "equity__raw__ratio_key__price_book",
                "sleeve": "equity",
                "family": "ratio_key",
                "kind": "raw",
            },
            {
                "factor_id": "equity__raw__ratio_key__price_sales",
                "sleeve": "equity",
                "family": "ratio_key",
                "kind": "raw",
            },
        ]
    )
    config = AnalysisConfig(min_train_days=50, factor_corr_threshold=0.90)

    cluster_df, reduced = cluster_factor_returns(factor_returns, factor_meta, config)

    keepers = cluster_df.loc[cluster_df["keep_factor"], "factor_id"].tolist()
    assert "equity__composite__value" in keepers
    assert "equity__composite__value" in reduced.columns
    assert "equity__raw__ratio_key__price_book" not in reduced.columns


def test_cluster_factor_returns_show_progress_emits_stage_label(capsys):
    index = pd.date_range("2026-01-01", periods=150, freq="B")
    factor_returns = pd.DataFrame(
        {
            "equity__composite__value": pd.Series(range(150), index=index, dtype=float),
            "equity__raw__ratio_key__price_book": pd.Series(
                range(150), index=index, dtype=float
            ),
        }
    )
    factor_meta = pd.DataFrame(
        [
            {
                "factor_id": "equity__composite__value",
                "sleeve": "equity",
                "family": "composite",
                "kind": "composite",
            },
            {
                "factor_id": "equity__raw__ratio_key__price_book",
                "sleeve": "equity",
                "family": "ratio_key",
                "kind": "raw",
            },
        ]
    )

    cluster_factor_returns(
        factor_returns,
        factor_meta,
        AnalysisConfig(min_train_days=50),
        show_progress=True,
    )

    captured = capsys.readouterr()
    assert "Factor clustering" in captured.err
