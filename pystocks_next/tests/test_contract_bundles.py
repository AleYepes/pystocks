from __future__ import annotations

import pandas as pd

from pystocks_next.feature_inputs.bundle import (
    DIVIDEND_EVENT_COLUMNS,
    DIVIDEND_SUMMARY_COLUMNS,
    MACRO_FEATURE_COLUMNS,
    PRICE_ELIGIBILITY_COLUMNS,
    PRICE_INPUT_COLUMNS,
    RISK_FREE_DAILY_COLUMNS,
    SNAPSHOT_FEATURE_REQUIRED_COLUMNS,
    SNAPSHOT_HOLDINGS_DIAGNOSTIC_COLUMNS,
    SNAPSHOT_RATIO_DIAGNOSTIC_COLUMNS,
    SNAPSHOT_TABLE_SUMMARY_COLUMNS,
    AnalysisInputBundle,
)
from pystocks_next.portfolio.inputs import (
    COVARIANCE_COLUMNS,
    ELIGIBILITY_COLUMNS,
    EXPECTED_RETURN_COLUMNS,
    EXPOSURE_COLUMNS,
    PortfolioInputBundle,
)


def test_analysis_input_bundle_empty_uses_explicit_contract_columns() -> None:
    bundle = AnalysisInputBundle.empty()

    assert tuple(bundle.prices.columns) == PRICE_INPUT_COLUMNS
    assert tuple(bundle.price_eligibility.columns) == PRICE_ELIGIBILITY_COLUMNS
    assert tuple(bundle.dividends.columns) == DIVIDEND_EVENT_COLUMNS
    assert tuple(bundle.dividend_summary.columns) == DIVIDEND_SUMMARY_COLUMNS
    assert tuple(bundle.snapshot_features.columns) == SNAPSHOT_FEATURE_REQUIRED_COLUMNS
    assert (
        tuple(bundle.snapshot_holdings_diagnostics.columns)
        == SNAPSHOT_HOLDINGS_DIAGNOSTIC_COLUMNS
    )
    assert (
        tuple(bundle.snapshot_ratio_diagnostics.columns)
        == SNAPSHOT_RATIO_DIAGNOSTIC_COLUMNS
    )
    assert (
        tuple(bundle.snapshot_table_summary.columns) == SNAPSHOT_TABLE_SUMMARY_COLUMNS
    )
    assert tuple(bundle.risk_free_daily.columns) == RISK_FREE_DAILY_COLUMNS
    assert tuple(bundle.macro_features.columns) == MACRO_FEATURE_COLUMNS


def test_analysis_input_bundle_from_frames_keeps_snapshot_feature_extensions() -> None:
    prices = pd.DataFrame(
        [
            {
                "conid": "100",
                "trade_date": "2026-01-02",
                "price_value": 10.0,
                "clean_price": 10.0,
                "raw_return": 0.01,
                "clean_return": 0.01,
                "is_valid_price": True,
                "is_stale_price": False,
                "is_outlier_return": False,
                "is_price_level_anomaly": False,
                "is_clean_price": True,
            }
        ]
    )
    snapshot_features = pd.DataFrame(
        [
            {
                "conid": "100",
                "effective_at": "2026-01-02",
                "sleeve": "equity",
                "profile__total_expense_ratio": 0.15,
            }
        ]
    )

    bundle = AnalysisInputBundle.from_frames(
        prices=prices,
        snapshot_features=snapshot_features,
    )

    assert tuple(bundle.prices.columns) == PRICE_INPUT_COLUMNS
    assert bundle.snapshot_features.columns.tolist() == [
        "conid",
        "effective_at",
        "sleeve",
        "profile__total_expense_ratio",
    ]


def test_portfolio_input_bundle_empty_uses_explicit_contract_columns() -> None:
    bundle = PortfolioInputBundle.empty()

    assert tuple(bundle.expected_returns.columns) == EXPECTED_RETURN_COLUMNS
    assert tuple(bundle.covariance.columns) == COVARIANCE_COLUMNS
    assert tuple(bundle.exposures.columns) == EXPOSURE_COLUMNS
    assert tuple(bundle.eligibility.columns) == ELIGIBILITY_COLUMNS


def test_portfolio_input_bundle_from_frames_reindexes_to_contract_shape() -> None:
    bundle = PortfolioInputBundle.from_frames(
        expected_returns=pd.DataFrame(
            [
                {
                    "as_of_date": "2026-01-31",
                    "conid": "100",
                    "expected_return": 0.08,
                }
            ]
        ),
        covariance=pd.DataFrame(
            [
                {
                    "as_of_date": "2026-01-31",
                    "left_conid": "100",
                    "right_conid": "200",
                    "covariance": 0.02,
                }
            ]
        ),
    )

    assert tuple(bundle.expected_returns.columns) == EXPECTED_RETURN_COLUMNS
    assert tuple(bundle.covariance.columns) == COVARIANCE_COLUMNS
    assert tuple(bundle.exposures.columns) == EXPOSURE_COLUMNS
    assert bundle.exposures.empty
