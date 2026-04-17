from __future__ import annotations

import pandas as pd

from pystocks_next.storage.reads import SNAPSHOT_TABLE_COLUMNS


def build_sample_raw_payload(
    *, conid: str = "123", as_of_date: str = "2026-01-02"
) -> dict[str, object]:
    return {
        "conid": conid,
        "as_of_date": as_of_date,
        "rows": [
            {"name": "equity", "value": 0.62},
            {"name": "fixed_income", "value": 0.28},
            {"name": "cash", "value": 0.10},
        ],
    }


def build_sample_price_history_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "conid": "200",
                "trade_date": "2026-01-03",
                "price": "11.2",
                "open": "11.0",
                "high": "11.4",
                "low": "10.8",
                "close": "11.3",
            },
            {
                "conid": "100",
                "trade_date": "2026-01-02",
                "price": "10.0",
                "open": "9.8",
                "high": "10.2",
                "low": "9.7",
                "close": "10.1",
            },
        ]
    )


def build_sample_dividend_events_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "conid": "200",
                "symbol": "BBB",
                "event_date": "2026-01-06",
                "amount": "0.22",
                "dividend_currency": "USD",
                "product_currency": "USD",
                "description": "Quarterly",
                "event_type": "cash",
                "declaration_date": "2025-12-20",
                "record_date": "2026-01-07",
                "payment_date": "2026-01-20",
            },
            {
                "conid": "100",
                "symbol": "AAA",
                "event_date": "2026-01-03",
                "amount": "0.11",
                "dividend_currency": "USD",
                "product_currency": "USD",
                "description": "Monthly",
                "event_type": "cash",
                "declaration_date": "2025-12-10",
                "record_date": "2026-01-04",
                "payment_date": "2026-01-12",
            },
        ]
    )


def build_sample_snapshot_tables() -> dict[str, pd.DataFrame]:
    tables = {
        name: pd.DataFrame(columns=pd.Index(columns))
        for name, columns in SNAPSHOT_TABLE_COLUMNS.items()
    }
    tables["profile_and_fees"] = pd.DataFrame(
        [
            {
                "conid": "200",
                "effective_at": "2026-01-03",
                "asset_type": "Equity",
                "classification": "ETF",
                "distribution_details": "Distributing",
                "domicile": "US",
                "fiscal_date": "2025-12-31",
                "fund_category": "Large Blend",
                "fund_management_company": "Alpha",
                "fund_manager_benchmark": "SP500",
                "fund_market_cap_focus": "Large",
                "geographical_focus": "US",
                "inception_date": "2020-01-01",
                "management_approach": "Passive",
                "management_expenses": "0.12",
                "manager_tenure": "5",
                "maturity_date": None,
                "objective_type": "Growth",
                "portfolio_manager": "Jane Doe",
                "redemption_charge_actual": "0",
                "redemption_charge_max": "0",
                "scheme": "Open End",
                "total_expense_ratio": "0.15",
                "total_net_assets_value": "1.2B",
                "total_net_assets_date": "2026-01-02",
                "objective": "Broad equity exposure",
                "jap_fund_warning": None,
                "theme_name": "Core",
            }
        ]
    )
    tables["holdings_asset_type"] = pd.DataFrame(
        [
            {
                "conid": "200",
                "effective_at": "2026-01-03",
                "equity": "0.85",
                "cash": "0.10",
                "fixed_income": "0.05",
                "other": "0.0",
            }
        ]
    )
    return tables


def build_sample_price_chart_payload() -> dict[str, object]:
    return {
        "history": {
            "series": [
                {
                    "name": "Price",
                    "plotData": [
                        {
                            "x": "20251231",
                            "debugY": "2025-12-31",
                            "y": "10.1",
                            "open": "9.9",
                            "high": "10.3",
                            "low": "9.8",
                            "close": "10.2",
                        },
                        {
                            "x": None,
                            "debugY": "2026-01-02",
                            "y": "10.4",
                            "open": "10.2",
                            "high": "10.5",
                            "low": "10.1",
                            "close": "10.3",
                        },
                    ],
                }
            ]
        }
    }


def build_sample_dividends_payload() -> dict[str, object]:
    return {
        "last_payed_dividend_currency": "USD",
        "history": {
            "series": [
                {
                    "name": "Dividend History",
                    "plotData": [
                        {
                            "x": "2026-01-03",
                            "ex_dividend_date": "2026-01-03",
                            "amount": "0.11",
                            "formatted_amount": "USD 0.11",
                            "description": "Monthly",
                            "type": "cash",
                            "declaration_date": "2025-12-10",
                            "record_date": "2026-01-04",
                            "payment_date": "2026-01-12",
                        },
                        {
                            "x": None,
                            "ex_dividend_date": "2026-01-10",
                            "y": "0.12",
                            "description": "Monthly",
                            "type": "cash",
                            "declaration_date": "2025-12-20",
                            "record_date": "2026-01-11",
                            "payment_date": "2026-01-19",
                        },
                    ],
                }
            ]
        },
    }


def build_sample_risk_free_sources_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "series_id": "DTB3",
                "source_name": "fred",
                "economy_code": "usa",
                "trade_date": "2026-01-02",
                "nominal_rate": 0.03,
            },
            {
                "series_id": "IR3TIB01CAM156N",
                "source_name": "fred",
                "economy_code": "can",
                "trade_date": "2026-01-02",
                "nominal_rate": 0.01,
            },
        ]
    )


def build_sample_risk_free_daily_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_date": "2026-01-02",
                "nominal_rate": 0.022,
                "daily_nominal_rate": 0.022 / 252.0,
                "source_count": 2,
                "observed_at": "2026-01-05T10:00:00+00:00",
            }
        ]
    )


def build_sample_world_bank_raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "economy_code": "usa",
                "indicator_id": "SP.POP.TOTL",
                "year": 2025,
                "value": 100.0,
            },
            {
                "economy_code": "usa",
                "indicator_id": "NY.GDP.PCAP.CD",
                "year": 2025,
                "value": 10.0,
            },
        ]
    )


def build_sample_world_bank_country_features_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "economy_code": "usa",
                "effective_at": "2025-12-31",
                "feature_year": 2025,
                "population_level": 100.0,
                "population_growth": 10.0,
                "population_acceleration": 1.0,
                "gdp_pcap_level": 10.0,
                "gdp_pcap_growth": 1.0,
                "gdp_pcap_acceleration": 0.1,
                "economic_output_gdp_level": 1000.0,
                "economic_output_gdp_growth": 5.0,
                "economic_output_gdp_acceleration": 0.5,
                "foreign_direct_investment_level": 1.0,
                "foreign_direct_investment_growth": 0.1,
                "foreign_direct_investment_acceleration": 0.01,
                "share_trade_volume_level": 0.5,
                "share_trade_volume_growth": 0.05,
                "share_trade_volume_acceleration": 0.005,
                "observed_at": "2026-01-05T10:00:00+00:00",
            }
        ]
    )
