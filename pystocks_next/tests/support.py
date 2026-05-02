from __future__ import annotations

import pandas as pd

from pystocks_next.storage.reads import SNAPSHOT_TABLE_COLUMNS


class RecordingProgressTracker:
    def __init__(
        self,
        events: list[tuple[str, str, object | None, object | None]],
        label: str,
    ) -> None:
        self._events = events
        self._label = label

    def advance(self, step: int = 1, *, detail: str | None = None) -> None:
        self._events.append(("advance", self._label, step, detail))

    def close(self, *, detail: str | None = None) -> None:
        self._events.append(("close", self._label, None, detail))


class RecordingProgressSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, object | None, object | None]] = []

    def stage(
        self,
        label: str,
        *,
        total: int | None = None,
        unit: str = "item",
    ) -> RecordingProgressTracker:
        self.events.append(("start", label, total, unit))
        return RecordingProgressTracker(self.events, label)


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
    tables["profile_fields"] = pd.DataFrame(
        [
            {
                "conid": "200",
                "effective_at": "2026-01-03",
                "field_id": "asset_type",
                "value_text": "Equity",
                "value_num": None,
                "value_date": None,
            }
        ]
    )
    tables["holdings_asset_type"] = pd.DataFrame(
        [
            {
                "conid": "200",
                "effective_at": "2026-01-03",
                "bucket_id": "equity",
                "value_num": "0.85",
                "vs_peers": "0.90",
            }
        ]
    )
    tables["holdings_debtor_quality"] = pd.DataFrame(
        [
            {
                "conid": "200",
                "effective_at": "2026-01-03",
                "bucket_id": "quality_aa",
                "value_num": "0.15",
                "vs_peers": None,
            }
        ]
    )
    tables["holdings_maturity"] = pd.DataFrame(
        [
            {
                "conid": "200",
                "effective_at": "2026-01-03",
                "bucket_id": "maturity_1_to_3_years",
                "value_num": "0.125",
                "vs_peers": None,
            }
        ]
    )
    tables["holdings_industry"] = pd.DataFrame(
        [
            {
                "conid": "200",
                "effective_at": "2026-01-03",
                "industry": "Technology",
                "value_num": "0.44",
                "vs_peers": "0.40",
            }
        ]
    )
    tables["ratios_key_ratios"] = pd.DataFrame(
        [
            {
                "conid": "200",
                "effective_at": "2026-01-03",
                "metric_id": "price_sales",
                "value_num": "3.63",
                "vs_peers": "0.0146",
            }
        ]
    )
    tables["dividends_industry_metrics"] = pd.DataFrame(
        [
            {
                "conid": "200",
                "effective_at": "2026-01-03",
                "metric_id": "dividend_yield",
                "value_num": "0.0122",
                "currency": "USD",
            }
        ]
    )
    tables["morningstar_summary"] = pd.DataFrame(
        [
            {
                "conid": "200",
                "effective_at": "2026-01-03",
                "metric_id": "medalist_rating",
                "title": "Medalist Rating",
                "derived_quantitatively": 0,
                "publish_date": "2026-01-03",
                "value_text": "Silver",
                "value_num": None,
            }
        ]
    )
    tables["lipper_ratings"] = pd.DataFrame(
        [
            {
                "conid": "200",
                "effective_at": "2026-01-03",
                "period": "overall",
                "metric_id": "total_return",
                "value_num": "5",
                "rating_label": "236 funds",
                "universe_name": "Sweden",
                "universe_as_of_date": "2026-01-30",
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


def build_sample_profile_and_fees_payload() -> dict[str, object]:
    return {
        "objective": "Broad equity exposure",
        "jap_fund_warning": False,
        "themes": [{"theme_name": "Core"}],
        "fund_and_profile": [
            {"name": "Asset Type", "value": "Equity"},
            {"name": "Classification", "value": "ETF"},
            {"name": "Distribution Details", "value": "Distributing"},
            {"name": "Domicile", "value": "US"},
            {"name": "Fund Category", "value": "Large Blend"},
            {"name": "Fund Management Company", "value": "Alpha"},
            {"name": "Fund Manager Benchmark", "value": "SP500"},
            {"name": "Fund Market Cap Focus", "value": "Large"},
            {"name": "Geographical Focus", "value": "US"},
            {"name": "Inception Date", "value": "2020-01-01"},
            {"name": "Management Approach", "value": "Passive"},
            {"name": "Management Expenses", "value": "0.12%"},
            {"name": "Manager Tenure", "value": "2021-01-01"},
            {"name": "Objective Type", "value": "Growth"},
            {"name": "Portfolio Manager", "value": "Jane Doe"},
            {"name": "Redemption Charge Actual", "value": "0%"},
            {"name": "Redemption Charge Max", "value": "0%"},
            {"name": "Scheme", "value": "Open End"},
            {"name": "Total Expense Ratio", "value": "0.15%"},
            {"name": "Total Net Assets (Month End)", "value": "$1.2B (2026-01-02)"},
        ],
    }


def build_sample_holdings_payload() -> dict[str, object]:
    return {
        "as_of_date": "2026-01-03",
        "allocation_self": [
            {"name": "Equity", "weight": "85%", "vs": 90.0},
            {"name": "Cash", "assets_pct": "10%"},
            {"name": "Fixed Income", "formatted_weight": "5%"},
        ],
        "industry": [
            {"name": "Technology", "weight": 44.8681, "vs": 40.0},
        ],
        "currency": [
            {"name": "US Dollar", "weight": 99.9604, "code": "USD", "vs": 98.5},
        ],
        "investor_country": [
            {
                "name": "United States",
                "weight": 97.3418,
                "country_code": "US",
                "vs": 96.0,
            },
        ],
        "debt_type": [
            {"name": "Sovereign Bond", "weight": "20%", "vs": "25%"},
        ],
        "debtor": [
            {"name": "% Quality/AA", "weight": "15%"},
            {"name": "% Quality/BBB", "weight": "8%"},
            {"name": "% Quality Not Rated", "weight": "2%"},
        ],
        "maturity": [
            {"name": "% Maturity 1 to 3 Years", "weight": "12.5%"},
            {"name": "% Maturity Less than 1 Year", "weight": "5.4%"},
        ],
        "geographic": {
            "us": "97.34%",
            "eu": "1.89%",
        },
        "top_10": [
            {"name": "NVIDIA CORPORATION", "assets_pct": "7.83%"},
        ],
    }


def build_sample_ratios_payload() -> dict[str, object]:
    return {
        "as_of_date": "2026-01-03",
        "ratios": [
            {
                "name": "Price/Sales",
                "name_tag": "price_sales",
                "value": 3.63,
                "vs": 0.0146,
            }
        ],
        "financials": [
            {
                "name": "Sales Growth 1 Year",
                "name_tag": "sales_growth_1_year",
                "value": 5.04,
                "vs": -0.15,
            }
        ],
        "fixed_income": [
            {
                "name": "Current Yield",
                "name_tag": "current_yield",
                "value": "3.17%",
                "vs": "0.05%",
            }
        ],
        "dividend": [
            {
                "name": "Dividend Yield",
                "name_tag": "dividend_yield",
                "value": 2.35,
                "vs": -0.08,
            }
        ],
        "zscore": [
            {
                "name": "1 Month",
                "name_tag": "1_month",
                "value": -0.04,
            }
        ],
    }


def build_sample_dividends_snapshot_payload() -> dict[str, object]:
    return {
        "as_of_date": "2026-01-03",
        "industry_average": {
            "dividend_yield": "1.22%",
            "annual_dividend": "25.65",
        },
        "industry_comparison": {
            "content": [
                {
                    "search_id": "div_yield",
                    "value": 0.0085,
                },
                {
                    "search_id": "div_per_share",
                    "value": 5.585599,
                },
            ]
        },
        "last_payed_dividend_currency": "USD",
    }


def build_sample_morningstar_payload() -> dict[str, object]:
    return {
        "as_of_date": "20260131",
        "q_full_report_id": "report_123",
        "summary": [
            {"id": "medalist_rating", "value": "Silver", "publish_date": "20260128"},
            {
                "id": "q_process",
                "title": "Process",
                "value": "Average",
                "q": True,
                "publish_date": "20260128",
            },
            {
                "id": "q_people",
                "title": "People",
                "value": "Average",
                "q": True,
                "publish_date": "20260128",
            },
            {
                "id": "q_parent",
                "title": "Parent",
                "value": "Average",
                "q": True,
                "publish_date": "20260128",
            },
            {"id": "morningstar_rating", "value": "4", "publish_date": "20260131"},
            {
                "id": "sustainability_rating",
                "value": "Average",
                "publish_date": "20251231",
            },
            {"id": "category", "value": "Large Blend"},
            {
                "id": "category_index",
                "value": "Morningstar US Large-Mid TR USD",
            },
        ],
    }


def build_sample_lipper_payload() -> dict[str, object]:
    return {
        "universes": [
            {
                "name": "Sweden",
                "as_of_date": 1769749200000,
                "overall": [
                    {
                        "name": "Total Return",
                        "name_tag": "total_return",
                        "rating": {"name": "236 funds", "value": 5},
                    }
                ],
                "3_year": [
                    {
                        "name": "Total Return",
                        "name_tag": "total_return",
                        "rating": {"name": "236 funds", "value": 4},
                    }
                ],
            }
        ],
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
