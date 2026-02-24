import pytest

from pystocks.fundamentals_normalizers import (
    extract_dividends_events,
    extract_factor_features,
    extract_ownership_trade_log,
)


def test_ratios_factor_feature_extraction():
    payload = {
        "ratios": [
            {
                "name": "Price/Sales",
                "name_tag": "price_sales",
                "value": 3.63,
                "vs": 0.1,
                "min": 2.0,
                "max": 5.0,
                "avg": 3.2,
                "percentile": 40.0,
            }
        ]
    }
    rows = extract_factor_features("ratios", payload, effective_at="2026-02-20")
    names = {r["feature_name"] for r in rows}
    assert "fundamentals_price_sales" in names
    assert "fundamentals_price_sales_vs" in names
    assert "fundamentals_price_sales_percentile" in names

def test_holdings_weighted_features():
    payload = {
        "allocation_self": [
            {"name": "Equity", "weight": 99.5},
            {"name": "Cash", "weight": 0.5},
        ],
        "industry": [
            {"name": "Technology", "weight": 44.8},
        ],
        "currency": [
            {"name": "USD", "weight": "98.2%"},
        ],
        "investor_country": [
            {"name": "United States", "weight": 75.0},
        ],
        "top_10_weight": "38.39%",
    }
    rows = extract_factor_features("holdings", payload, effective_at="2026-02-20")
    by_name = {r["feature_name"]: r["feature_value"] for r in rows}

    assert by_name["holding_types_equity"] == pytest.approx(0.995, rel=0, abs=1e-6)
    assert by_name["industries_technology"] == pytest.approx(0.448, rel=0, abs=1e-6)
    assert by_name["currencies_usd"] == pytest.approx(0.982, rel=0, abs=1e-6)
    assert by_name["countries_united_states"] == pytest.approx(0.75, rel=0, abs=1e-6)
    assert by_name["holding_top_10_weight"] == pytest.approx(0.3839, rel=0, abs=1e-6)

def test_morningstar_ordinal_mapping():
    payload = {
        "summary": [
            {"id": "medalist_rating", "value": "Bronze"},
            {"id": "sustainability_rating", "value": "Above_Average"},
            {"id": "morningstar_rating", "value": "2"},
        ]
    }
    rows = extract_factor_features("morningstar", payload, effective_at="2026-02-20")
    by_name = {r["feature_name"]: r["feature_value"] for r in rows}
    assert by_name["morningstar_medalist_rating"] == 3.0
    assert by_name["morningstar_sustainability_rating"] == 4.0
    assert by_name["morningstar_morningstar_rating"] == 2.0

def test_ownership_trade_log_drops_no_change():
    payload = {
        "trade_log": [
            {"action": "NO CHANGE", "shares": 0, "displayDate": {"t": "2026-01-31"}},
            {"action": "SELL", "shares": -10, "value": -100.0, "displayDate": {"t": "2026-01-31"}},
        ]
    }
    rows = extract_ownership_trade_log(payload, drop_no_change=True)
    assert len(rows) == 1
    assert rows[0]["action"] == "SELL"

def test_dividends_events_ignore_price_series():
    payload = {
        "history": {
            "series": [
                {
                    "name": "price",
                    "plotData": [
                        {"x": 1704067200000, "y": 100.0},
                    ],
                },
                {
                    "name": "dividends",
                    "plotData": [
                        {
                            "x": 1704067200000,
                            "amount": 1.23,
                            "type": "ACTUAL",
                            "ex_dividend_date": {"d": 1, "m": "JAN", "y": 2024},
                            "formatted_amount": "1.23 USD",
                        }
                    ],
                },
            ]
        }
    }
    rows = extract_dividends_events(payload)
    assert len(rows) == 1
    assert rows[0]["amount"] == 1.23
    assert rows[0]["currency"] == "USD"
