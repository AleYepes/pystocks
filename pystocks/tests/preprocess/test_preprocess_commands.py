import pandas as pd

import pystocks.analysis as analysis_module
import pystocks.preprocess.dividends as dividends_module
import pystocks.preprocess.price as price_module
import pystocks.preprocess.snapshots as snapshots_module
from pystocks.analysis import build_analysis_panel
from pystocks.preprocess.dividends import run_dividend_preprocess
from pystocks.preprocess.price import run_price_preprocess
from pystocks.preprocess.snapshots import run_snapshot_preprocess


def test_run_price_preprocess_writes_parquet_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        price_module,
        "load_price_history",
        lambda sqlite_path: pd.DataFrame(
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
            ]
        ),
    )

    result = run_price_preprocess(output_dir=tmp_path)

    assert result["status"] == "ok"
    assert (tmp_path / "analysis_daily_returns.parquet").exists()
    assert (tmp_path / "analysis_price_eligibility.parquet").exists()


def test_run_snapshot_preprocess_writes_parquet_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        snapshots_module,
        "load_snapshot_feature_tables",
        lambda sqlite_path: snapshots_module._normalize_snapshot_tables(
            {
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
                            "effective_at": "2026-01-31",
                            "equity": 1.0,
                            "fixed_income": 0.0,
                        }
                    ]
                ),
                "ratios_key_ratios": pd.DataFrame(
                    [
                        {
                            "conid": "a",
                            "effective_at": "2026-01-31",
                            "metric_id": "price_book",
                            "value_num": 1.5,
                        }
                    ]
                ),
            }
        ),
    )

    result = run_snapshot_preprocess(output_dir=tmp_path)

    assert result["status"] == "ok"
    assert (tmp_path / "analysis_snapshot_features.parquet").exists()
    assert (tmp_path / "analysis_snapshot_holdings_diagnostics.parquet").exists()
    assert (tmp_path / "analysis_snapshot_ratio_diagnostics.parquet").exists()
    assert (tmp_path / "analysis_snapshot_table_summary.parquet").exists()


def test_run_dividend_preprocess_writes_parquet_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dividends_module,
        "load_dividend_events",
        lambda sqlite_path: dividends_module._normalize_dividend_frame(
            pd.DataFrame(
                [
                    {
                        "conid": "x",
                        "symbol": "X",
                        "event_date": pd.Timestamp("2024-01-03"),
                        "amount": 1.5,
                        "dividend_currency": "USD",
                        "product_currency": "USD",
                        "description": "Regular Dividend",
                        "event_type": "ACTUAL",
                    }
                ]
            )
        ),
    )
    monkeypatch.setattr(
        dividends_module,
        "load_price_history",
        lambda sqlite_path: pd.DataFrame(
            [
                {
                    "conid": "x",
                    "trade_date": "2024-01-01",
                    "price": 99.0,
                    "open": 99.0,
                    "high": 99.0,
                    "low": 99.0,
                    "close": 99.0,
                },
                {
                    "conid": "x",
                    "trade_date": "2024-01-02",
                    "price": 100.0,
                    "open": 100.0,
                    "high": 100.0,
                    "low": 100.0,
                    "close": 100.0,
                },
            ]
        ),
    )

    result = run_dividend_preprocess(output_dir=tmp_path)

    assert result["status"] == "ok"
    assert (tmp_path / "analysis_dividend_events.parquet").exists()
    assert (tmp_path / "analysis_dividend_summary.parquet").exists()


def test_build_analysis_panel_writes_parquet_output(tmp_path, monkeypatch):
    monkeypatch.setattr(
        analysis_module,
        "load_saved_price_preprocess_results",
        lambda output_dir=None: {
            "prices": pd.DataFrame(
                [
                    {
                        "conid": "a",
                        "trade_date": pd.Timestamp("2026-01-31"),
                        "close": 10.0,
                    }
                ]
            ),
            "eligibility": pd.DataFrame([{"conid": "a", "eligible": True}]),
        },
    )
    monkeypatch.setattr(
        analysis_module,
        "load_snapshot_features",
        lambda output_dir=None: pd.DataFrame(
            [{"conid": "a", "effective_at": pd.Timestamp("2026-01-31")}]
        ),
    )
    monkeypatch.setattr(
        analysis_module,
        "build_analysis_panel_data",
        lambda snapshot_features, price_result, config, world_bank_country_features=None, show_progress=False: (
            pd.DataFrame([{"conid": "a", "rebalance_date": pd.Timestamp("2026-01-31")}])
        ),
    )
    monkeypatch.setattr(
        analysis_module,
        "load_risk_free_daily",
        lambda output_dir=None: pd.DataFrame(
            {"trade_date": [pd.Timestamp("2026-01-31")], "daily_nominal_rate": [0.0]}
        ),
    )
    monkeypatch.setattr(
        analysis_module,
        "load_world_bank_country_features",
        lambda output_dir=None: pd.DataFrame(
            [
                {
                    "economy_code": "USA",
                    "effective_at": pd.Timestamp("2025-12-31"),
                    "population_level": 1.0,
                    "population_growth": 0.0,
                    "gdp_pcap_level": 1.0,
                    "gdp_pcap_growth": 0.0,
                    "economic_output_gdp_level": 1.0,
                    "economic_output_gdp_growth": 0.0,
                    "foreign_direct_investment_level": 1.0,
                    "foreign_direct_investment_growth": 0.0,
                    "share_trade_volume_level": 1.0,
                    "share_trade_volume_growth": 0.0,
                    "observed_at": pd.Timestamp("2026-01-01"),
                }
            ]
        ),
    )

    result = build_analysis_panel(
        sqlite_path=tmp_path / "analysis.sqlite",
        output_dir=tmp_path,
    )

    assert result["status"] == "ok"
    assert (tmp_path / "analysis_snapshot_panel.parquet").exists()
