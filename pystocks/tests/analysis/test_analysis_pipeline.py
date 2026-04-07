import sqlite3

import numpy as np
import pandas as pd

import pystocks.analysis as analysis_module
from pystocks.analysis import (
    AnalysisConfig,
    _build_research_windows,
    build_analysis_panel,
    build_analysis_panel_data,
    cluster_factor_returns,
    run_analysis_pipeline,
    run_factor_research,
    run_factor_research_data,
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
    config = AnalysisConfig(
        rebalance_freq="M",
        include_macro_features=False,
        require_supplementary_data=False,
    )

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


def test_build_analysis_panel_adds_macro_features_from_country_weights():
    snapshot_features = pd.DataFrame(
        [
            {
                "conid": "a",
                "effective_at": pd.Timestamp("2026-01-31"),
                "profile__asset_type": "Equity",
                "profile__total_net_assets_num": 100.0,
                "country__usa": 0.6,
                "country__can": 0.4,
            }
        ]
    )
    price_result = {
        "prices": pd.DataFrame(
            [
                {
                    "conid": "a",
                    "trade_date": pd.Timestamp("2026-01-31"),
                    "clean_price": 10.0,
                    "clean_return": 0.01,
                    "is_clean_price": True,
                }
            ]
        ),
        "eligibility": pd.DataFrame([{"conid": "a", "eligible": True}]),
    }
    world_bank_country_features = pd.DataFrame(
        [
            {
                "economy_code": "USA",
                "effective_at": pd.Timestamp("2025-12-31"),
                "population_level": 10.0,
                "population_growth": 1.0,
                "gdp_pcap_level": 30.0,
                "gdp_pcap_growth": 3.0,
                "economic_output_gdp_level": 0.7,
                "economic_output_gdp_growth": 0.1,
                "foreign_direct_investment_level": 0.2,
                "foreign_direct_investment_growth": 0.02,
                "share_trade_volume_level": 0.5,
                "share_trade_volume_growth": 0.05,
                "observed_at": pd.Timestamp("2026-01-05"),
            },
            {
                "economy_code": "CAN",
                "effective_at": pd.Timestamp("2025-12-31"),
                "population_level": 20.0,
                "population_growth": 2.0,
                "gdp_pcap_level": 40.0,
                "gdp_pcap_growth": 4.0,
                "economic_output_gdp_level": 0.3,
                "economic_output_gdp_growth": 0.2,
                "foreign_direct_investment_level": 0.1,
                "foreign_direct_investment_growth": 0.01,
                "share_trade_volume_level": 0.25,
                "share_trade_volume_growth": 0.03,
                "observed_at": pd.Timestamp("2026-01-05"),
            },
        ]
    )
    config = AnalysisConfig(rebalance_freq="M", require_supplementary_data=True)

    panel = build_analysis_panel_data(
        snapshot_features,
        price_result,
        config,
        world_bank_country_features=world_bank_country_features,
    )

    row = panel.iloc[0]
    assert row["macro__population_level"] == 14.0
    assert row["macro__gdp_pcap_growth"] == 3.4


def test_build_analysis_panel_requires_macro_features_when_enabled():
    with np.testing.assert_raises_regex(RuntimeError, "World Bank"):
        build_analysis_panel_data(
            pd.DataFrame(
                [
                    {
                        "conid": "a",
                        "effective_at": pd.Timestamp("2026-01-31"),
                    }
                ]
            ),
            {
                "prices": pd.DataFrame(
                    [
                        {
                            "conid": "a",
                            "trade_date": pd.Timestamp("2026-01-31"),
                            "clean_price": 10.0,
                            "clean_return": 0.0,
                            "is_clean_price": True,
                        }
                    ]
                ),
                "eligibility": pd.DataFrame([{"conid": "a", "eligible": True}]),
            },
            AnalysisConfig(require_supplementary_data=True),
            world_bank_country_features=pd.DataFrame(),
        )


def test_build_analysis_panel_keeps_progress_bars_by_default(monkeypatch):
    leaves: list[bool] = []

    def fake_track_progress(
        iterable,
        *,
        show_progress=False,
        total=None,
        desc=None,
        unit=None,
        leave=False,
    ):
        leaves.append(bool(leave))
        return iterable

    monkeypatch.setattr(analysis_module, "track_progress", fake_track_progress)

    snapshot_features = pd.DataFrame(
        [
            {
                "conid": "a",
                "effective_at": pd.Timestamp("2026-01-31"),
                "profile__asset_type": "Equity",
                "profile__total_net_assets_num": 100.0,
            }
        ]
    )
    price_result = {
        "prices": pd.DataFrame(
            [
                {
                    "conid": "a",
                    "trade_date": pd.Timestamp("2026-01-31"),
                    "clean_price": 10.0,
                    "clean_return": 0.01,
                    "is_clean_price": True,
                }
            ]
        ),
        "eligibility": pd.DataFrame([{"conid": "a", "eligible": True}]),
    }

    panel = build_analysis_panel_data(
        snapshot_features,
        price_result,
        AnalysisConfig(
            include_macro_features=False,
            require_supplementary_data=False,
        ),
        show_progress=True,
    )

    assert not panel.empty
    assert leaves
    assert all(leaves)


def test_build_analysis_panel_can_disable_persistent_progress_bars(monkeypatch):
    leaves: list[bool] = []

    def fake_track_progress(
        iterable,
        *,
        show_progress=False,
        total=None,
        desc=None,
        unit=None,
        leave=False,
    ):
        leaves.append(bool(leave))
        return iterable

    monkeypatch.setattr(analysis_module, "track_progress", fake_track_progress)

    snapshot_features = pd.DataFrame(
        [
            {
                "conid": "a",
                "effective_at": pd.Timestamp("2026-01-31"),
                "profile__asset_type": "Equity",
                "profile__total_net_assets_num": 100.0,
            }
        ]
    )
    price_result = {
        "prices": pd.DataFrame(
            [
                {
                    "conid": "a",
                    "trade_date": pd.Timestamp("2026-01-31"),
                    "clean_price": 10.0,
                    "clean_return": 0.01,
                    "is_clean_price": True,
                }
            ]
        ),
        "eligibility": pd.DataFrame([{"conid": "a", "eligible": True}]),
    }

    panel = build_analysis_panel_data(
        snapshot_features,
        price_result,
        AnalysisConfig(
            include_macro_features=False,
            require_supplementary_data=False,
            persist_progress_bars=False,
        ),
        show_progress=True,
    )

    assert not panel.empty
    assert leaves
    assert not any(leaves)


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


def test_cluster_factor_returns_returns_empty_frames_when_history_is_too_short():
    index = pd.date_range("2026-01-01", periods=2, freq="B")
    factor_returns = pd.DataFrame(
        {
            "equity__raw__ratio_key__price_book": pd.Series(
                [0.01, 0.02],
                index=index,
                dtype=float,
            )
        }
    )
    factor_meta = pd.DataFrame(
        [
            {
                "factor_id": "equity__raw__ratio_key__price_book",
                "sleeve": "equity",
                "family": "ratio_key",
                "kind": "raw",
            }
        ]
    )

    cluster_df, reduced = cluster_factor_returns(
        factor_returns,
        factor_meta,
        AnalysisConfig(min_train_days=50),
    )

    assert cluster_df.empty
    assert list(cluster_df.columns) == [
        "factor_id",
        "sleeve",
        "cluster_id",
        "cluster_representative",
        "cluster_size",
        "keep_factor",
    ]
    assert reduced.empty


def test_build_research_windows_uses_fixed_training_windows():
    panel = pd.DataFrame(
        [
            {"rebalance_date": pd.Timestamp("2023-01-31")},
            {"rebalance_date": pd.Timestamp("2024-01-31")},
            {"rebalance_date": pd.Timestamp("2025-01-31")},
            {"rebalance_date": pd.Timestamp("2026-01-31")},
        ]
    )
    reduced_factors = pd.DataFrame(
        {"equity__benchmark__market_excess": [0.01, 0.02, 0.03]},
        index=pd.to_datetime(["2023-02-01", "2024-02-01", "2025-02-01"]),
    )
    config = AnalysisConfig(training_window_years=(3, 4), walk_forward_step_months=12)

    windows = _build_research_windows(panel, reduced_factors, config)

    assert windows == [
        (
            pd.Timestamp("2020-01-31"),
            pd.Timestamp("2023-01-31"),
            pd.Timestamp("2024-01-31"),
            3,
        ),
        (
            pd.Timestamp("2019-01-31"),
            pd.Timestamp("2023-01-31"),
            pd.Timestamp("2024-01-31"),
            4,
        ),
        (
            pd.Timestamp("2021-01-31"),
            pd.Timestamp("2024-01-31"),
            pd.Timestamp("2025-01-31"),
            3,
        ),
        (
            pd.Timestamp("2020-01-31"),
            pd.Timestamp("2024-01-31"),
            pd.Timestamp("2025-01-31"),
            4,
        ),
        (
            pd.Timestamp("2022-01-31"),
            pd.Timestamp("2025-01-31"),
            pd.Timestamp("2026-01-31"),
            3,
        ),
        (
            pd.Timestamp("2021-01-31"),
            pd.Timestamp("2025-01-31"),
            pd.Timestamp("2026-01-31"),
            4,
        ),
    ]


def _stub_price_result():
    return {
        "prices": pd.DataFrame(
            [{"conid": "a", "trade_date": pd.Timestamp("2026-01-31"), "close": 10.0}]
        ),
        "eligibility": pd.DataFrame([{"conid": "a", "eligible": True}]),
    }


def test_build_analysis_panel_preprocesses_inputs_once(tmp_path, monkeypatch):
    counts = {"prices": 0, "snapshots": 0, "risk_free": 0, "macro": 0}

    monkeypatch.setattr(
        analysis_module, "load_price_history", lambda sqlite_path: pd.DataFrame()
    )

    def fake_preprocess_price_history(price_df, config=None, show_progress=False):
        counts["prices"] += 1
        return _stub_price_result()

    def fake_load_snapshot_features(sqlite_path):
        counts["snapshots"] += 1
        return pd.DataFrame(
            [{"conid": "a", "effective_at": pd.Timestamp("2026-01-31")}]
        )

    monkeypatch.setattr(
        analysis_module, "preprocess_price_history", fake_preprocess_price_history
    )
    monkeypatch.setattr(
        analysis_module,
        "save_price_preprocess_results",
        lambda result, output_dir=None: {"prices_path": "prices.parquet"},
    )
    monkeypatch.setattr(
        analysis_module, "load_snapshot_features", fake_load_snapshot_features
    )
    monkeypatch.setattr(
        analysis_module,
        "load_risk_free_daily",
        lambda sqlite_path: (
            counts.__setitem__("risk_free", counts["risk_free"] + 1)
            or pd.DataFrame(
                {
                    "trade_date": [pd.Timestamp("2026-01-31")],
                    "daily_nominal_rate": [0.0],
                }
            )
        ),
    )
    monkeypatch.setattr(
        analysis_module,
        "load_world_bank_country_features",
        lambda sqlite_path: (
            counts.__setitem__("macro", counts["macro"] + 1)
            or pd.DataFrame(
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
            )
        ),
    )
    monkeypatch.setattr(
        analysis_module,
        "build_analysis_panel_data",
        lambda snapshot_features, price_result, config, world_bank_country_features=None, show_progress=False: (
            pd.DataFrame([{"conid": "a", "rebalance_date": pd.Timestamp("2026-01-31")}])
        ),
    )

    result = build_analysis_panel(
        sqlite_path=tmp_path / "analysis.sqlite",
        output_dir=tmp_path,
    )

    assert counts == {"prices": 1, "snapshots": 1, "risk_free": 1, "macro": 1}
    assert result["rows"] == 1
    assert (tmp_path / "analysis_snapshot_panel.parquet").exists()


def test_run_factor_research_preprocesses_inputs_once(tmp_path, monkeypatch):
    counts = {"prices": 0, "snapshots": 0, "risk_free": 0, "macro": 0}

    monkeypatch.setattr(
        analysis_module, "load_price_history", lambda sqlite_path: pd.DataFrame()
    )

    def fake_preprocess_price_history(price_df, config=None, show_progress=False):
        counts["prices"] += 1
        return _stub_price_result()

    def fake_load_snapshot_features(sqlite_path):
        counts["snapshots"] += 1
        return pd.DataFrame(
            [{"conid": "a", "effective_at": pd.Timestamp("2026-01-31")}]
        )

    monkeypatch.setattr(
        analysis_module, "preprocess_price_history", fake_preprocess_price_history
    )
    monkeypatch.setattr(
        analysis_module,
        "save_price_preprocess_results",
        lambda result, output_dir=None: {"prices_path": "prices.parquet"},
    )
    monkeypatch.setattr(
        analysis_module, "load_snapshot_features", fake_load_snapshot_features
    )
    monkeypatch.setattr(
        analysis_module,
        "load_risk_free_daily",
        lambda sqlite_path: (
            counts.__setitem__("risk_free", counts["risk_free"] + 1)
            or pd.DataFrame(
                {
                    "trade_date": [pd.Timestamp("2026-01-31")],
                    "daily_nominal_rate": [0.0],
                }
            )
        ),
    )
    monkeypatch.setattr(
        analysis_module,
        "load_world_bank_country_features",
        lambda sqlite_path: (
            counts.__setitem__("macro", counts["macro"] + 1)
            or pd.DataFrame(
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
            )
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
        "run_factor_research_data",
        lambda panel, prices, risk_free_daily, config, show_progress=False: {
            "factor_returns": pd.DataFrame(),
            "factor_meta": pd.DataFrame(),
            "factor_clusters": pd.DataFrame(),
            "factor_diagnostics": pd.DataFrame(),
            "baseline_returns": pd.DataFrame(),
            "baseline_members": pd.DataFrame(),
            "model_results": pd.DataFrame(),
            "factor_persistence": pd.DataFrame(),
            "current_betas": pd.DataFrame(),
            "asset_expected_returns": pd.DataFrame(),
            "asset_factor_betas": pd.DataFrame(),
        },
    )
    monkeypatch.setattr(
        analysis_module,
        "_write_output",
        lambda name, df, output_dir, sqlite_path, long_sql_df=None, tx=None: str(
            tmp_path / f"{name}.parquet"
        ),
    )

    result = run_factor_research(
        sqlite_path=tmp_path / "analysis.sqlite",
        output_dir=tmp_path,
    )

    assert counts == {"prices": 1, "snapshots": 1, "risk_free": 1, "macro": 1}
    assert result["snapshot_rows"] == 1


def test_write_output_skips_zero_column_sql_frames(tmp_path):
    sqlite_path = tmp_path / "analysis.sqlite"

    analysis_module._write_output(
        "analysis_zero_col_output",
        pd.DataFrame(),
        tmp_path,
        sqlite_path,
    )

    assert (tmp_path / "analysis_zero_col_output.parquet").exists()

    with sqlite3.connect(sqlite_path) as conn:
        table_exists = conn.execute(
            """
            SELECT EXISTS(
                SELECT 1
                FROM sqlite_master
                WHERE type = 'table' AND name = ?
            )
            """,
            ["analysis_zero_col_output"],
        ).fetchone()[0]

    assert table_exists == 0


def test_run_factor_research_data_uses_risk_free_excess(monkeypatch):
    panel = pd.DataFrame(
        [
            {
                "conid": "a",
                "sleeve": "equity",
                "rebalance_date": pd.Timestamp("2026-01-31"),
            },
            {
                "conid": "a",
                "sleeve": "equity",
                "rebalance_date": pd.Timestamp("2026-02-28"),
            },
            {
                "conid": "a",
                "sleeve": "equity",
                "rebalance_date": pd.Timestamp("2026-03-31"),
            },
        ]
    )
    prices = pd.DataFrame(
        [
            {
                "conid": "a",
                "trade_date": pd.Timestamp("2026-02-10"),
                "clean_return": 0.02,
                "is_clean_price": True,
            },
            {
                "conid": "a",
                "trade_date": pd.Timestamp("2026-03-10"),
                "clean_return": 0.03,
                "is_clean_price": True,
            },
        ]
    )
    risk_free_daily = pd.DataFrame(
        [
            {"trade_date": pd.Timestamp("2026-02-10"), "daily_nominal_rate": 0.01},
            {"trade_date": pd.Timestamp("2026-03-10"), "daily_nominal_rate": 0.01},
        ]
    )
    factor_returns = pd.DataFrame(
        {"equity__benchmark__market_excess": [0.1, 0.2]},
        index=pd.to_datetime(["2026-02-10", "2026-03-10"]),
    )
    factor_meta = pd.DataFrame(
        [
            {
                "factor_id": "equity__benchmark__market_excess",
                "sleeve": "equity",
                "family": "market_excess",
                "kind": "benchmark",
            }
        ]
    )
    captured = {}

    monkeypatch.setattr(
        analysis_module,
        "build_factor_returns",
        lambda *args, **kwargs: (
            factor_returns,
            factor_meta,
            pd.Series(0.0, index=factor_returns.index),
            pd.DataFrame(),
        ),
    )
    monkeypatch.setattr(
        analysis_module,
        "cluster_factor_returns",
        lambda factor_returns, factor_meta, config, show_progress=False: (
            pd.DataFrame(
                [
                    {
                        "factor_id": "equity__benchmark__market_excess",
                        "sleeve": "equity",
                        "cluster_id": "equity_1",
                        "cluster_representative": "equity__benchmark__market_excess",
                        "cluster_size": 1,
                        "keep_factor": True,
                    }
                ]
            ),
            factor_returns,
        ),
    )
    monkeypatch.setattr(
        analysis_module,
        "_build_research_windows",
        lambda panel, reduced_factors, config: [
            (
                pd.Timestamp("2025-02-28"),
                pd.Timestamp("2026-02-28"),
                pd.Timestamp("2026-03-31"),
                1,
            )
        ],
    )
    monkeypatch.setattr(
        analysis_module,
        "compute_current_betas_data",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    def fake_fit(X_train, y_train):
        captured["y_train"] = y_train.copy()

        class _Model:
            alpha_ = 0.1
            l1_ratio_ = 0.5
            n_iter_ = 10
            dual_gap_ = 0.0
            mse_path_ = np.array([[1.0, 2.0]])

        class _Pipeline:
            def predict(self, X):
                return np.zeros(len(X))

        return _Pipeline(), _Model(), 0.0, np.array([0.25])

    monkeypatch.setattr(analysis_module, "_fit_elastic_net", fake_fit)

    result = run_factor_research_data(
        panel,
        prices,
        risk_free_daily,
        AnalysisConfig(
            min_train_days=1,
            min_test_days=1,
            training_window_years=(1,),
            walk_forward_step_months=1,
            include_macro_features=False,
            require_supplementary_data=False,
        ),
    )

    assert np.allclose(captured["y_train"], np.array([0.01]))
    assert not result["model_results"].empty


def test_run_analysis_pipeline_delegates_to_run_factor_research(monkeypatch):
    sentinel = {"status": "ok", "factor_count": 3}

    monkeypatch.setattr(
        analysis_module,
        "build_analysis_panel",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("build_analysis_panel should not be called")
        ),
    )
    monkeypatch.setattr(
        analysis_module,
        "run_factor_research",
        lambda *args, **kwargs: sentinel,
    )

    assert run_analysis_pipeline() is sentinel
