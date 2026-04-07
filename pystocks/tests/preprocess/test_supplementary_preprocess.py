import pandas as pd
import pytest

from pystocks.preprocess.supplementary import (
    build_risk_free_series_weights,
    derive_risk_free_daily,
    load_risk_free_country_weights,
    preprocess_world_bank_country_features,
)


def test_derive_risk_free_daily_applies_series_weights_and_derives_daily_rate():
    source_df = pd.DataFrame(
        [
            {
                "series_id": "DTB3",
                "source_name": "fred",
                "trade_date": "2026-01-01",
                "nominal_rate": 0.03,
                "fetched_at": "2026-01-02T00:00:00+00:00",
            },
            {
                "series_id": "IR3TIB01CAM156N",
                "source_name": "fred",
                "trade_date": "2026-01-01",
                "nominal_rate": 0.01,
                "fetched_at": "2026-01-02T00:00:00+00:00",
            },
        ]
    )
    series_weights = pd.Series({"DTB3": 0.75, "IR3TIB01CAM156N": 0.25}, dtype=float)

    result = derive_risk_free_daily(source_df, series_weights=series_weights)

    row = result.iloc[0]
    assert row["nominal_rate"] == pytest.approx(0.025)
    assert row["daily_nominal_rate"] == pytest.approx(0.025 / 252.0)
    assert row["source_count"] == 2


def test_build_risk_free_series_weights_maps_country_weights_to_supported_series():
    country_weights = pd.DataFrame(
        [
            {"economy_code": "USA", "weight": 0.5},
            {"economy_code": "CAN", "weight": 0.3},
            {"economy_code": "ESP", "weight": 0.2},
        ]
    )

    result = build_risk_free_series_weights(country_weights)

    assert result["DTB3"] == pytest.approx(0.625)
    assert result["IR3TIB01CAM156N"] == pytest.approx(0.375)
    assert "IR3TIB01DEM156N" not in result.index


def test_load_risk_free_country_weights_uses_latest_snapshot_per_conid(tmp_path):
    import sqlite3

    from pystocks.storage.schema import init_storage

    db_path = tmp_path / "supplementary.sqlite"
    init_storage(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO holdings_investor_country (conid, effective_at, country_code, value_num)
            VALUES (?, ?, ?, ?)
            """,
            [
                ("1", "2026-01-31", "USA", 0.60),
                ("1", "2026-01-31", "CAN", 0.40),
                ("1", "2026-02-28", "USA", 0.20),
                ("1", "2026-02-28", "CAN", 0.80),
                ("2", "2026-02-15", "USA", 1.00),
            ],
        )
        conn.commit()

    result = load_risk_free_country_weights(db_path)
    weights = result.set_index("economy_code")["weight"]

    assert weights["USA"] == 0.60
    assert weights["CAN"] == 0.40


def test_preprocess_world_bank_country_features_derives_levels_and_growth():
    raw_df = pd.DataFrame(
        [
            {
                "economy_code": "USA",
                "indicator_id": "SP.POP.TOTL",
                "year": 2024,
                "value": 100.0,
                "fetched_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "economy_code": "USA",
                "indicator_id": "SP.POP.TOTL",
                "year": 2025,
                "value": 110.0,
                "fetched_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "economy_code": "USA",
                "indicator_id": "NY.GDP.MKTP.CD",
                "year": 2024,
                "value": 1000.0,
                "fetched_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "economy_code": "USA",
                "indicator_id": "NY.GDP.MKTP.CD",
                "year": 2025,
                "value": 1210.0,
                "fetched_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "economy_code": "USA",
                "indicator_id": "BX.KLT.DINV.WD.GD.ZS",
                "year": 2024,
                "value": 1.0,
                "fetched_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "economy_code": "USA",
                "indicator_id": "BX.KLT.DINV.WD.GD.ZS",
                "year": 2025,
                "value": 1.5,
                "fetched_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "economy_code": "USA",
                "indicator_id": "NE.IMP.GNFS.ZS",
                "year": 2024,
                "value": 40.0,
                "fetched_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "economy_code": "USA",
                "indicator_id": "NE.IMP.GNFS.ZS",
                "year": 2025,
                "value": 44.0,
                "fetched_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "economy_code": "USA",
                "indicator_id": "NE.EXP.GNFS.ZS",
                "year": 2024,
                "value": 60.0,
                "fetched_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "economy_code": "USA",
                "indicator_id": "NE.EXP.GNFS.ZS",
                "year": 2025,
                "value": 66.0,
                "fetched_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "economy_code": "CAN",
                "indicator_id": "SP.POP.TOTL",
                "year": 2024,
                "value": 50.0,
                "fetched_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "economy_code": "CAN",
                "indicator_id": "SP.POP.TOTL",
                "year": 2025,
                "value": 55.0,
                "fetched_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "economy_code": "CAN",
                "indicator_id": "NY.GDP.MKTP.CD",
                "year": 2024,
                "value": 500.0,
                "fetched_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "economy_code": "CAN",
                "indicator_id": "NY.GDP.MKTP.CD",
                "year": 2025,
                "value": 605.0,
                "fetched_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "economy_code": "CAN",
                "indicator_id": "NY.GDP.PCAP.CD",
                "year": 2024,
                "value": 10.0,
                "fetched_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "economy_code": "CAN",
                "indicator_id": "NY.GDP.PCAP.CD",
                "year": 2025,
                "value": 11.0,
                "fetched_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "economy_code": "CAN",
                "indicator_id": "BX.KLT.DINV.WD.GD.ZS",
                "year": 2024,
                "value": 2.0,
                "fetched_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "economy_code": "CAN",
                "indicator_id": "BX.KLT.DINV.WD.GD.ZS",
                "year": 2025,
                "value": 2.5,
                "fetched_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "economy_code": "CAN",
                "indicator_id": "NE.IMP.GNFS.ZS",
                "year": 2024,
                "value": 25.0,
                "fetched_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "economy_code": "CAN",
                "indicator_id": "NE.IMP.GNFS.ZS",
                "year": 2025,
                "value": 27.5,
                "fetched_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "economy_code": "CAN",
                "indicator_id": "NE.EXP.GNFS.ZS",
                "year": 2024,
                "value": 25.0,
                "fetched_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "economy_code": "CAN",
                "indicator_id": "NE.EXP.GNFS.ZS",
                "year": 2025,
                "value": 27.5,
                "fetched_at": "2026-01-01T00:00:00+00:00",
            },
        ]
    )

    result = preprocess_world_bank_country_features(raw_df)

    usa_2024 = result[
        (result["economy_code"] == "USA") & (result["feature_year"] == 2024)
    ].iloc[0]
    usa_2025 = result[
        (result["economy_code"] == "USA") & (result["feature_year"] == 2025)
    ].iloc[0]

    assert usa_2024["gdp_pcap_level"] == 10.0
    assert usa_2024["economic_output_gdp_level"] == 1000.0 / 1500.0
    assert usa_2024["share_trade_volume_level"] == 100.0 / 150.0
    assert usa_2025["population_growth"] == 10.0
    assert usa_2025["foreign_direct_investment_growth"] == 0.5
