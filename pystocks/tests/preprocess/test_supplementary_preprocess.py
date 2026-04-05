import pandas as pd

from pystocks.preprocess.supplementary import (
    derive_risk_free_daily,
    preprocess_world_bank_country_features,
)


def test_derive_risk_free_daily_averages_sources_and_derives_daily_rate():
    source_df = pd.DataFrame(
        [
            {
                "series_id": "A",
                "source_name": "fred",
                "trade_date": "2026-01-01",
                "nominal_rate": 0.03,
                "fetched_at": "2026-01-02T00:00:00+00:00",
            },
            {
                "series_id": "B",
                "source_name": "fred",
                "trade_date": "2026-01-01",
                "nominal_rate": 0.01,
                "fetched_at": "2026-01-02T00:00:00+00:00",
            },
        ]
    )

    result = derive_risk_free_daily(source_df)

    row = result.iloc[0]
    assert row["nominal_rate"] == 0.02
    assert row["daily_nominal_rate"] == 0.02 / 252.0
    assert row["source_count"] == 2


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
