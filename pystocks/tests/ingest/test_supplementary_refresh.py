import sqlite3

import pandas as pd
import pytest

import pystocks.ingest.supplementary as supplementary_module
from pystocks.storage.readers import load_world_bank_country_features
from pystocks.storage.schema import init_storage


def test_fetch_world_bank_raw_uses_wbgapi_and_skips_unrecognized(monkeypatch):
    class FakeEconomy:
        @staticmethod
        def list():
            return [{"id": "USA"}, {"id": "CAN"}]

    class FakeData:
        @staticmethod
        def fetch(series, economy="all", numericTimeKeys=False):
            assert series == ["SP.POP.TOTL"]
            assert economy == ["USA", "CAN"]
            assert numericTimeKeys is True
            return iter(
                [
                    {
                        "economy": "USA",
                        "series": "SP.POP.TOTL",
                        "time": 2025,
                        "value": 100.0,
                    },
                    {
                        "economy": "CAN",
                        "series": "SP.POP.TOTL",
                        "time": 2025,
                        "value": 50.0,
                    },
                ]
            )

    class FakeWorldBankModule:
        economy = FakeEconomy()
        data = FakeData()

    monkeypatch.setattr(
        supplementary_module, "_load_wbgapi_module", lambda: FakeWorldBankModule
    )

    result = supplementary_module.fetch_world_bank_raw(
        ["US", "Canada", "TWN"],
        indicator_map={"SP.POP.TOTL": "population"},
    )

    assert result["economy_code"].tolist() == ["CAN", "USA"]
    assert result["indicator_id"].tolist() == ["SP.POP.TOTL", "SP.POP.TOTL"]
    assert result["year"].tolist() == [2025, 2025]


def test_fetch_world_bank_raw_raises_clear_error_without_wbgapi(monkeypatch):
    def raise_missing():
        raise RuntimeError("wbgapi is required for World Bank refresh.")

    monkeypatch.setattr(supplementary_module, "_load_wbgapi_module", raise_missing)

    with pytest.raises(RuntimeError, match="wbgapi"):
        supplementary_module.fetch_world_bank_raw(
            ["US"], indicator_map={"SP.POP.TOTL": "population"}
        )


def test_refresh_supplementary_data_writes_normalized_tables(tmp_path, monkeypatch):
    db_path = tmp_path / "supplementary.sqlite"
    init_storage(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO holdings_investor_country (conid, effective_at, country_code, value_num)
            VALUES (?, ?, ?, ?)
            """,
            ["1", "2026-01-31", "USA", 0.6],
        )
        conn.execute(
            """
            INSERT INTO holdings_investor_country (conid, effective_at, country_code, value_num)
            VALUES (?, ?, ?, ?)
            """,
            ["2", "2026-01-31", "CAN", 0.4],
        )
        conn.commit()

    monkeypatch.setattr(
        supplementary_module,
        "fetch_risk_free_sources",
        lambda series_map=None: pd.DataFrame(
            [
                {
                    "series_id": "DTB3",
                    "source_name": "fred",
                    "trade_date": pd.Timestamp("2026-01-02"),
                    "nominal_rate": 0.03,
                    "economy_code": "USA",
                },
                {
                    "series_id": "IR3TIB01CAM156N",
                    "source_name": "fred",
                    "trade_date": pd.Timestamp("2026-01-02"),
                    "nominal_rate": 0.01,
                    "economy_code": "CAN",
                },
            ]
        ),
    )
    monkeypatch.setattr(
        supplementary_module,
        "fetch_world_bank_raw",
        lambda economy_codes, indicator_map=None: pd.DataFrame(
            [
                {
                    "economy_code": "USA",
                    "indicator_id": "SP.POP.TOTL",
                    "year": 2025,
                    "value": 100.0,
                },
                {
                    "economy_code": "USA",
                    "indicator_id": "NY.GDP.PCAP.CD",
                    "year": 2025,
                    "value": 10.0,
                },
                {
                    "economy_code": "USA",
                    "indicator_id": "NY.GDP.MKTP.CD",
                    "year": 2025,
                    "value": 1000.0,
                },
                {
                    "economy_code": "USA",
                    "indicator_id": "BX.KLT.DINV.WD.GD.ZS",
                    "year": 2025,
                    "value": 1.0,
                },
                {
                    "economy_code": "USA",
                    "indicator_id": "NE.IMP.GNFS.ZS",
                    "year": 2025,
                    "value": 40.0,
                },
                {
                    "economy_code": "USA",
                    "indicator_id": "NE.EXP.GNFS.ZS",
                    "year": 2025,
                    "value": 60.0,
                },
                {
                    "economy_code": "CAN",
                    "indicator_id": "SP.POP.TOTL",
                    "year": 2025,
                    "value": 50.0,
                },
                {
                    "economy_code": "CAN",
                    "indicator_id": "NY.GDP.PCAP.CD",
                    "year": 2025,
                    "value": 10.0,
                },
                {
                    "economy_code": "CAN",
                    "indicator_id": "NY.GDP.MKTP.CD",
                    "year": 2025,
                    "value": 500.0,
                },
                {
                    "economy_code": "CAN",
                    "indicator_id": "BX.KLT.DINV.WD.GD.ZS",
                    "year": 2025,
                    "value": 2.0,
                },
                {
                    "economy_code": "CAN",
                    "indicator_id": "NE.IMP.GNFS.ZS",
                    "year": 2025,
                    "value": 25.0,
                },
                {
                    "economy_code": "CAN",
                    "indicator_id": "NE.EXP.GNFS.ZS",
                    "year": 2025,
                    "value": 25.0,
                },
            ]
        ),
    )

    result = supplementary_module.refresh_supplementary_data(sqlite_path=db_path)

    assert result["status"] == "ok"

    with sqlite3.connect(db_path) as conn:
        risk_free_rows = conn.execute(
            "SELECT COUNT(*) FROM supplementary_risk_free_daily"
        ).fetchone()[0]
        risk_free_nominal_rate = conn.execute(
            "SELECT nominal_rate FROM supplementary_risk_free_daily"
        ).fetchone()[0]
        macro_rows = conn.execute(
            "SELECT COUNT(*) FROM supplementary_world_bank_country_features"
        ).fetchone()[0]
        log_rows = conn.execute(
            "SELECT COUNT(*) FROM supplementary_fetch_log"
        ).fetchone()[0]

    assert risk_free_rows == 1
    assert risk_free_nominal_rate == pytest.approx(0.022)
    assert macro_rows == 2
    assert log_rows == 3


def test_load_world_bank_country_features_tolerates_old_schema(tmp_path):
    db_path = tmp_path / "supplementary.sqlite"
    init_storage(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE supplementary_world_bank_country_features")
        conn.execute(
            """
            CREATE TABLE supplementary_world_bank_country_features (
                economy_code TEXT NOT NULL,
                effective_at TEXT NOT NULL,
                feature_year INTEGER NOT NULL,
                population_level REAL,
                population_growth REAL,
                gdp_pcap_level REAL,
                gdp_pcap_growth REAL,
                economic_output_gdp_level REAL,
                economic_output_gdp_growth REAL,
                foreign_direct_investment_level REAL,
                foreign_direct_investment_growth REAL,
                share_trade_volume_level REAL,
                share_trade_volume_growth REAL,
                observed_at TEXT NOT NULL,
                PRIMARY KEY (economy_code, feature_year)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO supplementary_world_bank_country_features (
                economy_code,
                effective_at,
                feature_year,
                population_level,
                population_growth,
                gdp_pcap_level,
                gdp_pcap_growth,
                economic_output_gdp_level,
                economic_output_gdp_growth,
                foreign_direct_investment_level,
                foreign_direct_investment_growth,
                share_trade_volume_level,
                share_trade_volume_growth,
                observed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "USA",
                "2025-12-31",
                2025,
                100.0,
                10.0,
                10.0,
                1.0,
                0.7,
                0.1,
                1.0,
                0.1,
                0.5,
                0.05,
                "2026-01-01T00:00:00+00:00",
            ),
        )
        conn.commit()

    result = load_world_bank_country_features(db_path)

    row = result.iloc[0]
    assert pd.isna(row["population_acceleration"])
    assert pd.isna(row["gdp_pcap_acceleration"])
    assert pd.isna(row["economic_output_gdp_acceleration"])
    assert pd.isna(row["foreign_direct_investment_acceleration"])
    assert pd.isna(row["share_trade_volume_acceleration"])
