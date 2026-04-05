import sqlite3

import pandas as pd

import pystocks.ingest.supplementary as supplementary_module
from pystocks.storage.schema import init_storage


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
                }
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
        macro_rows = conn.execute(
            "SELECT COUNT(*) FROM supplementary_world_bank_country_features"
        ).fetchone()[0]
        log_rows = conn.execute(
            "SELECT COUNT(*) FROM supplementary_fetch_log"
        ).fetchone()[0]

    assert risk_free_rows == 1
    assert macro_rows == 2
    assert log_rows == 3
