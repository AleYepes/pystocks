from __future__ import annotations

import pandas as pd
import pytest

from pystocks_next.feature_inputs import (
    build_analysis_input_bundle,
    build_supplementary_input_bundle,
)
from pystocks_next.storage import (
    load_latest_holdings_country_weights,
    write_holdings_snapshot,
    write_supplementary_risk_free_sources,
    write_supplementary_world_bank_raw,
)
from pystocks_next.tests.support import RecordingProgressSink
from pystocks_next.universe.products import UniverseInstrument, upsert_instruments


def _build_macro_raw_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (
        economy_code,
        population_base,
        gdp_base,
        fdi_base,
        imports_base,
        exports_base,
    ) in (
        ("USA", 100.0, 1000.0, 10.0, 90.0, 110.0),
        ("CAN", 40.0, 300.0, 5.0, 35.0, 45.0),
    ):
        for year, year_step in ((2023, 0.0), (2024, 1.0), (2025, 2.0)):
            rows.extend(
                [
                    {
                        "economy_code": economy_code,
                        "indicator_id": "SP.POP.TOTL",
                        "year": year,
                        "value": population_base + year_step,
                        "observed_at": "2026-01-05T10:00:00+00:00",
                    },
                    {
                        "economy_code": economy_code,
                        "indicator_id": "NY.GDP.MKTP.CD",
                        "year": year,
                        "value": gdp_base + (50.0 * year_step),
                        "observed_at": "2026-01-05T10:00:00+00:00",
                    },
                    {
                        "economy_code": economy_code,
                        "indicator_id": "BX.KLT.DINV.WD.GD.ZS",
                        "year": year,
                        "value": fdi_base + year_step,
                        "observed_at": "2026-01-05T10:00:00+00:00",
                    },
                    {
                        "economy_code": economy_code,
                        "indicator_id": "NE.IMP.GNFS.ZS",
                        "year": year,
                        "value": imports_base + (5.0 * year_step),
                        "observed_at": "2026-01-05T10:00:00+00:00",
                    },
                    {
                        "economy_code": economy_code,
                        "indicator_id": "NE.EXP.GNFS.ZS",
                        "year": year,
                        "value": exports_base + (5.0 * year_step),
                        "observed_at": "2026-01-05T10:00:00+00:00",
                    },
                ]
            )
    return pd.DataFrame(rows)


def test_build_supplementary_input_bundle_derives_weighted_risk_free_and_macro_features(
    sample_risk_free_sources_frame: pd.DataFrame,
) -> None:
    risk_free_sources = pd.concat(
        [
            sample_risk_free_sources_frame,
            pd.DataFrame(
                [
                    {
                        "series_id": "DTB3",
                        "source_name": "fred",
                        "economy_code": "usa",
                        "trade_date": "2026-01-03",
                        "nominal_rate": 0.04,
                        "observed_at": "2026-01-05T10:00:00+00:00",
                    },
                    {
                        "series_id": "IR3TIB01CAM156N",
                        "source_name": "fred",
                        "economy_code": "can",
                        "trade_date": "2026-01-03",
                        "nominal_rate": 0.02,
                        "observed_at": "2026-01-05T10:00:00+00:00",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    risk_free_sources["observed_at"] = "2026-01-05T10:00:00+00:00"
    country_weights = pd.DataFrame(
        [
            {"economy_code": "US", "weight": 0.75},
            {"economy_code": "CA", "weight": 0.25},
        ]
    )

    bundle = build_supplementary_input_bundle(
        risk_free_sources=risk_free_sources,
        world_bank_raw=_build_macro_raw_frame(),
        country_weights=country_weights,
    )

    assert bundle.risk_free_daily["trade_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-01-02",
        "2026-01-03",
    ]
    assert bundle.risk_free_daily["nominal_rate"].tolist() == pytest.approx(
        [0.025, 0.035]
    )
    assert bundle.risk_free_daily["source_count"].tolist() == [2.0, 2.0]

    usa_2025 = bundle.macro_features.loc[
        (bundle.macro_features["economy_code"] == "USA")
        & (bundle.macro_features["feature_year"] == 2025)
    ].iloc[0]
    can_2025 = bundle.macro_features.loc[
        (bundle.macro_features["economy_code"] == "CAN")
        & (bundle.macro_features["feature_year"] == 2025)
    ].iloc[0]

    assert usa_2025["gdp_pcap_level"] == pytest.approx(1100.0 / 102.0)
    assert can_2025["economic_output_gdp_level"] == pytest.approx(400.0 / 1500.0)
    assert can_2025["share_trade_volume_level"] == pytest.approx(100.0 / 320.0)


def test_build_analysis_input_bundle_reads_supplementary_inputs_from_storage(
    temp_store,
    sample_holdings_payload: dict[str, object],
    sample_risk_free_sources_frame: pd.DataFrame,
) -> None:
    upsert_instruments(
        temp_store,
        [UniverseInstrument(conid="100", symbol="AAA", currency="USD")],
    )
    observed_at = "2026-01-05T10:00:00+00:00"
    write_holdings_snapshot(
        temp_store,
        conid="100",
        payload=sample_holdings_payload,
        observed_at=observed_at,
    )
    write_supplementary_risk_free_sources(
        temp_store,
        frame=sample_risk_free_sources_frame.assign(observed_at=observed_at),
        observed_at=observed_at,
    )
    write_supplementary_world_bank_raw(
        temp_store,
        frame=_build_macro_raw_frame(),
        observed_at=observed_at,
    )

    bundle = build_analysis_input_bundle(conn=temp_store)
    holdings_weights = load_latest_holdings_country_weights(temp_store).frame

    assert not holdings_weights.empty
    assert bundle.risk_free_daily["trade_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-01-02"
    ]
    assert bundle.macro_features["economy_code"].tolist().count("USA") == 3
    assert bundle.macro_features["economy_code"].tolist().count("CAN") == 3


def test_build_analysis_input_bundle_reports_progress(temp_store) -> None:
    progress = RecordingProgressSink()

    bundle = build_analysis_input_bundle(conn=temp_store, progress=progress)

    assert bundle.prices.empty
    assert progress.events == [
        ("start", "Building feature inputs", 4, "step"),
        ("advance", "Building feature inputs", 1, "Built price inputs"),
        ("advance", "Building feature inputs", 1, "Built dividend inputs"),
        ("advance", "Building feature inputs", 1, "Built snapshot inputs"),
        ("advance", "Building feature inputs", 1, "Built supplementary inputs"),
        ("close", "Building feature inputs", None, "Feature inputs ready"),
    ]
