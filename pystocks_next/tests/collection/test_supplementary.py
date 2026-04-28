from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from pystocks_next.collection import refresh_supplementary_sources
from pystocks_next.storage import (
    load_risk_free_sources,
    load_world_bank_raw,
)
from pystocks_next.tests.support import RecordingProgressSink


def test_refresh_supplementary_sources_persists_raw_tables_and_fetch_logs(
    temp_store,
    sample_risk_free_sources_frame: pd.DataFrame,
    sample_world_bank_raw_frame: pd.DataFrame,
) -> None:
    captured_economies: list[str] = []

    def fake_world_bank_fetcher(economy_codes: Sequence[str]) -> pd.DataFrame:
        captured_economies.extend(str(code) for code in economy_codes)
        return sample_world_bank_raw_frame

    result = refresh_supplementary_sources(
        temp_store,
        economy_codes=["US"],
        risk_free_fetcher=lambda: sample_risk_free_sources_frame,
        world_bank_fetcher=fake_world_bank_fetcher,
    )

    stored_risk_free = load_risk_free_sources(temp_store).frame
    stored_world_bank = load_world_bank_raw(temp_store).frame
    fetch_log_count = temp_store.execute(
        "SELECT COUNT(*) FROM supplementary_fetch_log"
    ).fetchone()[0]

    assert result.status == "ok"
    assert result.risk_free_source_rows == 2
    assert result.world_bank_raw_rows == 2
    assert result.fetch_log_rows == 2
    assert result.economy_count == 1
    assert captured_economies == ["USA"]
    assert stored_risk_free["economy_code"].tolist() == ["USA", "CAN"]
    assert stored_world_bank["economy_code"].tolist() == ["USA", "USA"]
    assert fetch_log_count == 2


def test_refresh_supplementary_sources_returns_no_economies_without_targets(
    temp_store,
) -> None:
    result = refresh_supplementary_sources(
        temp_store,
        risk_free_fetcher=lambda: pd.DataFrame(),
        world_bank_fetcher=lambda economy_codes: pd.DataFrame(),
    )

    assert result.status == "no_economies"
    assert result.economy_count == 0
    assert result.fetch_log_rows == 0


def test_refresh_supplementary_sources_normalizes_and_filters_economy_codes(
    temp_store,
) -> None:
    captured_economies: list[str] = []

    def fake_world_bank_fetcher(economy_codes: Sequence[str]) -> pd.DataFrame:
        captured_economies.extend(str(code) for code in economy_codes)
        return pd.DataFrame(
            columns=pd.Index(["economy_code", "indicator_id", "year", "value"])
        )

    result = refresh_supplementary_sources(
        temp_store,
        economy_codes=["US", "jp", "Unidentified", "United Kingdom", "", "US"],
        risk_free_fetcher=lambda: pd.DataFrame(
            columns=pd.Index(
                [
                    "series_id",
                    "source_name",
                    "trade_date",
                    "nominal_rate",
                    "economy_code",
                ]
            )
        ),
        world_bank_fetcher=fake_world_bank_fetcher,
    )

    assert result.status == "ok"
    assert result.economy_count == 3
    assert captured_economies == ["USA", "JPN", "GBR"]


def test_refresh_supplementary_sources_reports_progress(
    temp_store,
    sample_risk_free_sources_frame: pd.DataFrame,
    sample_world_bank_raw_frame: pd.DataFrame,
) -> None:
    progress = RecordingProgressSink()

    refresh_supplementary_sources(
        temp_store,
        economy_codes=["US"],
        risk_free_fetcher=lambda: sample_risk_free_sources_frame,
        world_bank_fetcher=lambda economy_codes: sample_world_bank_raw_frame,
        progress=progress,
    )

    assert progress.events == [
        ("start", "Refreshing supplementary data", 4, "step"),
        (
            "advance",
            "Refreshing supplementary data",
            1,
            "Fetched risk-free source series",
        ),
        ("advance", "Refreshing supplementary data", 1, "Fetched World Bank series"),
        (
            "advance",
            "Refreshing supplementary data",
            1,
            "Persisted risk-free source series",
        ),
        (
            "advance",
            "Refreshing supplementary data",
            1,
            "Persisted supplementary rows for 1 economies",
        ),
        ("close", "Refreshing supplementary data", None, "4 rows written"),
    ]
