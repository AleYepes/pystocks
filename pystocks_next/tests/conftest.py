from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pystocks_next.storage.sqlite import connect_sqlite, initialize_operational_store
from pystocks_next.tests.support import (
    build_sample_dividend_events_frame,
    build_sample_dividends_payload,
    build_sample_holdings_payload,
    build_sample_price_chart_payload,
    build_sample_price_history_frame,
    build_sample_profile_and_fees_payload,
    build_sample_raw_payload,
    build_sample_risk_free_daily_frame,
    build_sample_risk_free_sources_frame,
    build_sample_snapshot_tables,
    build_sample_world_bank_country_features_frame,
    build_sample_world_bank_raw_frame,
)


@pytest.fixture
def temp_store_path(tmp_path: Path) -> Path:
    db_path = tmp_path / "pystocks_next.sqlite"
    initialize_operational_store(db_path)
    return db_path


@pytest.fixture
def temp_store(temp_store_path: Path):
    with connect_sqlite(temp_store_path) as conn:
        yield conn


@pytest.fixture
def sample_raw_payload() -> dict[str, object]:
    return build_sample_raw_payload()


@pytest.fixture
def sample_price_history_frame() -> pd.DataFrame:
    return build_sample_price_history_frame()


@pytest.fixture
def sample_dividend_events_frame() -> pd.DataFrame:
    return build_sample_dividend_events_frame()


@pytest.fixture
def sample_snapshot_tables() -> dict[str, pd.DataFrame]:
    return build_sample_snapshot_tables()


@pytest.fixture
def sample_price_chart_payload() -> dict[str, object]:
    return build_sample_price_chart_payload()


@pytest.fixture
def sample_dividends_payload() -> dict[str, object]:
    return build_sample_dividends_payload()


@pytest.fixture
def sample_profile_and_fees_payload() -> dict[str, object]:
    return build_sample_profile_and_fees_payload()


@pytest.fixture
def sample_holdings_payload() -> dict[str, object]:
    return build_sample_holdings_payload()


@pytest.fixture
def sample_risk_free_sources_frame() -> pd.DataFrame:
    return build_sample_risk_free_sources_frame()


@pytest.fixture
def sample_risk_free_daily_frame() -> pd.DataFrame:
    return build_sample_risk_free_daily_frame()


@pytest.fixture
def sample_world_bank_raw_frame() -> pd.DataFrame:
    return build_sample_world_bank_raw_frame()


@pytest.fixture
def sample_world_bank_country_features_frame() -> pd.DataFrame:
    return build_sample_world_bank_country_features_frame()
