from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import cast

import pytest

from pystocks_next.collection.fundamentals import (
    CollectedEndpointPayload,
    EndpointFetchResult,
    FundamentalsCollector,
    FundamentalsConidOutcome,
    FundamentalsPersistResult,
)
from pystocks_next.storage import (
    UnresolvedEffectiveAtError,
    load_dividend_events,
    load_price_history,
    load_snapshot_feature_tables,
)
from pystocks_next.storage.writes import write_price_chart_series
from pystocks_next.tests.support import RecordingProgressSink
from pystocks_next.universe import UniverseInstrument, upsert_instruments


class _DummySession:
    class _DummyClient:
        class _DummyResponse:
            status_code = 500

            def json(self) -> object:
                return {}

        async def get(self, url: str):
            del url
            return self._DummyResponse()

    async def validate_auth_state(self, *, timeout_s: float = 20.0) -> bool:
        del timeout_s
        return True

    async def login(
        self,
        *,
        headless: bool = True,
        force_browser: bool = False,
    ) -> bool:
        del headless, force_browser
        return True

    async def reauthenticate(self, *, headless: bool = False) -> bool:
        del headless
        return True

    @asynccontextmanager
    async def get_client(self, *, timeout_s: float = 20.0):
        del timeout_s
        yield self._DummyClient()


def test_select_price_chart_period_matches_legacy_windows() -> None:
    collector = FundamentalsCollector(
        session=_DummySession(),
        latest_price_effective_at_by_conid={
            "100": None,
            "101": date(2026, 2, 26),
            "102": date(2025, 12, 9),
        },
    )

    assert (
        collector.select_price_chart_period("100", as_of_date=date(2026, 2, 27))
        == "MAX"
    )
    assert (
        collector.select_price_chart_period("101", as_of_date=date(2026, 2, 27)) == "1W"
    )
    assert (
        collector.select_price_chart_period("102", as_of_date=date(2026, 2, 27)) == "3M"
    )


def test_useful_payload_heuristics_preserve_sparse_real_signals() -> None:
    collector = FundamentalsCollector(session=_DummySession())

    assert collector.is_useful_payload({"as_of_date": "2026-02-20"}, "holdings")
    assert collector.is_useful_payload(
        {"as_of_date": 1769835600000, "fixed_income": [{"name": "YTM", "value": 4.45}]},
        "ratios",
    )


def test_collect_conid_returns_structured_skip_for_landing_only_instrument() -> None:
    collector = FundamentalsCollector(session=_DummySession())

    async def fake_fetch_endpoint(_client, endpoint: str) -> EndpointFetchResult:
        assert endpoint == "landing/123?widgets=objective,keyProfile"
        return EndpointFetchResult(
            status_code=200,
            payload={"key_profile": {"data": {"ticker": "ABC"}}},
        )

    collector.fetch_endpoint = fake_fetch_endpoint  # type: ignore[method-assign]

    result = asyncio.run(
        collector.collect_conid(
            cast(_DummySession._DummyClient, _DummySession._DummyClient()), conid="123"
        )
    )

    assert result.conid == "123"
    assert result.status == "skipped"
    assert result.skip_reason == "skip_missing_total_net_assets"


def test_run_requests_visible_login_when_auth_is_missing(temp_store) -> None:
    class _AuthMissingSession(_DummySession):
        def __init__(self) -> None:
            self.login_calls: list[tuple[bool, bool]] = []

        async def validate_auth_state(self, *, timeout_s: float = 20.0) -> bool:
            del timeout_s
            return False

        async def login(
            self,
            *,
            headless: bool = True,
            force_browser: bool = False,
        ) -> bool:
            self.login_calls.append((headless, force_browser))
            return False

    session = _AuthMissingSession()
    collector = FundamentalsCollector(session=session)

    result = asyncio.run(collector.run(temp_store, explicit_conids=["100"]))

    assert result.status == "auth_required"
    assert result.aborted is True
    assert session.login_calls == [(False, True)]


def test_run_persists_first_slice_end_to_end(
    temp_store,
    sample_profile_and_fees_payload: dict[str, object],
    sample_holdings_payload: dict[str, object],
    sample_ratios_payload: dict[str, object],
    sample_dividends_payload: dict[str, object],
    sample_price_chart_payload: dict[str, object],
    sample_morningstar_payload: dict[str, object],
    sample_lipper_payload: dict[str, object],
    tmp_path: Path,
) -> None:
    upsert_instruments(
        temp_store, [UniverseInstrument(conid="100", symbol="AAA", currency="USD")]
    )
    write_price_chart_series(
        temp_store,
        conid="100",
        payload=sample_price_chart_payload,
        observed_at="2026-01-04T10:00:00+00:00",
    )

    collector = FundamentalsCollector(session=_DummySession())

    async def fake_fetch_endpoint(_client, endpoint: str) -> EndpointFetchResult:
        payload_by_endpoint = {
            "landing/100?widgets=objective,keyProfile": {
                "as_of_date": "2026-01-03",
                "key_profile": {"data": {"total_net_assets": "1.2B"}},
            },
            "mf_profile_and_fees/100?sustainability=UK&lang=en": sample_profile_and_fees_payload,
            "mf_holdings/100": sample_holdings_payload,
            "mf_ratios_fundamentals/100": sample_ratios_payload,
            "mf_lip_ratings/100": sample_lipper_payload,
            "dividends/100": sample_dividends_payload,
            "mstar/fund/detail?conid=100": sample_morningstar_payload,
            "mf_performance_chart/100?chart_period=1W": sample_price_chart_payload,
        }
        return EndpointFetchResult(
            status_code=200, payload=payload_by_endpoint[endpoint]
        )

    collector.fetch_endpoint = fake_fetch_endpoint  # type: ignore[method-assign]
    result = asyncio.run(
        collector.run(
            temp_store,
            explicit_conids=["100"],
            telemetry_output_path=tmp_path / "fundamentals_run_telemetry.json",
        )
    )

    price_history = load_price_history(temp_store).frame
    dividend_events = load_dividend_events(temp_store).frame
    snapshot_tables = load_snapshot_feature_tables(temp_store).tables

    assert result.status == "ok"
    assert result.total_targeted_conids == 1
    assert result.processed_conids == 1
    assert result.saved_snapshots >= 5
    assert result.inserted_events == 2
    assert result.series_latest_rows_upserted == 2
    assert result.telemetry_path is not None
    assert result.latest_telemetry_path is not None
    assert price_history["conid"].tolist().count("100") == 2
    assert dividend_events["conid"].tolist() == ["100", "100"]
    assert not snapshot_tables["profile_and_fees"].empty
    assert not snapshot_tables["holdings_asset_type"].empty
    assert not snapshot_tables["ratios_key_ratios"].empty


def test_run_reports_progress_for_processed_conids(temp_store) -> None:
    collector = FundamentalsCollector(session=_DummySession())
    progress = RecordingProgressSink()

    async def fake_collect_conid(_client, conid: str) -> FundamentalsConidOutcome:
        return FundamentalsConidOutcome(
            conid=conid,
            status="saved",
            observed_at="2026-01-04T10:00:00+00:00",
        )

    collector.collect_conid = fake_collect_conid  # type: ignore[method-assign]
    collector.persist_outcome = lambda conn, outcome: FundamentalsPersistResult(  # type: ignore[method-assign]
        saved_snapshots=1
    )

    result = asyncio.run(
        collector.run(
            temp_store,
            explicit_conids=["100", "101"],
            progress=progress,
        )
    )

    assert result.processed_conids == 2
    assert progress.events == [
        ("start", "Collecting fundamentals", 2, "conid"),
        ("advance", "Collecting fundamentals", 1, "100 saved, 1 snapshots saved"),
        ("advance", "Collecting fundamentals", 1, "101 saved, 2 snapshots saved"),
        ("close", "Collecting fundamentals", None, "2/2 conids, 2 snapshots saved"),
    ]


def test_run_writes_persistence_failure_artifact_and_report(
    temp_store,
    sample_morningstar_payload: dict[str, object],
    tmp_path: Path,
) -> None:
    collector = FundamentalsCollector(session=_DummySession())

    async def fake_collect_conid(_client, conid: str) -> FundamentalsConidOutcome:
        return FundamentalsConidOutcome(
            conid=conid,
            status="success",
            observed_at="2026-01-04T10:00:00+00:00",
            endpoint_payloads=(
                CollectedEndpointPayload(
                    endpoint_name="morningstar",
                    endpoint_family="mstar",
                    request_path=f"mstar/fund/detail?conid={conid}",
                    conid=conid,
                    observed_at="2026-01-04T10:00:00+00:00",
                    payload=sample_morningstar_payload,
                    status_code=200,
                    is_useful=True,
                ),
            ),
        )

    def fake_persist_outcome(_conn, _outcome) -> FundamentalsPersistResult:
        raise UnresolvedEffectiveAtError(
            "morningstar_snapshot", "source_as_of_date is required"
        )

    collector.collect_conid = fake_collect_conid  # type: ignore[method-assign]
    collector.persist_outcome = fake_persist_outcome  # type: ignore[method-assign]
    telemetry_path = tmp_path / "fundamentals_run_telemetry.json"

    with pytest.raises(UnresolvedEffectiveAtError) as exc_info:
        asyncio.run(
            collector.run(
                temp_store,
                explicit_conids=["100"],
                telemetry_output_path=telemetry_path,
            )
        )

    report_payload = json.loads(telemetry_path.read_text())
    artifact_paths = sorted((tmp_path / "persist_failures").glob("*.json"))

    assert artifact_paths
    assert "Persistence failure artifact:" in "\n".join(exc_info.value.__notes__)
    assert "Telemetry report:" in "\n".join(exc_info.value.__notes__)
    assert report_payload["run_stats"]["status"] == "failed"
    assert report_payload["run_stats"]["failed_conid"] == "100"
    assert len(report_payload["persistence_failures"]) == 1
    assert report_payload["persistence_failures"][0]["artifact_path"] == str(
        artifact_paths[0]
    )

    artifact_payload = json.loads(artifact_paths[0].read_text())
    assert artifact_payload["conid"] == "100"
    assert artifact_payload["exception_type"] == "UnresolvedEffectiveAtError"
    assert artifact_payload["endpoint_payloads"][0]["endpoint_name"] == "morningstar"
    assert (
        artifact_payload["endpoint_payloads"][0]["payload"]
        == sample_morningstar_payload
    )
