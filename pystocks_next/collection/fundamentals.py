from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Collection, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qs

import httpx

from ..progress import ProgressSink
from ..storage import (
    UnresolvedEffectiveAtError,
    load_latest_price_effective_at_by_conid,
    new_capture_batch_id,
    parse_date_candidate,
    write_dividend_events_series,
    write_dividends_snapshot,
    write_holdings_snapshot,
    write_lipper_ratings_snapshot,
    write_morningstar_snapshot,
    write_price_chart_series,
    write_profile_and_fees_snapshot,
    write_ratios_snapshot,
)
from ..universe import select_explicit_targets, select_governed_targets
from .session import CollectionSession
from .telemetry import CollectionTelemetry

_SKIP_MISSING_TOTAL_NET_ASSETS = "skip_missing_total_net_assets"


def _sanitize_path_segment(value: str) -> str:
    sanitized = "".join(char if char.isalnum() else "_" for char in value.strip())
    return sanitized.strip("_") or "unknown"


class EndpointHttpResponse(Protocol):
    status_code: int

    def json(self) -> object: ...


class EndpointHttpClient(Protocol):
    async def get(self, url: str, /) -> EndpointHttpResponse: ...


class CollectionSessionLike(Protocol):
    async def validate_auth_state(self, *, timeout_s: float = 20.0) -> bool: ...

    async def login(
        self,
        *,
        headless: bool = True,
        force_browser: bool = False,
    ) -> bool: ...

    async def reauthenticate(self, *, headless: bool = False) -> bool: ...

    def get_client(
        self,
        *,
        timeout_s: float = 20.0,
    ) -> AbstractAsyncContextManager[EndpointHttpClient]: ...


@dataclass(frozen=True, slots=True)
class EndpointFetchResult:
    status_code: int
    payload: Any


@dataclass(frozen=True, slots=True)
class CollectedEndpointPayload:
    endpoint_name: str
    endpoint_family: str
    request_path: str
    conid: str
    observed_at: str
    payload: Any
    status_code: int
    is_useful: bool


@dataclass(frozen=True, slots=True)
class FundamentalsPersistResult:
    saved_snapshots: int = 0
    inserted_events: int = 0
    overwritten_events: int = 0
    unchanged_events: int = 0
    series_raw_rows_written: int = 0
    series_latest_rows_upserted: int = 0


@dataclass(frozen=True, slots=True)
class FundamentalsConidOutcome:
    conid: str
    status: str
    observed_at: str
    skip_reason: str | None = None
    endpoint_payloads: tuple[CollectedEndpointPayload, ...] = ()
    storage_result: FundamentalsPersistResult | None = None


@dataclass(frozen=True, slots=True)
class FundamentalsCollectionResult:
    status: str
    total_targeted_conids: int
    processed_conids: int
    saved_snapshots: int
    inserted_events: int
    overwritten_events: int
    unchanged_events: int
    series_raw_rows_written: int
    series_latest_rows_upserted: int
    auth_retries: int
    aborted: bool
    telemetry_path: str | None = None
    latest_telemetry_path: str | None = None


class EndpointPersistenceError(RuntimeError):
    def __init__(self, payload: CollectedEndpointPayload, cause: Exception) -> None:
        super().__init__(str(cause))
        self.payload = payload
        self.cause = cause


class FundamentalsCollector:
    """Thin fundamentals runner for the first collection slice."""

    PRICE_CHART_PERIOD_WINDOWS = [
        ("1W", 14),
        ("1M", 45),
        ("3M", 120),
        ("6M", 240),
        ("1Y", 400),
        ("3Y", 1200),
        ("5Y", 2200),
        ("10Y", 4000),
    ]
    RESULT_ENDPOINT_FAMILIES = {
        "profile_and_fees": "mf_profile_and_fees",
        "holdings": "mf_holdings",
        "ratios": "mf_ratios_fundamentals",
        "lipper_ratings": "mf_lip_ratings",
        "dividends": "dividends",
        "morningstar": "mstar/fund/detail",
        "price_chart": "mf_performance_chart",
    }

    def __init__(
        self,
        *,
        session: CollectionSessionLike | None = None,
        telemetry: CollectionTelemetry | None = None,
        latest_price_effective_at_by_conid: Mapping[str, date | None] | None = None,
    ) -> None:
        self.session = session or CollectionSession()
        self.telemetry = telemetry or CollectionTelemetry()
        self._latest_price_effective_at_by_conid = dict(
            latest_price_effective_at_by_conid or {}
        )

    def _endpoint_family(self, endpoint: str) -> str:
        endpoint = endpoint.lstrip("/")
        if endpoint.startswith("fundamentals/"):
            endpoint = endpoint[len("fundamentals/") :]
        if endpoint.startswith("landing/"):
            return "landing"
        if endpoint.startswith("mf_profile_and_fees/"):
            return "mf_profile_and_fees"
        if endpoint.startswith("mf_holdings/"):
            return "mf_holdings"
        if endpoint.startswith("mf_ratios_fundamentals/"):
            return "mf_ratios_fundamentals"
        if endpoint.startswith("mf_lip_ratings/"):
            return "mf_lip_ratings"
        if endpoint.startswith("dividends/"):
            return "dividends"
        if endpoint.startswith("mstar/fund/detail"):
            return "mstar/fund/detail"
        if endpoint.startswith("mf_performance_chart/"):
            return "mf_performance_chart"
        if endpoint.startswith("sma/request?"):
            query = endpoint.split("?", 1)[1] if "?" in endpoint else ""
            request_type = parse_qs(query).get("type", ["unknown"])[0]
            return f"sma/request?type={request_type}"
        return endpoint.split("?", 1)[0]

    def _has_any_value(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (list, dict, tuple, set)):
            return len(value) > 0
        return True

    def is_useful_payload(self, payload: Any, endpoint_name: str) -> bool:
        if not isinstance(payload, dict):
            return False
        kind_alias = {
            "profile_and_fees": "profile",
            "lipper_ratings": "lipper",
            "dividends": "divs",
            "morningstar": "mstar",
            "price_chart": "price_chart",
        }
        kind = str(kind_alias.get(endpoint_name, endpoint_name))
        checks = {
            "profile": ["fund_and_profile", "objective"],
            "holdings": [
                "as_of_date",
                "asOfDate",
                "allocation_self",
                "top_10",
                "industry",
                "currency",
                "investor_country",
                "debt_type",
                "debtor",
                "maturity",
                "geographic",
                "top_10_weight",
            ],
            "ratios": [
                "as_of_date",
                "asOfDate",
                "ratios",
                "financials",
                "fixed_income",
                "dividend",
                "zscore",
            ],
            "lipper": ["universes"],
            "divs": [
                "history",
                "industry_average",
                "no_div_data_marker",
                "last_payed_dividend_amount",
            ],
            "mstar": ["summary", "commentary"],
            "price_chart": ["plot", "history"],
        }
        for key in checks.get(kind, []):
            if self._has_any_value(payload.get(key)):
                return True
        return False

    def _landing_has_total_net_assets(self, landing_data: Any) -> bool:
        if not isinstance(landing_data, dict):
            return False
        key_profile = landing_data.get("key_profile") or landing_data.get("keyProfile")
        if not isinstance(key_profile, dict):
            return False
        profile_data = key_profile.get("data")
        if not isinstance(profile_data, dict):
            return False
        if "total_net_assets" not in profile_data:
            return False
        return self._has_any_value(profile_data.get("total_net_assets"))

    def _should_skip_fanout(self, landing_data: Any) -> tuple[bool, str | None]:
        if not self._landing_has_total_net_assets(landing_data):
            return True, "total_net_assets missing"
        return False, None

    def _build_landing_endpoint(self, conid: str) -> str:
        return f"landing/{conid}?widgets=objective,keyProfile"

    def _build_price_chart_endpoint(
        self, conid: str, *, chart_period: str = "MAX"
    ) -> str:
        return f"mf_performance_chart/{conid}?chart_period={chart_period}"

    def _get_latest_price_effective_at(self, conid: str) -> date | None:
        return self._latest_price_effective_at_by_conid.get(str(conid))

    def select_price_chart_period(
        self,
        conid: str,
        *,
        as_of_date: date | datetime | None = None,
    ) -> str:
        latest_effective_at = self._get_latest_price_effective_at(conid)
        if latest_effective_at is None:
            return "MAX"

        as_of = parse_date_candidate(as_of_date) or datetime.now(tz=UTC).date()
        if isinstance(as_of, datetime):
            as_of = as_of.date()
        missing_days = (as_of - latest_effective_at).days
        if missing_days <= 0:
            return "1W"
        target_days = missing_days + 7
        for period, max_days in self.PRICE_CHART_PERIOD_WINDOWS:
            if target_days <= max_days:
                return period
        return "MAX"

    async def fetch_endpoint(
        self,
        client: EndpointHttpClient,
        endpoint: str,
    ) -> EndpointFetchResult:
        path_prefix = (
            ""
            if (
                "/" in endpoint
                and endpoint.split("/")[0] in ["fundamentals", "mstar", "sma", "impact"]
            )
            else "fundamentals/"
        )
        url = f"/tws.proxy/{path_prefix}{endpoint}"
        try:
            response = await client.get(url)
        except httpx.RequestError:
            self.telemetry.record_call(self._endpoint_family(endpoint), 0)
            return EndpointFetchResult(status_code=0, payload=None)

        self.telemetry.record_call(
            self._endpoint_family(endpoint), response.status_code
        )
        if response.status_code != 200:
            return EndpointFetchResult(status_code=response.status_code, payload=None)
        try:
            payload = response.json()
        except ValueError:
            payload = None
        return EndpointFetchResult(status_code=200, payload=payload)

    async def collect_conid(
        self,
        client: EndpointHttpClient,
        *,
        conid: str,
    ) -> FundamentalsConidOutcome:
        observed_at = datetime.now(tz=UTC).isoformat()
        landing_endpoint = self._build_landing_endpoint(conid)
        landing = await self.fetch_endpoint(client, landing_endpoint)
        if landing.status_code in {401, 403}:
            return FundamentalsConidOutcome(
                conid=conid,
                status="auth_error",
                observed_at=observed_at,
            )
        if not isinstance(landing.payload, dict):
            return FundamentalsConidOutcome(
                conid=conid,
                status="failed",
                observed_at=observed_at,
            )

        skip_fanout, skip_reason = self._should_skip_fanout(landing.payload)
        if skip_fanout:
            return FundamentalsConidOutcome(
                conid=conid,
                status="skipped",
                observed_at=observed_at,
                skip_reason=_SKIP_MISSING_TOTAL_NET_ASSETS if skip_reason else None,
            )

        landing_as_of_date = landing.payload.get("as_of_date") or landing.payload.get(
            "asOfDate"
        )
        chart_period = self.select_price_chart_period(
            conid,
            as_of_date=landing_as_of_date,
        )
        task_items = [
            (
                "profile_and_fees",
                f"mf_profile_and_fees/{conid}?sustainability=UK&lang=en",
            ),
            ("holdings", f"mf_holdings/{conid}"),
            ("ratios", f"mf_ratios_fundamentals/{conid}"),
            ("lipper_ratings", f"mf_lip_ratings/{conid}"),
            ("dividends", f"dividends/{conid}"),
            ("morningstar", f"mstar/fund/detail?conid={conid}"),
            (
                "price_chart",
                self._build_price_chart_endpoint(conid, chart_period=chart_period),
            ),
        ]
        responses = await asyncio.gather(
            *[self.fetch_endpoint(client, endpoint) for _, endpoint in task_items]
        )

        collected: list[CollectedEndpointPayload] = []
        for (name, endpoint), response in zip(task_items, responses, strict=True):
            if response.status_code in {401, 403}:
                return FundamentalsConidOutcome(
                    conid=conid,
                    status="auth_error",
                    observed_at=observed_at,
                )
            is_useful = self.is_useful_payload(response.payload, name)
            include_payload = is_useful or (
                name in {"profile_and_fees", "price_chart"}
                and isinstance(response.payload, dict)
            )
            if include_payload:
                collected.append(
                    CollectedEndpointPayload(
                        endpoint_name=name,
                        endpoint_family=self.RESULT_ENDPOINT_FAMILIES[name],
                        request_path=endpoint,
                        conid=conid,
                        observed_at=observed_at,
                        payload=response.payload,
                        status_code=response.status_code,
                        is_useful=is_useful,
                    )
                )
            if is_useful:
                self.telemetry.record_useful_payload(
                    self.RESULT_ENDPOINT_FAMILIES[name]
                )

        return FundamentalsConidOutcome(
            conid=conid,
            status="success",
            observed_at=observed_at,
            endpoint_payloads=tuple(collected),
        )

    def persist_outcome(
        self,
        conn: sqlite3.Connection,
        outcome: FundamentalsConidOutcome,
    ) -> FundamentalsPersistResult:
        if outcome.status != "success":
            return FundamentalsPersistResult()

        batch_id = new_capture_batch_id()
        saved_snapshots = 0
        inserted_events = 0
        series_latest_rows_upserted = 0

        for payload in outcome.endpoint_payloads:
            if payload.status_code != 200 or not isinstance(payload.payload, dict):
                continue
            try:
                if payload.endpoint_name == "profile_and_fees":
                    write_profile_and_fees_snapshot(
                        conn,
                        conid=payload.conid,
                        payload=payload.payload,
                        observed_at=payload.observed_at,
                        capture_batch_id=batch_id,
                    )
                    saved_snapshots += 1
                elif payload.endpoint_name == "holdings" and payload.is_useful:
                    write_holdings_snapshot(
                        conn,
                        conid=payload.conid,
                        payload=payload.payload,
                        observed_at=payload.observed_at,
                        capture_batch_id=batch_id,
                    )
                    saved_snapshots += 1
                elif payload.endpoint_name == "ratios" and payload.is_useful:
                    write_ratios_snapshot(
                        conn,
                        conid=payload.conid,
                        payload=payload.payload,
                        observed_at=payload.observed_at,
                        capture_batch_id=batch_id,
                    )
                    saved_snapshots += 1
                elif payload.endpoint_name == "lipper_ratings" and payload.is_useful:
                    write_lipper_ratings_snapshot(
                        conn,
                        conid=payload.conid,
                        payload=payload.payload,
                        observed_at=payload.observed_at,
                        capture_batch_id=batch_id,
                    )
                    saved_snapshots += 1
                elif payload.endpoint_name == "morningstar" and payload.is_useful:
                    write_morningstar_snapshot(
                        conn,
                        conid=payload.conid,
                        payload=payload.payload,
                        observed_at=payload.observed_at,
                        capture_batch_id=batch_id,
                    )
                    saved_snapshots += 1
                elif payload.endpoint_name == "dividends" and payload.is_useful:
                    try:
                        write_dividends_snapshot(
                            conn,
                            conid=payload.conid,
                            payload=payload.payload,
                            observed_at=payload.observed_at,
                            capture_batch_id=batch_id,
                        )
                        saved_snapshots += 1
                    except UnresolvedEffectiveAtError:
                        pass
                    dividend_series_result = write_dividend_events_series(
                        conn,
                        conid=payload.conid,
                        payload=payload.payload,
                        observed_at=payload.observed_at,
                        capture_batch_id=batch_id,
                    )
                    inserted_events += dividend_series_result.rows_inserted
                elif payload.endpoint_name == "price_chart":
                    price_series_result = write_price_chart_series(
                        conn,
                        conid=payload.conid,
                        payload=payload.payload,
                        observed_at=payload.observed_at,
                        capture_batch_id=batch_id,
                    )
                    series_latest_rows_upserted += price_series_result.rows_upserted
            except Exception as exc:
                raise EndpointPersistenceError(payload, exc) from exc

        return FundamentalsPersistResult(
            saved_snapshots=saved_snapshots,
            inserted_events=inserted_events,
            series_latest_rows_upserted=series_latest_rows_upserted,
        )

    def _write_persistence_failure_artifact(
        self,
        *,
        outcome: FundamentalsConidOutcome,
        failing_payload: CollectedEndpointPayload,
        telemetry_output_path: Path | None,
        exc: Exception,
    ) -> Path | None:
        if telemetry_output_path is None:
            return None

        artifact_dir = telemetry_output_path.parent / "persist_failures"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_name = (
            f"{_sanitize_path_segment(outcome.conid)}_"
            f"{_sanitize_path_segment(type(exc).__name__)}_"
            f"{_sanitize_path_segment(new_capture_batch_id())}.json"
        )
        artifact_path = artifact_dir / artifact_name
        payload = {
            "conid": outcome.conid,
            "status": outcome.status,
            "observed_at": outcome.observed_at,
            "skip_reason": outcome.skip_reason,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "failing_endpoint": {
                "endpoint_name": failing_payload.endpoint_name,
                "endpoint_family": failing_payload.endpoint_family,
                "request_path": failing_payload.request_path,
                "conid": failing_payload.conid,
                "observed_at": failing_payload.observed_at,
                "status_code": failing_payload.status_code,
                "is_useful": failing_payload.is_useful,
                "payload": failing_payload.payload,
            },
            "available_endpoints": [
                item.endpoint_name for item in outcome.endpoint_payloads
            ],
        }
        artifact_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str)
        )
        return artifact_path

    async def run(
        self,
        conn: sqlite3.Connection,
        *,
        explicit_conids: Sequence[str] | None = None,
        limit: int | None = None,
        start_index: int = 0,
        skip_conids: Collection[str] | None = None,
        force: bool = False,
        telemetry_output_path: Path | None = None,
        max_auth_retries: int = 1,
        progress: ProgressSink | None = None,
    ) -> FundamentalsCollectionResult:
        if explicit_conids is not None:
            targets = select_explicit_targets(list(explicit_conids))
        else:
            targets = select_governed_targets(conn)

        if not force and skip_conids:
            skipped = {str(conid) for conid in skip_conids}
            targets = [conid for conid in targets if conid not in skipped]

        if start_index:
            targets = targets[start_index:]
        if limit is not None:
            targets = targets[:limit]

        latest_dates = load_latest_price_effective_at_by_conid(conn, conids=targets)
        self._latest_price_effective_at_by_conid.update(latest_dates)

        is_authenticated = await self.session.validate_auth_state()
        if not is_authenticated:
            try:
                is_authenticated = await self.session.login(
                    headless=False,
                    force_browser=True,
                )
            except NotImplementedError:
                is_authenticated = False
        if not is_authenticated:
            return FundamentalsCollectionResult(
                status="auth_required",
                total_targeted_conids=len(targets),
                processed_conids=0,
                saved_snapshots=0,
                inserted_events=0,
                overwritten_events=0,
                unchanged_events=0,
                series_raw_rows_written=0,
                series_latest_rows_upserted=0,
                auth_retries=0,
                aborted=True,
            )

        auth_retries = 0
        processed_conids = 0
        saved_snapshots = 0
        inserted_events = 0
        overwritten_events = 0
        unchanged_events = 0
        series_raw_rows_written = 0
        series_latest_rows_upserted = 0
        aborted = False

        pending = list(targets)
        tracker = (
            progress.stage(
                "Collecting fundamentals",
                total=len(targets),
                unit="conid",
            )
            if progress is not None
            else None
        )
        while pending:
            conid = pending.pop(0)
            async with self.session.get_client() as client:
                outcome = await self.collect_conid(client, conid=conid)
            if outcome.status == "auth_error":
                if auth_retries >= max_auth_retries:
                    aborted = True
                    break
                try:
                    reauthenticated = await self.session.reauthenticate()
                except NotImplementedError:
                    reauthenticated = False
                if not reauthenticated:
                    aborted = True
                    break
                auth_retries += 1
                pending.insert(0, conid)
                if tracker is not None:
                    tracker.advance(step=0, detail=f"reauthenticated for {conid}")
                continue

            try:
                storage_result = self.persist_outcome(conn, outcome)
            except EndpointPersistenceError as exc:
                cause = exc.cause
                if tracker is not None:
                    tracker.close(
                        detail=f"failed at {conid} after {processed_conids} conids"
                    )
                artifact_path = self._write_persistence_failure_artifact(
                    outcome=outcome,
                    failing_payload=exc.payload,
                    telemetry_output_path=telemetry_output_path,
                    exc=cause,
                )
                self.telemetry.record_persistence_failure(
                    conid=exc.payload.conid,
                    endpoint_name=exc.payload.endpoint_name,
                    endpoint_family=exc.payload.endpoint_family,
                    request_path=exc.payload.request_path,
                    observed_at=exc.payload.observed_at,
                    status_code=exc.payload.status_code,
                    is_useful=exc.payload.is_useful,
                    exception_type=type(cause).__name__,
                    exception_message=str(cause),
                    artifact_path=str(artifact_path)
                    if artifact_path is not None
                    else None,
                )
                if telemetry_output_path is not None:
                    self.telemetry.write_report(
                        telemetry_output_path,
                        run_stats={
                            "status": "failed",
                            "failed_conid": conid,
                            "total_targeted_conids": len(targets),
                            "processed_conids": processed_conids,
                            "saved_snapshots": saved_snapshots,
                            "inserted_events": inserted_events,
                            "overwritten_events": overwritten_events,
                            "unchanged_events": unchanged_events,
                            "series_raw_rows_written": series_raw_rows_written,
                            "series_latest_rows_upserted": series_latest_rows_upserted,
                            "auth_retries": auth_retries,
                            "aborted": aborted,
                        },
                    )
                if artifact_path is not None:
                    cause.add_note(f"Persistence failure artifact: {artifact_path}")
                if telemetry_output_path is not None:
                    cause.add_note(f"Telemetry report: {telemetry_output_path}")
                raise cause.with_traceback(cause.__traceback__)
            processed_conids += 1
            saved_snapshots += storage_result.saved_snapshots
            inserted_events += storage_result.inserted_events
            overwritten_events += storage_result.overwritten_events
            unchanged_events += storage_result.unchanged_events
            series_raw_rows_written += storage_result.series_raw_rows_written
            series_latest_rows_upserted += storage_result.series_latest_rows_upserted
            if tracker is not None:
                tracker.advance(
                    detail=(
                        f"{conid} {outcome.status}, {saved_snapshots} snapshots saved"
                    )
                )

        telemetry_path_text: str | None = None
        latest_telemetry_path_text: str | None = None
        result = FundamentalsCollectionResult(
            status="ok" if not aborted else "aborted",
            total_targeted_conids=len(targets),
            processed_conids=processed_conids,
            saved_snapshots=saved_snapshots,
            inserted_events=inserted_events,
            overwritten_events=overwritten_events,
            unchanged_events=unchanged_events,
            series_raw_rows_written=series_raw_rows_written,
            series_latest_rows_upserted=series_latest_rows_upserted,
            auth_retries=auth_retries,
            aborted=aborted,
        )
        if tracker is not None:
            tracker.close(
                detail=(
                    f"{processed_conids}/{len(targets)} conids, "
                    f"{saved_snapshots} snapshots saved"
                )
            )
        if telemetry_output_path is not None:
            telemetry_path, latest_path = self.telemetry.write_report(
                telemetry_output_path,
                run_stats=asdict(result),
            )
            telemetry_path_text = str(telemetry_path)
            latest_telemetry_path_text = str(latest_path)

        return FundamentalsCollectionResult(
            status=result.status,
            total_targeted_conids=result.total_targeted_conids,
            processed_conids=result.processed_conids,
            saved_snapshots=result.saved_snapshots,
            inserted_events=result.inserted_events,
            overwritten_events=result.overwritten_events,
            unchanged_events=result.unchanged_events,
            series_raw_rows_written=result.series_raw_rows_written,
            series_latest_rows_upserted=result.series_latest_rows_upserted,
            auth_retries=result.auth_retries,
            aborted=result.aborted,
            telemetry_path=telemetry_path_text,
            latest_telemetry_path=latest_telemetry_path_text,
        )
