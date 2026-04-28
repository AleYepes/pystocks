from __future__ import annotations

import json
from pathlib import Path

from pystocks_next.collection.telemetry import CollectionTelemetry


def test_write_report_matches_legacy_summary_shape(tmp_path: Path) -> None:
    telemetry = CollectionTelemetry(run_started_at="2026-04-05T15:00:00+00:00")
    telemetry.record_call("mf_holdings", 200)
    telemetry.record_call("mf_holdings", 200)
    telemetry.record_useful_payload("mf_holdings")
    telemetry.record_persistence_failure(
        conid="100",
        endpoint_name="morningstar",
        endpoint_family="mstar",
        request_path="mstar/fund/detail?conid=100",
        observed_at="2026-04-05T15:05:00+00:00",
        status_code=200,
        is_useful=True,
        exception_type="UnresolvedEffectiveAtError",
        exception_message="morningstar_snapshot: source_as_of_date is required",
        artifact_path="/tmp/persist_failure.json",
    )

    report_path, latest_path = telemetry.write_report(
        tmp_path / "fundamentals_run_telemetry.json",
        run_stats={
            "processed_conids": 8,
            "saved_snapshots": 7,
        },
    )

    assert report_path.exists()
    assert latest_path.exists()

    payload = json.loads(report_path.read_text())
    assert payload["run_stats"]["processed_conids"] == 8
    assert payload["endpoint_summary"] == [
        {
            "endpoint": "mf_holdings",
            "call_count": 2,
            "useful_payload_count": 1,
            "useful_payload_rate": 0.5,
            "status_codes": {"200": 2},
        }
    ]
    assert payload["persistence_failures"] == [
        {
            "artifact_path": "/tmp/persist_failure.json",
            "conid": "100",
            "endpoint_family": "mstar",
            "endpoint_name": "morningstar",
            "exception_message": "morningstar_snapshot: source_as_of_date is required",
            "exception_type": "UnresolvedEffectiveAtError",
            "is_useful": True,
            "observed_at": "2026-04-05T15:05:00+00:00",
            "request_path": "mstar/fund/detail?conid=100",
            "status_code": 200,
        }
    ]
