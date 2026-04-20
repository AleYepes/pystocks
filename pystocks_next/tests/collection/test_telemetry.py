from __future__ import annotations

import json
from pathlib import Path

from pystocks_next.collection.telemetry import CollectionTelemetry


def test_write_report_matches_legacy_summary_shape(tmp_path: Path) -> None:
    telemetry = CollectionTelemetry(run_started_at="2026-04-05T15:00:00+00:00")
    telemetry.record_call("mf_holdings", 200)
    telemetry.record_call("mf_holdings", 200)
    telemetry.record_useful_payload("mf_holdings")

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
