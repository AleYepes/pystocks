import json
from collections import Counter, defaultdict

from pystocks.ingest.fundamentals import FundamentalScraper


def test_save_telemetry_writes_json_without_sqlite_persistence(tmp_path):
    scraper = FundamentalScraper.__new__(FundamentalScraper)
    scraper.telemetry = {
        "run_started_at": "2026-04-05T15:00:00+00:00",
        "endpoint_calls": Counter({"mf_holdings": 2}),
        "endpoint_useful_payloads": Counter({"mf_holdings": 1}),
        "status_codes": defaultdict(Counter, {"mf_holdings": Counter({"200": 2})}),
    }
    scraper.research_dir = tmp_path
    scraper.store = object()

    telemetry_path, latest_path = scraper.save_telemetry(
        total_targeted=10,
        processed_conids=8,
        saved_snapshots=7,
        inserted_events=5,
        overwritten_events=1,
        unchanged_events=1,
        series_raw_rows_written=2,
        series_latest_rows_upserted=2,
        auth_retries=0,
        aborted=False,
    )

    assert telemetry_path.exists()
    assert latest_path.exists()

    payload = json.loads(telemetry_path.read_text())
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
