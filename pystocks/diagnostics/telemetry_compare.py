import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import RESEARCH_DIR


def _load_telemetry(
    path: str | Path,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)

    endpoint_map = {}
    for row in data.get("endpoint_summary", []):
        endpoint = row.get("endpoint")
        if not endpoint:
            continue
        endpoint_map[endpoint] = {
            "call_count": int(row.get("call_count", 0)),
            "useful_payload_count": int(row.get("useful_payload_count", 0)),
            "useful_payload_rate": float(row.get("useful_payload_rate", 0.0)),
        }
    run_stats = data.get("run_stats", {})
    return endpoint_map, run_stats


def compare(
    baseline_path: str | Path,
    candidate_path: str | Path,
    output_csv: str | None = None,
) -> str:
    baseline, baseline_stats = _load_telemetry(baseline_path)
    candidate, candidate_stats = _load_telemetry(candidate_path)

    endpoints = sorted(set(baseline.keys()) | set(candidate.keys()))
    rows = []
    totals = {
        "baseline_calls_total": 0,
        "candidate_calls_total": 0,
        "delta_calls_total": 0,
        "baseline_useful_total": 0,
        "candidate_useful_total": 0,
        "delta_useful_total": 0,
    }
    for endpoint in endpoints:
        b = baseline.get(endpoint, {})
        c = candidate.get(endpoint, {})
        b_calls = int(b.get("call_count", 0))
        c_calls = int(c.get("call_count", 0))
        b_useful = int(b.get("useful_payload_count", 0))
        c_useful = int(c.get("useful_payload_count", 0))
        b_rate = float(b.get("useful_payload_rate", 0.0))
        c_rate = float(c.get("useful_payload_rate", 0.0))
        delta_calls = c_calls - b_calls
        delta_useful = c_useful - b_useful

        totals["baseline_calls_total"] += b_calls
        totals["candidate_calls_total"] += c_calls
        totals["delta_calls_total"] += delta_calls
        totals["baseline_useful_total"] += b_useful
        totals["candidate_useful_total"] += c_useful
        totals["delta_useful_total"] += delta_useful

        rows.append(
            {
                "endpoint": endpoint,
                "baseline_calls": b_calls,
                "candidate_calls": c_calls,
                "delta_calls": delta_calls,
                "baseline_useful_payloads": b_useful,
                "candidate_useful_payloads": c_useful,
                "delta_useful_payloads": delta_useful,
                "baseline_useful_rate": b_rate,
                "candidate_useful_rate": c_rate,
                "delta_useful_rate": c_rate - b_rate,
            }
        )

    df = pd.DataFrame(rows).sort_values("delta_calls")

    output_dir = RESEARCH_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_path = output_dir / f"fundamentals_telemetry_compare_{ts}.csv"

    df.to_csv(output_path, index=False)

    print("--- Telemetry Compare Totals ---")
    for key, value in totals.items():
        print(f"{key}: {value}")

    print("\n--- Run Stats (baseline) ---")
    for key in [
        "total_targeted_conids",
        "processed_conids",
        "saved_snapshots",
        "auth_retries",
        "aborted",
    ]:
        print(f"{key}: {baseline_stats.get(key)}")

    print("\n--- Run Stats (candidate) ---")
    for key in [
        "total_targeted_conids",
        "processed_conids",
        "saved_snapshots",
        "auth_retries",
        "aborted",
    ]:
        print(f"{key}: {candidate_stats.get(key)}")

    print(f"\nWrote endpoint comparison to: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    import fire

    fire.Fire(compare)
