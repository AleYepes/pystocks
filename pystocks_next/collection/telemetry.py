from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EndpointTelemetrySummary:
    endpoint: str
    call_count: int
    useful_payload_count: int
    useful_payload_rate: float
    status_codes: dict[str, int]


@dataclass(slots=True)
class CollectionTelemetry:
    run_started_at: str = field(
        default_factory=lambda: datetime.now(tz=UTC).isoformat()
    )
    endpoint_calls: Counter[str] = field(default_factory=Counter)
    endpoint_useful_payloads: Counter[str] = field(default_factory=Counter)
    status_codes: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )

    def record_call(self, endpoint_family: str, status_code: int) -> None:
        self.endpoint_calls[endpoint_family] += 1
        self.status_codes[endpoint_family][str(status_code)] += 1

    def record_useful_payload(self, endpoint_family: str) -> None:
        self.endpoint_useful_payloads[endpoint_family] += 1

    def build_endpoint_summary(self) -> list[EndpointTelemetrySummary]:
        endpoints = sorted(
            set(self.endpoint_calls)
            | set(self.endpoint_useful_payloads)
            | set(self.status_codes)
        )
        summary: list[EndpointTelemetrySummary] = []
        for endpoint in endpoints:
            call_count = int(self.endpoint_calls[endpoint])
            useful_payload_count = int(self.endpoint_useful_payloads[endpoint])
            useful_payload_rate = (
                useful_payload_count / call_count if call_count > 0 else 0.0
            )
            summary.append(
                EndpointTelemetrySummary(
                    endpoint=endpoint,
                    call_count=call_count,
                    useful_payload_count=useful_payload_count,
                    useful_payload_rate=useful_payload_rate,
                    status_codes=dict(sorted(self.status_codes[endpoint].items())),
                )
            )
        return summary

    def build_report(self, *, run_stats: Mapping[str, object]) -> dict[str, object]:
        return {
            "run_started_at": self.run_started_at,
            "run_stats": dict(run_stats),
            "endpoint_summary": [asdict(row) for row in self.build_endpoint_summary()],
        }

    def write_report(
        self,
        output_path: Path,
        *,
        run_stats: Mapping[str, object],
    ) -> tuple[Path, Path]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        latest_path = output_path.with_name(
            f"{output_path.stem}_latest{output_path.suffix}"
        )
        payload = self.build_report(run_stats=run_stats)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        latest_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return output_path, latest_path
