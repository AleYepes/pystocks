import hashlib
import json
import logging
import re
import shutil
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import zstandard as zstd
from tqdm import tqdm

from .config import (
    FUNDAMENTALS_BLOBS_DIR,
    FUNDAMENTALS_DIR,
    FUNDAMENTALS_DUCKDB_PATH,
    FUNDAMENTALS_EVENTS_DB_PATH,
    FUNDAMENTALS_PARQUET_DIR,
)
from .fundamentals_normalizers import (
    extract_dividends_events,
    extract_dividends_industry_metrics,
    extract_ownership_trade_log,
    normalize_dividends_snapshot,
    normalize_ownership_snapshot,
)

logger = logging.getLogger(__name__)


ENDPOINT_KEYS = [
    "landing",
    "profile_and_fees",
    "holdings",
    "ratios",
    "lipper_ratings",
    "dividends",
    "morningstar",
    "performance",
    "risk_stats",
    "sentiment_search",
    "ownership",
    "sentiment",
    "esg",
]

COMPLEX_ENDPOINTS = {"dividends", "ownership"}
COMPLEX_ENDPOINT_ANALYTICS = {
    "dividends": ["dividends_events", "dividends_industry_metrics"],
    "ownership": ["ownership_trade_log"],
}

ANALYTICS_BASE_VIEW_SCHEMA = [
    ("analytics_row_id", "BIGINT"),
    ("analytics_name", "VARCHAR"),
    ("row_key", "VARCHAR"),
    ("endpoint_event_id", "BIGINT"),
    ("endpoint", "VARCHAR"),
    ("conid", "VARCHAR"),
    ("observed_at", "VARCHAR"),
    ("effective_at", "VARCHAR"),
    ("payload_hash", "VARCHAR"),
    ("source_file", "VARCHAR"),
    ("inserted_at", "VARCHAR"),
]

ANALYTICS_VIEW_EXTRAS = {
    "dividends_events": [
        ("event_date", "VARCHAR"),
        ("amount", "DOUBLE"),
        ("currency", "VARCHAR"),
        ("description", "VARCHAR"),
        ("event_type", "VARCHAR"),
        ("declaration_date", "VARCHAR"),
        ("record_date", "VARCHAR"),
        ("payment_date", "VARCHAR"),
    ],
    "dividends_industry_metrics": [
        ("metric_id", "VARCHAR"),
        ("value", "DOUBLE"),
        ("formatted_value", "VARCHAR"),
    ],
    "ownership_trade_log": [
        ("trade_date", "VARCHAR"),
        ("action", "VARCHAR"),
        ("shares", "DOUBLE"),
        ("value", "DOUBLE"),
        ("holding", "DOUBLE"),
        ("party", "VARCHAR"),
        ("source", "VARCHAR"),
        ("insider", "VARCHAR"),
    ],
}

# Use discrete endpoint dates only. Do not derive effective date from
# chart/time-series payload internals.
DISCRETE_ENDPOINTS = {
    "profile_and_fees",
    "holdings",
    "ratios",
    "lipper_ratings",
    "dividends",
    "morningstar",
    "ownership",
    "esg",
}


def _sanitize_segment(value):
    segment = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return segment or "field"


def _slugify_endpoint(endpoint):
    return _sanitize_segment(endpoint.replace("/", "_").replace("?", "_").replace("=", "_"))


def _canonical_json_bytes(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _parse_date_candidate(value):
    if value is None:
        return None

    if isinstance(value, (int, float)):
        ts = float(value)
        if ts <= 0:
            return None
        iv = int(ts)
        if 19000101 <= iv <= 29991231:
            s = str(iv)
            try:
                return datetime.strptime(s, "%Y%m%d").date()
            except Exception:
                pass
        if ts > 1e12:
            ts = ts / 1000.0
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc).date()
        except Exception:
            return None

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if len(s) == 8 and s.isdigit():
            try:
                return datetime.strptime(s, "%Y%m%d").date()
            except Exception:
                pass
        if s.isdigit():
            return _parse_date_candidate(int(s))
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
        except Exception:
            pass
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(s, fmt).date()
            except Exception:
                continue
    return None


def _extract_as_of_date(payload):
    if not isinstance(payload, dict):
        return None

    for key in ("as_of_date", "asOfDate"):
        if key in payload:
            parsed = _parse_date_candidate(payload.get(key))
            if parsed:
                return parsed
    return None


def _collect_nested_as_of_dates(payload):
    dates = []

    def walk(node):
        if isinstance(node, dict):
            for key in ("as_of_date", "asOfDate"):
                if key in node:
                    parsed = _parse_date_candidate(node.get(key))
                    if parsed is not None:
                        dates.append(parsed)
            for value in node.values():
                walk(value)
            return
        if isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)
    return dates


_DATE_IN_TEXT_RE = re.compile(r"(?<!\d)(\d{4}[/-]\d{2}[/-]\d{2})(?!\d)")


def _parse_ymd_text(value):
    if not value:
        return None
    m = _DATE_IN_TEXT_RE.search(str(value))
    if not m:
        return None
    return _parse_date_candidate(m.group(1))


def _extract_profile_embedded_dates(payload):
    if not isinstance(payload, dict):
        return []
    out = []
    for row in payload.get("fund_and_profile", []) or []:
        if not isinstance(row, dict):
            continue
        parsed = _parse_ymd_text(row.get("value"))
        if parsed is not None:
            out.append(parsed)
    return out


def _extract_morningstar_publish_dates(payload):
    out = []

    def walk(node):
        if isinstance(node, dict):
            for key in ("publish_date", "publishDate"):
                if key in node:
                    parsed = _parse_date_candidate(node.get(key))
                    if parsed is not None:
                        out.append(parsed)
            for value in node.values():
                walk(value)
            return
        if isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)
    return out


def _empty_select_sql(columns):
    projection = ",\n                        ".join(
        f"CAST(NULL AS {column_type}) AS {column_name}" for column_name, column_type in columns
    )
    return f"SELECT\n                        {projection}\n                    WHERE FALSE"


def _extract_endpoint_effective_date(endpoint, payload):
    if not isinstance(payload, (dict, list)):
        return None

    # Morningstar payloads can carry stale top-level as_of_date while section-level
    # publish_date reflects the UI-visible report recency. Prefer publish_date.
    if endpoint == "morningstar" and isinstance(payload, dict):
        morningstar_publish_dates = _extract_morningstar_publish_dates(payload)
        if morningstar_publish_dates:
            return max(morningstar_publish_dates)

    if isinstance(payload, dict):
        direct = _extract_as_of_date(payload)
        if direct is not None:
            return direct

    nested_as_of = _collect_nested_as_of_dates(payload)
    candidates = list(nested_as_of)

    if endpoint == "profile_and_fees":
        candidates.extend(_extract_profile_embedded_dates(payload))

    if candidates:
        return max(candidates)

    return None


def _normalize_scalar(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _flatten_payload_scalars(payload):
    flat = {}

    def walk(value, prefix):
        if isinstance(value, dict):
            for k, v in value.items():
                child = _sanitize_segment(k)
                child_prefix = f"{prefix}__{child}" if prefix else child
                walk(v, child_prefix)
            return

        if isinstance(value, list):
            if prefix:
                flat[f"f__{prefix}__len"] = len(value)
                if value and all(not isinstance(v, (dict, list)) for v in value):
                    flat[f"f__{prefix}__json"] = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
            return

        if prefix:
            flat[f"f__{prefix}"] = _normalize_scalar(value)

    walk(payload, "")
    return flat


class FundamentalsStore:
    """
    Storage backend:
    - CAS compressed blobs for full raw endpoint payloads.
    - Endpoint-partitioned Parquet event rows for fast analytics.
    - Analytics dataset-partitioned Parquet rows for normalized time-series features.
    - SQLite manifest for dedupe/index.
    - DuckDB views over Parquet partitions.
    """

    def __init__(
        self,
        fundamentals_dir=FUNDAMENTALS_DIR,
        blobs_dir=FUNDAMENTALS_BLOBS_DIR,
        parquet_dir=FUNDAMENTALS_PARQUET_DIR,
        events_db_path=FUNDAMENTALS_EVENTS_DB_PATH,
        duckdb_path=FUNDAMENTALS_DUCKDB_PATH,
    ):
        self.fundamentals_dir = Path(fundamentals_dir)
        self.blobs_dir = Path(blobs_dir)
        self.parquet_dir = Path(parquet_dir)
        self.events_db_path = Path(events_db_path)
        self.duckdb_path = Path(duckdb_path)

        self.fundamentals_dir.mkdir(parents=True, exist_ok=True)
        self.blobs_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)

        self._compressor = zstd.ZstdCompressor(level=10)
        self._decompressor = zstd.ZstdDecompressor()
        self._init_manifest_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.events_db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_manifest_db(self):
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS blobs (
                    hash TEXT PRIMARY KEY,
                    blob_path TEXT NOT NULL UNIQUE,
                    raw_size_bytes INTEGER NOT NULL,
                    compressed_size_bytes INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS endpoint_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conid TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    endpoint_slug TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    effective_source TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    blob_path TEXT NOT NULL,
                    payload_size_raw INTEGER NOT NULL,
                    payload_size_compressed INTEGER NOT NULL,
                    source_file TEXT,
                    inserted_at TEXT NOT NULL,
                    UNIQUE(conid, endpoint, effective_at, payload_hash)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS analytics_rows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analytics_name TEXT NOT NULL,
                    row_key TEXT NOT NULL,
                    endpoint_event_id INTEGER NOT NULL,
                    endpoint TEXT NOT NULL,
                    conid TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    parquet_path TEXT,
                    source_file TEXT,
                    inserted_at TEXT NOT NULL,
                    UNIQUE(analytics_name, row_key, payload_hash)
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_endpoint_events_endpoint ON endpoint_events(endpoint)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_endpoint_events_effective_at ON endpoint_events(effective_at)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_endpoint_events_conid_endpoint ON endpoint_events(conid, endpoint)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_analytics_rows_name ON analytics_rows(analytics_name)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_analytics_rows_event_id ON analytics_rows(endpoint_event_id)"
            )
            conn.commit()

    def _blob_path_for_hash(self, hash_hex):
        return self.blobs_dir / hash_hex[:2] / hash_hex[2:4] / f"{hash_hex}.json.zst"

    def _store_blob(self, payload):
        raw_bytes = _canonical_json_bytes(payload)
        payload_hash = hashlib.sha256(raw_bytes).hexdigest()
        blob_path = self._blob_path_for_hash(payload_hash)
        blob_path.parent.mkdir(parents=True, exist_ok=True)

        if not blob_path.exists():
            compressed = self._compressor.compress(raw_bytes)
            with open(blob_path, "wb") as f:
                f.write(compressed)
            compressed_size = len(compressed)
        else:
            compressed_size = blob_path.stat().st_size

        raw_size = len(raw_bytes)
        now_iso = datetime.now(timezone.utc).isoformat()

        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO blobs (
                    hash, blob_path, raw_size_bytes, compressed_size_bytes, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (payload_hash, str(blob_path), raw_size, compressed_size, now_iso),
            )
            conn.commit()

        return {
            "hash": payload_hash,
            "blob_path": str(blob_path),
            "raw_size": raw_size,
            "compressed_size": compressed_size,
        }

    def _load_blob_payload(self, blob_path):
        p = Path(blob_path)
        with open(p, "rb") as f:
            raw = self._decompressor.decompress(f.read())
        return json.loads(raw.decode("utf-8"))

    def _resolve_effective_dates(self, endpoint_payloads, observed_at):
        discrete_dates = {}
        for endpoint in DISCRETE_ENDPOINTS:
            payload = endpoint_payloads.get(endpoint)
            if payload is None:
                continue
            date_value = _extract_endpoint_effective_date(endpoint, payload)
            if date_value is not None:
                discrete_dates[endpoint] = date_value

        ratios_date = discrete_dates.get("ratios")
        modal_discrete = None
        if discrete_dates:
            counts = Counter(discrete_dates.values())
            top_n = max(counts.values())
            modal_discrete = max(d for d, c in counts.items() if c == top_n)

        resolved = {}
        for endpoint in endpoint_payloads.keys():
            own = discrete_dates.get(endpoint)
            if own is not None:
                resolved[endpoint] = (own.isoformat(), f"{endpoint}.own_date")
                continue
            if ratios_date is not None:
                resolved[endpoint] = (ratios_date.isoformat(), "ratios.as_of_date_fallback")
                continue
            if modal_discrete is not None:
                resolved[endpoint] = (modal_discrete.isoformat(), "modal_discrete_as_of_fallback")
                continue
            resolved[endpoint] = (observed_at.date().isoformat(), "observed_at_fallback")

        return resolved

    def _endpoint_payloads_from_snapshot(self, snapshot):
        payloads = {}
        for endpoint in ENDPOINT_KEYS:
            value = snapshot.get(endpoint)
            if isinstance(value, (dict, list)):
                payloads[endpoint] = value
        return payloads

    def _write_endpoint_event_parquet(self, row, endpoint_slug, effective_at):
        dt = datetime.fromisoformat(f"{effective_at}T00:00:00+00:00")
        partition_dir = (
            self.parquet_dir
            / f"endpoint={endpoint_slug}"
            / f"year={dt.year:04d}"
            / f"month={dt.month:02d}"
        )
        partition_dir.mkdir(parents=True, exist_ok=True)

        file_name = f"{effective_at}_{row['conid']}_{row['payload_hash'][:12]}_{row['event_id']}.parquet"
        file_path = partition_dir / file_name
        pd.DataFrame([row]).to_parquet(file_path, index=False, engine="pyarrow", compression="zstd")
        return file_path

    def _partition_date_for_analytics(self, analytics_name, row, lineage_meta):
        candidate = None
        if analytics_name == "dividends_events":
            candidate = row.get("event_date")
        elif analytics_name == "ownership_trade_log":
            candidate = row.get("trade_date")
        elif analytics_name == "dividends_industry_metrics":
            candidate = row.get("metric_date")

        parsed = _parse_date_candidate(candidate)
        if parsed is not None:
            return parsed.isoformat()

        fallback = _parse_date_candidate(lineage_meta.get("effective_at"))
        if fallback is not None:
            return fallback.isoformat()
        return datetime.now(timezone.utc).date().isoformat()

    def _write_analytics_row_parquet(self, row, analytics_slug, partition_date, row_id):
        dt = datetime.fromisoformat(f"{partition_date}T00:00:00+00:00")
        partition_dir = (
            self.parquet_dir
            / f"analytics={analytics_slug}"
            / f"year={dt.year:04d}"
            / f"month={dt.month:02d}"
        )
        partition_dir.mkdir(parents=True, exist_ok=True)

        file_name = (
            f"{partition_date}_{row['conid']}_{row['payload_hash'][:12]}_"
            f"{row['row_key'][:12]}_{row_id}.parquet"
        )
        file_path = partition_dir / file_name
        pd.DataFrame([row]).to_parquet(file_path, index=False, engine="pyarrow", compression="zstd")
        return file_path

    def _compute_row_key(self, analytics_name, row, lineage_meta):
        if row.get("row_key"):
            return str(row.get("row_key"))

        conid = str(lineage_meta.get("conid"))
        payload_hash = str(lineage_meta.get("payload_hash"))

        if analytics_name == "dividends_events":
            basis = [
                conid,
                row.get("event_date"),
                row.get("amount"),
                row.get("description"),
                payload_hash,
            ]
        elif analytics_name == "ownership_trade_log":
            basis = [
                conid,
                row.get("trade_date"),
                row.get("party"),
                row.get("action"),
                row.get("shares"),
                row.get("value"),
                payload_hash,
            ]
        elif analytics_name == "dividends_industry_metrics":
            basis = [
                conid,
                row.get("metric_id"),
                row.get("value"),
                lineage_meta.get("effective_at"),
                payload_hash,
            ]
        else:
            basis = [conid, json.dumps(row, sort_keys=True, separators=(",", ":")), payload_hash]

        material = "|".join(str(x) for x in basis)
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _build_endpoint_analytics_row(self, endpoint, payload, base_row):
        row = dict(base_row)

        if endpoint == "dividends":
            canonical = normalize_dividends_snapshot(payload)
            for key, value in canonical.items():
                row[f"a__{_sanitize_segment(key)}"] = _normalize_scalar(value)
            return row

        if endpoint == "ownership":
            canonical = normalize_ownership_snapshot(payload)
            for key, value in canonical.items():
                row[f"a__{_sanitize_segment(key)}"] = _normalize_scalar(value)
            return row

        row.update(_flatten_payload_scalars(payload))
        return row

    def persist_analytics_rows(self, analytics_name, rows, lineage_meta, source_file=None):
        if not rows:
            return {
                "analytics_name": analytics_name,
                "inserted_rows": 0,
                "duplicate_rows": 0,
            }

        inserted_rows = 0
        duplicate_rows = 0
        analytics_slug = _slugify_endpoint(analytics_name)

        for row in rows:
            row = row if isinstance(row, dict) else {"value": row}
            row_key = self._compute_row_key(analytics_name, row, lineage_meta)
            now_iso = datetime.now(timezone.utc).isoformat()

            with self._get_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO analytics_rows (
                        analytics_name, row_key, endpoint_event_id, endpoint, conid,
                        observed_at, effective_at, payload_hash, parquet_path,
                        source_file, inserted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(analytics_name, row_key, payload_hash) DO NOTHING
                    """,
                    (
                        analytics_name,
                        row_key,
                        int(lineage_meta["endpoint_event_id"]),
                        lineage_meta["endpoint"],
                        str(lineage_meta["conid"]),
                        lineage_meta["observed_at"],
                        lineage_meta["effective_at"],
                        lineage_meta["payload_hash"],
                        None,
                        source_file,
                        now_iso,
                    ),
                )
                inserted = cur.rowcount > 0
                row_id = cur.lastrowid
                conn.commit()

            if not inserted:
                duplicate_rows += 1
                continue

            full_row = {
                "analytics_row_id": row_id,
                "analytics_name": analytics_name,
                "row_key": row_key,
                "endpoint_event_id": int(lineage_meta["endpoint_event_id"]),
                "endpoint": lineage_meta["endpoint"],
                "conid": str(lineage_meta["conid"]),
                "observed_at": lineage_meta["observed_at"],
                "effective_at": lineage_meta["effective_at"],
                "payload_hash": lineage_meta["payload_hash"],
                "source_file": source_file,
                "inserted_at": now_iso,
            }
            for key, value in row.items():
                full_row[_sanitize_segment(key)] = _normalize_scalar(value)

            partition_date = self._partition_date_for_analytics(analytics_name, full_row, lineage_meta)
            parquet_path = self._write_analytics_row_parquet(
                full_row,
                analytics_slug=analytics_slug,
                partition_date=partition_date,
                row_id=row_id,
            )

            with self._get_conn() as conn:
                conn.execute(
                    "UPDATE analytics_rows SET parquet_path = ? WHERE id = ?",
                    (str(parquet_path), row_id),
                )
                conn.commit()

            inserted_rows += 1

        return {
            "analytics_name": analytics_name,
            "inserted_rows": inserted_rows,
            "duplicate_rows": duplicate_rows,
        }

    def _persist_complex_endpoint_analytics(self, endpoint, payload, lineage_meta, source_file=None):
        analytics_payloads = []
        if endpoint == "dividends":
            analytics_payloads.append(("dividends_events", extract_dividends_events(payload)))
            analytics_payloads.append(("dividends_industry_metrics", extract_dividends_industry_metrics(payload)))
        elif endpoint == "ownership":
            analytics_payloads.append(("ownership_trade_log", extract_ownership_trade_log(payload, drop_no_change=True)))

        inserted_rows = 0
        duplicate_rows = 0
        per_dataset = []

        for analytics_name, rows in analytics_payloads:
            result = self.persist_analytics_rows(
                analytics_name=analytics_name,
                rows=rows,
                lineage_meta=lineage_meta,
                source_file=source_file,
            )
            per_dataset.append(result)
            inserted_rows += int(result.get("inserted_rows", 0))
            duplicate_rows += int(result.get("duplicate_rows", 0))

        return {
            "analytics_rows_inserted": inserted_rows,
            "analytics_rows_duplicate": duplicate_rows,
            "analytics_datasets": per_dataset,
        }

    def persist_endpoint_payload(
        self,
        conid,
        endpoint,
        payload,
        observed_at,
        effective_at,
        effective_source,
        source_file=None,
    ):
        blob_info = self._store_blob(payload)
        endpoint_slug = _slugify_endpoint(endpoint)
        now_iso = datetime.now(timezone.utc).isoformat()

        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO endpoint_events (
                    conid, endpoint, endpoint_slug, observed_at, effective_at, effective_source,
                    payload_hash, blob_path, payload_size_raw, payload_size_compressed,
                    source_file, inserted_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(conid, endpoint, effective_at, payload_hash) DO NOTHING
                """,
                (
                    str(conid),
                    endpoint,
                    endpoint_slug,
                    observed_at,
                    effective_at,
                    effective_source,
                    blob_info["hash"],
                    blob_info["blob_path"],
                    blob_info["raw_size"],
                    blob_info["compressed_size"],
                    source_file,
                    now_iso,
                ),
            )
            inserted = cur.rowcount > 0
            event_id = cur.lastrowid
            conn.commit()

        if not inserted:
            return {
                "inserted": False,
                "duplicate": True,
                "endpoint": endpoint,
                "analytics_rows_inserted": 0,
                "analytics_rows_duplicate": 0,
                "analytics_datasets": [],
            }

        base_row = {
            "event_id": event_id,
            "conid": str(conid),
            "endpoint": endpoint,
            "endpoint_slug": endpoint_slug,
            "observed_at": observed_at,
            "effective_at": effective_at,
            "effective_source": effective_source,
            "payload_hash": blob_info["hash"],
            "blob_path": blob_info["blob_path"],
            "payload_size_raw": blob_info["raw_size"],
            "payload_size_compressed": blob_info["compressed_size"],
            "source_file": source_file,
            "inserted_at": now_iso,
        }

        endpoint_row = self._build_endpoint_analytics_row(endpoint, payload, base_row)
        parquet_path = self._write_endpoint_event_parquet(endpoint_row, endpoint_slug, effective_at)

        analytics_rows_inserted = 0
        analytics_rows_duplicate = 0
        analytics_datasets = []

        if endpoint in COMPLEX_ENDPOINTS:
            analytics_result = self._persist_complex_endpoint_analytics(
                endpoint=endpoint,
                payload=payload,
                lineage_meta={
                    "endpoint_event_id": event_id,
                    "endpoint": endpoint,
                    "conid": str(conid),
                    "observed_at": observed_at,
                    "effective_at": effective_at,
                    "payload_hash": blob_info["hash"],
                },
                source_file=source_file,
            )
            analytics_rows_inserted = int(analytics_result.get("analytics_rows_inserted", 0))
            analytics_rows_duplicate = int(analytics_result.get("analytics_rows_duplicate", 0))
            analytics_datasets = analytics_result.get("analytics_datasets", [])

        return {
            "inserted": True,
            "duplicate": False,
            "endpoint": endpoint,
            "parquet_path": str(parquet_path),
            "analytics_rows_inserted": analytics_rows_inserted,
            "analytics_rows_duplicate": analytics_rows_duplicate,
            "analytics_datasets": analytics_datasets,
        }

    def persist_combined_snapshot(self, snapshot, source_file=None, refresh_duckdb=False):
        if not isinstance(snapshot, dict):
            return {"inserted_events": 0, "duplicate_events": 0, "status": "invalid_snapshot"}

        conid = str(snapshot.get("conid", "")).strip()
        if not conid:
            return {"inserted_events": 0, "duplicate_events": 0, "status": "missing_conid"}

        observed_at_raw = snapshot.get("scraped_at")
        observed_dt = None
        if isinstance(observed_at_raw, str):
            try:
                observed_dt = datetime.fromisoformat(observed_at_raw.replace("Z", "+00:00"))
            except Exception:
                observed_dt = None
        if observed_dt is None:
            observed_dt = datetime.now(timezone.utc)
        observed_iso = observed_dt.astimezone(timezone.utc).isoformat()

        endpoint_payloads = self._endpoint_payloads_from_snapshot(snapshot)
        if not endpoint_payloads:
            return {"inserted_events": 0, "duplicate_events": 0, "status": "no_endpoint_payloads"}

        effective_map = self._resolve_effective_dates(endpoint_payloads, observed_dt)

        inserted_events = 0
        duplicate_events = 0
        analytics_rows_inserted = 0
        analytics_rows_duplicate = 0
        per_endpoint = []

        for endpoint, payload in endpoint_payloads.items():
            effective_at, effective_source = effective_map[endpoint]
            result = self.persist_endpoint_payload(
                conid=conid,
                endpoint=endpoint,
                payload=payload,
                observed_at=observed_iso,
                effective_at=effective_at,
                effective_source=effective_source,
                source_file=source_file,
            )
            per_endpoint.append(result)
            analytics_rows_inserted += int(result.get("analytics_rows_inserted", 0))
            analytics_rows_duplicate += int(result.get("analytics_rows_duplicate", 0))

            if result.get("inserted"):
                inserted_events += 1
            elif result.get("duplicate"):
                duplicate_events += 1

        if refresh_duckdb:
            self.refresh_duckdb_views()

        return {
            "inserted_events": inserted_events,
            "duplicate_events": duplicate_events,
            "analytics_rows_inserted": analytics_rows_inserted,
            "analytics_rows_duplicate": analytics_rows_duplicate,
            "status": "ok",
            "per_endpoint": per_endpoint,
        }

    def refresh_duckdb_views(self):
        endpoint_slugs = []
        analytics_names = []
        known_analytics_names = sorted({name for names in COMPLEX_ENDPOINT_ANALYTICS.values() for name in names})

        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT endpoint_slug FROM endpoint_events ORDER BY endpoint_slug")
            endpoint_slugs = [row[0] for row in cur.fetchall()]

            cur.execute("SELECT DISTINCT analytics_name FROM analytics_rows ORDER BY analytics_name")
            analytics_names = [row[0] for row in cur.fetchall()]

        endpoint_files = list(self.parquet_dir.glob("endpoint=*/year=*/month=*/*.parquet"))
        analytics_files = list(self.parquet_dir.glob("analytics=*/year=*/month=*/*.parquet"))

        db = duckdb.connect(str(self.duckdb_path))
        try:
            if endpoint_files:
                all_pattern = f"{self.parquet_dir.as_posix()}/endpoint=*/year=*/month=*/*.parquet"
                db.execute(
                    f"""
                    CREATE OR REPLACE VIEW endpoint_events_all AS
                    SELECT * FROM read_parquet('{all_pattern}', union_by_name=true)
                    """
                )
            else:
                db.execute(
                    """
                    CREATE OR REPLACE VIEW endpoint_events_all AS
                    SELECT
                        CAST(NULL AS BIGINT) AS event_id,
                        CAST(NULL AS VARCHAR) AS conid,
                        CAST(NULL AS VARCHAR) AS endpoint,
                        CAST(NULL AS VARCHAR) AS endpoint_slug,
                        CAST(NULL AS VARCHAR) AS observed_at,
                        CAST(NULL AS VARCHAR) AS effective_at,
                        CAST(NULL AS VARCHAR) AS effective_source,
                        CAST(NULL AS VARCHAR) AS payload_hash,
                        CAST(NULL AS VARCHAR) AS blob_path,
                        CAST(NULL AS BIGINT) AS payload_size_raw,
                        CAST(NULL AS BIGINT) AS payload_size_compressed,
                        CAST(NULL AS VARCHAR) AS source_file,
                        CAST(NULL AS VARCHAR) AS inserted_at
                    WHERE FALSE
                    """
                )

            for slug in endpoint_slugs:
                view_name = f"endpoint_{slug}"
                pattern_path = self.parquet_dir / f"endpoint={slug}"
                if list(pattern_path.glob("year=*/month=*/*.parquet")):
                    pattern = f"{self.parquet_dir.as_posix()}/endpoint={slug}/year=*/month=*/*.parquet"
                    db.execute(
                        f"""
                        CREATE OR REPLACE VIEW {view_name} AS
                        SELECT * FROM read_parquet('{pattern}', union_by_name=true)
                        """
                    )
                else:
                    db.execute(
                        f"""
                        CREATE OR REPLACE VIEW {view_name} AS
                        SELECT * FROM endpoint_events_all WHERE FALSE
                        """
                    )

            db.execute(
                """
                CREATE OR REPLACE VIEW endpoint_catalog AS
                SELECT endpoint, endpoint_slug, COUNT(*) AS n_events,
                       MIN(effective_at) AS min_effective_at,
                       MAX(effective_at) AS max_effective_at
                FROM endpoint_events_all
                GROUP BY endpoint, endpoint_slug
                ORDER BY endpoint
                """
            )

            if analytics_files:
                analytics_pattern = f"{self.parquet_dir.as_posix()}/analytics=*/year=*/month=*/*.parquet"
                db.execute(
                    f"""
                    CREATE OR REPLACE VIEW analytics_rows_all AS
                    SELECT * FROM read_parquet('{analytics_pattern}', union_by_name=true)
                    """
                )
            else:
                db.execute(
                    """
                    CREATE OR REPLACE VIEW analytics_rows_all AS
                    SELECT
                        CAST(NULL AS BIGINT) AS analytics_row_id,
                        CAST(NULL AS VARCHAR) AS analytics_name,
                        CAST(NULL AS VARCHAR) AS row_key,
                        CAST(NULL AS BIGINT) AS endpoint_event_id,
                        CAST(NULL AS VARCHAR) AS endpoint,
                        CAST(NULL AS VARCHAR) AS conid,
                        CAST(NULL AS VARCHAR) AS observed_at,
                        CAST(NULL AS VARCHAR) AS effective_at,
                        CAST(NULL AS VARCHAR) AS payload_hash,
                        CAST(NULL AS VARCHAR) AS source_file,
                        CAST(NULL AS VARCHAR) AS inserted_at
                    WHERE FALSE
                    """
                )

            analytics_view_names = sorted(set(analytics_names).union(known_analytics_names))
            for analytics_name in analytics_view_names:
                slug = _slugify_endpoint(analytics_name)
                view_name = f"analytics_{slug}"
                pattern_path = self.parquet_dir / f"analytics={slug}"
                if list(pattern_path.glob("year=*/month=*/*.parquet")):
                    pattern = f"{self.parquet_dir.as_posix()}/analytics={slug}/year=*/month=*/*.parquet"
                    db.execute(
                        f"""
                        CREATE OR REPLACE VIEW {view_name} AS
                        SELECT * FROM read_parquet('{pattern}', union_by_name=true)
                        """
                    )
                else:
                    schema = ANALYTICS_BASE_VIEW_SCHEMA + ANALYTICS_VIEW_EXTRAS.get(analytics_name, [])
                    empty_sql = _empty_select_sql(schema)
                    db.execute(
                        f"""
                        CREATE OR REPLACE VIEW {view_name} AS
                        {empty_sql}
                        """
                    )

            db.execute(
                """
                CREATE OR REPLACE VIEW analytics_catalog AS
                SELECT analytics_name, COUNT(*) AS n_rows,
                       MIN(effective_at) AS min_effective_at,
                       MAX(effective_at) AS max_effective_at
                FROM analytics_rows_all
                GROUP BY analytics_name
                ORDER BY analytics_name
                """
            )
        finally:
            db.close()

        return {
            "endpoint_views": len(endpoint_slugs),
            "analytics_views": len(set(analytics_names).union(known_analytics_names)),
            "duckdb_path": str(self.duckdb_path),
        }

    def _rewrite_complex_endpoint_parquet_row(self, event_row, payload):
        base_row = {
            "event_id": int(event_row["id"]),
            "conid": str(event_row["conid"]),
            "endpoint": event_row["endpoint"],
            "endpoint_slug": event_row["endpoint_slug"],
            "observed_at": event_row["observed_at"],
            "effective_at": event_row["effective_at"],
            "effective_source": event_row["effective_source"],
            "payload_hash": event_row["payload_hash"],
            "blob_path": event_row["blob_path"],
            "payload_size_raw": event_row["payload_size_raw"],
            "payload_size_compressed": event_row["payload_size_compressed"],
            "source_file": event_row["source_file"],
            "inserted_at": event_row["inserted_at"],
        }
        endpoint_row = self._build_endpoint_analytics_row(event_row["endpoint"], payload, base_row)
        self._write_endpoint_event_parquet(endpoint_row, event_row["endpoint_slug"], event_row["effective_at"])

    def rebuild_complex_analytics(self, endpoints=None, refresh_duckdb=True):
        if endpoints is None:
            endpoints = sorted(COMPLEX_ENDPOINTS)
        elif isinstance(endpoints, str):
            endpoints = [e.strip() for e in endpoints.split(",") if e.strip()]

        endpoints = [e for e in endpoints if e in COMPLEX_ENDPOINTS]
        if not endpoints:
            return {
                "status": "no_endpoints",
                "rewritten_endpoint_rows": 0,
                "inserted_analytics_rows": 0,
                "duplicate_analytics_rows": 0,
                "failed": 0,
            }

        analytics_names = sorted({name for ep in endpoints for name in COMPLEX_ENDPOINT_ANALYTICS.get(ep, [])})

        for endpoint in endpoints:
            endpoint_dir = self.parquet_dir / f"endpoint={_slugify_endpoint(endpoint)}"
            if endpoint_dir.exists():
                shutil.rmtree(endpoint_dir)

        for analytics_name in analytics_names:
            analytics_dir = self.parquet_dir / f"analytics={_slugify_endpoint(analytics_name)}"
            if analytics_dir.exists():
                shutil.rmtree(analytics_dir)

        if analytics_names:
            placeholders = ",".join("?" for _ in analytics_names)
            with self._get_conn() as conn:
                conn.execute(
                    f"DELETE FROM analytics_rows WHERE analytics_name IN ({placeholders})",
                    tuple(analytics_names),
                )
                conn.commit()

        placeholders = ",".join("?" for _ in endpoints)
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT id, conid, endpoint, endpoint_slug, observed_at, effective_at,
                       effective_source, payload_hash, blob_path, payload_size_raw,
                       payload_size_compressed, source_file, inserted_at
                FROM endpoint_events
                WHERE endpoint IN ({placeholders})
                ORDER BY id
                """,
                tuple(endpoints),
            )
            event_rows = [dict(r) for r in cur.fetchall()]

        rewritten_endpoint_rows = 0
        inserted_analytics_rows = 0
        duplicate_analytics_rows = 0
        failed = 0

        for event_row in tqdm(event_rows, desc="Rebuilding complex endpoint analytics"):
            try:
                payload = self._load_blob_payload(event_row["blob_path"])
                self._rewrite_complex_endpoint_parquet_row(event_row, payload)
                rewritten_endpoint_rows += 1

                analytics_result = self._persist_complex_endpoint_analytics(
                    endpoint=event_row["endpoint"],
                    payload=payload,
                    lineage_meta={
                        "endpoint_event_id": int(event_row["id"]),
                        "endpoint": event_row["endpoint"],
                        "conid": str(event_row["conid"]),
                        "observed_at": event_row["observed_at"],
                        "effective_at": event_row["effective_at"],
                        "payload_hash": event_row["payload_hash"],
                    },
                    source_file=event_row.get("source_file"),
                )
                inserted_analytics_rows += int(analytics_result.get("analytics_rows_inserted", 0))
                duplicate_analytics_rows += int(analytics_result.get("analytics_rows_duplicate", 0))
            except Exception as e:
                failed += 1
                logger.error(f"Failed rebuilding analytics for event_id={event_row.get('id')}: {e}")

        if refresh_duckdb:
            self.refresh_duckdb_views()

        return {
            "status": "ok",
            "endpoints": endpoints,
            "events_seen": len(event_rows),
            "rewritten_endpoint_rows": rewritten_endpoint_rows,
            "inserted_analytics_rows": inserted_analytics_rows,
            "duplicate_analytics_rows": duplicate_analytics_rows,
            "failed": failed,
            "duckdb_path": str(self.duckdb_path),
        }

def refresh_views():
    store = FundamentalsStore()
    result = store.refresh_duckdb_views()
    for k, v in result.items():
        print(f"{k}: {v}")
    return result


def backfill_complex_analytics(endpoints=None):
    store = FundamentalsStore()
    result = store.rebuild_complex_analytics(endpoints=endpoints, refresh_duckdb=True)
    for k, v in result.items():
        print(f"{k}: {v}")
    return result


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "refresh_views": refresh_views,
            "backfill_complex_analytics": backfill_complex_analytics,
        }
    )
