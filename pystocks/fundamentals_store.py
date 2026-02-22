import hashlib
import json
import logging
import re
import sqlite3
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

# Use discrete endpoint dates only. Do not derive effective date from
# chart/time-series payload internals.
DISCRETE_ENDPOINTS = {
    "profile_and_fees",
    "holdings",
    "ratios",
    "lipper_ratings",
    "dividends",
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
        # YYYYMMDD integer format
        iv = int(ts)
        if 19000101 <= iv <= 29991231:
            s = str(iv)
            try:
                return datetime.strptime(s, "%Y%m%d").date()
            except Exception:
                pass
        # Heuristic: ms epoch
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
                # Preserve scalar lists as compact JSON for optional use.
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
                "CREATE INDEX IF NOT EXISTS idx_endpoint_events_endpoint ON endpoint_events(endpoint)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_endpoint_events_effective_at ON endpoint_events(effective_at)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_endpoint_events_conid_endpoint ON endpoint_events(conid, endpoint)"
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

    def _resolve_effective_dates(self, endpoint_payloads, observed_at):
        discrete_dates = {}
        for endpoint in DISCRETE_ENDPOINTS:
            payload = endpoint_payloads.get(endpoint)
            if payload is None:
                continue
            date_value = _extract_as_of_date(payload)
            if date_value is not None:
                discrete_dates[endpoint] = date_value

        ratios_date = discrete_dates.get("ratios")
        earliest_discrete = min(discrete_dates.values()) if discrete_dates else None

        resolved = {}
        for endpoint in endpoint_payloads.keys():
            own = discrete_dates.get(endpoint)
            if own is not None:
                resolved[endpoint] = (own.isoformat(), f"{endpoint}.as_of_date")
                continue
            if ratios_date is not None:
                resolved[endpoint] = (ratios_date.isoformat(), "ratios.as_of_date_fallback")
                continue
            if earliest_discrete is not None:
                resolved[endpoint] = (earliest_discrete.isoformat(), "earliest_discrete_as_of_fallback")
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
            return {"inserted": False, "duplicate": True, "endpoint": endpoint}

        row = {
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
        row.update(_flatten_payload_scalars(payload))
        parquet_path = self._write_endpoint_event_parquet(row, endpoint_slug, effective_at)
        return {"inserted": True, "duplicate": False, "endpoint": endpoint, "parquet_path": str(parquet_path)}

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
            if result.get("inserted"):
                inserted_events += 1
            elif result.get("duplicate"):
                duplicate_events += 1

        if refresh_duckdb:
            self.refresh_duckdb_views()

        return {
            "inserted_events": inserted_events,
            "duplicate_events": duplicate_events,
            "status": "ok",
            "per_endpoint": per_endpoint,
        }

    def refresh_duckdb_views(self):
        endpoint_slugs = []
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT endpoint_slug FROM endpoint_events ORDER BY endpoint_slug")
            endpoint_slugs = [row[0] for row in cur.fetchall()]

        parquet_files = list(self.parquet_dir.glob("endpoint=*/year=*/month=*/*.parquet"))
        db = duckdb.connect(str(self.duckdb_path))
        try:
            if parquet_files:
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
                pattern = f"{self.parquet_dir.as_posix()}/endpoint={slug}/year=*/month=*/*.parquet"
                db.execute(
                    f"""
                    CREATE OR REPLACE VIEW {view_name} AS
                    SELECT * FROM read_parquet('{pattern}', union_by_name=true)
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
        finally:
            db.close()

        return {"endpoint_views": len(endpoint_slugs), "duckdb_path": str(self.duckdb_path)}

    def migrate_legacy_json(self, delete_legacy=True, limit=None, refresh_duckdb=True):
        legacy_files = []
        for child in sorted(self.fundamentals_dir.iterdir()):
            if child.is_dir() and child.name.isdigit():
                legacy_files.extend(sorted(child.glob("*.json")))

        if limit:
            legacy_files = legacy_files[:limit]

        migrated = 0
        failed = 0
        deleted = 0
        inserted_events = 0
        duplicate_events = 0

        for path in tqdm(legacy_files, desc="Migrating legacy fundamentals JSON"):
            try:
                payload = json.loads(path.read_text())
                result = self.persist_combined_snapshot(
                    payload,
                    source_file=str(path),
                    refresh_duckdb=False,
                )
                inserted_events += int(result.get("inserted_events", 0))
                duplicate_events += int(result.get("duplicate_events", 0))
                migrated += 1

                if delete_legacy:
                    path.unlink(missing_ok=True)
                    deleted += 1
            except Exception as e:
                failed += 1
                logger.error(f"Failed to migrate {path}: {e}")

        if delete_legacy:
            for child in sorted(self.fundamentals_dir.iterdir()):
                if child.is_dir() and child.name.isdigit():
                    try:
                        if not any(child.iterdir()):
                            child.rmdir()
                    except Exception:
                        pass

        if refresh_duckdb:
            self.refresh_duckdb_views()

        return {
            "legacy_files_seen": len(legacy_files),
            "migrated_files": migrated,
            "failed_files": failed,
            "deleted_legacy_files": deleted,
            "inserted_events": inserted_events,
            "duplicate_events": duplicate_events,
            "events_db_path": str(self.events_db_path),
            "duckdb_path": str(self.duckdb_path),
        }


def migrate_legacy_json(delete_legacy=True, limit=None):
    store = FundamentalsStore()
    result = store.migrate_legacy_json(delete_legacy=delete_legacy, limit=limit, refresh_duckdb=True)
    for k, v in result.items():
        print(f"{k}: {v}")
    return result


def refresh_views():
    store = FundamentalsStore()
    result = store.refresh_duckdb_views()
    for k, v in result.items():
        print(f"{k}: {v}")
    return result


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "migrate_legacy_json": migrate_legacy_json,
            "refresh_views": refresh_views,
        }
    )
