import hashlib
import json
import logging
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import zstandard as zstd

from .config import (
    DIVIDENDS_EVENTS_PARQUET_DIR,
    FACTOR_FEATURES_PARQUET_DIR,
    FUNDAMENTALS_BLOBS_DIR,
    FUNDAMENTALS_DIR,
    FUNDAMENTALS_DUCKDB_PATH,
    FUNDAMENTALS_EVENTS_DB_PATH,
    FUNDAMENTALS_PARQUET_DIR,
    OWNERSHIP_TRADE_LOG_PARQUET_DIR,
    PRICE_CHART_PARQUET_DIR,
    SENTIMENT_SEARCH_PARQUET_DIR,
)
from .fundamentals_normalizers import (
    extract_dividends_events,
    extract_factor_features,
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
    "price_chart",
    "performance",
    "sentiment_search",
    "ownership",
    "esg",
]

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
            try:
                return datetime.strptime(str(iv), "%Y%m%d").date()
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


def _extract_endpoint_effective_date(endpoint, payload):
    if not isinstance(payload, (dict, list)):
        return None

    if endpoint == "morningstar" and isinstance(payload, dict):
        publish_dates = _extract_morningstar_publish_dates(payload)
        if publish_dates:
            return max(publish_dates)

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


def _sanitize_sentiment_search_payload(payload):
    if not isinstance(payload, dict):
        return payload
    sentiment = payload.get("sentiment")
    if not isinstance(sentiment, list):
        return payload

    drop_keys = {
        "high",
        "low",
        "price",
        "price_change_p",
        "close",
        "open",
        "price_change",
    }
    for row in sentiment:
        if not isinstance(row, dict):
            continue
        for key in drop_keys:
            row.pop(key, None)
    return payload


def _extract_price_chart_rows(payload):
    if not isinstance(payload, dict):
        return []

    plot = payload.get("plot")
    if not isinstance(plot, dict):
        return []

    series_list = plot.get("series")
    if not isinstance(series_list, list):
        return []

    selected_series = None
    for series in series_list:
        if not isinstance(series, dict):
            continue
        name = str(series.get("name") or series.get("title") or "").strip().lower()
        if "price" in name:
            selected_series = series
            break
    if selected_series is None and series_list:
        first = series_list[0]
        if isinstance(first, dict):
            selected_series = first
    if selected_series is None:
        return []

    points = selected_series.get("plotData")
    if not isinstance(points, list):
        return []

    rows = []
    for point in points:
        if not isinstance(point, dict):
            continue
        trade_date = _parse_date_candidate(point.get("x"))
        if trade_date is None:
            trade_date = _parse_date_candidate(point.get("debugY"))

        row = {
            "trade_date": trade_date.isoformat() if trade_date is not None else None,
            "timestamp_ms": point.get("x"),
            "price": point.get("y"),
            "open": point.get("open"),
            "high": point.get("high"),
            "low": point.get("low"),
            "close": point.get("close"),
            "debug_y": point.get("debugY"),
        }
        if any(v is not None for v in row.values()):
            rows.append(row)

    return rows


def _extract_sentiment_search_rows(payload):
    if not isinstance(payload, dict):
        return []
    sentiment = payload.get("sentiment")
    if not isinstance(sentiment, list):
        return []

    rows = []
    for point in sentiment:
        if not isinstance(point, dict):
            continue

        dt = _parse_date_candidate(point.get("datetime"))
        row = {
            "trade_date": dt.isoformat() if dt is not None else None,
            "datetime_ms": point.get("datetime"),
        }

        for key, value in point.items():
            if key == "datetime":
                continue
            row[key] = value

        if any(v is not None for v in row.values()):
            rows.append(row)

    return rows


def _to_scalar(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class FundamentalsStore:
    """
    Storage backend:
    - CAS compressed blobs for full raw endpoint payloads.
    - Slim endpoint snapshot parquet rows for lineage/audit.
    - Unified long factor-feature parquet rows.
    - Dedicated series parquet stores.
    - SQLite manifest for blobs + endpoint events.
    - DuckDB views over parquet partitions.
    """

    def __init__(
        self,
        fundamentals_dir=FUNDAMENTALS_DIR,
        blobs_dir=FUNDAMENTALS_BLOBS_DIR,
        parquet_dir=FUNDAMENTALS_PARQUET_DIR,
        factor_features_parquet_dir=FACTOR_FEATURES_PARQUET_DIR,
        price_chart_parquet_dir=PRICE_CHART_PARQUET_DIR,
        sentiment_search_parquet_dir=SENTIMENT_SEARCH_PARQUET_DIR,
        ownership_trade_log_parquet_dir=OWNERSHIP_TRADE_LOG_PARQUET_DIR,
        dividends_events_parquet_dir=DIVIDENDS_EVENTS_PARQUET_DIR,
        events_db_path=FUNDAMENTALS_EVENTS_DB_PATH,
        duckdb_path=FUNDAMENTALS_DUCKDB_PATH,
    ):
        self.fundamentals_dir = Path(fundamentals_dir)
        self.blobs_dir = Path(blobs_dir)
        self.parquet_dir = Path(parquet_dir)
        self.factor_features_parquet_dir = Path(factor_features_parquet_dir)
        self.price_chart_parquet_dir = Path(price_chart_parquet_dir)
        self.sentiment_search_parquet_dir = Path(sentiment_search_parquet_dir)
        self.ownership_trade_log_parquet_dir = Path(ownership_trade_log_parquet_dir)
        self.dividends_events_parquet_dir = Path(dividends_events_parquet_dir)
        self.events_db_path = Path(events_db_path)
        self.duckdb_path = Path(duckdb_path)

        self.fundamentals_dir.mkdir(parents=True, exist_ok=True)
        self.blobs_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self.factor_features_parquet_dir.mkdir(parents=True, exist_ok=True)
        self.price_chart_parquet_dir.mkdir(parents=True, exist_ok=True)
        self.sentiment_search_parquet_dir.mkdir(parents=True, exist_ok=True)
        self.ownership_trade_log_parquet_dir.mkdir(parents=True, exist_ok=True)
        self.dividends_events_parquet_dir.mkdir(parents=True, exist_ok=True)

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
            # Legacy table: no longer used after unified factor-feature normalization.
            cur.execute("DROP TABLE IF EXISTS analytics_rows")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_endpoint_events_endpoint ON endpoint_events(endpoint)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_endpoint_events_effective_at ON endpoint_events(effective_at)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_endpoint_events_conid_endpoint ON endpoint_events(conid, endpoint)")
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
                if endpoint == "sentiment_search" and isinstance(value, dict):
                    value = _sanitize_sentiment_search_payload(value)
                payloads[endpoint] = value
        return payloads

    def _build_endpoint_snapshot_row(self, base_row, payload):
        row = dict(base_row)
        if isinstance(payload, dict):
            row["s__payload_kind"] = "dict"
            row["s__top_level_keys_len"] = len(payload)
        elif isinstance(payload, list):
            row["s__payload_kind"] = "list"
            row["s__top_level_keys_len"] = len(payload)
        else:
            row["s__payload_kind"] = type(payload).__name__
            row["s__top_level_keys_len"] = 0

        if base_row.get("endpoint") == "dividends" and isinstance(payload, dict):
            canonical = normalize_dividends_snapshot(payload)
            for key in ("response_type", "has_history", "history_points", "embedded_price_points"):
                value = canonical.get(key)
                if isinstance(value, bool):
                    value = 1 if value else 0
                row[f"s__dividends_{_sanitize_segment(key)}"] = _to_scalar(value)

        if base_row.get("endpoint") == "ownership" and isinstance(payload, dict):
            canonical = normalize_ownership_snapshot(payload)
            for key in (
                "owners_types_count",
                "institutional_owners_count",
                "insider_owners_count",
                "trade_log_count_raw",
                "trade_log_count_kept",
                "ownership_history_price_points",
            ):
                row[f"s__ownership_{_sanitize_segment(key)}"] = _to_scalar(canonical.get(key))

        return row

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

    def _write_factor_features_parquet(self, rows, lineage_meta, endpoint_slug):
        conid = str(lineage_meta.get("conid"))
        event_id = int(lineage_meta.get("endpoint_event_id"))
        payload_hash = str(lineage_meta.get("payload_hash"))
        effective_at = str(lineage_meta.get("effective_at"))

        enriched_rows = []
        for row in rows:
            feature_name = _sanitize_segment(row.get("feature_name"))
            feature_group = _sanitize_segment(row.get("feature_group"))
            feature_value = row.get("feature_value")
            feature_source = str(row.get("feature_source") or "")
            if feature_name == "field" or feature_value is None:
                continue

            row_key_material = "|".join(
                [
                    str(endpoint_slug),
                    str(conid),
                    str(feature_name),
                    str(effective_at),
                    str(payload_hash),
                ]
            )
            row_key = hashlib.sha256(row_key_material.encode("utf-8")).hexdigest()

            enriched_rows.append(
                {
                    "endpoint_event_id": event_id,
                    "endpoint": lineage_meta.get("endpoint"),
                    "conid": conid,
                    "observed_at": lineage_meta.get("observed_at"),
                    "effective_at": effective_at,
                    "payload_hash": payload_hash,
                    "feature_name": feature_name,
                    "feature_group": feature_group,
                    "feature_value": float(feature_value),
                    "feature_source": feature_source,
                    "source_file": lineage_meta.get("source_file"),
                    "inserted_at": lineage_meta.get("inserted_at"),
                    "row_key": row_key,
                }
            )

        if not enriched_rows:
            return None, 0

        partition_dir = (
            self.factor_features_parquet_dir
            / f"endpoint={endpoint_slug}"
            / f"conid={conid}"
        )
        partition_dir.mkdir(parents=True, exist_ok=True)

        file_name = f"{effective_at}_{payload_hash[:12]}_{event_id}.parquet"
        file_path = partition_dir / file_name

        pd.DataFrame(enriched_rows).to_parquet(file_path, index=False, engine="pyarrow", compression="zstd")
        return file_path, len(enriched_rows)

    def _write_price_chart_series_parquet(self, rows, lineage_meta):
        return self._write_series_parquet(
            rows=rows,
            lineage_meta=lineage_meta,
            endpoint="price_chart",
            base_dir=self.price_chart_parquet_dir,
        )

    def _write_sentiment_search_series_parquet(self, rows, lineage_meta):
        return self._write_series_parquet(
            rows=rows,
            lineage_meta=lineage_meta,
            endpoint="sentiment_search",
            base_dir=self.sentiment_search_parquet_dir,
        )

    def _write_ownership_trade_log_series_parquet(self, rows, lineage_meta):
        return self._write_series_parquet(
            rows=rows,
            lineage_meta=lineage_meta,
            endpoint="ownership",
            base_dir=self.ownership_trade_log_parquet_dir,
        )

    def _write_dividends_events_series_parquet(self, rows, lineage_meta):
        return self._write_series_parquet(
            rows=rows,
            lineage_meta=lineage_meta,
            endpoint="dividends",
            base_dir=self.dividends_events_parquet_dir,
        )

    def _write_series_parquet(self, rows, lineage_meta, endpoint, base_dir):
        conid = str(lineage_meta.get("conid"))
        event_id = int(lineage_meta.get("endpoint_event_id"))
        payload_hash = str(lineage_meta.get("payload_hash"))
        effective_at = str(lineage_meta.get("effective_at"))

        enriched_rows = []
        for row in rows:
            full_row = {
                "endpoint_event_id": event_id,
                "endpoint": endpoint,
                "conid": conid,
                "observed_at": lineage_meta.get("observed_at"),
                "effective_at": effective_at,
                "payload_hash": payload_hash,
                "source_file": lineage_meta.get("source_file"),
                "inserted_at": lineage_meta.get("inserted_at"),
            }
            for key, value in (row or {}).items():
                full_row[_sanitize_segment(key)] = _to_scalar(value)
            full_row["row_key"] = self._series_row_key(endpoint, full_row)
            enriched_rows.append(full_row)

        if not enriched_rows:
            return None, 0

        partition_dir = Path(base_dir) / f"conid={conid}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        target_path = partition_dir / "series.parquet"

        existing_files = sorted(partition_dir.glob("*.parquet"))
        existing_df = pd.DataFrame()
        if existing_files:
            frames = []
            for path in existing_files:
                try:
                    frames.append(pd.read_parquet(path))
                except Exception as e:
                    logger.warning(f"Failed reading existing series parquet {path}: {e}")
            if frames:
                existing_df = pd.concat(frames, ignore_index=True, sort=False)

        incoming_df = pd.DataFrame(enriched_rows)
        if existing_df.empty:
            combined = incoming_df
        else:
            combined = pd.concat([existing_df, incoming_df], ignore_index=True, sort=False)

        if "row_key" not in combined.columns:
            combined["row_key"] = None

        missing_row_key = combined["row_key"].isna() | (
            combined["row_key"].astype(str).str.strip() == ""
        )
        if bool(missing_row_key.any()):
            combined.loc[missing_row_key, "row_key"] = combined.loc[missing_row_key].apply(
                lambda r: self._series_row_key(endpoint, r),
                axis=1,
            )

        combined = combined.drop_duplicates(subset=["row_key"], keep="last")
        combined = self._sort_series_rows(endpoint, combined)

        combined.to_parquet(target_path, index=False, engine="pyarrow", compression="zstd")

        for stale_file in existing_files:
            if stale_file != target_path and stale_file.exists():
                stale_file.unlink()

        return target_path, len(enriched_rows)

    def _base_row_from_event_meta(self, event_meta):
        return {
            "event_id": int(event_meta["event_id"]),
            "conid": str(event_meta["conid"]),
            "endpoint": event_meta["endpoint"],
            "endpoint_slug": event_meta["endpoint_slug"],
            "observed_at": event_meta["observed_at"],
            "effective_at": event_meta["effective_at"],
            "effective_source": event_meta["effective_source"],
            "payload_hash": event_meta["payload_hash"],
            "blob_path": event_meta["blob_path"],
            "payload_size_raw": event_meta["payload_size_raw"],
            "payload_size_compressed": event_meta["payload_size_compressed"],
            "source_file": event_meta.get("source_file"),
            "inserted_at": event_meta["inserted_at"],
        }

    def _lineage_meta_from_event_meta(self, event_meta):
        return {
            "endpoint_event_id": int(event_meta["event_id"]),
            "endpoint": event_meta["endpoint"],
            "conid": str(event_meta["conid"]),
            "observed_at": event_meta["observed_at"],
            "effective_at": event_meta["effective_at"],
            "payload_hash": event_meta["payload_hash"],
            "source_file": event_meta.get("source_file"),
            "inserted_at": event_meta["inserted_at"],
        }

    def _expected_endpoint_snapshot_path(self, event_meta):
        effective_at = str(event_meta["effective_at"])
        dt = datetime.fromisoformat(f"{effective_at}T00:00:00+00:00")
        partition_dir = (
            self.parquet_dir
            / f"endpoint={event_meta['endpoint_slug']}"
            / f"year={dt.year:04d}"
            / f"month={dt.month:02d}"
        )
        file_name = (
            f"{effective_at}_{event_meta['conid']}_"
            f"{str(event_meta['payload_hash'])[:12]}_{int(event_meta['event_id'])}.parquet"
        )
        return partition_dir / file_name

    def _expected_factor_features_path(self, event_meta):
        partition_dir = (
            self.factor_features_parquet_dir
            / f"endpoint={event_meta['endpoint_slug']}"
            / f"conid={event_meta['conid']}"
        )
        file_name = (
            f"{event_meta['effective_at']}_"
            f"{str(event_meta['payload_hash'])[:12]}_{int(event_meta['event_id'])}.parquet"
        )
        return partition_dir / file_name

    def _expected_series_path(self, event_meta, base_dir):
        return Path(base_dir) / f"conid={event_meta['conid']}" / "series.parquet"

    def _series_row_key(self, endpoint, row):
        def value(name):
            v = row.get(name) if hasattr(row, "get") else None
            if v is None:
                return ""
            if isinstance(v, float) and pd.isna(v):
                return ""
            text = str(v).strip()
            if not text or text.lower() == "nan":
                return ""
            return text

        if endpoint == "price_chart":
            ident = value("trade_date") or value("timestamp_ms") or value("debug_y")
            return f"price_chart|{ident}"
        if endpoint == "sentiment_search":
            ident = value("datetime_ms") or value("trade_date")
            return f"sentiment_search|{ident}"
        if endpoint == "ownership":
            return "|".join(
                [
                    "ownership",
                    value("trade_date"),
                    value("action"),
                    value("party"),
                    value("source"),
                    value("insider"),
                    value("shares"),
                    value("value"),
                    value("holding"),
                ]
            )
        if endpoint == "dividends":
            return "|".join(
                [
                    "dividends",
                    value("event_date") or value("trade_date"),
                    value("amount"),
                    value("currency"),
                    value("event_type"),
                    value("declaration_date"),
                    value("record_date"),
                    value("payment_date"),
                    value("description"),
                ]
            )

        skip = {
            "endpoint_event_id",
            "endpoint",
            "conid",
            "observed_at",
            "effective_at",
            "payload_hash",
            "source_file",
            "inserted_at",
            "row_key",
        }
        payload = []
        for k in sorted(row.keys()):
            if k in skip:
                continue
            payload.append(f"{k}={value(k)}")
        return f"{endpoint}|{'|'.join(payload)}"

    def _sort_series_rows(self, endpoint, df):
        sort_map = {
            "price_chart": ["trade_date", "timestamp_ms", "debug_y"],
            "sentiment_search": ["trade_date", "datetime_ms"],
            "ownership": ["trade_date", "action", "party", "source", "insider", "shares", "value", "holding"],
            "dividends": [
                "event_date",
                "trade_date",
                "payment_date",
                "record_date",
                "declaration_date",
                "amount",
                "currency",
                "event_type",
            ],
        }
        cols = [c for c in sort_map.get(endpoint, ["row_key"]) if c in df.columns]
        if not cols:
            return df.reset_index(drop=True)
        try:
            return df.sort_values(by=cols, kind="stable", na_position="last").reset_index(drop=True)
        except Exception:
            if "row_key" in df.columns:
                return df.sort_values(by=["row_key"], kind="stable").reset_index(drop=True)
            return df.reset_index(drop=True)

    def _get_existing_event(self, conid, endpoint, effective_at, payload_hash):
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                    id,
                    conid,
                    endpoint,
                    endpoint_slug,
                    observed_at,
                    effective_at,
                    effective_source,
                    payload_hash,
                    blob_path,
                    payload_size_raw,
                    payload_size_compressed,
                    source_file,
                    inserted_at
                FROM endpoint_events
                WHERE conid = ?
                  AND endpoint = ?
                  AND effective_at = ?
                  AND payload_hash = ?
                LIMIT 1
                """,
                (str(conid), endpoint, effective_at, payload_hash),
            )
            row = cur.fetchone()
            if row is None:
                return None
            out = dict(row)
            out["event_id"] = int(out.pop("id"))
            return out

    def _materialize_event_artifacts(self, event_meta, payload, only_missing=False):
        base_row = self._base_row_from_event_meta(event_meta)
        lineage_meta = self._lineage_meta_from_event_meta(event_meta)
        endpoint = base_row["endpoint"]
        endpoint_slug = base_row["endpoint_slug"]
        effective_at = base_row["effective_at"]

        snapshot_path = self._expected_endpoint_snapshot_path(event_meta)
        if (not only_missing) or (not snapshot_path.exists()):
            snapshot_row = self._build_endpoint_snapshot_row(base_row, payload)
            snapshot_path = Path(
                self._write_endpoint_event_parquet(snapshot_row, endpoint_slug, effective_at)
            )

        factor_rows = extract_factor_features(endpoint, payload, effective_at=effective_at)
        factor_path = None
        factor_rows_written = 0
        if factor_rows:
            expected_factor_path = self._expected_factor_features_path(event_meta)
            if (not only_missing) or (not expected_factor_path.exists()):
                written_factor_path, factor_rows_written = self._write_factor_features_parquet(
                    rows=factor_rows,
                    lineage_meta=lineage_meta,
                    endpoint_slug=endpoint_slug,
                )
                if written_factor_path is not None:
                    factor_path = Path(written_factor_path)
            else:
                factor_path = expected_factor_path

        series_path = None
        series_rows_written = 0
        if endpoint == "price_chart":
            rows = _extract_price_chart_rows(payload)
            expected_series_path = self._expected_series_path(event_meta, self.price_chart_parquet_dir)
            if rows and ((not only_missing) or (not expected_series_path.exists())):
                written_series_path, series_rows_written = self._write_price_chart_series_parquet(
                    rows,
                    lineage_meta=lineage_meta,
                )
                if written_series_path is not None:
                    series_path = Path(written_series_path)
            elif expected_series_path.exists():
                series_path = expected_series_path
        elif endpoint == "sentiment_search":
            sanitized = _sanitize_sentiment_search_payload(payload)
            rows = _extract_sentiment_search_rows(sanitized)
            expected_series_path = self._expected_series_path(event_meta, self.sentiment_search_parquet_dir)
            if rows and ((not only_missing) or (not expected_series_path.exists())):
                written_series_path, series_rows_written = self._write_sentiment_search_series_parquet(
                    rows,
                    lineage_meta=lineage_meta,
                )
                if written_series_path is not None:
                    series_path = Path(written_series_path)
            elif expected_series_path.exists():
                series_path = expected_series_path
        elif endpoint == "ownership":
            rows = extract_ownership_trade_log(payload, drop_no_change=True)
            expected_series_path = self._expected_series_path(event_meta, self.ownership_trade_log_parquet_dir)
            if rows and ((not only_missing) or (not expected_series_path.exists())):
                written_series_path, series_rows_written = self._write_ownership_trade_log_series_parquet(
                    rows,
                    lineage_meta=lineage_meta,
                )
                if written_series_path is not None:
                    series_path = Path(written_series_path)
            elif expected_series_path.exists():
                series_path = expected_series_path
        elif endpoint == "dividends":
            rows = extract_dividends_events(payload)
            expected_series_path = self._expected_series_path(event_meta, self.dividends_events_parquet_dir)
            if rows and ((not only_missing) or (not expected_series_path.exists())):
                written_series_path, series_rows_written = self._write_dividends_events_series_parquet(
                    rows,
                    lineage_meta=lineage_meta,
                )
                if written_series_path is not None:
                    series_path = Path(written_series_path)
            elif expected_series_path.exists():
                series_path = expected_series_path

        return {
            "snapshot_path": str(snapshot_path) if snapshot_path is not None else None,
            "factor_path": str(factor_path) if factor_path is not None else None,
            "factor_rows_written": int(factor_rows_written),
            "series_path": str(series_path) if series_path is not None else None,
            "series_rows_written": int(series_rows_written),
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
            existing_event = self._get_existing_event(
                conid=conid,
                endpoint=endpoint,
                effective_at=effective_at,
                payload_hash=blob_info["hash"],
            )
            artifacts = {
                "snapshot_path": None,
                "factor_path": None,
                "factor_rows_written": 0,
                "series_path": None,
                "series_rows_written": 0,
            }
            if existing_event is not None:
                artifacts = self._materialize_event_artifacts(
                    event_meta=existing_event,
                    payload=payload,
                    only_missing=True,
                )
            return {
                "inserted": False,
                "duplicate": True,
                "endpoint": endpoint,
                "snapshot_path": artifacts["snapshot_path"],
                "factor_path": artifacts["factor_path"],
                "factor_rows_written": int(artifacts["factor_rows_written"]),
                "series_path": artifacts["series_path"],
                "series_rows_written": int(artifacts["series_rows_written"]),
            }

        event_meta = {
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
        artifacts = self._materialize_event_artifacts(
            event_meta=event_meta,
            payload=payload,
            only_missing=False,
        )

        return {
            "inserted": True,
            "duplicate": False,
            "endpoint": endpoint,
            "snapshot_path": artifacts["snapshot_path"],
            "factor_path": artifacts["factor_path"],
            "factor_rows_written": int(artifacts["factor_rows_written"]),
            "series_path": artifacts["series_path"],
            "series_rows_written": int(artifacts["series_rows_written"]),
        }

    def persist_combined_snapshot(self, snapshot, source_file=None, refresh_duckdb=False):
        if not isinstance(snapshot, dict):
            return {"inserted_events": 0, "duplicate_events": 0, "status": "invalid_snapshot"}

        conid = snapshot.get("conid")
        if conid is None:
            return {"inserted_events": 0, "duplicate_events": 0, "status": "missing_conid"}

        observed_at = snapshot.get("scraped_at") or datetime.now(timezone.utc).isoformat()
        try:
            observed_dt = datetime.fromisoformat(str(observed_at).replace("Z", "+00:00"))
            if observed_dt.tzinfo is None:
                observed_dt = observed_dt.replace(tzinfo=timezone.utc)
        except Exception:
            observed_dt = datetime.now(timezone.utc)
            observed_at = observed_dt.isoformat()

        endpoint_payloads = self._endpoint_payloads_from_snapshot(snapshot)
        if not endpoint_payloads:
            return {"inserted_events": 0, "duplicate_events": 0, "status": "no_payloads"}

        effective_map = self._resolve_effective_dates(endpoint_payloads, observed_dt)

        inserted_events = 0
        duplicate_events = 0
        factor_rows_written = 0
        series_rows_written = 0
        per_endpoint = {}

        for endpoint, payload in endpoint_payloads.items():
            effective_at, effective_source = effective_map[endpoint]
            result = self.persist_endpoint_payload(
                conid=conid,
                endpoint=endpoint,
                payload=payload,
                observed_at=observed_at,
                effective_at=effective_at,
                effective_source=effective_source,
                source_file=source_file,
            )

            if result.get("duplicate"):
                duplicate_events += 1
            else:
                inserted_events += 1

            factor_rows_written += int(result.get("factor_rows_written", 0))
            series_rows_written += int(result.get("series_rows_written", 0))
            per_endpoint[endpoint] = result

        if refresh_duckdb:
            self.refresh_duckdb_views()

        return {
            "inserted_events": inserted_events,
            "duplicate_events": duplicate_events,
            "factor_rows_written": factor_rows_written,
            "series_rows_written": series_rows_written,
            "status": "ok",
            "per_endpoint": per_endpoint,
        }

    def _create_empty_endpoint_events_view(self, db):
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

    def _create_empty_factor_features_view(self, db):
        db.execute(
            """
            CREATE OR REPLACE VIEW factor_features_all AS
            SELECT
                CAST(NULL AS BIGINT) AS endpoint_event_id,
                CAST(NULL AS VARCHAR) AS endpoint,
                CAST(NULL AS VARCHAR) AS conid,
                CAST(NULL AS VARCHAR) AS observed_at,
                CAST(NULL AS VARCHAR) AS effective_at,
                CAST(NULL AS VARCHAR) AS payload_hash,
                CAST(NULL AS VARCHAR) AS feature_name,
                CAST(NULL AS VARCHAR) AS feature_group,
                CAST(NULL AS DOUBLE) AS feature_value,
                CAST(NULL AS VARCHAR) AS feature_source,
                CAST(NULL AS VARCHAR) AS source_file,
                CAST(NULL AS VARCHAR) AS inserted_at,
                CAST(NULL AS VARCHAR) AS row_key
            WHERE FALSE
            """
        )

    def _create_empty_price_series_view(self, db):
        db.execute(
            """
            CREATE OR REPLACE VIEW price_chart_series_all AS
            SELECT
                CAST(NULL AS BIGINT) AS endpoint_event_id,
                CAST(NULL AS VARCHAR) AS endpoint,
                CAST(NULL AS VARCHAR) AS conid,
                CAST(NULL AS VARCHAR) AS observed_at,
                CAST(NULL AS VARCHAR) AS effective_at,
                CAST(NULL AS VARCHAR) AS payload_hash,
                CAST(NULL AS VARCHAR) AS source_file,
                CAST(NULL AS VARCHAR) AS inserted_at,
                CAST(NULL AS VARCHAR) AS trade_date,
                CAST(NULL AS BIGINT) AS timestamp_ms,
                CAST(NULL AS DOUBLE) AS price,
                CAST(NULL AS DOUBLE) AS open,
                CAST(NULL AS DOUBLE) AS high,
                CAST(NULL AS DOUBLE) AS low,
                CAST(NULL AS DOUBLE) AS close,
                CAST(NULL AS BIGINT) AS debug_y,
                CAST(NULL AS VARCHAR) AS row_key
            WHERE FALSE
            """
        )

    def _create_empty_sentiment_series_view(self, db):
        db.execute(
            """
            CREATE OR REPLACE VIEW sentiment_search_series_all AS
            SELECT
                CAST(NULL AS BIGINT) AS endpoint_event_id,
                CAST(NULL AS VARCHAR) AS endpoint,
                CAST(NULL AS VARCHAR) AS conid,
                CAST(NULL AS VARCHAR) AS observed_at,
                CAST(NULL AS VARCHAR) AS effective_at,
                CAST(NULL AS VARCHAR) AS payload_hash,
                CAST(NULL AS VARCHAR) AS source_file,
                CAST(NULL AS VARCHAR) AS inserted_at,
                CAST(NULL AS VARCHAR) AS trade_date,
                CAST(NULL AS BIGINT) AS datetime_ms,
                CAST(NULL AS VARCHAR) AS row_key
            WHERE FALSE
            """
        )

    def _create_empty_ownership_trade_log_view(self, db):
        db.execute(
            """
            CREATE OR REPLACE VIEW ownership_trade_log_series_all AS
            SELECT
                CAST(NULL AS BIGINT) AS endpoint_event_id,
                CAST(NULL AS VARCHAR) AS endpoint,
                CAST(NULL AS VARCHAR) AS conid,
                CAST(NULL AS VARCHAR) AS observed_at,
                CAST(NULL AS VARCHAR) AS effective_at,
                CAST(NULL AS VARCHAR) AS payload_hash,
                CAST(NULL AS VARCHAR) AS source_file,
                CAST(NULL AS VARCHAR) AS inserted_at,
                CAST(NULL AS VARCHAR) AS trade_date,
                CAST(NULL AS VARCHAR) AS action,
                CAST(NULL AS DOUBLE) AS shares,
                CAST(NULL AS DOUBLE) AS value,
                CAST(NULL AS DOUBLE) AS holding,
                CAST(NULL AS VARCHAR) AS party,
                CAST(NULL AS VARCHAR) AS source,
                CAST(NULL AS VARCHAR) AS insider,
                CAST(NULL AS VARCHAR) AS row_key
            WHERE FALSE
            """
        )

    def _create_empty_dividends_events_view(self, db):
        db.execute(
            """
            CREATE OR REPLACE VIEW dividends_events_series_all AS
            SELECT
                CAST(NULL AS BIGINT) AS endpoint_event_id,
                CAST(NULL AS VARCHAR) AS endpoint,
                CAST(NULL AS VARCHAR) AS conid,
                CAST(NULL AS VARCHAR) AS observed_at,
                CAST(NULL AS VARCHAR) AS effective_at,
                CAST(NULL AS VARCHAR) AS payload_hash,
                CAST(NULL AS VARCHAR) AS source_file,
                CAST(NULL AS VARCHAR) AS inserted_at,
                CAST(NULL AS VARCHAR) AS trade_date,
                CAST(NULL AS VARCHAR) AS event_date,
                CAST(NULL AS DOUBLE) AS amount,
                CAST(NULL AS VARCHAR) AS currency,
                CAST(NULL AS VARCHAR) AS description,
                CAST(NULL AS VARCHAR) AS event_type,
                CAST(NULL AS VARCHAR) AS declaration_date,
                CAST(NULL AS VARCHAR) AS record_date,
                CAST(NULL AS VARCHAR) AS payment_date,
                CAST(NULL AS VARCHAR) AS row_key
            WHERE FALSE
            """
        )

    def refresh_duckdb_views(self):
        endpoint_slugs = []
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT endpoint_slug FROM endpoint_events ORDER BY endpoint_slug")
            endpoint_slugs = [row[0] for row in cur.fetchall()]

        endpoint_files = list(self.parquet_dir.glob("endpoint=*/year=*/month=*/*.parquet"))
        factor_feature_files = list(self.factor_features_parquet_dir.glob("endpoint=*/conid=*/*.parquet"))
        price_chart_files = list(self.price_chart_parquet_dir.glob("conid=*/*.parquet"))
        sentiment_search_files = list(self.sentiment_search_parquet_dir.glob("conid=*/*.parquet"))
        ownership_trade_files = list(self.ownership_trade_log_parquet_dir.glob("conid=*/*.parquet"))
        dividends_events_files = list(self.dividends_events_parquet_dir.glob("conid=*/*.parquet"))

        db = duckdb.connect(str(self.duckdb_path))
        try:
            # Retire legacy analytics views from previous schema versions.
            legacy_views = db.execute(
                """
                SELECT table_name
                FROM information_schema.views
                WHERE table_schema = 'main'
                  AND (
                      table_name = 'analytics_rows_all'
                      OR table_name = 'analytics_catalog'
                      OR table_name LIKE 'analytics_%'
                  )
                """
            ).fetchall()
            for (view_name,) in legacy_views:
                safe_name = str(view_name).replace('"', "")
                db.execute(f'DROP VIEW IF EXISTS "{safe_name}"')

            if endpoint_files:
                all_pattern = f"{self.parquet_dir.as_posix()}/endpoint=*/year=*/month=*/*.parquet"
                db.execute(
                    f"""
                    CREATE OR REPLACE VIEW endpoint_events_all AS
                    SELECT * FROM read_parquet('{all_pattern}', union_by_name=true)
                    """
                )
            else:
                self._create_empty_endpoint_events_view(db)

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

            if factor_feature_files:
                factor_pattern = f"{self.factor_features_parquet_dir.as_posix()}/endpoint=*/conid=*/*.parquet"
                db.execute(
                    f"""
                    CREATE OR REPLACE VIEW factor_features_all AS
                    SELECT * REPLACE (CAST(conid AS VARCHAR) AS conid)
                    FROM read_parquet('{factor_pattern}', union_by_name=true)
                    """
                )
            else:
                self._create_empty_factor_features_view(db)

            db.execute(
                """
                CREATE OR REPLACE VIEW factor_features_catalog AS
                SELECT
                    feature_group,
                    feature_name,
                    COUNT(*) AS n_rows,
                    COUNT(DISTINCT conid) AS n_conids,
                    MIN(effective_at) AS min_effective_at,
                    MAX(effective_at) AS max_effective_at
                FROM factor_features_all
                GROUP BY feature_group, feature_name
                ORDER BY feature_group, feature_name
                """
            )

            db.execute(
                """
                CREATE OR REPLACE VIEW factor_features_latest AS
                SELECT * EXCLUDE (rn)
                FROM (
                    SELECT
                        *,
                        ROW_NUMBER() OVER (
                            PARTITION BY conid, feature_name
                            ORDER BY TRY_CAST(effective_at AS DATE) DESC,
                                     TRY_CAST(observed_at AS TIMESTAMP) DESC,
                                     endpoint_event_id DESC
                        ) AS rn
                    FROM factor_features_all
                )
                WHERE rn = 1
                """
            )

            if price_chart_files:
                price_pattern = f"{self.price_chart_parquet_dir.as_posix()}/conid=*/*.parquet"
                db.execute(
                    f"""
                    CREATE OR REPLACE VIEW price_chart_series_all AS
                    SELECT * REPLACE (CAST(conid AS VARCHAR) AS conid)
                    FROM read_parquet('{price_pattern}', union_by_name=true)
                    """
                )
            else:
                self._create_empty_price_series_view(db)

            db.execute(
                """
                CREATE OR REPLACE VIEW price_chart_series_catalog AS
                SELECT conid, COUNT(*) AS n_rows,
                       MIN(trade_date) AS min_trade_date,
                       MAX(trade_date) AS max_trade_date
                FROM price_chart_series_all
                GROUP BY conid
                ORDER BY conid
                """
            )

            if sentiment_search_files:
                sentiment_pattern = f"{self.sentiment_search_parquet_dir.as_posix()}/conid=*/*.parquet"
                db.execute(
                    f"""
                    CREATE OR REPLACE VIEW sentiment_search_series_all AS
                    SELECT * REPLACE (CAST(conid AS VARCHAR) AS conid)
                    FROM read_parquet('{sentiment_pattern}', union_by_name=true)
                    """
                )
            else:
                self._create_empty_sentiment_series_view(db)

            db.execute(
                """
                CREATE OR REPLACE VIEW sentiment_search_series_catalog AS
                WITH endpoint_conids AS (
                    SELECT conid, COUNT(*) AS n_events, MAX(observed_at) AS last_observed_at
                    FROM endpoint_events_all
                    WHERE endpoint = 'sentiment_search'
                    GROUP BY conid
                ),
                series_rollup AS (
                    SELECT conid, COUNT(*) AS n_rows,
                           MIN(trade_date) AS min_trade_date,
                           MAX(trade_date) AS max_trade_date
                    FROM sentiment_search_series_all
                    GROUP BY conid
                )
                SELECT
                    e.conid,
                    COALESCE(s.n_rows, 0) AS n_rows,
                    s.min_trade_date,
                    s.max_trade_date,
                    e.n_events,
                    e.last_observed_at
                FROM endpoint_conids e
                LEFT JOIN series_rollup s
                    ON s.conid = e.conid
                ORDER BY e.conid
                """
            )

            if ownership_trade_files:
                ownership_pattern = f"{self.ownership_trade_log_parquet_dir.as_posix()}/conid=*/*.parquet"
                db.execute(
                    f"""
                    CREATE OR REPLACE VIEW ownership_trade_log_series_all AS
                    SELECT * REPLACE (CAST(conid AS VARCHAR) AS conid)
                    FROM read_parquet('{ownership_pattern}', union_by_name=true)
                    """
                )
            else:
                self._create_empty_ownership_trade_log_view(db)

            db.execute(
                """
                CREATE OR REPLACE VIEW ownership_trade_log_series_catalog AS
                WITH endpoint_conids AS (
                    SELECT conid, COUNT(*) AS n_events, MAX(observed_at) AS last_observed_at
                    FROM endpoint_events_all
                    WHERE endpoint = 'ownership'
                    GROUP BY conid
                ),
                series_rollup AS (
                    SELECT conid, COUNT(*) AS n_rows,
                           MIN(trade_date) AS min_trade_date,
                           MAX(trade_date) AS max_trade_date
                    FROM ownership_trade_log_series_all
                    GROUP BY conid
                )
                SELECT
                    e.conid,
                    COALESCE(s.n_rows, 0) AS n_rows,
                    s.min_trade_date,
                    s.max_trade_date,
                    e.n_events,
                    e.last_observed_at
                FROM endpoint_conids e
                LEFT JOIN series_rollup s
                    ON s.conid = e.conid
                ORDER BY e.conid
                """
            )

            if dividends_events_files:
                dividends_pattern = f"{self.dividends_events_parquet_dir.as_posix()}/conid=*/*.parquet"
                db.execute(
                    f"""
                    CREATE OR REPLACE VIEW dividends_events_series_all AS
                    SELECT * REPLACE (CAST(conid AS VARCHAR) AS conid)
                    FROM read_parquet('{dividends_pattern}', union_by_name=true)
                    """
                )
            else:
                self._create_empty_dividends_events_view(db)

            db.execute(
                """
                CREATE OR REPLACE VIEW dividends_events_series_catalog AS
                WITH endpoint_conids AS (
                    SELECT conid, COUNT(*) AS n_events, MAX(observed_at) AS last_observed_at
                    FROM endpoint_events_all
                    WHERE endpoint = 'dividends'
                    GROUP BY conid
                ),
                series_rollup AS (
                    SELECT conid, COUNT(*) AS n_rows,
                           MIN(trade_date) AS min_trade_date,
                           MAX(trade_date) AS max_trade_date
                    FROM dividends_events_series_all
                    GROUP BY conid
                )
                SELECT
                    e.conid,
                    COALESCE(s.n_rows, 0) AS n_rows,
                    s.min_trade_date,
                    s.max_trade_date,
                    e.n_events,
                    e.last_observed_at
                FROM endpoint_conids e
                LEFT JOIN series_rollup s
                    ON s.conid = e.conid
                ORDER BY e.conid
                """
            )

            db.execute(
                """
                CREATE OR REPLACE VIEW price_features_daily AS
                WITH dedup AS (
                    SELECT
                        conid,
                        TRY_CAST(trade_date AS DATE) AS trade_date,
                        COALESCE(close, price) AS close_price,
                        endpoint_event_id,
                        observed_at,
                        effective_at
                    FROM price_chart_series_all
                    WHERE trade_date IS NOT NULL
                      AND COALESCE(close, price) IS NOT NULL
                ),
                latest_daily AS (
                    SELECT * EXCLUDE (rn)
                    FROM (
                        SELECT
                            *,
                            ROW_NUMBER() OVER (
                                PARTITION BY conid, trade_date
                                ORDER BY TRY_CAST(effective_at AS DATE) DESC,
                                         TRY_CAST(observed_at AS TIMESTAMP) DESC,
                                         endpoint_event_id DESC
                            ) AS rn
                        FROM dedup
                        WHERE trade_date IS NOT NULL
                    )
                    WHERE rn = 1
                ),
                returns AS (
                    SELECT
                        conid,
                        trade_date,
                        close_price,
                        close_price / NULLIF(LAG(close_price) OVER (PARTITION BY conid ORDER BY trade_date), 0) - 1 AS pct_change
                    FROM latest_daily
                )
                SELECT
                    conid,
                    trade_date,
                    close_price,
                    pct_change,
                    AVG(pct_change) OVER (
                        PARTITION BY conid ORDER BY trade_date
                        ROWS BETWEEN 62 PRECEDING AND CURRENT ROW
                    ) AS momentum_3mo,
                    AVG(pct_change) OVER (
                        PARTITION BY conid ORDER BY trade_date
                        ROWS BETWEEN 125 PRECEDING AND CURRENT ROW
                    ) AS momentum_6mo,
                    AVG(pct_change) OVER (
                        PARTITION BY conid ORDER BY trade_date
                        ROWS BETWEEN 251 PRECEDING AND CURRENT ROW
                    ) AS momentum_1y,
                    EXP(SUM(CASE WHEN pct_change IS NULL OR pct_change <= -0.999999 THEN NULL ELSE LN(1 + pct_change) END) OVER (
                        PARTITION BY conid ORDER BY trade_date
                        ROWS BETWEEN 62 PRECEDING AND CURRENT ROW
                    )) - 1 AS rs_3mo,
                    EXP(SUM(CASE WHEN pct_change IS NULL OR pct_change <= -0.999999 THEN NULL ELSE LN(1 + pct_change) END) OVER (
                        PARTITION BY conid ORDER BY trade_date
                        ROWS BETWEEN 125 PRECEDING AND CURRENT ROW
                    )) - 1 AS rs_6mo,
                    EXP(SUM(CASE WHEN pct_change IS NULL OR pct_change <= -0.999999 THEN NULL ELSE LN(1 + pct_change) END) OVER (
                        PARTITION BY conid ORDER BY trade_date
                        ROWS BETWEEN 251 PRECEDING AND CURRENT ROW
                    )) - 1 AS rs_1y
                FROM returns
                """
            )

            db.execute(
                """
                CREATE OR REPLACE VIEW price_features_long_daily AS
                SELECT
                    conid,
                    trade_date,
                    'price_pct_change' AS feature_name,
                    'price' AS feature_group,
                    pct_change AS feature_value,
                    'price_features_daily:pct_change' AS feature_source,
                    CAST(NULL AS BIGINT) AS endpoint_event_id,
                    'price_chart' AS endpoint,
                    CAST(trade_date AS VARCHAR) AS effective_at,
                    CAST(NULL AS VARCHAR) AS observed_at,
                    CAST(NULL AS VARCHAR) AS payload_hash
                FROM price_features_daily
                WHERE pct_change IS NOT NULL

                UNION ALL

                SELECT
                    conid,
                    trade_date,
                    'momentum_3mo' AS feature_name,
                    'price' AS feature_group,
                    momentum_3mo AS feature_value,
                    'price_features_daily:momentum_3mo' AS feature_source,
                    CAST(NULL AS BIGINT) AS endpoint_event_id,
                    'price_chart' AS endpoint,
                    CAST(trade_date AS VARCHAR) AS effective_at,
                    CAST(NULL AS VARCHAR) AS observed_at,
                    CAST(NULL AS VARCHAR) AS payload_hash
                FROM price_features_daily
                WHERE momentum_3mo IS NOT NULL

                UNION ALL

                SELECT
                    conid,
                    trade_date,
                    'momentum_6mo' AS feature_name,
                    'price' AS feature_group,
                    momentum_6mo AS feature_value,
                    'price_features_daily:momentum_6mo' AS feature_source,
                    CAST(NULL AS BIGINT) AS endpoint_event_id,
                    'price_chart' AS endpoint,
                    CAST(trade_date AS VARCHAR) AS effective_at,
                    CAST(NULL AS VARCHAR) AS observed_at,
                    CAST(NULL AS VARCHAR) AS payload_hash
                FROM price_features_daily
                WHERE momentum_6mo IS NOT NULL

                UNION ALL

                SELECT
                    conid,
                    trade_date,
                    'momentum_1y' AS feature_name,
                    'price' AS feature_group,
                    momentum_1y AS feature_value,
                    'price_features_daily:momentum_1y' AS feature_source,
                    CAST(NULL AS BIGINT) AS endpoint_event_id,
                    'price_chart' AS endpoint,
                    CAST(trade_date AS VARCHAR) AS effective_at,
                    CAST(NULL AS VARCHAR) AS observed_at,
                    CAST(NULL AS VARCHAR) AS payload_hash
                FROM price_features_daily
                WHERE momentum_1y IS NOT NULL

                UNION ALL

                SELECT
                    conid,
                    trade_date,
                    'rs_3mo' AS feature_name,
                    'price' AS feature_group,
                    rs_3mo AS feature_value,
                    'price_features_daily:rs_3mo' AS feature_source,
                    CAST(NULL AS BIGINT) AS endpoint_event_id,
                    'price_chart' AS endpoint,
                    CAST(trade_date AS VARCHAR) AS effective_at,
                    CAST(NULL AS VARCHAR) AS observed_at,
                    CAST(NULL AS VARCHAR) AS payload_hash
                FROM price_features_daily
                WHERE rs_3mo IS NOT NULL

                UNION ALL

                SELECT
                    conid,
                    trade_date,
                    'rs_6mo' AS feature_name,
                    'price' AS feature_group,
                    rs_6mo AS feature_value,
                    'price_features_daily:rs_6mo' AS feature_source,
                    CAST(NULL AS BIGINT) AS endpoint_event_id,
                    'price_chart' AS endpoint,
                    CAST(trade_date AS VARCHAR) AS effective_at,
                    CAST(NULL AS VARCHAR) AS observed_at,
                    CAST(NULL AS VARCHAR) AS payload_hash
                FROM price_features_daily
                WHERE rs_6mo IS NOT NULL

                UNION ALL

                SELECT
                    conid,
                    trade_date,
                    'rs_1y' AS feature_name,
                    'price' AS feature_group,
                    rs_1y AS feature_value,
                    'price_features_daily:rs_1y' AS feature_source,
                    CAST(NULL AS BIGINT) AS endpoint_event_id,
                    'price_chart' AS endpoint,
                    CAST(trade_date AS VARCHAR) AS effective_at,
                    CAST(NULL AS VARCHAR) AS observed_at,
                    CAST(NULL AS VARCHAR) AS payload_hash
                FROM price_features_daily
                WHERE rs_1y IS NOT NULL
                """
            )

            db.execute(
                """
                CREATE OR REPLACE VIEW factor_panel_long_daily AS
                WITH calendar AS (
                    SELECT DISTINCT conid, trade_date
                    FROM price_features_daily
                    WHERE trade_date IS NOT NULL
                ),
                ranked_asof AS (
                    SELECT
                        c.conid,
                        c.trade_date,
                        f.feature_name,
                        f.feature_group,
                        f.feature_value,
                        f.feature_source,
                        f.endpoint_event_id,
                        f.endpoint,
                        f.effective_at,
                        f.observed_at,
                        f.payload_hash,
                        ROW_NUMBER() OVER (
                            PARTITION BY c.conid, c.trade_date, f.feature_name
                            ORDER BY TRY_CAST(f.effective_at AS DATE) DESC,
                                     TRY_CAST(f.observed_at AS TIMESTAMP) DESC,
                                     f.endpoint_event_id DESC
                        ) AS rn
                    FROM calendar c
                    JOIN factor_features_all f
                      ON f.conid = c.conid
                     AND TRY_CAST(f.effective_at AS DATE) <= c.trade_date
                )
                SELECT
                    conid,
                    trade_date,
                    feature_name,
                    feature_group,
                    feature_value,
                    feature_source,
                    endpoint,
                    endpoint_event_id,
                    effective_at,
                    observed_at,
                    payload_hash
                FROM ranked_asof
                WHERE rn = 1

                UNION ALL

                SELECT
                    conid,
                    trade_date,
                    feature_name,
                    feature_group,
                    feature_value,
                    feature_source,
                    endpoint,
                    endpoint_event_id,
                    effective_at,
                    observed_at,
                    payload_hash
                FROM price_features_long_daily
                """
            )
        finally:
            db.close()

        return {
            "endpoint_views": len(endpoint_slugs),
            "factor_views": 4,
            "price_chart_views": 2,
            "sentiment_search_views": 2,
            "ownership_trade_log_views": 2,
            "dividends_events_views": 2,
            "duckdb_path": str(self.duckdb_path),
        }

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
            "refresh_views": refresh_views,
        }
    )
