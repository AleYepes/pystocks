import hashlib
import json
import logging
import math
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import zstandard as zstd

from .config import SQLITE_DB_PATH
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

SERIES_ENDPOINTS = {"price_chart", "sentiment_search", "ownership", "dividends"}

_MONTH_MAP = {
    "JAN": 1,
    "JANUARY": 1,
    "FEB": 2,
    "FEBRUARY": 2,
    "MAR": 3,
    "MARCH": 3,
    "APR": 4,
    "APRIL": 4,
    "MAY": 5,
    "JUN": 6,
    "JUNE": 6,
    "JUL": 7,
    "JULY": 7,
    "AUG": 8,
    "AUGUST": 8,
    "SEP": 9,
    "SEPT": 9,
    "SEPTEMBER": 9,
    "OCT": 10,
    "OCTOBER": 10,
    "NOV": 11,
    "NOVEMBER": 11,
    "DEC": 12,
    "DECEMBER": 12,
}

_DATE_IN_TEXT_RE = re.compile(r"(?<!\d)(\d{4}[/-]\d{2}[/-]\d{2})(?!\d)")
_NUM_WITH_SUFFIX_RE = re.compile(r"^[\s\$€£¥]*([+-]?\d+(?:\.\d+)?)\s*([KMBT])?\b", re.IGNORECASE)


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


def _parse_ymd_text(value):
    if not value:
        return None
    m = _DATE_IN_TEXT_RE.search(str(value))
    if not m:
        return None
    return _parse_date_candidate(m.group(1))


def _to_iso_date(value):
    if value is None:
        return None

    if isinstance(value, dict):
        if "t" in value:
            parsed = _parse_date_candidate(value.get("t"))
            if parsed is not None:
                return parsed.isoformat()
        d = value.get("d")
        m = value.get("m")
        y = value.get("y")
        if y is not None and m is not None and d is not None:
            try:
                mi = int(m) if isinstance(m, (int, float)) else _MONTH_MAP.get(str(m).strip().upper())
                if mi is None:
                    return None
                return datetime(int(y), int(mi), int(d), tzinfo=timezone.utc).date().isoformat()
            except Exception:
                return None
        return None

    parsed = _parse_date_candidate(value)
    return parsed.isoformat() if parsed is not None else None


def _parse_number(value, percent_as_fraction=False):
    if value is None:
        return None

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float)):
        out = float(value)
        if math.isfinite(out):
            return out
        return None

    s = str(value).strip()
    if not s:
        return None

    if s.lower() in {"na", "n/a", "none", "null", "unknown", "nan", "inf", "-inf"}:
        return None

    is_percent = "%" in s
    plain = s.replace(",", "").replace("%", "").strip()

    try:
        out = float(plain)
        if is_percent and percent_as_fraction:
            out /= 100.0
        if math.isfinite(out):
            return out
        return None
    except Exception:
        pass

    m = _NUM_WITH_SUFFIX_RE.match(s)
    if not m:
        return None

    num = float(m.group(1))
    suffix = (m.group(2) or "").upper()
    mul = {"": 1.0, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}.get(suffix, 1.0)
    out = num * mul
    if is_percent and percent_as_fraction:
        out /= 100.0
    return out if math.isfinite(out) else None


def _to_fraction_weight(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        if abs(v) > 1.0:
            return v / 100.0
        return v

    s = str(value).strip()
    if not s:
        return None
    if "%" in s:
        p = _parse_number(s, percent_as_fraction=True)
        return p

    p = _parse_number(s)
    if p is None:
        return None
    if abs(p) > 1.0:
        return p / 100.0
    return p


def _to_int_bool(value):
    if isinstance(value, bool):
        return 1 if value else 0
    if value in {0, 1}:
        return int(value)
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
    out = {k: v for k, v in payload.items()}
    out_sentiment = []
    for row in sentiment:
        if not isinstance(row, dict):
            continue
        row_copy = {k: v for k, v in row.items() if k not in drop_keys}
        out_sentiment.append(row_copy)
    out["sentiment"] = out_sentiment
    return out


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
            "sscore": point.get("sscore"),
            "sdelta": point.get("sdelta"),
            "svolatility": point.get("svolatility"),
            "sdispersion": point.get("sdispersion"),
            "svscore": point.get("svscore"),
            "svolume": point.get("svolume"),
            "smean": point.get("smean"),
            "sbuzz": point.get("sbuzz"),
        }

        if any(v is not None for v in row.values()):
            rows.append(row)

    return rows


def _collect_scalar_paths(node, prefix=""):
    out = []
    if isinstance(node, dict):
        for key, value in node.items():
            key_name = _sanitize_segment(key)
            path = f"{prefix}.{key_name}" if prefix else key_name
            if isinstance(value, (dict, list)):
                continue
            out.append((path, value))
    return out


class FundamentalsStore:
    def __init__(self, sqlite_path=SQLITE_DB_PATH):
        self.sqlite_path = Path(sqlite_path)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._compressor = zstd.ZstdCompressor(level=10)
        self._decompressor = zstd.ZstdDecompressor()
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(str(self.sqlite_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    schema_version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS products (
                    conid TEXT PRIMARY KEY,
                    symbol TEXT,
                    exchange TEXT,
                    isin TEXT,
                    currency TEXT,
                    name TEXT,
                    last_scraped_fundamentals TEXT,
                    last_status_fundamentals TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS raw_payload_blobs (
                    payload_hash TEXT PRIMARY KEY,
                    compression TEXT NOT NULL,
                    raw_size_bytes INTEGER NOT NULL,
                    compressed_size_bytes INTEGER NOT NULL,
                    payload_blob BLOB NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ingest_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_started_at TEXT NOT NULL,
                    run_finished_at TEXT NOT NULL,
                    total_targeted_conids INTEGER NOT NULL,
                    processed_conids INTEGER NOT NULL,
                    saved_snapshots INTEGER NOT NULL,
                    inserted_events INTEGER NOT NULL,
                    overwritten_events INTEGER NOT NULL,
                    unchanged_events INTEGER NOT NULL,
                    series_raw_rows_written INTEGER NOT NULL,
                    series_latest_rows_upserted INTEGER NOT NULL,
                    auth_retries INTEGER NOT NULL,
                    aborted INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ingest_run_endpoint_rollups (
                    run_id INTEGER NOT NULL,
                    endpoint TEXT NOT NULL,
                    call_count INTEGER NOT NULL,
                    useful_payload_count INTEGER NOT NULL,
                    useful_payload_rate REAL NOT NULL,
                    status_2xx INTEGER NOT NULL,
                    status_4xx INTEGER NOT NULL,
                    status_5xx INTEGER NOT NULL,
                    status_other INTEGER NOT NULL,
                    PRIMARY KEY (run_id, endpoint),
                    FOREIGN KEY (run_id) REFERENCES ingest_runs(run_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS endpoint_scalar_extras (
                    endpoint TEXT NOT NULL,
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    path TEXT NOT NULL,
                    value_text TEXT,
                    value_num REAL,
                    value_bool INTEGER,
                    value_date TEXT,
                    PRIMARY KEY (endpoint, conid, effective_at, path),
                    FOREIGN KEY (conid) REFERENCES products(conid)
                );

                CREATE TABLE IF NOT EXISTS landing_snapshots (
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    source_file TEXT,
                    inserted_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    total_net_assets_text TEXT,
                    has_mstar INTEGER,
                    has_ownership INTEGER,
                    has_mf_esg INTEGER,
                    PRIMARY KEY (conid, effective_at),
                    FOREIGN KEY (conid) REFERENCES products(conid),
                    FOREIGN KEY (payload_hash) REFERENCES raw_payload_blobs(payload_hash)
                );

                CREATE TABLE IF NOT EXISTS landing_key_profile_fields (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    field_key TEXT NOT NULL,
                    field_value_text TEXT,
                    field_value_num REAL,
                    FOREIGN KEY (conid) REFERENCES products(conid)
                );

                CREATE TABLE IF NOT EXISTS landing_section_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    section_name TEXT NOT NULL,
                    metric_key TEXT NOT NULL,
                    metric_name TEXT,
                    value_num REAL,
                    vs_num REAL,
                    rank_num REAL,
                    value_text TEXT,
                    value_fmt TEXT,
                    annualized INTEGER,
                    FOREIGN KEY (conid) REFERENCES products(conid)
                );

                CREATE TABLE IF NOT EXISTS landing_top10_holdings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    rank_num INTEGER,
                    name TEXT,
                    ticker TEXT,
                    assets_pct_text TEXT,
                    assets_pct_num REAL,
                    FOREIGN KEY (conid) REFERENCES products(conid)
                );

                CREATE TABLE IF NOT EXISTS landing_top10_holding_conids (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    parent_rank INTEGER,
                    holding_conid TEXT,
                    FOREIGN KEY (conid) REFERENCES products(conid)
                );

                CREATE TABLE IF NOT EXISTS profile_fees_snapshots (
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    source_file TEXT,
                    inserted_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    symbol TEXT,
                    objective TEXT,
                    jap_fund_warning INTEGER,
                    PRIMARY KEY (conid, effective_at),
                    FOREIGN KEY (conid) REFERENCES products(conid),
                    FOREIGN KEY (payload_hash) REFERENCES raw_payload_blobs(payload_hash)
                );

                CREATE TABLE IF NOT EXISTS profile_fees_fund_profile_fields (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    field_name TEXT,
                    name_tag TEXT,
                    value_text TEXT,
                    value_num REAL,
                    value_date TEXT,
                    FOREIGN KEY (conid) REFERENCES products(conid)
                );

                CREATE TABLE IF NOT EXISTS profile_fees_report_fields (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    report_name TEXT,
                    report_as_of_date TEXT,
                    field_name TEXT,
                    field_value_text TEXT,
                    field_value_num REAL,
                    is_summary INTEGER,
                    FOREIGN KEY (conid) REFERENCES products(conid)
                );

                CREATE TABLE IF NOT EXISTS profile_fees_expense_allocations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    expense_name TEXT,
                    ratio_text TEXT,
                    value_num REAL,
                    FOREIGN KEY (conid) REFERENCES products(conid)
                );

                CREATE TABLE IF NOT EXISTS profile_fees_themes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    theme_name TEXT,
                    FOREIGN KEY (conid) REFERENCES products(conid)
                );

                CREATE TABLE IF NOT EXISTS holdings_snapshots (
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    source_file TEXT,
                    inserted_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    as_of_date TEXT,
                    top_10_weight REAL,
                    PRIMARY KEY (conid, effective_at),
                    FOREIGN KEY (conid) REFERENCES products(conid),
                    FOREIGN KEY (payload_hash) REFERENCES raw_payload_blobs(payload_hash)
                );

                CREATE TABLE IF NOT EXISTS holdings_bucket_weights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    bucket_type TEXT NOT NULL,
                    name TEXT,
                    code TEXT,
                    country_code TEXT,
                    rank_num REAL,
                    weight_num REAL,
                    vs_num REAL,
                    formatted_weight TEXT,
                    FOREIGN KEY (conid) REFERENCES products(conid)
                );

                CREATE TABLE IF NOT EXISTS holdings_top10 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    rank_num INTEGER,
                    name TEXT,
                    ticker TEXT,
                    assets_pct_text TEXT,
                    assets_pct_num REAL,
                    FOREIGN KEY (conid) REFERENCES products(conid)
                );

                CREATE TABLE IF NOT EXISTS holdings_top10_conids (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    parent_rank INTEGER,
                    holding_conid TEXT,
                    FOREIGN KEY (conid) REFERENCES products(conid)
                );

                CREATE TABLE IF NOT EXISTS holdings_geographic_weights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    region TEXT,
                    weight_num REAL,
                    FOREIGN KEY (conid) REFERENCES products(conid)
                );

                CREATE TABLE IF NOT EXISTS ratios_snapshots (
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    source_file TEXT,
                    inserted_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    as_of_date TEXT,
                    title_vs TEXT,
                    PRIMARY KEY (conid, effective_at),
                    FOREIGN KEY (conid) REFERENCES products(conid),
                    FOREIGN KEY (payload_hash) REFERENCES raw_payload_blobs(payload_hash)
                );

                CREATE TABLE IF NOT EXISTS ratios_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    section TEXT,
                    metric_id TEXT,
                    metric_name TEXT,
                    value_num REAL,
                    value_fmt TEXT,
                    vs_num REAL,
                    min_num REAL,
                    max_num REAL,
                    avg_num REAL,
                    percentile_num REAL,
                    min_fmt TEXT,
                    max_fmt TEXT,
                    avg_fmt TEXT,
                    FOREIGN KEY (conid) REFERENCES products(conid)
                );

                CREATE TABLE IF NOT EXISTS lipper_ratings_snapshots (
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    source_file TEXT,
                    inserted_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    universe_count INTEGER,
                    PRIMARY KEY (conid, effective_at),
                    FOREIGN KEY (conid) REFERENCES products(conid),
                    FOREIGN KEY (payload_hash) REFERENCES raw_payload_blobs(payload_hash)
                );

                CREATE TABLE IF NOT EXISTS lipper_ratings_values (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    period TEXT,
                    metric_id TEXT,
                    metric_name TEXT,
                    rating_value REAL,
                    rating_label TEXT,
                    universe_name TEXT,
                    universe_as_of_date TEXT,
                    FOREIGN KEY (conid) REFERENCES products(conid)
                );

                CREATE TABLE IF NOT EXISTS dividends_snapshots (
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    source_file TEXT,
                    inserted_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    response_type TEXT,
                    has_history INTEGER,
                    history_points INTEGER,
                    embedded_price_points INTEGER,
                    no_div_data_marker REAL,
                    no_div_data_period REAL,
                    no_dividend_text TEXT,
                    last_paid_date TEXT,
                    last_paid_amount REAL,
                    last_paid_currency TEXT,
                    dividend_yield REAL,
                    annual_dividend REAL,
                    paying_companies REAL,
                    paying_companies_percent REAL,
                    dividend_ttm REAL,
                    dividend_yield_ttm REAL,
                    PRIMARY KEY (conid, effective_at),
                    FOREIGN KEY (conid) REFERENCES products(conid),
                    FOREIGN KEY (payload_hash) REFERENCES raw_payload_blobs(payload_hash)
                );

                CREATE TABLE IF NOT EXISTS dividends_industry_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    metric_id TEXT,
                    value_num REAL,
                    formatted_value TEXT,
                    FOREIGN KEY (conid) REFERENCES products(conid)
                );

                CREATE TABLE IF NOT EXISTS morningstar_snapshots (
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    source_file TEXT,
                    inserted_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    as_of_date TEXT,
                    q_full_report_id TEXT,
                    PRIMARY KEY (conid, effective_at),
                    FOREIGN KEY (conid) REFERENCES products(conid),
                    FOREIGN KEY (payload_hash) REFERENCES raw_payload_blobs(payload_hash)
                );

                CREATE TABLE IF NOT EXISTS morningstar_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    metric_id TEXT,
                    title TEXT,
                    value_text TEXT,
                    value_num REAL,
                    publish_date TEXT,
                    q INTEGER,
                    FOREIGN KEY (conid) REFERENCES products(conid)
                );

                CREATE TABLE IF NOT EXISTS morningstar_commentary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    item_id TEXT,
                    title TEXT,
                    subtitle TEXT,
                    subsection_id TEXT,
                    publish_date TEXT,
                    q INTEGER,
                    text TEXT,
                    author_name TEXT,
                    FOREIGN KEY (conid) REFERENCES products(conid)
                );

                CREATE TABLE IF NOT EXISTS performance_snapshots (
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    source_file TEXT,
                    inserted_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    title_vs TEXT,
                    PRIMARY KEY (conid, effective_at),
                    FOREIGN KEY (conid) REFERENCES products(conid),
                    FOREIGN KEY (payload_hash) REFERENCES raw_payload_blobs(payload_hash)
                );

                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    section TEXT,
                    metric_id TEXT,
                    metric_name TEXT,
                    name_tag_arg TEXT,
                    value_num REAL,
                    value_fmt TEXT,
                    vs_num REAL,
                    min_num REAL,
                    max_num REAL,
                    avg_num REAL,
                    percentile_num REAL,
                    min_fmt TEXT,
                    max_fmt TEXT,
                    avg_fmt TEXT,
                    FOREIGN KEY (conid) REFERENCES products(conid)
                );

                CREATE TABLE IF NOT EXISTS ownership_snapshots (
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    source_file TEXT,
                    inserted_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    owners_types_count INTEGER,
                    institutional_owners_count INTEGER,
                    insider_owners_count INTEGER,
                    trade_log_count_raw INTEGER,
                    trade_log_count_kept INTEGER,
                    has_ownership_history INTEGER,
                    ownership_history_price_points INTEGER,
                    institutional_total_value_text TEXT,
                    institutional_total_shares_text TEXT,
                    institutional_total_pct_text TEXT,
                    institutional_total_pct_num REAL,
                    insider_total_value_text TEXT,
                    insider_total_shares_text TEXT,
                    insider_total_pct_text TEXT,
                    insider_total_pct_num REAL,
                    PRIMARY KEY (conid, effective_at),
                    FOREIGN KEY (conid) REFERENCES products(conid),
                    FOREIGN KEY (payload_hash) REFERENCES raw_payload_blobs(payload_hash)
                );

                CREATE TABLE IF NOT EXISTS ownership_owners_types (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    type TEXT,
                    display_type TEXT,
                    float_value REAL,
                    display_float TEXT,
                    FOREIGN KEY (conid) REFERENCES products(conid)
                );

                CREATE TABLE IF NOT EXISTS ownership_holders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    holder_group TEXT,
                    holder_name TEXT,
                    holder_type TEXT,
                    display_value TEXT,
                    display_shares TEXT,
                    display_pct TEXT,
                    pct_num REAL,
                    FOREIGN KEY (conid) REFERENCES products(conid)
                );

                CREATE TABLE IF NOT EXISTS esg_snapshots (
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    source_file TEXT,
                    inserted_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    as_of_date TEXT,
                    coverage REAL,
                    source TEXT,
                    symbol TEXT,
                    no_settings INTEGER,
                    PRIMARY KEY (conid, effective_at),
                    FOREIGN KEY (conid) REFERENCES products(conid),
                    FOREIGN KEY (payload_hash) REFERENCES raw_payload_blobs(payload_hash)
                );

                CREATE TABLE IF NOT EXISTS esg_nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    node_path TEXT,
                    parent_path TEXT,
                    depth INTEGER,
                    node_name TEXT,
                    node_value REAL,
                    FOREIGN KEY (conid) REFERENCES products(conid)
                );

                CREATE TABLE IF NOT EXISTS price_chart_snapshots (
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    source_file TEXT,
                    inserted_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    points_count INTEGER,
                    min_trade_date TEXT,
                    max_trade_date TEXT,
                    PRIMARY KEY (conid, effective_at),
                    FOREIGN KEY (conid) REFERENCES products(conid),
                    FOREIGN KEY (payload_hash) REFERENCES raw_payload_blobs(payload_hash)
                );

                CREATE TABLE IF NOT EXISTS sentiment_search_snapshots (
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    source_file TEXT,
                    inserted_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    points_count INTEGER,
                    min_trade_date TEXT,
                    max_trade_date TEXT,
                    PRIMARY KEY (conid, effective_at),
                    FOREIGN KEY (conid) REFERENCES products(conid),
                    FOREIGN KEY (payload_hash) REFERENCES raw_payload_blobs(payload_hash)
                );

                CREATE TABLE IF NOT EXISTS price_chart_series_raw (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    inserted_at TEXT NOT NULL,
                    row_key TEXT NOT NULL,
                    trade_date TEXT,
                    timestamp_ms INTEGER,
                    price REAL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    debug_y INTEGER,
                    FOREIGN KEY (conid) REFERENCES products(conid)
                );

                CREATE TABLE IF NOT EXISTS price_chart_series_latest (
                    conid TEXT NOT NULL,
                    row_key TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    trade_date TEXT,
                    timestamp_ms INTEGER,
                    price REAL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    debug_y INTEGER,
                    PRIMARY KEY (conid, row_key),
                    FOREIGN KEY (conid) REFERENCES products(conid)
                );

                CREATE TABLE IF NOT EXISTS sentiment_search_series_raw (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    inserted_at TEXT NOT NULL,
                    row_key TEXT NOT NULL,
                    trade_date TEXT,
                    datetime_ms INTEGER,
                    sscore REAL,
                    sdelta REAL,
                    svolatility REAL,
                    sdispersion REAL,
                    svscore REAL,
                    svolume REAL,
                    smean REAL,
                    sbuzz REAL,
                    FOREIGN KEY (conid) REFERENCES products(conid)
                );

                CREATE TABLE IF NOT EXISTS sentiment_search_series_latest (
                    conid TEXT NOT NULL,
                    row_key TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    trade_date TEXT,
                    datetime_ms INTEGER,
                    sscore REAL,
                    sdelta REAL,
                    svolatility REAL,
                    sdispersion REAL,
                    svscore REAL,
                    svolume REAL,
                    smean REAL,
                    sbuzz REAL,
                    PRIMARY KEY (conid, row_key),
                    FOREIGN KEY (conid) REFERENCES products(conid)
                );

                CREATE TABLE IF NOT EXISTS ownership_trade_log_series_raw (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    inserted_at TEXT NOT NULL,
                    row_key TEXT NOT NULL,
                    trade_date TEXT,
                    action TEXT,
                    shares REAL,
                    value REAL,
                    holding REAL,
                    party TEXT,
                    source TEXT,
                    insider TEXT,
                    FOREIGN KEY (conid) REFERENCES products(conid)
                );

                CREATE TABLE IF NOT EXISTS ownership_trade_log_series_latest (
                    conid TEXT NOT NULL,
                    row_key TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    trade_date TEXT,
                    action TEXT,
                    shares REAL,
                    value REAL,
                    holding REAL,
                    party TEXT,
                    source TEXT,
                    insider TEXT,
                    PRIMARY KEY (conid, row_key),
                    FOREIGN KEY (conid) REFERENCES products(conid)
                );

                CREATE TABLE IF NOT EXISTS dividends_events_series_raw (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conid TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    inserted_at TEXT NOT NULL,
                    row_key TEXT NOT NULL,
                    trade_date TEXT,
                    event_date TEXT,
                    amount REAL,
                    currency TEXT,
                    description TEXT,
                    event_type TEXT,
                    declaration_date TEXT,
                    record_date TEXT,
                    payment_date TEXT,
                    FOREIGN KEY (conid) REFERENCES products(conid)
                );

                CREATE TABLE IF NOT EXISTS dividends_events_series_latest (
                    conid TEXT NOT NULL,
                    row_key TEXT NOT NULL,
                    effective_at TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    trade_date TEXT,
                    event_date TEXT,
                    amount REAL,
                    currency TEXT,
                    description TEXT,
                    event_type TEXT,
                    declaration_date TEXT,
                    record_date TEXT,
                    payment_date TEXT,
                    PRIMARY KEY (conid, row_key),
                    FOREIGN KEY (conid) REFERENCES products(conid)
                );
                """
            )

            conn.execute(
                """
                INSERT OR IGNORE INTO schema_meta (schema_version, applied_at)
                VALUES (1, ?)
                """,
                [datetime.now(timezone.utc).isoformat()],
            )

            conn.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_landing_snapshots_hash ON landing_snapshots(payload_hash);
                CREATE INDEX IF NOT EXISTS idx_profile_fees_snapshots_hash ON profile_fees_snapshots(payload_hash);
                CREATE INDEX IF NOT EXISTS idx_holdings_snapshots_hash ON holdings_snapshots(payload_hash);
                CREATE INDEX IF NOT EXISTS idx_ratios_snapshots_hash ON ratios_snapshots(payload_hash);
                CREATE INDEX IF NOT EXISTS idx_lipper_snapshots_hash ON lipper_ratings_snapshots(payload_hash);
                CREATE INDEX IF NOT EXISTS idx_dividends_snapshots_hash ON dividends_snapshots(payload_hash);
                CREATE INDEX IF NOT EXISTS idx_morningstar_snapshots_hash ON morningstar_snapshots(payload_hash);
                CREATE INDEX IF NOT EXISTS idx_performance_snapshots_hash ON performance_snapshots(payload_hash);
                CREATE INDEX IF NOT EXISTS idx_ownership_snapshots_hash ON ownership_snapshots(payload_hash);
                CREATE INDEX IF NOT EXISTS idx_esg_snapshots_hash ON esg_snapshots(payload_hash);
                CREATE INDEX IF NOT EXISTS idx_price_snapshots_hash ON price_chart_snapshots(payload_hash);
                CREATE INDEX IF NOT EXISTS idx_sentiment_snapshots_hash ON sentiment_search_snapshots(payload_hash);

                CREATE INDEX IF NOT EXISTS idx_price_latest_trade_date ON price_chart_series_latest(trade_date);
                CREATE INDEX IF NOT EXISTS idx_sentiment_latest_trade_date ON sentiment_search_series_latest(trade_date);
                CREATE INDEX IF NOT EXISTS idx_ownership_latest_trade_date ON ownership_trade_log_series_latest(trade_date);
                CREATE INDEX IF NOT EXISTS idx_dividends_latest_trade_date ON dividends_events_series_latest(event_date);
                """
            )

    def _ensure_product(self, conn, conid):
        now_iso = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT INTO products (conid, updated_at)
            VALUES (?, ?)
            ON CONFLICT(conid) DO NOTHING
            """,
            [str(conid), now_iso],
        )

    def _store_blob(self, conn, payload):
        raw_bytes = _canonical_json_bytes(payload)
        payload_hash = hashlib.sha256(raw_bytes).hexdigest()

        existing = conn.execute(
            """
            SELECT payload_hash
            FROM raw_payload_blobs
            WHERE payload_hash = ?
            """,
            [payload_hash],
        ).fetchone()

        if existing is None:
            compressed = self._compressor.compress(raw_bytes)
            conn.execute(
                """
                INSERT INTO raw_payload_blobs (
                    payload_hash,
                    compression,
                    raw_size_bytes,
                    compressed_size_bytes,
                    payload_blob,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    payload_hash,
                    "zstd",
                    len(raw_bytes),
                    len(compressed),
                    compressed,
                    datetime.now(timezone.utc).isoformat(),
                ],
            )

            return {
                "hash": payload_hash,
                "raw_size": len(raw_bytes),
                "compressed_size": len(compressed),
            }

        row = conn.execute(
            """
            SELECT raw_size_bytes, compressed_size_bytes
            FROM raw_payload_blobs
            WHERE payload_hash = ?
            """,
            [payload_hash],
        ).fetchone()

        return {
            "hash": payload_hash,
            "raw_size": int(row["raw_size_bytes"]),
            "compressed_size": int(row["compressed_size_bytes"]),
        }

    def _load_blob_payload(self, payload_hash):
        with self._get_conn() as conn:
            row = conn.execute(
                """
                SELECT payload_blob
                FROM raw_payload_blobs
                WHERE payload_hash = ?
                """,
                [str(payload_hash)],
            ).fetchone()
            if row is None:
                return None
            raw = self._decompressor.decompress(row["payload_blob"])
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

    def _main_table_for_endpoint(self, endpoint):
        return {
            "landing": "landing_snapshots",
            "profile_and_fees": "profile_fees_snapshots",
            "holdings": "holdings_snapshots",
            "ratios": "ratios_snapshots",
            "lipper_ratings": "lipper_ratings_snapshots",
            "dividends": "dividends_snapshots",
            "morningstar": "morningstar_snapshots",
            "price_chart": "price_chart_snapshots",
            "performance": "performance_snapshots",
            "sentiment_search": "sentiment_search_snapshots",
            "ownership": "ownership_snapshots",
            "esg": "esg_snapshots",
        }[endpoint]

    def _upsert_snapshot_row(
        self,
        conn,
        table,
        conid,
        effective_at,
        observed_at,
        payload_hash,
        source_file,
        now_iso,
        extra,
    ):
        base = {
            "conid": str(conid),
            "effective_at": str(effective_at),
            "observed_at": str(observed_at),
            "payload_hash": str(payload_hash),
            "source_file": source_file,
            "inserted_at": now_iso,
            "updated_at": now_iso,
        }
        row = dict(base)
        row.update(extra)

        cols = list(row.keys())
        placeholders = ", ".join(["?"] * len(cols))
        update_cols = [c for c in cols if c not in {"conid", "effective_at", "inserted_at"}]
        update_sql = ", ".join([f"{c}=excluded.{c}" for c in update_cols])

        conn.execute(
            f"""
            INSERT INTO {table} ({", ".join(cols)})
            VALUES ({placeholders})
            ON CONFLICT(conid, effective_at) DO UPDATE SET
                {update_sql}
            """,
            [row[c] for c in cols],
        )

    def _delete_children(self, conn, tables, conid, effective_at):
        for table in tables:
            conn.execute(
                f"DELETE FROM {table} WHERE conid = ? AND effective_at = ?",
                [str(conid), str(effective_at)],
            )

    def _insert_rows(self, conn, table, rows):
        if not rows:
            return
        cols = list(rows[0].keys())
        placeholders = ", ".join(["?"] * len(cols))
        conn.executemany(
            f"INSERT INTO {table} ({", ".join(cols)}) VALUES ({placeholders})",
            [[row.get(c) for c in cols] for row in rows],
        )

    def _store_endpoint_scalar_extras(self, conn, endpoint, conid, effective_at, payload):
        conn.execute(
            """
            DELETE FROM endpoint_scalar_extras
            WHERE endpoint = ? AND conid = ? AND effective_at = ?
            """,
            [endpoint, str(conid), str(effective_at)],
        )

        if not isinstance(payload, dict):
            return

        rows = []
        for path, value in _collect_scalar_paths(payload):
            rows.append(
                {
                    "endpoint": endpoint,
                    "conid": str(conid),
                    "effective_at": str(effective_at),
                    "path": path,
                    "value_text": str(value) if value is not None else None,
                    "value_num": _parse_number(value),
                    "value_bool": _to_int_bool(value),
                    "value_date": _to_iso_date(value),
                }
            )

        self._insert_rows(conn, "endpoint_scalar_extras", rows)

    def _upsert_landing(self, conn, conid, effective_at, observed_at, payload_hash, source_file, now_iso, payload):
        key_profile = payload.get("key_profile") or payload.get("keyProfile") or {}
        key_profile_data = key_profile.get("data") if isinstance(key_profile, dict) else {}
        total_net_assets_text = None
        if isinstance(key_profile_data, dict):
            total_net_assets_text = key_profile_data.get("total_net_assets")

        self._upsert_snapshot_row(
            conn,
            "landing_snapshots",
            conid,
            effective_at,
            observed_at,
            payload_hash,
            source_file,
            now_iso,
            {
                "total_net_assets_text": total_net_assets_text,
                "has_mstar": 1 if bool(payload.get("mstar")) else 0,
                "has_ownership": 1 if bool(payload.get("ownership")) else 0,
                "has_mf_esg": 1 if bool(payload.get("mf_esg")) else 0,
            },
        )

        self._delete_children(
            conn,
            [
                "landing_key_profile_fields",
                "landing_section_metrics",
                "landing_top10_holdings",
                "landing_top10_holding_conids",
            ],
            conid,
            effective_at,
        )

        key_profile_rows = []
        if isinstance(key_profile_data, dict):
            for key, value in key_profile_data.items():
                if isinstance(value, (dict, list)):
                    continue
                key_profile_rows.append(
                    {
                        "conid": str(conid),
                        "effective_at": str(effective_at),
                        "field_key": _sanitize_segment(key),
                        "field_value_text": str(value) if value is not None else None,
                        "field_value_num": _parse_number(value),
                    }
                )
        self._insert_rows(conn, "landing_key_profile_fields", key_profile_rows)

        section_rows = []
        for section_name, section_payload in payload.items():
            if not isinstance(section_payload, dict):
                continue
            data = section_payload.get("data")
            if not isinstance(data, list):
                continue
            for item in data:
                if not isinstance(item, dict):
                    continue
                metric_key = _sanitize_segment(item.get("name_tag") or item.get("id") or item.get("name"))
                section_rows.append(
                    {
                        "conid": str(conid),
                        "effective_at": str(effective_at),
                        "section_name": _sanitize_segment(section_name),
                        "metric_key": metric_key,
                        "metric_name": item.get("name"),
                        "value_num": _parse_number(item.get("value")),
                        "vs_num": _parse_number(item.get("vs_value") if "vs_value" in item else item.get("vs")),
                        "rank_num": _parse_number(item.get("rank")),
                        "value_text": str(item.get("value")) if item.get("value") is not None else None,
                        "value_fmt": item.get("value_fmt"),
                        "annualized": _to_int_bool(item.get("annualized")),
                    }
                )
        self._insert_rows(conn, "landing_section_metrics", section_rows)

        top10_rows = []
        top10_conids_rows = []
        top10 = (
            payload.get("top10", {})
            .get("data", {})
            .get("top10", [])
            if isinstance(payload.get("top10"), dict)
            else []
        )
        if isinstance(top10, list):
            for idx, item in enumerate(top10):
                if not isinstance(item, dict):
                    continue
                rank_num = int(_parse_number(item.get("rank")) or (idx + 1))
                top10_rows.append(
                    {
                        "conid": str(conid),
                        "effective_at": str(effective_at),
                        "rank_num": rank_num,
                        "name": item.get("name"),
                        "ticker": item.get("ticker"),
                        "assets_pct_text": item.get("assets_pct"),
                        "assets_pct_num": _to_fraction_weight(item.get("assets_pct")),
                    }
                )
                for c in item.get("conids", []) if isinstance(item.get("conids"), list) else []:
                    top10_conids_rows.append(
                        {
                            "conid": str(conid),
                            "effective_at": str(effective_at),
                            "parent_rank": rank_num,
                            "holding_conid": str(c),
                        }
                    )
        self._insert_rows(conn, "landing_top10_holdings", top10_rows)
        self._insert_rows(conn, "landing_top10_holding_conids", top10_conids_rows)

    def _upsert_profile_fees(self, conn, conid, effective_at, observed_at, payload_hash, source_file, now_iso, payload):
        self._upsert_snapshot_row(
            conn,
            "profile_fees_snapshots",
            conid,
            effective_at,
            observed_at,
            payload_hash,
            source_file,
            now_iso,
            {
                "symbol": payload.get("symbol"),
                "objective": payload.get("objective"),
                "jap_fund_warning": _to_int_bool(payload.get("jap_fund_warning")),
            },
        )

        self._delete_children(
            conn,
            [
                "profile_fees_fund_profile_fields",
                "profile_fees_report_fields",
                "profile_fees_expense_allocations",
                "profile_fees_themes",
            ],
            conid,
            effective_at,
        )

        profile_rows = []
        for item in payload.get("fund_and_profile", []) if isinstance(payload.get("fund_and_profile"), list) else []:
            if not isinstance(item, dict):
                continue
            value = item.get("value")
            profile_rows.append(
                {
                    "conid": str(conid),
                    "effective_at": str(effective_at),
                    "field_name": item.get("name"),
                    "name_tag": item.get("name_tag"),
                    "value_text": str(value) if value is not None else None,
                    "value_num": _parse_number(value, percent_as_fraction=True),
                    "value_date": _to_iso_date(value),
                }
            )
        self._insert_rows(conn, "profile_fees_fund_profile_fields", profile_rows)

        report_rows = []
        for report in payload.get("reports", []) if isinstance(payload.get("reports"), list) else []:
            if not isinstance(report, dict):
                continue
            report_name = report.get("name")
            report_as_of = _to_iso_date(report.get("as_of_date"))
            fields = report.get("fields", []) if isinstance(report.get("fields"), list) else []
            for field in fields:
                if not isinstance(field, dict):
                    continue
                value = field.get("value")
                report_rows.append(
                    {
                        "conid": str(conid),
                        "effective_at": str(effective_at),
                        "report_name": report_name,
                        "report_as_of_date": report_as_of,
                        "field_name": field.get("name") or field.get("name_tag"),
                        "field_value_text": str(value) if value is not None else None,
                        "field_value_num": _parse_number(value, percent_as_fraction=True),
                        "is_summary": _to_int_bool(field.get("is_summary")),
                    }
                )
        self._insert_rows(conn, "profile_fees_report_fields", report_rows)

        expense_rows = []
        for item in payload.get("expenses_allocation", []) if isinstance(payload.get("expenses_allocation"), list) else []:
            if not isinstance(item, dict):
                continue
            expense_rows.append(
                {
                    "conid": str(conid),
                    "effective_at": str(effective_at),
                    "expense_name": item.get("name"),
                    "ratio_text": str(item.get("ratio")) if item.get("ratio") is not None else None,
                    "value_num": _parse_number(item.get("value"), percent_as_fraction=True),
                }
            )
        self._insert_rows(conn, "profile_fees_expense_allocations", expense_rows)

        theme_rows = []
        themes = payload.get("themes", []) if isinstance(payload.get("themes"), list) else []
        for theme in themes:
            if isinstance(theme, dict):
                theme_name = theme.get("name") or theme.get("title") or theme.get("theme")
            else:
                theme_name = str(theme) if theme is not None else None
            if not theme_name:
                continue
            theme_rows.append(
                {
                    "conid": str(conid),
                    "effective_at": str(effective_at),
                    "theme_name": theme_name,
                }
            )
        self._insert_rows(conn, "profile_fees_themes", theme_rows)

    def _upsert_holdings(self, conn, conid, effective_at, observed_at, payload_hash, source_file, now_iso, payload):
        self._upsert_snapshot_row(
            conn,
            "holdings_snapshots",
            conid,
            effective_at,
            observed_at,
            payload_hash,
            source_file,
            now_iso,
            {
                "as_of_date": _to_iso_date(payload.get("as_of_date")),
                "top_10_weight": _to_fraction_weight(payload.get("top_10_weight")),
            },
        )

        self._delete_children(
            conn,
            [
                "holdings_bucket_weights",
                "holdings_top10",
                "holdings_top10_conids",
                "holdings_geographic_weights",
            ],
            conid,
            effective_at,
        )

        bucket_rows = []
        sections = [
            "allocation_self",
            "industry",
            "currency",
            "investor_country",
            "debt_type",
            "debtor",
            "maturity",
        ]
        for section in sections:
            values = payload.get(section, []) if isinstance(payload.get(section), list) else []
            for item in values:
                if not isinstance(item, dict):
                    continue
                weight_value = item.get("weight")
                if weight_value is None:
                    weight_value = item.get("assets_pct")
                if weight_value is None:
                    weight_value = item.get("formatted_weight")

                bucket_rows.append(
                    {
                        "conid": str(conid),
                        "effective_at": str(effective_at),
                        "bucket_type": section,
                        "name": item.get("name") or item.get("type"),
                        "code": item.get("code"),
                        "country_code": item.get("country_code"),
                        "rank_num": _parse_number(item.get("rank")),
                        "weight_num": _to_fraction_weight(weight_value),
                        "vs_num": _parse_number(item.get("vs"), percent_as_fraction=True),
                        "formatted_weight": item.get("formatted_weight"),
                    }
                )
        self._insert_rows(conn, "holdings_bucket_weights", bucket_rows)

        top10_rows = []
        top10_conids_rows = []
        for idx, item in enumerate(payload.get("top_10", []) if isinstance(payload.get("top_10"), list) else []):
            if not isinstance(item, dict):
                continue
            rank_num = int(_parse_number(item.get("rank")) or (idx + 1))
            top10_rows.append(
                {
                    "conid": str(conid),
                    "effective_at": str(effective_at),
                    "rank_num": rank_num,
                    "name": item.get("name"),
                    "ticker": item.get("ticker"),
                    "assets_pct_text": item.get("assets_pct"),
                    "assets_pct_num": _to_fraction_weight(item.get("assets_pct")),
                }
            )

            for c in item.get("conids", []) if isinstance(item.get("conids"), list) else []:
                top10_conids_rows.append(
                    {
                        "conid": str(conid),
                        "effective_at": str(effective_at),
                        "parent_rank": rank_num,
                        "holding_conid": str(c),
                    }
                )
        self._insert_rows(conn, "holdings_top10", top10_rows)
        self._insert_rows(conn, "holdings_top10_conids", top10_conids_rows)

        geographic_rows = []
        geographic = payload.get("geographic") if isinstance(payload.get("geographic"), dict) else {}
        for key, value in geographic.items():
            if isinstance(value, (dict, list)):
                continue
            geographic_rows.append(
                {
                    "conid": str(conid),
                    "effective_at": str(effective_at),
                    "region": _sanitize_segment(key),
                    "weight_num": _to_fraction_weight(value),
                }
            )
        self._insert_rows(conn, "holdings_geographic_weights", geographic_rows)

    def _upsert_ratios(self, conn, conid, effective_at, observed_at, payload_hash, source_file, now_iso, payload):
        self._upsert_snapshot_row(
            conn,
            "ratios_snapshots",
            conid,
            effective_at,
            observed_at,
            payload_hash,
            source_file,
            now_iso,
            {
                "as_of_date": _to_iso_date(payload.get("as_of_date")),
                "title_vs": payload.get("title_vs"),
            },
        )

        self._delete_children(conn, ["ratios_metrics"], conid, effective_at)

        metric_rows = []
        for section in ("ratios", "financials", "fixed_income", "dividend", "zscore"):
            values = payload.get(section, []) if isinstance(payload.get(section), list) else []
            for item in values:
                if not isinstance(item, dict):
                    continue
                metric_rows.append(
                    {
                        "conid": str(conid),
                        "effective_at": str(effective_at),
                        "section": section,
                        "metric_id": _sanitize_segment(item.get("name_tag") or item.get("id") or item.get("name")),
                        "metric_name": item.get("name"),
                        "value_num": _parse_number(item.get("value")),
                        "value_fmt": item.get("value_fmt"),
                        "vs_num": _parse_number(item.get("vs")),
                        "min_num": _parse_number(item.get("min")),
                        "max_num": _parse_number(item.get("max")),
                        "avg_num": _parse_number(item.get("avg")),
                        "percentile_num": _parse_number(item.get("percentile")),
                        "min_fmt": item.get("min_fmt"),
                        "max_fmt": item.get("max_fmt"),
                        "avg_fmt": item.get("avg_fmt"),
                    }
                )
        self._insert_rows(conn, "ratios_metrics", metric_rows)

    def _upsert_lipper(self, conn, conid, effective_at, observed_at, payload_hash, source_file, now_iso, payload):
        universes = payload.get("universes", []) if isinstance(payload.get("universes"), list) else []

        self._upsert_snapshot_row(
            conn,
            "lipper_ratings_snapshots",
            conid,
            effective_at,
            observed_at,
            payload_hash,
            source_file,
            now_iso,
            {
                "universe_count": len(universes),
            },
        )

        self._delete_children(conn, ["lipper_ratings_values"], conid, effective_at)

        rows = []
        for universe in universes:
            if not isinstance(universe, dict):
                continue
            universe_name = universe.get("name")
            universe_as_of = _to_iso_date(universe.get("as_of_date") or universe.get("asOfDate"))
            for period_key, period_items in universe.items():
                if period_key in {"as_of_date", "asOfDate", "name", "title"}:
                    continue
                if not isinstance(period_items, list):
                    continue
                period = _sanitize_segment(period_key)
                for item in period_items:
                    if not isinstance(item, dict):
                        continue
                    rating = item.get("rating") if isinstance(item.get("rating"), dict) else {}
                    rows.append(
                        {
                            "conid": str(conid),
                            "effective_at": str(effective_at),
                            "period": period,
                            "metric_id": _sanitize_segment(item.get("name_tag") or item.get("id") or item.get("name")),
                            "metric_name": item.get("name"),
                            "rating_value": _parse_number(rating.get("value")),
                            "rating_label": rating.get("name"),
                            "universe_name": universe_name,
                            "universe_as_of_date": universe_as_of,
                        }
                    )
        self._insert_rows(conn, "lipper_ratings_values", rows)

    def _upsert_dividends(self, conn, conid, effective_at, observed_at, payload_hash, source_file, now_iso, payload):
        snapshot = normalize_dividends_snapshot(payload)

        self._upsert_snapshot_row(
            conn,
            "dividends_snapshots",
            conid,
            effective_at,
            observed_at,
            payload_hash,
            source_file,
            now_iso,
            {
                "response_type": snapshot.get("response_type"),
                "has_history": _to_int_bool(snapshot.get("has_history")),
                "history_points": int(snapshot.get("history_points") or 0),
                "embedded_price_points": int(snapshot.get("embedded_price_points") or 0),
                "no_div_data_marker": _parse_number(snapshot.get("no_div_data_marker")),
                "no_div_data_period": _parse_number(snapshot.get("no_div_data_period")),
                "no_dividend_text": payload.get("no_dividend_text"),
                "last_paid_date": snapshot.get("last_paid_date"),
                "last_paid_amount": _parse_number(snapshot.get("last_paid_amount")),
                "last_paid_currency": snapshot.get("last_paid_currency"),
                "dividend_yield": _parse_number(snapshot.get("dividend_yield")),
                "annual_dividend": _parse_number(snapshot.get("annual_dividend")),
                "paying_companies": _parse_number(snapshot.get("paying_companies")),
                "paying_companies_percent": _parse_number(snapshot.get("paying_companies_percent")),
                "dividend_ttm": _parse_number(snapshot.get("dividend_ttm")),
                "dividend_yield_ttm": _parse_number(snapshot.get("dividend_yield_ttm")),
            },
        )

        self._delete_children(conn, ["dividends_industry_metrics"], conid, effective_at)

        rows = []
        for item in extract_dividends_industry_metrics(payload):
            rows.append(
                {
                    "conid": str(conid),
                    "effective_at": str(effective_at),
                    "metric_id": _sanitize_segment(item.get("metric_id")),
                    "value_num": _parse_number(item.get("value")),
                    "formatted_value": item.get("formatted_value"),
                }
            )
        self._insert_rows(conn, "dividends_industry_metrics", rows)

    def _upsert_morningstar(self, conn, conid, effective_at, observed_at, payload_hash, source_file, now_iso, payload):
        self._upsert_snapshot_row(
            conn,
            "morningstar_snapshots",
            conid,
            effective_at,
            observed_at,
            payload_hash,
            source_file,
            now_iso,
            {
                "as_of_date": _to_iso_date(payload.get("as_of_date") or payload.get("asOfDate")),
                "q_full_report_id": payload.get("q_full_report_id"),
            },
        )

        self._delete_children(conn, ["morningstar_summary", "morningstar_commentary"], conid, effective_at)

        summary_rows = []
        for item in payload.get("summary", []) if isinstance(payload.get("summary"), list) else []:
            if not isinstance(item, dict):
                continue
            summary_rows.append(
                {
                    "conid": str(conid),
                    "effective_at": str(effective_at),
                    "metric_id": _sanitize_segment(item.get("id") or item.get("title") or "metric"),
                    "title": item.get("title"),
                    "value_text": str(item.get("value")) if item.get("value") is not None else None,
                    "value_num": _parse_number(item.get("value")),
                    "publish_date": _to_iso_date(item.get("publish_date")),
                    "q": _to_int_bool(item.get("q")),
                }
            )
        self._insert_rows(conn, "morningstar_summary", summary_rows)

        commentary_rows = []
        for item in payload.get("commentary", []) if isinstance(payload.get("commentary"), list) else []:
            if not isinstance(item, dict):
                continue
            author = item.get("author") if isinstance(item.get("author"), dict) else {}
            commentary_rows.append(
                {
                    "conid": str(conid),
                    "effective_at": str(effective_at),
                    "item_id": item.get("id"),
                    "title": item.get("title"),
                    "subtitle": item.get("subtitle"),
                    "subsection_id": item.get("subsection_id"),
                    "publish_date": _to_iso_date(item.get("publish_date")),
                    "q": _to_int_bool(item.get("q")),
                    "text": item.get("text"),
                    "author_name": author.get("name"),
                }
            )
        self._insert_rows(conn, "morningstar_commentary", commentary_rows)

    def _upsert_performance(self, conn, conid, effective_at, observed_at, payload_hash, source_file, now_iso, payload):
        self._upsert_snapshot_row(
            conn,
            "performance_snapshots",
            conid,
            effective_at,
            observed_at,
            payload_hash,
            source_file,
            now_iso,
            {
                "title_vs": payload.get("title_vs"),
            },
        )

        self._delete_children(conn, ["performance_metrics"], conid, effective_at)

        rows = []
        for section in ("cumulative", "annualized", "yield", "risk", "statistic"):
            values = payload.get(section, []) if isinstance(payload.get(section), list) else []
            for item in values:
                if not isinstance(item, dict):
                    continue
                rows.append(
                    {
                        "conid": str(conid),
                        "effective_at": str(effective_at),
                        "section": section,
                        "metric_id": _sanitize_segment(item.get("name_tag") or item.get("id") or item.get("name")),
                        "metric_name": item.get("name"),
                        "name_tag_arg": str(item.get("name_tag_arg")) if item.get("name_tag_arg") is not None else None,
                        "value_num": _parse_number(item.get("value")),
                        "value_fmt": item.get("value_fmt"),
                        "vs_num": _parse_number(item.get("vs")),
                        "min_num": _parse_number(item.get("min")),
                        "max_num": _parse_number(item.get("max")),
                        "avg_num": _parse_number(item.get("avg")),
                        "percentile_num": _parse_number(item.get("percentile")),
                        "min_fmt": item.get("min_fmt"),
                        "max_fmt": item.get("max_fmt"),
                        "avg_fmt": item.get("avg_fmt"),
                    }
                )
        self._insert_rows(conn, "performance_metrics", rows)

    def _upsert_ownership(self, conn, conid, effective_at, observed_at, payload_hash, source_file, now_iso, payload):
        snapshot = normalize_ownership_snapshot(payload)

        self._upsert_snapshot_row(
            conn,
            "ownership_snapshots",
            conid,
            effective_at,
            observed_at,
            payload_hash,
            source_file,
            now_iso,
            {
                "owners_types_count": int(snapshot.get("owners_types_count") or 0),
                "institutional_owners_count": int(snapshot.get("institutional_owners_count") or 0),
                "insider_owners_count": int(snapshot.get("insider_owners_count") or 0),
                "trade_log_count_raw": int(snapshot.get("trade_log_count_raw") or 0),
                "trade_log_count_kept": int(snapshot.get("trade_log_count_kept") or 0),
                "has_ownership_history": _to_int_bool(snapshot.get("has_ownership_history")),
                "ownership_history_price_points": int(snapshot.get("ownership_history_price_points") or 0),
                "institutional_total_value_text": snapshot.get("institutional_total_value"),
                "institutional_total_shares_text": snapshot.get("institutional_total_shares"),
                "institutional_total_pct_text": snapshot.get("institutional_total_pct"),
                "institutional_total_pct_num": _parse_number(snapshot.get("institutional_total_pct_num")),
                "insider_total_value_text": snapshot.get("insider_total_value"),
                "insider_total_shares_text": snapshot.get("insider_total_shares"),
                "insider_total_pct_text": snapshot.get("insider_total_pct"),
                "insider_total_pct_num": _parse_number(snapshot.get("insider_total_pct_num")),
            },
        )

        self._delete_children(conn, ["ownership_owners_types", "ownership_holders"], conid, effective_at)

        owners_type_rows = []
        for item in payload.get("owners_types", []) if isinstance(payload.get("owners_types"), list) else []:
            if not isinstance(item, dict):
                continue
            type_info = item.get("type") if isinstance(item.get("type"), dict) else {}
            owners_type_rows.append(
                {
                    "conid": str(conid),
                    "effective_at": str(effective_at),
                    "type": type_info.get("type"),
                    "display_type": type_info.get("display_type"),
                    "float_value": _parse_number(item.get("float")),
                    "display_float": item.get("display_float"),
                }
            )
        self._insert_rows(conn, "ownership_owners_types", owners_type_rows)

        holder_rows = []
        groups = [
            ("institutional", payload.get("institutional_owners", [])),
            ("insider", payload.get("insider_owners", [])),
        ]
        for holder_group, values in groups:
            if not isinstance(values, list):
                continue
            for item in values:
                if not isinstance(item, dict):
                    continue
                holder_rows.append(
                    {
                        "conid": str(conid),
                        "effective_at": str(effective_at),
                        "holder_group": holder_group,
                        "holder_name": item.get("name"),
                        "holder_type": item.get("type"),
                        "display_value": item.get("display_value"),
                        "display_shares": item.get("display_shares"),
                        "display_pct": item.get("display_pct"),
                        "pct_num": _parse_number(item.get("display_pct"), percent_as_fraction=True),
                    }
                )
        self._insert_rows(conn, "ownership_holders", holder_rows)

    def _upsert_esg(self, conn, conid, effective_at, observed_at, payload_hash, source_file, now_iso, payload):
        self._upsert_snapshot_row(
            conn,
            "esg_snapshots",
            conid,
            effective_at,
            observed_at,
            payload_hash,
            source_file,
            now_iso,
            {
                "as_of_date": _to_iso_date(payload.get("as_of_date") or payload.get("asOfDate")),
                "coverage": _parse_number(payload.get("coverage")),
                "source": payload.get("source"),
                "symbol": payload.get("symbol"),
                "no_settings": _to_int_bool(payload.get("no_settings")),
            },
        )

        self._delete_children(conn, ["esg_nodes"], conid, effective_at)

        rows = []

        def walk(nodes, parent_path, depth):
            if not isinstance(nodes, list):
                return
            for idx, node in enumerate(nodes):
                if not isinstance(node, dict):
                    continue
                name = str(node.get("name") or f"node_{idx}")
                seg = f"{idx}_{_sanitize_segment(name)}"
                node_path = f"{parent_path}/{seg}" if parent_path else seg
                rows.append(
                    {
                        "conid": str(conid),
                        "effective_at": str(effective_at),
                        "node_path": node_path,
                        "parent_path": parent_path if parent_path else None,
                        "depth": depth,
                        "node_name": name,
                        "node_value": _parse_number(node.get("value")),
                    }
                )
                walk(node.get("children"), node_path, depth + 1)

        walk(payload.get("content"), "", 0)
        self._insert_rows(conn, "esg_nodes", rows)

    def _upsert_price_chart_snapshot(self, conn, conid, effective_at, observed_at, payload_hash, source_file, now_iso, payload):
        rows = _extract_price_chart_rows(payload)
        dates = [r.get("trade_date") for r in rows if r.get("trade_date")]
        min_date = min(dates) if dates else None
        max_date = max(dates) if dates else None

        self._upsert_snapshot_row(
            conn,
            "price_chart_snapshots",
            conid,
            effective_at,
            observed_at,
            payload_hash,
            source_file,
            now_iso,
            {
                "points_count": len(rows),
                "min_trade_date": min_date,
                "max_trade_date": max_date,
            },
        )

    def _upsert_sentiment_snapshot(self, conn, conid, effective_at, observed_at, payload_hash, source_file, now_iso, payload):
        rows = _extract_sentiment_search_rows(payload)
        dates = [r.get("trade_date") for r in rows if r.get("trade_date")]
        min_date = min(dates) if dates else None
        max_date = max(dates) if dates else None

        self._upsert_snapshot_row(
            conn,
            "sentiment_search_snapshots",
            conid,
            effective_at,
            observed_at,
            payload_hash,
            source_file,
            now_iso,
            {
                "points_count": len(rows),
                "min_trade_date": min_date,
                "max_trade_date": max_date,
            },
        )

    def _series_is_newer(self, new_effective_at, new_observed_at, new_payload_hash, existing_row):
        old_effective_at = str(existing_row["effective_at"] or "")
        old_observed_at = str(existing_row["observed_at"] or "")
        old_payload_hash = str(existing_row["payload_hash"] or "")

        new_effective_at = str(new_effective_at or "")
        new_observed_at = str(new_observed_at or "")
        new_payload_hash = str(new_payload_hash or "")

        if new_effective_at > old_effective_at:
            return True
        if new_effective_at < old_effective_at:
            return False

        if new_observed_at > old_observed_at:
            return True
        if new_observed_at < old_observed_at:
            return False

        return new_payload_hash > old_payload_hash

    def _write_price_chart_series(self, conn, conid, effective_at, observed_at, payload_hash, inserted_at, payload):
        rows = _extract_price_chart_rows(payload)
        raw_written = 0
        latest_upserted = 0

        for row in rows:
            row_key = row.get("trade_date") or row.get("timestamp_ms") or row.get("debug_y")
            row_key = f"price_chart|{row_key}"

            conn.execute(
                """
                INSERT INTO price_chart_series_raw (
                    conid,
                    effective_at,
                    observed_at,
                    payload_hash,
                    inserted_at,
                    row_key,
                    trade_date,
                    timestamp_ms,
                    price,
                    open,
                    high,
                    low,
                    close,
                    debug_y
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    str(conid),
                    str(effective_at),
                    str(observed_at),
                    str(payload_hash),
                    inserted_at,
                    str(row_key),
                    row.get("trade_date"),
                    row.get("timestamp_ms"),
                    _parse_number(row.get("price")),
                    _parse_number(row.get("open")),
                    _parse_number(row.get("high")),
                    _parse_number(row.get("low")),
                    _parse_number(row.get("close")),
                    int(row.get("debug_y")) if row.get("debug_y") is not None else None,
                ],
            )
            raw_written += 1

            existing = conn.execute(
                """
                SELECT effective_at, observed_at, payload_hash
                FROM price_chart_series_latest
                WHERE conid = ? AND row_key = ?
                """,
                [str(conid), str(row_key)],
            ).fetchone()

            if existing is None:
                conn.execute(
                    """
                    INSERT INTO price_chart_series_latest (
                        conid,
                        row_key,
                        effective_at,
                        observed_at,
                        payload_hash,
                        updated_at,
                        trade_date,
                        timestamp_ms,
                        price,
                        open,
                        high,
                        low,
                        close,
                        debug_y
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        str(conid),
                        str(row_key),
                        str(effective_at),
                        str(observed_at),
                        str(payload_hash),
                        inserted_at,
                        row.get("trade_date"),
                        row.get("timestamp_ms"),
                        _parse_number(row.get("price")),
                        _parse_number(row.get("open")),
                        _parse_number(row.get("high")),
                        _parse_number(row.get("low")),
                        _parse_number(row.get("close")),
                        int(row.get("debug_y")) if row.get("debug_y") is not None else None,
                    ],
                )
                latest_upserted += 1
            elif self._series_is_newer(effective_at, observed_at, payload_hash, existing):
                conn.execute(
                    """
                    UPDATE price_chart_series_latest
                    SET effective_at = ?,
                        observed_at = ?,
                        payload_hash = ?,
                        updated_at = ?,
                        trade_date = ?,
                        timestamp_ms = ?,
                        price = ?,
                        open = ?,
                        high = ?,
                        low = ?,
                        close = ?,
                        debug_y = ?
                    WHERE conid = ? AND row_key = ?
                    """,
                    [
                        str(effective_at),
                        str(observed_at),
                        str(payload_hash),
                        inserted_at,
                        row.get("trade_date"),
                        row.get("timestamp_ms"),
                        _parse_number(row.get("price")),
                        _parse_number(row.get("open")),
                        _parse_number(row.get("high")),
                        _parse_number(row.get("low")),
                        _parse_number(row.get("close")),
                        int(row.get("debug_y")) if row.get("debug_y") is not None else None,
                        str(conid),
                        str(row_key),
                    ],
                )
                latest_upserted += 1

        return raw_written, latest_upserted

    def _write_sentiment_series(self, conn, conid, effective_at, observed_at, payload_hash, inserted_at, payload):
        rows = _extract_sentiment_search_rows(payload)
        raw_written = 0
        latest_upserted = 0

        for row in rows:
            row_key = row.get("datetime_ms") or row.get("trade_date")
            row_key = f"sentiment_search|{row_key}"

            conn.execute(
                """
                INSERT INTO sentiment_search_series_raw (
                    conid,
                    effective_at,
                    observed_at,
                    payload_hash,
                    inserted_at,
                    row_key,
                    trade_date,
                    datetime_ms,
                    sscore,
                    sdelta,
                    svolatility,
                    sdispersion,
                    svscore,
                    svolume,
                    smean,
                    sbuzz
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    str(conid),
                    str(effective_at),
                    str(observed_at),
                    str(payload_hash),
                    inserted_at,
                    str(row_key),
                    row.get("trade_date"),
                    int(row.get("datetime_ms")) if row.get("datetime_ms") is not None else None,
                    _parse_number(row.get("sscore")),
                    _parse_number(row.get("sdelta")),
                    _parse_number(row.get("svolatility")),
                    _parse_number(row.get("sdispersion")),
                    _parse_number(row.get("svscore")),
                    _parse_number(row.get("svolume")),
                    _parse_number(row.get("smean")),
                    _parse_number(row.get("sbuzz")),
                ],
            )
            raw_written += 1

            existing = conn.execute(
                """
                SELECT effective_at, observed_at, payload_hash
                FROM sentiment_search_series_latest
                WHERE conid = ? AND row_key = ?
                """,
                [str(conid), str(row_key)],
            ).fetchone()

            if existing is None:
                conn.execute(
                    """
                    INSERT INTO sentiment_search_series_latest (
                        conid,
                        row_key,
                        effective_at,
                        observed_at,
                        payload_hash,
                        updated_at,
                        trade_date,
                        datetime_ms,
                        sscore,
                        sdelta,
                        svolatility,
                        sdispersion,
                        svscore,
                        svolume,
                        smean,
                        sbuzz
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        str(conid),
                        str(row_key),
                        str(effective_at),
                        str(observed_at),
                        str(payload_hash),
                        inserted_at,
                        row.get("trade_date"),
                        int(row.get("datetime_ms")) if row.get("datetime_ms") is not None else None,
                        _parse_number(row.get("sscore")),
                        _parse_number(row.get("sdelta")),
                        _parse_number(row.get("svolatility")),
                        _parse_number(row.get("sdispersion")),
                        _parse_number(row.get("svscore")),
                        _parse_number(row.get("svolume")),
                        _parse_number(row.get("smean")),
                        _parse_number(row.get("sbuzz")),
                    ],
                )
                latest_upserted += 1
            elif self._series_is_newer(effective_at, observed_at, payload_hash, existing):
                conn.execute(
                    """
                    UPDATE sentiment_search_series_latest
                    SET effective_at = ?,
                        observed_at = ?,
                        payload_hash = ?,
                        updated_at = ?,
                        trade_date = ?,
                        datetime_ms = ?,
                        sscore = ?,
                        sdelta = ?,
                        svolatility = ?,
                        sdispersion = ?,
                        svscore = ?,
                        svolume = ?,
                        smean = ?,
                        sbuzz = ?
                    WHERE conid = ? AND row_key = ?
                    """,
                    [
                        str(effective_at),
                        str(observed_at),
                        str(payload_hash),
                        inserted_at,
                        row.get("trade_date"),
                        int(row.get("datetime_ms")) if row.get("datetime_ms") is not None else None,
                        _parse_number(row.get("sscore")),
                        _parse_number(row.get("sdelta")),
                        _parse_number(row.get("svolatility")),
                        _parse_number(row.get("sdispersion")),
                        _parse_number(row.get("svscore")),
                        _parse_number(row.get("svolume")),
                        _parse_number(row.get("smean")),
                        _parse_number(row.get("sbuzz")),
                        str(conid),
                        str(row_key),
                    ],
                )
                latest_upserted += 1

        return raw_written, latest_upserted

    def _write_ownership_trade_log_series(self, conn, conid, effective_at, observed_at, payload_hash, inserted_at, payload):
        rows = extract_ownership_trade_log(payload, drop_no_change=True)
        raw_written = 0
        latest_upserted = 0

        for row in rows:
            row_key = "|".join(
                [
                    "ownership",
                    str(row.get("trade_date") or ""),
                    str(row.get("action") or ""),
                    str(row.get("party") or ""),
                    str(row.get("source") or ""),
                    str(row.get("insider") or ""),
                    str(row.get("shares") or ""),
                    str(row.get("value") or ""),
                    str(row.get("holding") or ""),
                ]
            )

            conn.execute(
                """
                INSERT INTO ownership_trade_log_series_raw (
                    conid,
                    effective_at,
                    observed_at,
                    payload_hash,
                    inserted_at,
                    row_key,
                    trade_date,
                    action,
                    shares,
                    value,
                    holding,
                    party,
                    source,
                    insider
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    str(conid),
                    str(effective_at),
                    str(observed_at),
                    str(payload_hash),
                    inserted_at,
                    row_key,
                    row.get("trade_date"),
                    row.get("action"),
                    _parse_number(row.get("shares")),
                    _parse_number(row.get("value")),
                    _parse_number(row.get("holding")),
                    row.get("party"),
                    row.get("source"),
                    str(row.get("insider")) if row.get("insider") is not None else None,
                ],
            )
            raw_written += 1

            existing = conn.execute(
                """
                SELECT effective_at, observed_at, payload_hash
                FROM ownership_trade_log_series_latest
                WHERE conid = ? AND row_key = ?
                """,
                [str(conid), row_key],
            ).fetchone()

            if existing is None:
                conn.execute(
                    """
                    INSERT INTO ownership_trade_log_series_latest (
                        conid,
                        row_key,
                        effective_at,
                        observed_at,
                        payload_hash,
                        updated_at,
                        trade_date,
                        action,
                        shares,
                        value,
                        holding,
                        party,
                        source,
                        insider
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        str(conid),
                        row_key,
                        str(effective_at),
                        str(observed_at),
                        str(payload_hash),
                        inserted_at,
                        row.get("trade_date"),
                        row.get("action"),
                        _parse_number(row.get("shares")),
                        _parse_number(row.get("value")),
                        _parse_number(row.get("holding")),
                        row.get("party"),
                        row.get("source"),
                        str(row.get("insider")) if row.get("insider") is not None else None,
                    ],
                )
                latest_upserted += 1
            elif self._series_is_newer(effective_at, observed_at, payload_hash, existing):
                conn.execute(
                    """
                    UPDATE ownership_trade_log_series_latest
                    SET effective_at = ?,
                        observed_at = ?,
                        payload_hash = ?,
                        updated_at = ?,
                        trade_date = ?,
                        action = ?,
                        shares = ?,
                        value = ?,
                        holding = ?,
                        party = ?,
                        source = ?,
                        insider = ?
                    WHERE conid = ? AND row_key = ?
                    """,
                    [
                        str(effective_at),
                        str(observed_at),
                        str(payload_hash),
                        inserted_at,
                        row.get("trade_date"),
                        row.get("action"),
                        _parse_number(row.get("shares")),
                        _parse_number(row.get("value")),
                        _parse_number(row.get("holding")),
                        row.get("party"),
                        row.get("source"),
                        str(row.get("insider")) if row.get("insider") is not None else None,
                        str(conid),
                        row_key,
                    ],
                )
                latest_upserted += 1

        return raw_written, latest_upserted

    def _write_dividends_events_series(self, conn, conid, effective_at, observed_at, payload_hash, inserted_at, payload):
        rows = extract_dividends_events(payload)
        raw_written = 0
        latest_upserted = 0

        for row in rows:
            row_key = "|".join(
                [
                    "dividends",
                    str(row.get("event_date") or row.get("trade_date") or ""),
                    str(row.get("amount") or ""),
                    str(row.get("currency") or ""),
                    str(row.get("event_type") or ""),
                    str(row.get("declaration_date") or ""),
                    str(row.get("record_date") or ""),
                    str(row.get("payment_date") or ""),
                    str(row.get("description") or ""),
                ]
            )

            conn.execute(
                """
                INSERT INTO dividends_events_series_raw (
                    conid,
                    effective_at,
                    observed_at,
                    payload_hash,
                    inserted_at,
                    row_key,
                    trade_date,
                    event_date,
                    amount,
                    currency,
                    description,
                    event_type,
                    declaration_date,
                    record_date,
                    payment_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    str(conid),
                    str(effective_at),
                    str(observed_at),
                    str(payload_hash),
                    inserted_at,
                    row_key,
                    row.get("trade_date"),
                    row.get("event_date"),
                    _parse_number(row.get("amount")),
                    row.get("currency"),
                    row.get("description"),
                    row.get("event_type"),
                    row.get("declaration_date"),
                    row.get("record_date"),
                    row.get("payment_date"),
                ],
            )
            raw_written += 1

            existing = conn.execute(
                """
                SELECT effective_at, observed_at, payload_hash
                FROM dividends_events_series_latest
                WHERE conid = ? AND row_key = ?
                """,
                [str(conid), row_key],
            ).fetchone()

            if existing is None:
                conn.execute(
                    """
                    INSERT INTO dividends_events_series_latest (
                        conid,
                        row_key,
                        effective_at,
                        observed_at,
                        payload_hash,
                        updated_at,
                        trade_date,
                        event_date,
                        amount,
                        currency,
                        description,
                        event_type,
                        declaration_date,
                        record_date,
                        payment_date
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        str(conid),
                        row_key,
                        str(effective_at),
                        str(observed_at),
                        str(payload_hash),
                        inserted_at,
                        row.get("trade_date"),
                        row.get("event_date"),
                        _parse_number(row.get("amount")),
                        row.get("currency"),
                        row.get("description"),
                        row.get("event_type"),
                        row.get("declaration_date"),
                        row.get("record_date"),
                        row.get("payment_date"),
                    ],
                )
                latest_upserted += 1
            elif self._series_is_newer(effective_at, observed_at, payload_hash, existing):
                conn.execute(
                    """
                    UPDATE dividends_events_series_latest
                    SET effective_at = ?,
                        observed_at = ?,
                        payload_hash = ?,
                        updated_at = ?,
                        trade_date = ?,
                        event_date = ?,
                        amount = ?,
                        currency = ?,
                        description = ?,
                        event_type = ?,
                        declaration_date = ?,
                        record_date = ?,
                        payment_date = ?
                    WHERE conid = ? AND row_key = ?
                    """,
                    [
                        str(effective_at),
                        str(observed_at),
                        str(payload_hash),
                        inserted_at,
                        row.get("trade_date"),
                        row.get("event_date"),
                        _parse_number(row.get("amount")),
                        row.get("currency"),
                        row.get("description"),
                        row.get("event_type"),
                        row.get("declaration_date"),
                        row.get("record_date"),
                        row.get("payment_date"),
                        str(conid),
                        row_key,
                    ],
                )
                latest_upserted += 1

        return raw_written, latest_upserted

    def _write_series(self, conn, endpoint, conid, effective_at, observed_at, payload_hash, inserted_at, payload):
        if endpoint == "price_chart":
            return self._write_price_chart_series(conn, conid, effective_at, observed_at, payload_hash, inserted_at, payload)
        if endpoint == "sentiment_search":
            return self._write_sentiment_series(conn, conid, effective_at, observed_at, payload_hash, inserted_at, payload)
        if endpoint == "ownership":
            return self._write_ownership_trade_log_series(conn, conid, effective_at, observed_at, payload_hash, inserted_at, payload)
        if endpoint == "dividends":
            return self._write_dividends_events_series(conn, conid, effective_at, observed_at, payload_hash, inserted_at, payload)
        return 0, 0

    def _persist_endpoint(
        self,
        conn,
        conid,
        endpoint,
        payload,
        observed_at,
        effective_at,
        effective_source,
        source_file,
    ):
        now_iso = datetime.now(timezone.utc).isoformat()
        self._ensure_product(conn, conid)

        blob_info = self._store_blob(conn, payload)
        payload_hash = blob_info["hash"]

        table = self._main_table_for_endpoint(endpoint)
        existing = conn.execute(
            f"SELECT payload_hash FROM {table} WHERE conid = ? AND effective_at = ?",
            [str(conid), str(effective_at)],
        ).fetchone()

        if existing is None:
            state = "inserted"
        elif str(existing["payload_hash"]) == str(payload_hash):
            state = "unchanged"
        else:
            state = "overwritten"

        if state == "unchanged":
            return {
                "endpoint": endpoint,
                "state": state,
                "effective_source": effective_source,
                "series_raw_rows_written": 0,
                "series_latest_rows_upserted": 0,
            }

        handlers = {
            "landing": self._upsert_landing,
            "profile_and_fees": self._upsert_profile_fees,
            "holdings": self._upsert_holdings,
            "ratios": self._upsert_ratios,
            "lipper_ratings": self._upsert_lipper,
            "dividends": self._upsert_dividends,
            "morningstar": self._upsert_morningstar,
            "price_chart": self._upsert_price_chart_snapshot,
            "performance": self._upsert_performance,
            "sentiment_search": self._upsert_sentiment_snapshot,
            "ownership": self._upsert_ownership,
            "esg": self._upsert_esg,
        }

        handler = handlers[endpoint]
        handler(
            conn,
            conid,
            effective_at,
            observed_at,
            payload_hash,
            source_file,
            now_iso,
            payload,
        )

        self._store_endpoint_scalar_extras(conn, endpoint, conid, effective_at, payload)

        series_raw_rows_written = 0
        series_latest_rows_upserted = 0
        if endpoint in SERIES_ENDPOINTS:
            series_raw_rows_written, series_latest_rows_upserted = self._write_series(
                conn,
                endpoint,
                conid,
                effective_at,
                observed_at,
                payload_hash,
                now_iso,
                payload,
            )

        return {
            "endpoint": endpoint,
            "state": state,
            "effective_source": effective_source,
            "series_raw_rows_written": int(series_raw_rows_written),
            "series_latest_rows_upserted": int(series_latest_rows_upserted),
        }

    def persist_combined_snapshot(self, snapshot, source_file=None, refresh_duckdb=False):
        if not isinstance(snapshot, dict):
            return {
                "inserted_events": 0,
                "overwritten_events": 0,
                "unchanged_events": 0,
                "series_raw_rows_written": 0,
                "series_latest_rows_upserted": 0,
                "status": "invalid_snapshot",
            }

        conid = snapshot.get("conid")
        if conid is None:
            return {
                "inserted_events": 0,
                "overwritten_events": 0,
                "unchanged_events": 0,
                "series_raw_rows_written": 0,
                "series_latest_rows_upserted": 0,
                "status": "missing_conid",
            }

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
            return {
                "inserted_events": 0,
                "overwritten_events": 0,
                "unchanged_events": 0,
                "series_raw_rows_written": 0,
                "series_latest_rows_upserted": 0,
                "status": "no_payloads",
            }

        effective_map = self._resolve_effective_dates(endpoint_payloads, observed_dt)

        inserted_events = 0
        overwritten_events = 0
        unchanged_events = 0
        series_raw_rows_written = 0
        series_latest_rows_upserted = 0
        per_endpoint = {}

        with self._get_conn() as conn:
            for endpoint, payload in endpoint_payloads.items():
                effective_at, effective_source = effective_map[endpoint]
                result = self._persist_endpoint(
                    conn=conn,
                    conid=conid,
                    endpoint=endpoint,
                    payload=payload,
                    observed_at=observed_at,
                    effective_at=effective_at,
                    effective_source=effective_source,
                    source_file=source_file,
                )

                state = result["state"]
                if state == "inserted":
                    inserted_events += 1
                elif state == "overwritten":
                    overwritten_events += 1
                else:
                    unchanged_events += 1

                series_raw_rows_written += int(result.get("series_raw_rows_written", 0))
                series_latest_rows_upserted += int(result.get("series_latest_rows_upserted", 0))
                per_endpoint[endpoint] = result

            conn.commit()

        if refresh_duckdb:
            self.refresh_duckdb_views()

        return {
            "inserted_events": inserted_events,
            "overwritten_events": overwritten_events,
            "unchanged_events": unchanged_events,
            "series_raw_rows_written": series_raw_rows_written,
            "series_latest_rows_upserted": series_latest_rows_upserted,
            "status": "ok",
            "per_endpoint": per_endpoint,
        }

    def persist_ingest_run(self, run_stats, endpoint_summary):
        run_stats = run_stats or {}
        endpoint_summary = endpoint_summary or []

        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO ingest_runs (
                    run_started_at,
                    run_finished_at,
                    total_targeted_conids,
                    processed_conids,
                    saved_snapshots,
                    inserted_events,
                    overwritten_events,
                    unchanged_events,
                    series_raw_rows_written,
                    series_latest_rows_upserted,
                    auth_retries,
                    aborted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    run_stats.get("run_started_at") or datetime.now(timezone.utc).isoformat(),
                    run_stats.get("run_finished_at") or datetime.now(timezone.utc).isoformat(),
                    int(run_stats.get("total_targeted_conids", 0)),
                    int(run_stats.get("processed_conids", 0)),
                    int(run_stats.get("saved_snapshots", 0)),
                    int(run_stats.get("inserted_events", 0)),
                    int(run_stats.get("overwritten_events", 0)),
                    int(run_stats.get("unchanged_events", 0)),
                    int(run_stats.get("series_raw_rows_written", 0)),
                    int(run_stats.get("series_latest_rows_upserted", 0)),
                    int(run_stats.get("auth_retries", 0)),
                    1 if bool(run_stats.get("aborted", False)) else 0,
                ],
            )
            run_id = int(cur.lastrowid)

            for row in endpoint_summary:
                endpoint = str(row.get("endpoint") or "")
                if not endpoint:
                    continue
                call_count = int(row.get("call_count", 0))
                useful_payload_count = int(row.get("useful_payload_count", 0))
                useful_payload_rate = float(row.get("useful_payload_rate", 0.0)) if call_count else 0.0
                status_codes = row.get("status_codes") or {}

                status_2xx = 0
                status_4xx = 0
                status_5xx = 0
                status_other = 0
                for code, count in status_codes.items():
                    try:
                        code_int = int(code)
                    except Exception:
                        status_other += int(count)
                        continue

                    if 200 <= code_int < 300:
                        status_2xx += int(count)
                    elif 400 <= code_int < 500:
                        status_4xx += int(count)
                    elif 500 <= code_int < 600:
                        status_5xx += int(count)
                    else:
                        status_other += int(count)

                conn.execute(
                    """
                    INSERT INTO ingest_run_endpoint_rollups (
                        run_id,
                        endpoint,
                        call_count,
                        useful_payload_count,
                        useful_payload_rate,
                        status_2xx,
                        status_4xx,
                        status_5xx,
                        status_other
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        run_id,
                        endpoint,
                        call_count,
                        useful_payload_count,
                        useful_payload_rate,
                        status_2xx,
                        status_4xx,
                        status_5xx,
                        status_other,
                    ],
                )

            conn.commit()
            return run_id

    def refresh_duckdb_views(self):
        with self._get_conn() as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("VACUUM")

            tables = [
                "products",
                "raw_payload_blobs",
                "landing_snapshots",
                "profile_fees_snapshots",
                "holdings_snapshots",
                "ratios_snapshots",
                "lipper_ratings_snapshots",
                "dividends_snapshots",
                "morningstar_snapshots",
                "performance_snapshots",
                "ownership_snapshots",
                "esg_snapshots",
                "price_chart_snapshots",
                "sentiment_search_snapshots",
                "price_chart_series_raw",
                "price_chart_series_latest",
                "sentiment_search_series_raw",
                "sentiment_search_series_latest",
                "ownership_trade_log_series_raw",
                "ownership_trade_log_series_latest",
                "dividends_events_series_raw",
                "dividends_events_series_latest",
            ]

            counts = {}
            for table in tables:
                row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
                counts[table] = int(row["n"])

            return {
                "sqlite_path": str(self.sqlite_path),
                "vacuumed": True,
                "table_counts": counts,
            }


def refresh_views():
    store = FundamentalsStore()
    result = store.refresh_duckdb_views()
    for key, value in result.items():
        print(f"{key}: {value}")
    return result


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "refresh_views": refresh_views,
        }
    )
