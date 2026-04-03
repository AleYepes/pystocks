import hashlib
import json
import logging
import math
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import zstandard as zstd

from ..config import SQLITE_DB_PATH
from ._sqlite import open_connection
from .normalize import (
    extract_dividends_events,
    extract_ownership_trade_log,
    normalize_dividends_snapshot,
    normalize_ownership_snapshot,
)
from .schema import init_storage

logger = logging.getLogger(__name__)


ENDPOINT_KEYS = [
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
_NUM_WITH_SUFFIX_RE = re.compile(
    r"^[\s\$€£¥]*([+-]?\d+(?:\.\d+)?)\s*([KMBT])?\b", re.IGNORECASE
)
_TOTAL_NET_ASSETS_DATE_BLOCK_RE = re.compile(
    r"\(\s*\d{4}[/-]\d{2}[/-]\d{2}\s*[\)\.]?\s*"
)

_PROFILE_AND_FEES_FIELD_COLUMN_TYPES = {
    "Asset Type": ("asset_type", "text"),
    "Classification": ("classification", "text"),
    "Distribution Details": ("distribution_details", "text"),
    "Domicile": ("domicile", "text"),
    "Fiscal Date": ("fiscal_date", "text"),
    "Fund Category": ("fund_category", "text"),
    "Fund Management Company": ("fund_management_company", "text"),
    "Fund Manager Benchmark": ("fund_manager_benchmark", "text"),
    "Fund Market Cap Focus": ("fund_market_cap_focus", "text"),
    "Geographical Focus": ("geographical_focus", "text"),
    "Inception Date": ("inception_date", "date"),
    "Launch Opening Price": ("inception_date", "date"),
    "Management Approach": ("management_approach", "text"),
    "Management Expenses": ("management_expenses", "percent"),
    "Manager Tenure": ("manager_tenure", "date"),
    "Maturity Date": ("maturity_date", "date"),
    "Objective Type": ("objective_type", "text"),
    "Portfolio Manager": ("portfolio_manager", "text"),
    "Redemption Charge Actual": ("redemption_charge_actual", "percent"),
    "Redemption Charge Max": ("redemption_charge_max", "percent"),
    "Scheme": ("scheme", "text"),
    "Total Expense Ratio": ("total_expense_ratio", "percent"),
    "Total Net Assets (Month End)": ("total_net_assets_value", "text"),
}

_PROFILE_AND_FEES_PIVOT_COLUMNS = [
    "asset_type",
    "classification",
    "distribution_details",
    "domicile",
    "fiscal_date",
    "fund_category",
    "fund_management_company",
    "fund_manager_benchmark",
    "fund_market_cap_focus",
    "geographical_focus",
    "inception_date",
    "management_approach",
    "management_expenses",
    "manager_tenure",
    "maturity_date",
    "objective_type",
    "portfolio_manager",
    "redemption_charge_actual",
    "redemption_charge_max",
    "scheme",
    "total_expense_ratio",
    "total_net_assets_value",
    "total_net_assets_date",
    "objective",
    "jap_fund_warning",
    "theme_name",
]

_PROFILE_AND_FEES_REPORT_FIELD_COLUMN_TYPES = {
    "Administrator Expenses": "administrator_expenses",
    "Advisor Expenses": "advisor_expenses",
    "Audit Expenses": "audit_expenses",
    "Audit and Legal Expenses": "audit_and_legal_expenses",
    "Custodian Expenses": "custodian_expenses",
    "Director Expense": "director_expense",
    "Management Fees": "management_fees",
    "Misc. Expenses": "misc_expenses",
    "Non-Management Expenses": "non_management_expenses",
    "Other Expense": "other_expense",
    "Other Non-Management Fees": "other_non_management_fees",
    "Postage and Printing Expenses": "postage_and_printing_expenses",
    "Prospectus Gross Expense Ratio": "prospectus_gross_expense_ratio",
    "Prospectus Gross Management Fee Ratio": "prospectus_gross_management_fee_ratio",
    "Prospectus Gross Other Expense Ratio": "prospectus_gross_other_expense_ratio",
    "Prospectus Net Expense Ratio": "prospectus_net_expense_ratio",
    "Prospectus Net Management Fee Ratio": "prospectus_net_management_fee_ratio",
    "Prospectus Net Other Expense Ratio": "prospectus_net_other_expense_ratio",
    "Registration Expenses": "registration_expenses",
    "Total Expense": "total_expense",
    "Total Gross Expense": "total_gross_expense",
    "Total Net Expense": "total_net_expense",
}

_PROFILE_AND_FEES_REPORT_PIVOT_COLUMNS = sorted(
    set(_PROFILE_AND_FEES_REPORT_FIELD_COLUMN_TYPES.values())
)

_PROFILE_AND_FEES_STYLEBOX_X = ("value", "core", "growth")
_PROFILE_AND_FEES_STYLEBOX_Y = ("large", "multi", "mid", "small")
_PROFILE_AND_FEES_STYLEBOX_COLUMNS = [
    f"{x}_{y}"
    for x in _PROFILE_AND_FEES_STYLEBOX_X
    for y in _PROFILE_AND_FEES_STYLEBOX_Y
]
_PROFILE_AND_FEES_STYLEBOX_COORD_COLUMNS = {
    (xi, yi): f"{x}_{y}"
    for xi, x in enumerate(_PROFILE_AND_FEES_STYLEBOX_X)
    for yi, y in enumerate(_PROFILE_AND_FEES_STYLEBOX_Y)
}

_HOLDINGS_SPLIT_TABLES = [
    "holdings_asset_type",
    "holdings_industry",
    "holdings_currency",
    "holdings_investor_country",
    "holdings_debt_type",
    "holdings_debtor_quality",
    "holdings_maturity",
]

_HOLDINGS_ASSET_TYPE_SOURCE_TO_COLUMN = {
    "equity": "equity",
    "cash": "cash",
    "fixed_income": "fixed_income",
    "other": "other",
}
_HOLDINGS_ASSET_TYPE_COLUMNS = tuple(_HOLDINGS_ASSET_TYPE_SOURCE_TO_COLUMN.values())

_HOLDINGS_DEBTOR_QUALITY_SOURCE_TO_COLUMN = {
    "quality_aaa": "quality_aaa",
    "quality_aa": "quality_aa",
    "quality_a": "quality_a",
    "quality_bbb": "quality_bbb",
    "quality_bb": "quality_bb",
    "quality_b": "quality_b",
    "quality_ccc": "quality_ccc",
    "quality_cc": "quality_cc",
    "quality_c": "quality_c",
    "quality_d": "quality_d",
    "quality_not_rated": "quality_not_rated",
    "quality_not_available": "quality_not_available",
}
_HOLDINGS_DEBTOR_QUALITY_COLUMNS = tuple(
    _HOLDINGS_DEBTOR_QUALITY_SOURCE_TO_COLUMN.values()
)

_HOLDINGS_MATURITY_SOURCE_TO_COLUMN = {
    "maturity_less_than_1_year": "maturity_less_than_1_year",
    "maturity_1_to_3_years": "maturity_1_to_3_years",
    "maturity_3_to_5_years": "maturity_3_to_5_years",
    "maturity_5_to_10_years": "maturity_5_to_10_years",
    "maturity_10_to_20_years": "maturity_10_to_20_years",
    "maturity_20_to_30_years": "maturity_20_to_30_years",
    "maturity_greater_than_30_years": "maturity_greater_than_30_years",
    "maturity_other": "maturity_other",
    "less_than_1_year": "maturity_less_than_1_year",
    "1_to_3_years": "maturity_1_to_3_years",
    "3_to_5_years": "maturity_3_to_5_years",
    "5_to_10_years": "maturity_5_to_10_years",
    "10_to_20_years": "maturity_10_to_20_years",
    "20_to_30_years": "maturity_20_to_30_years",
    "greater_than_30_years": "maturity_greater_than_30_years",
    "other": "maturity_other",
}
_HOLDINGS_MATURITY_COLUMNS = (
    "maturity_less_than_1_year",
    "maturity_1_to_3_years",
    "maturity_3_to_5_years",
    "maturity_5_to_10_years",
    "maturity_10_to_20_years",
    "maturity_20_to_30_years",
    "maturity_greater_than_30_years",
    "maturity_other",
)

_RATIOS_SECTION_TABLES = {
    "ratios": "ratios_key_ratios",
    "financials": "ratios_financials",
    "fixed_income": "ratios_fixed_income",
    "dividend": "ratios_dividend",
    "zscore": "ratios_zscore",
}

_MORNINGSTAR_SUMMARY_COLUMNS = (
    "medalist_rating",
    "process",
    "people",
    "parent",
    "morningstar_rating",
    "sustainability_rating",
    "category",
    "category_index",
)

_MORNINGSTAR_SUMMARY_ID_TO_COLUMN = {
    "medalist_rating": "medalist_rating",
    "process": "process",
    "people": "people",
    "parent": "parent",
    "morningstar_rating": "morningstar_rating",
    "sustainability_rating": "sustainability_rating",
    "category": "category",
    "category_index": "category_index",
}

_ESG_CODE_TO_COLUMN = {
    "TRESGS": "esg_overall_score",
    "TRESGCS": "esg_combined_score",
    "TRESGCCS": "esg_controversies_score",
    "TRESGENS": "environmental_overall_score",
    "TRESGENRRS": "environmental_resource_use_score",
    "TRESGENERS": "environmental_emissions_score",
    "TRESGENPIS": "environmental_innovation_score",
    "TRESGSOS": "social_overall_score",
    "TRESGSOWOS": "social_workforce_score",
    "TRESGSOHRS": "social_human_rights_score",
    "TRESGSOCOS": "social_community_score",
    "TRESGSOPRS": "social_product_responsibility_score",
    "TRESGCGS": "governance_overall_score",
    "TRESGCGBDS": "governance_management_score",
    "TRESGCGSRS": "governance_shareholders_score",
    "TRESGCGVSS": "governance_csr_strategy_score",
}

_ESG_SCORE_COLUMNS = tuple(_ESG_CODE_TO_COLUMN.values())

_DIVIDENDS_INDUSTRY_METRIC_COLUMNS = (
    "dividend_yield",
    "annual_dividend",
    "dividend_ttm",
    "dividend_yield_ttm",
)


def _sanitize_segment(value):
    segment = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return segment or "field"


def _canonical_json_bytes(payload):
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


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
            return datetime.fromtimestamp(ts, tz=UTC).date()
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


def _split_total_net_assets_value(value):
    if value is None:
        return None, None

    raw_text = str(value).strip()
    if not raw_text:
        return None, None

    parsed_date = _parse_ymd_text(raw_text)
    date_iso = parsed_date.isoformat() if parsed_date is not None else None

    clean_text = _TOTAL_NET_ASSETS_DATE_BLOCK_RE.sub("", raw_text)
    clean_text = _DATE_IN_TEXT_RE.sub("", clean_text)
    clean_text = clean_text.replace("()", "").strip()
    clean_text = re.sub(r"\s+", " ", clean_text).strip(" .,\t\r\n()")
    if not clean_text:
        clean_text = None

    return clean_text, date_iso


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
                mi = (
                    int(m)
                    if isinstance(m, (int, float))
                    else _MONTH_MAP.get(str(m).strip().upper())
                )
                if mi is None:
                    return None
                return datetime(int(y), int(mi), int(d), tzinfo=UTC).date().isoformat()
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


def _to_fraction_percent(value):
    if value is None:
        return None
    if isinstance(value, str) and "%" in value:
        return _parse_number(value, percent_as_fraction=True)
    parsed = _parse_number(value)
    if parsed is None:
        return None
    return parsed / 100.0


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


def _sanitize_dividends_payload(payload):
    if not isinstance(payload, dict):
        return payload
    history = payload.get("history")
    if not isinstance(history, dict):
        return payload
    series = history.get("series")
    if not isinstance(series, list):
        return payload

    filtered_series = []
    for node in series:
        if not isinstance(node, dict):
            continue
        name = str(node.get("name") or node.get("title") or "").strip().lower()
        if "price" in name:
            continue
        filtered_series.append(node)

    out = {k: v for k, v in payload.items()}
    out_history = {k: v for k, v in history.items()}
    out_history["series"] = filtered_series
    out["history"] = out_history
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
        x_date = _parse_date_candidate(point.get("x"))
        debug_date = _parse_date_candidate(point.get("debugY"))
        trade_date = x_date or debug_date

        row = {
            "effective_at": trade_date.isoformat() if trade_date is not None else None,
            "price": point.get("y"),
            "open": point.get("open"),
            "high": point.get("high"),
            "low": point.get("low"),
            "close": point.get("close"),
        }
        if any(v is not None for v in row.values()):
            row["debug_mismatch"] = int(
                x_date is not None
                and debug_date is not None
                and abs((x_date - debug_date).days) > 1
            )
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
        effective_at = dt.isoformat() if dt is not None else None
        if effective_at is None:
            continue
        row = {
            "effective_at": effective_at,
            "sscore": point.get("sscore"),
            "sdelta": point.get("sdelta"),
            "svolatility": point.get("svolatility"),
            "sdispersion": point.get("sdispersion"),
            "svscore": point.get("svscore"),
            "svolume": point.get("svolume"),
            "smean": point.get("smean"),
            "sbuzz": point.get("sbuzz"),
        }

        if any(
            row.get(k) is not None
            for k in (
                "sscore",
                "sdelta",
                "svolatility",
                "sdispersion",
                "svscore",
                "svolume",
                "smean",
                "sbuzz",
            )
        ):
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
        return open_connection(self.sqlite_path, row_factory=sqlite3.Row)

    def _init_db(self):
        init_storage(self.sqlite_path)

    def _ensure_product(self, conn, conid):
        now_iso = datetime.now(UTC).isoformat()
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
                    datetime.now(UTC).isoformat(),
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

    def get_latest_price_series_effective_at(self, conid):
        try:
            with self._get_conn() as conn:
                row = conn.execute(
                    """
                    SELECT MAX(effective_at) AS max_effective_at
                    FROM price_chart_series
                    WHERE conid = ?
                    """,
                    [str(conid)],
                ).fetchone()
        except sqlite3.OperationalError:
            return None

        if row is None:
            return None
        parsed = _parse_date_candidate(row["max_effective_at"])
        return parsed

    def _resolve_effective_dates(self, endpoint_payloads, _observed_at):
        ratios_payload = endpoint_payloads.get("ratios")

        ratios_date = (
            _extract_as_of_date(ratios_payload)
            if isinstance(ratios_payload, dict)
            else None
        )

        if ratios_date is not None:
            effective_at = ratios_date.isoformat()
            effective_source = "ratios.as_of_date_anchor"
        else:
            effective_at = None
            effective_source = "ratios.as_of_date_missing"

        return {
            endpoint: (effective_at, effective_source)
            for endpoint in endpoint_payloads.keys()
        }

    def _endpoint_payloads_from_snapshot(self, snapshot):
        payloads = {}
        for endpoint in ENDPOINT_KEYS:
            value = snapshot.get(endpoint)
            if isinstance(value, (dict, list)):
                if endpoint == "sentiment_search" and isinstance(value, dict):
                    value = _sanitize_sentiment_search_payload(value)
                if endpoint == "dividends" and isinstance(value, dict):
                    value = _sanitize_dividends_payload(value)
                payloads[endpoint] = value
        return payloads

    def _main_table_for_endpoint(self, endpoint):
        return {
            "profile_and_fees": "profile_and_fees_snapshots",
            "holdings": "holdings_snapshots",
            "ratios": "ratios_snapshots",
            "lipper_ratings": "lipper_ratings_snapshots",
            "dividends": "dividends_snapshots",
            "morningstar": "morningstar_snapshots",
            "price_chart": "price_chart_snapshots",
            "performance": "performance_snapshots",
            "sentiment_search": "sentiment_snapshots",
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
        include_source_file=False,
    ):
        base = {
            "conid": str(conid),
            "effective_at": str(effective_at),
            "observed_at": str(observed_at),
            "payload_hash": str(payload_hash),
            "inserted_at": now_iso,
            "updated_at": now_iso,
        }
        if include_source_file:
            base["source_file"] = source_file
        row = dict(base)
        row.update(extra)

        cols = list(row.keys())
        placeholders = ", ".join(["?"] * len(cols))
        update_cols = [
            c for c in cols if c not in {"conid", "effective_at", "inserted_at"}
        ]
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

    def _upsert_row(self, conn, table, row, primary_keys):
        cols = list(row.keys())
        placeholders = ", ".join(["?"] * len(cols))
        update_cols = [c for c in cols if c not in set(primary_keys)]
        update_sql = ", ".join([f"{c}=excluded.{c}" for c in update_cols])
        conn.execute(
            f"""
            INSERT INTO {table} ({", ".join(cols)})
            VALUES ({placeholders})
            ON CONFLICT({", ".join(primary_keys)}) DO UPDATE SET
                {update_sql}
            """,
            [row.get(c) for c in cols],
        )

    def _insert_rows(self, conn, table, rows):
        if not rows:
            return
        cols = list(rows[0].keys())
        placeholders = ", ".join(["?"] * len(cols))
        conn.executemany(
            f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})",
            [[row.get(c) for c in cols] for row in rows],
        )

    def _store_endpoint_scalar_extras(
        self, conn, endpoint, conid, effective_at, payload
    ):
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

    def _upsert_profile_fees(
        self,
        conn,
        conid,
        effective_at,
        observed_at,
        payload_hash,
        source_file,
        now_iso,
        payload,
    ):
        self._upsert_snapshot_row(
            conn,
            "profile_and_fees_snapshots",
            conid,
            effective_at,
            observed_at,
            payload_hash,
            source_file,
            now_iso,
            {},
            include_source_file=False,
        )

        self._delete_children(
            conn,
            [
                "profile_and_fees_reports",
                "profile_and_fees_stylebox",
            ],
            conid,
            effective_at,
        )

        themes = (
            payload.get("themes", []) if isinstance(payload.get("themes"), list) else []
        )
        theme_names = []
        for theme in themes:
            if isinstance(theme, dict):
                theme_name = (
                    theme.get("name") or theme.get("title") or theme.get("theme")
                )
            else:
                theme_name = str(theme) if theme is not None else None
            if theme_name is None:
                continue
            cleaned = str(theme_name).strip()
            if cleaned:
                theme_names.append(cleaned)

        profile_row = {
            "conid": str(conid),
            "effective_at": str(effective_at),
            **{col: None for col in _PROFILE_AND_FEES_PIVOT_COLUMNS},
        }
        profile_row["objective"] = (
            str(payload.get("objective"))
            if payload.get("objective") is not None
            else None
        )
        profile_row["jap_fund_warning"] = _to_int_bool(payload.get("jap_fund_warning"))
        profile_row["theme_name"] = (
            " | ".join(dict.fromkeys(theme_names)) if theme_names else None
        )

        for item in (
            payload.get("fund_and_profile", [])
            if isinstance(payload.get("fund_and_profile"), list)
            else []
        ):
            if not isinstance(item, dict):
                continue
            field_name = item.get("name")
            if field_name not in _PROFILE_AND_FEES_FIELD_COLUMN_TYPES:
                continue
            value = item.get("value")
            column_name, value_type = _PROFILE_AND_FEES_FIELD_COLUMN_TYPES[field_name]

            if field_name == "Total Net Assets (Month End)":
                net_assets_value, net_assets_date = _split_total_net_assets_value(value)
                profile_row["total_net_assets_value"] = net_assets_value
                profile_row["total_net_assets_date"] = net_assets_date
                continue

            if value_type == "text":
                profile_row[column_name] = str(value) if value is not None else None
            elif value_type == "percent":
                profile_row[column_name] = _parse_number(
                    value, percent_as_fraction=True
                )
            elif value_type == "date":
                parsed_date = _to_iso_date(value)
                if field_name == "Launch Opening Price":
                    if (
                        profile_row.get("inception_date") is None
                        and parsed_date is not None
                    ):
                        profile_row["inception_date"] = parsed_date
                else:
                    profile_row[column_name] = parsed_date

        self._upsert_row(
            conn, "profile_and_fees", profile_row, ["conid", "effective_at"]
        )

        report_rows = []
        for report in (
            payload.get("reports", [])
            if isinstance(payload.get("reports"), list)
            else []
        ):
            if not isinstance(report, dict):
                continue
            report_row = {
                "conid": str(conid),
                "effective_at": str(effective_at),
                "report_name": report.get("name"),
                "report_as_of_date": _to_iso_date(report.get("as_of_date")),
                **{col: None for col in _PROFILE_AND_FEES_REPORT_PIVOT_COLUMNS},
            }
            fields = (
                report.get("fields", [])
                if isinstance(report.get("fields"), list)
                else []
            )
            for field in fields:
                if not isinstance(field, dict):
                    continue
                field_name = field.get("name") or field.get("name_tag")
                field_name = str(field_name) if field_name is not None else None
                column_name = (
                    _PROFILE_AND_FEES_REPORT_FIELD_COLUMN_TYPES.get(field_name)
                    if field_name is not None
                    else None
                )
                if column_name is None and field_name:
                    normalized = _sanitize_segment(field_name)
                    if normalized in _PROFILE_AND_FEES_REPORT_PIVOT_COLUMNS:
                        column_name = normalized
                if column_name is None:
                    continue
                report_row[column_name] = _parse_number(
                    field.get("value"), percent_as_fraction=True
                )
            if not any(
                report_row.get(col) is not None
                for col in _PROFILE_AND_FEES_REPORT_PIVOT_COLUMNS
            ):
                continue
            report_rows.append(report_row)
        self._insert_rows(conn, "profile_and_fees_reports", report_rows)

        mstar = payload.get("mstar")
        if isinstance(mstar, dict):
            stylebox_row = {
                "conid": str(conid),
                "effective_at": str(effective_at),
                **{col: 0 for col in _PROFILE_AND_FEES_STYLEBOX_COLUMNS},
            }
            # Bitmask: 1 = Historical, 2 = Current
            hist = mstar.get("hist")
            if isinstance(hist, list):
                for pair in hist:
                    if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                        continue
                    try:
                        x_idx, y_idx = int(pair[0]), int(pair[1])
                    except (ValueError, TypeError):
                        continue
                    column_name = _PROFILE_AND_FEES_STYLEBOX_COORD_COLUMNS.get(
                        (x_idx, y_idx)
                    )
                    if column_name:
                        stylebox_row[column_name] |= 1

            selected = mstar.get("selected")
            if isinstance(selected, list):
                for pair in selected:
                    if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                        continue
                    try:
                        x_idx, y_idx = int(pair[0]), int(pair[1])
                    except (ValueError, TypeError):
                        continue
                    column_name = _PROFILE_AND_FEES_STYLEBOX_COORD_COLUMNS.get(
                        (x_idx, y_idx)
                    )
                    if column_name:
                        stylebox_row[column_name] |= 2

            self._upsert_row(
                conn,
                "profile_and_fees_stylebox",
                stylebox_row,
                ["conid", "effective_at"],
            )

    def _upsert_holdings(
        self,
        conn,
        conid,
        effective_at,
        observed_at,
        payload_hash,
        source_file,
        now_iso,
        payload,
    ):
        as_of_date = _extract_as_of_date(payload)
        conn.execute(
            """
            DELETE FROM holdings_snapshots
            WHERE conid = ? AND effective_at = ?
            """,
            [str(conid), str(effective_at)],
        )
        self._insert_rows(
            conn,
            "holdings_snapshots",
            [
                {
                    "conid": str(conid),
                    "effective_at": str(effective_at),
                    "observed_at": str(observed_at),
                    "payload_hash": str(payload_hash),
                    "inserted_at": now_iso,
                    "updated_at": now_iso,
                    "as_of_date": as_of_date.isoformat()
                    if as_of_date is not None
                    else None,
                }
            ],
        )

        self._delete_children(
            conn,
            ["holdings_top10", *_HOLDINGS_SPLIT_TABLES, "holdings_geographic_weights"],
            conid,
            effective_at,
        )

        top10_rows = []
        for item in (
            payload.get("top_10", []) if isinstance(payload.get("top_10"), list) else []
        ):
            if not isinstance(item, dict):
                continue
            conids = item.get("conids", [])
            conid_list = conids if isinstance(conids, list) else []
            conid_text = ",".join(
                str(value) for value in conid_list if value is not None
            )
            top10_rows.append(
                {
                    "conid": str(conid),
                    "effective_at": str(effective_at),
                    "name": item.get("name"),
                    "ticker": item.get("ticker"),
                    "holding_weight_num": _to_fraction_weight(item.get("assets_pct")),
                    "holding_conids": conid_text or None,
                }
            )
        self._insert_rows(conn, "holdings_top10", top10_rows)

        asset_type_row = {
            "conid": str(conid),
            "effective_at": str(effective_at),
            **{col: None for col in _HOLDINGS_ASSET_TYPE_COLUMNS},
        }
        for item in (
            payload.get("allocation_self", [])
            if isinstance(payload.get("allocation_self"), list)
            else []
        ):
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("type")
            if name is None:
                continue
            key = _sanitize_segment(name)
            column = _HOLDINGS_ASSET_TYPE_SOURCE_TO_COLUMN.get(key)
            if column is None:
                column = "other"
            weight_value = item.get("weight")
            if weight_value is None:
                weight_value = item.get("assets_pct")
            if weight_value is None:
                weight_value = item.get("formatted_weight")
            parsed_weight = _to_fraction_weight(weight_value)
            if parsed_weight is None:
                continue
            if asset_type_row[column] is None:
                asset_type_row[column] = parsed_weight
            else:
                asset_type_row[column] += parsed_weight
        if any(asset_type_row[col] is not None for col in _HOLDINGS_ASSET_TYPE_COLUMNS):
            self._insert_rows(conn, "holdings_asset_type", [asset_type_row])

        section_table_specs = [
            ("industry", "holdings_industry", "industry", None),
            ("currency", "holdings_currency", "currency", "code"),
            (
                "investor_country",
                "holdings_investor_country",
                "country",
                "country_code",
            ),
            ("debt_type", "holdings_debt_type", "debt_type", None),
        ]
        for section, table_name, name_column, extra_column in section_table_specs:
            rows = []
            values = (
                payload.get(section, [])
                if isinstance(payload.get(section), list)
                else []
            )
            for item in values:
                if not isinstance(item, dict):
                    continue
                name = item.get("name") or item.get("type")
                if name is None:
                    continue
                weight_value = item.get("weight")
                if weight_value is None:
                    weight_value = item.get("assets_pct")
                if weight_value is None:
                    weight_value = item.get("formatted_weight")
                row = {
                    "conid": str(conid),
                    "effective_at": str(effective_at),
                    name_column: str(name),
                    "value_num": _to_fraction_weight(weight_value),
                    "industry_avg": _to_fraction_percent(item.get("vs")),
                }
                if extra_column:
                    extra_value = item.get(extra_column)
                    row[extra_column] = (
                        str(extra_value) if extra_value is not None else None
                    )
                rows.append(row)
            self._insert_rows(conn, table_name, rows)

        debtor_quality_row = {
            "conid": str(conid),
            "effective_at": str(effective_at),
            **{
                column_name: None
                for column_name in (
                    *_HOLDINGS_DEBTOR_QUALITY_COLUMNS,
                    *[
                        f"{col}_industry_avg"
                        for col in _HOLDINGS_DEBTOR_QUALITY_COLUMNS
                    ],
                )
            },
        }
        for item in (
            payload.get("debtor", []) if isinstance(payload.get("debtor"), list) else []
        ):
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("type")
            if name is None:
                continue
            column = _HOLDINGS_DEBTOR_QUALITY_SOURCE_TO_COLUMN.get(
                _sanitize_segment(name)
            )
            if column is None:
                continue
            weight_value = item.get("weight")
            if weight_value is None:
                weight_value = item.get("assets_pct")
            if weight_value is None:
                weight_value = item.get("formatted_weight")
            debtor_quality_row[column] = _to_fraction_weight(weight_value)
            debtor_quality_row[f"{column}_industry_avg"] = _to_fraction_percent(
                item.get("vs")
            )
        if any(
            debtor_quality_row[col] is not None
            for col in _HOLDINGS_DEBTOR_QUALITY_COLUMNS
        ):
            self._insert_rows(conn, "holdings_debtor_quality", [debtor_quality_row])

        maturity_row = {
            "conid": str(conid),
            "effective_at": str(effective_at),
            **{
                column_name: None
                for column_name in (
                    *_HOLDINGS_MATURITY_COLUMNS,
                    *[f"{col}_industry_avg" for col in _HOLDINGS_MATURITY_COLUMNS],
                )
            },
        }
        for item in (
            payload.get("maturity", [])
            if isinstance(payload.get("maturity"), list)
            else []
        ):
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("type")
            if name is None:
                continue
            column = _HOLDINGS_MATURITY_SOURCE_TO_COLUMN.get(_sanitize_segment(name))
            if column is None:
                continue
            weight_value = item.get("weight")
            if weight_value is None:
                weight_value = item.get("assets_pct")
            if weight_value is None:
                weight_value = item.get("formatted_weight")
            maturity_row[column] = _to_fraction_weight(weight_value)
            maturity_row[f"{column}_industry_avg"] = _to_fraction_percent(
                item.get("vs")
            )
        if any(maturity_row[col] is not None for col in _HOLDINGS_MATURITY_COLUMNS):
            self._insert_rows(conn, "holdings_maturity", [maturity_row])

        geographic_rows = []
        geographic = (
            payload.get("geographic")
            if isinstance(payload.get("geographic"), dict)
            else {}
        )
        for key, value in geographic.items():
            if isinstance(value, (dict, list)):
                continue
            geographic_rows.append(
                {
                    "conid": str(conid),
                    "effective_at": str(effective_at),
                    "region": _sanitize_segment(key),
                    "value_num": _to_fraction_weight(value),
                }
            )
        self._insert_rows(conn, "holdings_geographic_weights", geographic_rows)

    def _upsert_ratios(
        self,
        conn,
        conid,
        effective_at,
        observed_at,
        payload_hash,
        source_file,
        now_iso,
        payload,
    ):
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
            },
        )

        self._delete_children(
            conn, list(_RATIOS_SECTION_TABLES.values()), conid, effective_at
        )

        for section, table_name in _RATIOS_SECTION_TABLES.items():
            metric_rows = []
            values = (
                payload.get(section, [])
                if isinstance(payload.get(section), list)
                else []
            )
            for item in values:
                if not isinstance(item, dict):
                    continue
                metric_rows.append(
                    {
                        "conid": str(conid),
                        "effective_at": str(effective_at),
                        "metric_id": _sanitize_segment(
                            item.get("name_tag") or item.get("id") or item.get("name")
                        ),
                        "value_num": _parse_number(item.get("value")),
                        "vs_num": _parse_number(item.get("vs")),
                        "min_num": _parse_number(item.get("min")),
                        "max_num": _parse_number(item.get("max")),
                        "avg_num": _parse_number(item.get("avg")),
                        "percentile_num": _parse_number(item.get("percentile")),
                    }
                )
            self._insert_rows(conn, table_name, metric_rows)

    def _upsert_lipper(
        self,
        conn,
        conid,
        effective_at,
        observed_at,
        payload_hash,
        source_file,
        now_iso,
        payload,
    ):
        universes = (
            payload.get("universes", [])
            if isinstance(payload.get("universes"), list)
            else []
        )

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
            include_source_file=False,
        )

        self._delete_children(conn, ["lipper_ratings"], conid, effective_at)

        rows = []
        for universe in universes:
            if not isinstance(universe, dict):
                continue
            universe_name = universe.get("name")
            universe_as_of = _to_iso_date(
                universe.get("as_of_date") or universe.get("asOfDate")
            )
            for period_key, period_items in universe.items():
                if period_key in {"as_of_date", "asOfDate", "name", "title"}:
                    continue
                if not isinstance(period_items, list):
                    continue
                period = _sanitize_segment(period_key)
                for item in period_items:
                    if not isinstance(item, dict):
                        continue
                    rating_raw = item.get("rating")
                    rating = rating_raw if isinstance(rating_raw, dict) else {}
                    rows.append(
                        {
                            "conid": str(conid),
                            "effective_at": str(effective_at),
                            "period": period,
                            "metric_id": _sanitize_segment(
                                item.get("name_tag")
                                or item.get("id")
                                or item.get("name")
                            ),
                            "rating_value": _parse_number(rating.get("value")),
                            "rating_label": rating.get("name"),
                            "universe_name": universe_name,
                            "universe_as_of_date": universe_as_of,
                        }
                    )
        self._insert_rows(conn, "lipper_ratings", rows)

    def _upsert_dividends(
        self,
        conn,
        conid,
        effective_at,
        observed_at,
        payload_hash,
        source_file,
        now_iso,
        payload,
    ):
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
            {},
        )

        self._delete_children(conn, ["dividends_industry_metrics"], conid, effective_at)

        currency = snapshot.get("last_paid_currency")
        if currency is None:
            event_rows = extract_dividends_events(payload)
            for row in event_rows:
                if row.get("currency"):
                    currency = row.get("currency")
                    break

        metrics_row = {
            "conid": str(conid),
            "effective_at": str(effective_at),
            "currency": currency,
        }
        for column in _DIVIDENDS_INDUSTRY_METRIC_COLUMNS:
            metrics_row[column] = _parse_number(snapshot.get(column))
        self._upsert_row(
            conn, "dividends_industry_metrics", metrics_row, ["conid", "effective_at"]
        )

    def _upsert_morningstar(
        self,
        conn,
        conid,
        effective_at,
        observed_at,
        payload_hash,
        source_file,
        now_iso,
        payload,
    ):
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
                "as_of_date": _to_iso_date(
                    payload.get("as_of_date") or payload.get("asOfDate")
                ),
                "q_full_report_id": payload.get("q_full_report_id"),
            },
            include_source_file=False,
        )

        self._delete_children(
            conn, ["morningstar_summary", "morningstar_commentary"], conid, effective_at
        )

        summary_row = {
            "conid": str(conid),
            "effective_at": str(effective_at),
            **{col: None for col in _MORNINGSTAR_SUMMARY_COLUMNS},
        }
        for item in (
            payload.get("summary", [])
            if isinstance(payload.get("summary"), list)
            else []
        ):
            if not isinstance(item, dict):
                continue
            metric_id = _sanitize_segment(
                item.get("id") or item.get("title") or "metric"
            )
            column_name = _MORNINGSTAR_SUMMARY_ID_TO_COLUMN.get(metric_id)
            if column_name is None:
                continue
            value = item.get("value")
            if column_name == "morningstar_rating":
                summary_row[column_name] = _parse_number(value)
            else:
                summary_row[column_name] = str(value) if value is not None else None
        if any(
            summary_row.get(col) is not None for col in _MORNINGSTAR_SUMMARY_COLUMNS
        ):
            self._upsert_row(
                conn, "morningstar_summary", summary_row, ["conid", "effective_at"]
            )

        commentary_rows = []
        for item in (
            payload.get("commentary", [])
            if isinstance(payload.get("commentary"), list)
            else []
        ):
            if not isinstance(item, dict):
                continue
            author_raw = item.get("author")
            author = author_raw if isinstance(author_raw, dict) else {}
            text = item.get("text")
            text = str(text) if text is not None else None
            if text is not None:
                text = text.strip()
            if not text:
                continue
            commentary_rows.append(
                {
                    "conid": str(conid),
                    "effective_at": str(effective_at),
                    "item_id": item.get("id"),
                    "subsection_id": item.get("subsection_id"),
                    "publish_date": _to_iso_date(item.get("publish_date")),
                    "text": text,
                    "author_name": author.get("name"),
                }
            )
        self._insert_rows(conn, "morningstar_commentary", commentary_rows)

    def _upsert_performance(
        self,
        conn,
        conid,
        effective_at,
        observed_at,
        payload_hash,
        source_file,
        now_iso,
        payload,
    ):
        self._upsert_snapshot_row(
            conn,
            "performance_snapshots",
            conid,
            effective_at,
            observed_at,
            payload_hash,
            source_file,
            now_iso,
            {},
        )

        self._delete_children(conn, ["performance"], conid, effective_at)

        rows = []
        for section in ("cumulative", "annualized", "yield", "risk", "statistic"):
            values = (
                payload.get(section, [])
                if isinstance(payload.get(section), list)
                else []
            )
            for item in values:
                if not isinstance(item, dict):
                    continue
                rows.append(
                    {
                        "conid": str(conid),
                        "effective_at": str(effective_at),
                        "section": section,
                        "metric_id": _sanitize_segment(
                            item.get("name_tag") or item.get("id") or item.get("name")
                        ),
                        "value_num": _parse_number(item.get("value")),
                        "vs_num": _parse_number(item.get("vs")),
                        "min_num": _parse_number(item.get("min")),
                        "max_num": _parse_number(item.get("max")),
                        "avg_num": _parse_number(item.get("avg")),
                    }
                )
        self._insert_rows(conn, "performance", rows)

    def _upsert_ownership(
        self,
        conn,
        conid,
        effective_at,
        observed_at,
        payload_hash,
        source_file,
        now_iso,
        payload,
    ):
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
                "institutional_owners_count": int(
                    snapshot.get("institutional_owners_count") or 0
                ),
                "insider_owners_count": int(snapshot.get("insider_owners_count") or 0),
                "trade_log_count_raw": int(snapshot.get("trade_log_count_raw") or 0),
                "trade_log_count_kept": int(snapshot.get("trade_log_count_kept") or 0),
                "has_ownership_history": _to_int_bool(
                    snapshot.get("has_ownership_history")
                ),
                "ownership_history_price_points": int(
                    snapshot.get("ownership_history_price_points") or 0
                ),
                "institutional_total_value_text": snapshot.get(
                    "institutional_total_value"
                ),
                "institutional_total_shares_text": snapshot.get(
                    "institutional_total_shares"
                ),
                "institutional_total_pct_text": snapshot.get("institutional_total_pct"),
                "institutional_total_pct_num": _parse_number(
                    snapshot.get("institutional_total_pct_num")
                ),
                "insider_total_value_text": snapshot.get("insider_total_value"),
                "insider_total_shares_text": snapshot.get("insider_total_shares"),
                "insider_total_pct_text": snapshot.get("insider_total_pct"),
                "insider_total_pct_num": _parse_number(
                    snapshot.get("insider_total_pct_num")
                ),
            },
        )

        self._delete_children(
            conn, ["ownership_owners_types", "ownership_holders"], conid, effective_at
        )

        owners_type_rows = []
        for item in (
            payload.get("owners_types", [])
            if isinstance(payload.get("owners_types"), list)
            else []
        ):
            if not isinstance(item, dict):
                continue
            type_info_raw = item.get("type")
            type_info = type_info_raw if isinstance(type_info_raw, dict) else {}
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
                holder_type = item.get("type")
                if isinstance(holder_type, dict):
                    holder_type = holder_type.get("type") or holder_type.get(
                        "display_type"
                    )
                holder_rows.append(
                    {
                        "conid": str(conid),
                        "effective_at": str(effective_at),
                        "holder_group": holder_group,
                        "holder_name": item.get("name"),
                        "holder_type": holder_type,
                        "display_value": item.get("display_value"),
                        "display_shares": item.get("display_shares"),
                        "display_pct": item.get("display_pct"),
                        "pct_num": _parse_number(
                            item.get("display_pct"), percent_as_fraction=True
                        ),
                    }
                )
        self._insert_rows(conn, "ownership_holders", holder_rows)

    def _upsert_esg(
        self,
        conn,
        conid,
        effective_at,
        observed_at,
        payload_hash,
        source_file,
        now_iso,
        payload,
    ):
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
                "as_of_date": _to_iso_date(
                    payload.get("as_of_date") or payload.get("asOfDate")
                ),
            },
        )

        values_by_column: dict[str, float | None] = {
            column: None for column in _ESG_SCORE_COLUMNS
        }

        def walk(nodes):
            if not isinstance(nodes, list):
                return
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                name = str(node.get("name") or "").strip()
                target_column = _ESG_CODE_TO_COLUMN.get(name)
                if target_column is not None:
                    values_by_column[target_column] = _parse_number(node.get("value"))
                walk(node.get("children"))

        walk(payload.get("content"))

        row = {
            "conid": str(conid),
            "effective_at": str(effective_at),
            "coverage": _parse_number(payload.get("coverage")),
            "source": payload.get("source"),
        }
        row.update(values_by_column)
        self._upsert_row(conn, "esg", row, ["conid", "effective_at"])

    def _upsert_price_chart_snapshot(
        self,
        conn,
        conid,
        effective_at,
        observed_at,
        payload_hash,
        source_file,
        now_iso,
        payload,
    ):
        rows = _extract_price_chart_rows(payload)
        dates = [r.get("effective_at") for r in rows if r.get("effective_at")]
        debug_mismatch_count = sum(int(r.get("debug_mismatch") or 0) for r in rows)
        if debug_mismatch_count > 0:
            logger.warning(
                "price_chart date mismatch between x and debugY: conid=%s effective_at=%s mismatches=%s",
                str(conid),
                str(effective_at),
                int(debug_mismatch_count),
            )
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
            include_source_file=False,
        )

    def _upsert_sentiment_snapshot(
        self,
        conn,
        conid,
        effective_at,
        observed_at,
        payload_hash,
        source_file,
        now_iso,
        payload,
    ):
        rows = _extract_sentiment_search_rows(payload)
        dates = [r.get("effective_at") for r in rows if r.get("effective_at")]
        min_date = min(dates) if dates else None
        max_date = max(dates) if dates else None

        self._upsert_snapshot_row(
            conn,
            "sentiment_snapshots",
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

    def _series_is_newer(
        self, new_effective_at, new_observed_at, new_payload_hash, existing_row
    ):
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

    def _write_price_chart_series(
        self, conn, conid, effective_at, observed_at, payload_hash, inserted_at, payload
    ):
        rows = _extract_price_chart_rows(payload)
        raw_written = 0
        latest_upserted = 0

        for row in rows:
            point_effective_at = row.get("effective_at")
            if not point_effective_at:
                continue

            conn.execute(
                """
                INSERT INTO price_chart_series (
                    conid,
                    effective_at,
                    price,
                    open,
                    high,
                    low,
                    close
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(conid, effective_at) DO UPDATE SET
                    price = excluded.price,
                    open = excluded.open,
                    high = excluded.high,
                    low = excluded.low,
                    close = excluded.close
                """,
                [
                    str(conid),
                    str(point_effective_at),
                    _parse_number(row.get("price")),
                    _parse_number(row.get("open")),
                    _parse_number(row.get("high")),
                    _parse_number(row.get("low")),
                    _parse_number(row.get("close")),
                ],
            )
            raw_written += 1

        return raw_written, latest_upserted

    def _write_sentiment_series(
        self, conn, conid, effective_at, observed_at, payload_hash, inserted_at, payload
    ):
        rows = _extract_sentiment_search_rows(payload)
        raw_written = 0
        latest_upserted = 0

        for row in rows:
            point_effective_at = row.get("effective_at")
            if not point_effective_at:
                continue

            conn.execute(
                """
                INSERT INTO sentiment_series (
                    conid,
                    effective_at,
                    sscore,
                    sdelta,
                    svolatility,
                    sdispersion,
                    svscore,
                    svolume,
                    smean,
                    sbuzz
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(conid, effective_at) DO UPDATE SET
                    sscore = excluded.sscore,
                    sdelta = excluded.sdelta,
                    svolatility = excluded.svolatility,
                    sdispersion = excluded.sdispersion,
                    svscore = excluded.svscore,
                    svolume = excluded.svolume,
                    smean = excluded.smean,
                    sbuzz = excluded.sbuzz
                """,
                [
                    str(conid),
                    str(point_effective_at),
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

        return raw_written, latest_upserted

    def _write_ownership_trade_log_series(
        self, conn, conid, effective_at, observed_at, payload_hash, inserted_at, payload
    ):
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
                        str(row.get("insider"))
                        if row.get("insider") is not None
                        else None,
                    ],
                )
                latest_upserted += 1
            elif self._series_is_newer(
                effective_at, observed_at, payload_hash, existing
            ):
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
                        str(row.get("insider"))
                        if row.get("insider") is not None
                        else None,
                        str(conid),
                        row_key,
                    ],
                )
                latest_upserted += 1

        return raw_written, latest_upserted

    def _write_dividends_events_series(
        self, conn, conid, effective_at, observed_at, payload_hash, inserted_at, payload
    ):
        rows = extract_dividends_events(payload)
        raw_written = 0
        latest_upserted = 0

        for row in rows:
            point_effective_at = row.get("trade_date") or row.get("event_date")
            if not point_effective_at:
                continue

            event_date = row.get("event_date")
            if event_date:
                point_date = _parse_date_candidate(point_effective_at)
                debug_date = _parse_date_candidate(event_date)
                if (
                    point_date is not None
                    and debug_date is not None
                    and abs((point_date - debug_date).days) > 1
                ):
                    logger.warning(
                        "dividends date mismatch between trade_date and event_date: conid=%s effective_at=%s trade_date=%s event_date=%s",
                        str(conid),
                        str(effective_at),
                        str(point_effective_at),
                        str(event_date),
                    )

            existing = conn.execute(
                """
                SELECT 1
                FROM dividends_events_series
                WHERE conid = ?
                  AND effective_at = ?
                  AND COALESCE(amount, -999999999.0) = COALESCE(?, -999999999.0)
                  AND COALESCE(currency, '') = COALESCE(?, '')
                  AND COALESCE(description, '') = COALESCE(?, '')
                  AND COALESCE(event_type, '') = COALESCE(?, '')
                  AND COALESCE(declaration_date, '') = COALESCE(?, '')
                  AND COALESCE(record_date, '') = COALESCE(?, '')
                  AND COALESCE(payment_date, '') = COALESCE(?, '')
                LIMIT 1
                """,
                [
                    str(conid),
                    str(point_effective_at),
                    _parse_number(row.get("amount")),
                    row.get("currency"),
                    row.get("description"),
                    row.get("event_type"),
                    row.get("declaration_date"),
                    row.get("record_date"),
                    row.get("payment_date"),
                ],
            ).fetchone()
            if existing is not None:
                continue

            conn.execute(
                """
                INSERT INTO dividends_events_series (
                    conid,
                    effective_at,
                    amount,
                    currency,
                    description,
                    event_type,
                    declaration_date,
                    record_date,
                    payment_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    str(conid),
                    str(point_effective_at),
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

        return raw_written, latest_upserted

    def _write_series(
        self,
        conn,
        endpoint,
        conid,
        effective_at,
        observed_at,
        payload_hash,
        inserted_at,
        payload,
    ):
        if endpoint == "price_chart":
            return self._write_price_chart_series(
                conn,
                conid,
                effective_at,
                observed_at,
                payload_hash,
                inserted_at,
                payload,
            )
        if endpoint == "sentiment_search":
            return self._write_sentiment_series(
                conn,
                conid,
                effective_at,
                observed_at,
                payload_hash,
                inserted_at,
                payload,
            )
        if endpoint == "ownership":
            return self._write_ownership_trade_log_series(
                conn,
                conid,
                effective_at,
                observed_at,
                payload_hash,
                inserted_at,
                payload,
            )
        if endpoint == "dividends":
            return self._write_dividends_events_series(
                conn,
                conid,
                effective_at,
                observed_at,
                payload_hash,
                inserted_at,
                payload,
            )
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
        now_iso = datetime.now(UTC).isoformat()
        self._ensure_product(conn, conid)

        blob_info = self._store_blob(conn, payload)
        payload_hash = blob_info["hash"]

        table = self._main_table_for_endpoint(endpoint)
        if endpoint == "holdings":
            existing = conn.execute(
                f"SELECT payload_hash FROM {table} WHERE conid = ? AND effective_at = ? ORDER BY rowid DESC LIMIT 1",
                [str(conid), str(effective_at)],
            ).fetchone()
        else:
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

        if state == "unchanged" and endpoint in SERIES_ENDPOINTS:
            return {
                "endpoint": endpoint,
                "state": state,
                "effective_source": effective_source,
                "series_raw_rows_written": 0,
                "series_latest_rows_upserted": 0,
            }

        handlers = {
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

    def persist_combined_snapshot(self, snapshot, source_file=None):
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

        observed_at = snapshot.get("scraped_at") or datetime.now(UTC).isoformat()
        try:
            observed_dt = datetime.fromisoformat(
                str(observed_at).replace("Z", "+00:00")
            )
            if observed_dt.tzinfo is None:
                observed_dt = observed_dt.replace(tzinfo=UTC)
        except Exception:
            observed_dt = datetime.now(UTC)
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
            saved_any_endpoint = False
            for endpoint, payload in endpoint_payloads.items():
                effective_at, effective_source = effective_map[endpoint]
                if effective_at is None:
                    per_endpoint[endpoint] = {
                        "endpoint": endpoint,
                        "state": "skipped_missing_effective_at",
                        "effective_source": effective_source,
                        "series_raw_rows_written": 0,
                        "series_latest_rows_upserted": 0,
                    }
                    continue
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
                series_latest_rows_upserted += int(
                    result.get("series_latest_rows_upserted", 0)
                )
                per_endpoint[endpoint] = result
                saved_any_endpoint = True

            if saved_any_endpoint:
                conn.commit()

        status = "ok" if saved_any_endpoint else "missing_ratios_effective_at"
        return {
            "inserted_events": inserted_events,
            "overwritten_events": overwritten_events,
            "unchanged_events": unchanged_events,
            "series_raw_rows_written": series_raw_rows_written,
            "series_latest_rows_upserted": series_latest_rows_upserted,
            "status": status,
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
                    run_stats.get("run_started_at") or datetime.now(UTC).isoformat(),
                    run_stats.get("run_finished_at") or datetime.now(UTC).isoformat(),
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
            lastrowid = cur.lastrowid
            if lastrowid is None:
                raise RuntimeError("Failed to persist telemetry run row.")
            run_id = int(lastrowid)

            for row in endpoint_summary:
                endpoint = str(row.get("endpoint") or "")
                if not endpoint:
                    continue
                call_count = int(row.get("call_count", 0))
                useful_payload_count = int(row.get("useful_payload_count", 0))
                useful_payload_rate = (
                    float(row.get("useful_payload_rate", 0.0)) if call_count else 0.0
                )
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
