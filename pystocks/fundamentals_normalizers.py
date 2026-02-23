import json
import re
from datetime import datetime, timezone

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


def _safe_list(value):
    return value if isinstance(value, list) else []


def _safe_dict(value):
    return value if isinstance(value, dict) else {}


def _sanitize_metric_id(value):
    s = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return s or "metric"


def _parse_percent(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    s = s.replace("%", "").replace(",", "")
    try:
        return float(s)
    except Exception:
        return None


def _parse_number(value, percent_as_fraction=False):
    if value is None:
        return None

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()
    if not s:
        return None

    is_percent = s.endswith("%")
    s = s.replace(",", "").replace("%", "")
    try:
        out = float(s)
    except Exception:
        return None

    if is_percent and percent_as_fraction:
        return out / 100.0
    return out


_PERCENT_FRACTION_METRICS = {
    "dividend_yield",
    "dividend_yield_ttm",
    "div_yield",
    "paying_companies_percent",
}

_KNOWN_DIVIDENDS_NUMERIC_METRICS = {
    "dividend_yield",
    "dividend_yield_ttm",
    "div_yield",
    "annual_dividend",
    "dividend_ttm",
    "div_per_share",
    "paying_companies",
    "paying_companies_percent",
}


def _coerce_dividends_metric_value(metric_id, value):
    metric = _sanitize_metric_id(metric_id)
    parsed = _parse_number(value, percent_as_fraction=metric in _PERCENT_FRACTION_METRICS)
    if parsed is not None:
        return parsed
    if metric in _KNOWN_DIVIDENDS_NUMERIC_METRICS:
        return None
    return value


def _to_iso_date(value):
    if value is None:
        return None

    if isinstance(value, dict):
        text_date = value.get("t")
        if text_date:
            return _to_iso_date(text_date)

        y = value.get("y")
        m = value.get("m")
        d = value.get("d")
        if y and m and d:
            try:
                if isinstance(m, str):
                    m_num = _MONTH_MAP.get(m.strip().upper())
                    if m_num is None:
                        m_num = int(m)
                else:
                    m_num = int(m)
                return datetime(int(y), int(m_num), int(d)).date().isoformat()
            except Exception:
                return None
        return None

    if isinstance(value, (int, float)):
        ts = float(value)
        if ts <= 0:
            return None
        iv = int(ts)
        if 19000101 <= iv <= 29991231:
            try:
                return datetime.strptime(str(iv), "%Y%m%d").date().isoformat()
            except Exception:
                pass
        if ts > 1e12:
            ts = ts / 1000.0
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
        except Exception:
            return None

    s = str(value).strip()
    if not s:
        return None
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return s
    if len(s) == 10 and s[4] == "/" and s[7] == "/":
        return s.replace("/", "-")
    if len(s) == 8 and s.isdigit():
        try:
            return datetime.strptime(s, "%Y%m%d").date().isoformat()
        except Exception:
            pass
    if s.isdigit():
        return _to_iso_date(int(s))
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date().isoformat()
    except Exception:
        return None


def _extract_currency(formatted_amount):
    if not formatted_amount:
        return None
    m = re.search(r"\b([A-Z]{3})\b", str(formatted_amount))
    return m.group(1) if m else None


def _metric_id(item):
    if not isinstance(item, dict):
        return "metric"
    for key in ("search_id", "title_tag", "name_tag", "title", "name"):
        if item.get(key):
            return _sanitize_metric_id(item.get(key))
    return "metric"


def _series_points(history_payload, needle):
    points = 0
    for series in _safe_list(_safe_dict(history_payload).get("series")):
        if not isinstance(series, dict):
            continue
        name = (series.get("name") or series.get("title") or "").strip().lower()
        if needle in name:
            points += len(_safe_list(series.get("plotData")))
    return points


def normalize_dividends_snapshot(payload):
    payload = _safe_dict(payload)

    industry_average = _safe_dict(payload.get("industry_average"))
    industry_comparison = _safe_dict(payload.get("industry_comparison"))
    comparison_content = _safe_list(industry_comparison.get("content"))

    history = _safe_dict(payload.get("history"))
    dividend_history_points = _series_points(history, "dividend")
    embedded_price_points = _series_points(history, "price")

    out = {
        "response_type": "long" if history else "short",
        "has_history": bool(history),
        "history_points": int(dividend_history_points),
        "embedded_price_points": int(embedded_price_points),
        "no_div_data_marker": payload.get("no_div_data_marker"),
        "no_div_data_period": payload.get("no_div_data_period"),
        "dividend_yield": _coerce_dividends_metric_value("dividend_yield", industry_average.get("dividend_yield")),
        "annual_dividend": _coerce_dividends_metric_value("annual_dividend", industry_average.get("annual_dividend")),
        "paying_companies": _coerce_dividends_metric_value("paying_companies", industry_average.get("paying_companies")),
        "paying_companies_percent": _coerce_dividends_metric_value(
            "paying_companies_percent",
            industry_average.get("paying_companies_percent"),
        ),
        "dividend_ttm": None,
        "dividend_yield_ttm": None,
        "last_paid_date": _to_iso_date(payload.get("last_payed_dividend_date")),
        "last_paid_amount": _parse_number(payload.get("last_payed_dividend_amount")),
        "last_paid_currency": payload.get("last_payed_dividend_currency"),
    }

    for item in comparison_content:
        if not isinstance(item, dict):
            continue
        metric_id = _metric_id(item)
        if metric_id in {"div_yield", "dividend_yield_ttm", "dividend_yield"}:
            out["dividend_yield_ttm"] = _coerce_dividends_metric_value(metric_id, item.get("value"))
        elif metric_id in {"div_per_share", "dividend_ttm", "annual_dividend"}:
            out["dividend_ttm"] = _coerce_dividends_metric_value(metric_id, item.get("value"))

    if out["dividend_ttm"] is None and industry_average.get("annual_dividend") is not None:
        out["dividend_ttm"] = _coerce_dividends_metric_value("annual_dividend", industry_average.get("annual_dividend"))
    if out["dividend_yield_ttm"] is None and industry_average.get("dividend_yield") is not None:
        out["dividend_yield_ttm"] = _coerce_dividends_metric_value("dividend_yield", industry_average.get("dividend_yield"))

    return out


def extract_dividends_events(payload):
    payload = _safe_dict(payload)
    history = _safe_dict(payload.get("history"))
    series = _safe_list(history.get("series"))

    fallback_currency = payload.get("last_payed_dividend_currency")
    rows = []

    for node in series:
        if not isinstance(node, dict):
            continue
        name = (node.get("name") or node.get("title") or "").strip().lower()
        if "dividend" not in name:
            continue

        for point in _safe_list(node.get("plotData")):
            if not isinstance(point, dict):
                continue

            event_date = _to_iso_date(point.get("ex_dividend_date")) or _to_iso_date(point.get("x"))
            amount = point.get("amount")
            if amount is None:
                amount = point.get("y")
            amount_value = _parse_number(amount)
            if amount_value is not None:
                amount = amount_value

            row = {
                "event_date": event_date,
                "amount": amount,
                "currency": _extract_currency(point.get("formatted_amount")) or fallback_currency,
                "description": point.get("description"),
                "event_type": point.get("type"),
                "declaration_date": _to_iso_date(point.get("declaration_date")),
                "record_date": _to_iso_date(point.get("record_date")),
                "payment_date": _to_iso_date(point.get("payment_date")),
            }

            if any(v is not None for v in row.values()):
                rows.append(row)

    return rows


def extract_dividends_industry_metrics(payload):
    payload = _safe_dict(payload)
    rows = []

    industry_average = _safe_dict(payload.get("industry_average"))
    field_map = {
        "dividend_yield": "dividend_yield",
        "annual_dividend": "annual_dividend",
        "paying_companies": "paying_companies",
        "paying_companies_percent": "paying_companies_percent",
    }

    for source_key, metric_id in field_map.items():
        if source_key not in industry_average:
            continue
        raw_value = industry_average.get(source_key)
        value = _coerce_dividends_metric_value(metric_id, raw_value)
        rows.append(
            {
                "metric_id": metric_id,
                "value": value,
                "formatted_value": str(raw_value) if raw_value is not None else None,
            }
        )

    industry_comparison = _safe_dict(payload.get("industry_comparison"))
    for item in _safe_list(industry_comparison.get("content")):
        if not isinstance(item, dict):
            continue
        metric_id = _metric_id(item)
        raw_value = item.get("value")
        formatted_value = item.get("formatted_value")
        if formatted_value is None and raw_value is not None:
            formatted_value = str(raw_value)
        rows.append(
            {
                "metric_id": metric_id,
                "value": _coerce_dividends_metric_value(metric_id, raw_value),
                "formatted_value": formatted_value,
            }
        )

    return rows


def normalize_ownership_snapshot(payload):
    payload = _safe_dict(payload)

    trade_log = _safe_list(payload.get("trade_log"))
    kept_count = 0
    for row in trade_log:
        action = str(_safe_dict(row).get("action") or "").strip().upper()
        if action != "NO CHANGE":
            kept_count += 1

    owners_types = _safe_list(payload.get("owners_types"))
    institutional_owners = _safe_list(payload.get("institutional_owners"))
    insider_owners = _safe_list(payload.get("insider_owners"))

    history = _safe_dict(payload.get("ownership_history"))
    embedded_price_points = _series_points(history, "price")

    inst_total = _safe_dict(payload.get("institutional_total"))
    insider_total = _safe_dict(payload.get("insider_total"))

    owners_types_summary = []
    for row in owners_types:
        row = _safe_dict(row)
        type_info = _safe_dict(row.get("type"))
        owners_types_summary.append(
            {
                "type": type_info.get("type"),
                "display_type": type_info.get("display_type"),
                "float": row.get("float"),
                "display_float": row.get("display_float"),
            }
        )

    out = {
        "owners_types_count": len(owners_types),
        "institutional_owners_count": len(institutional_owners),
        "insider_owners_count": len(insider_owners),
        "trade_log_count_raw": len(trade_log),
        "trade_log_count_kept": kept_count,
        "has_ownership_history": bool(history),
        "ownership_history_price_points": int(embedded_price_points),
        "institutional_total_value": inst_total.get("display_value"),
        "institutional_total_shares": inst_total.get("display_shares"),
        "institutional_total_pct": inst_total.get("display_pct"),
        "institutional_total_pct_num": _parse_percent(inst_total.get("display_pct")),
        "insider_total_value": insider_total.get("display_value"),
        "insider_total_shares": insider_total.get("display_shares"),
        "insider_total_pct": insider_total.get("display_pct"),
        "insider_total_pct_num": _parse_percent(insider_total.get("display_pct")),
        "owners_types_json": json.dumps(owners_types_summary, separators=(",", ":"), ensure_ascii=False),
    }

    return out


def extract_ownership_trade_log(payload, drop_no_change=True):
    payload = _safe_dict(payload)
    rows = []

    for item in _safe_list(payload.get("trade_log")):
        item = _safe_dict(item)
        action = str(item.get("action") or "").strip().upper()
        if drop_no_change and action == "NO CHANGE":
            continue

        row = {
            "trade_date": _to_iso_date(item.get("displayDate")),
            "action": action or None,
            "shares": item.get("shares"),
            "value": item.get("value"),
            "holding": item.get("holding"),
            "party": item.get("party"),
            "source": item.get("source"),
            "insider": item.get("insider"),
        }

        if any(v is not None for v in row.values()):
            rows.append(row)

    return rows


__all__ = [
    "normalize_dividends_snapshot",
    "extract_dividends_events",
    "extract_dividends_industry_metrics",
    "normalize_ownership_snapshot",
    "extract_ownership_trade_log",
]
