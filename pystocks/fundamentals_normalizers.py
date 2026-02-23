import json
import math
import re
from collections import defaultdict
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

_NUM_WITH_SUFFIX_RE = re.compile(r"^[\s\$€£¥]*([+-]?\d+(?:\.\d+)?)\s*([KMBT])?\b", re.IGNORECASE)
_DATE_LIKE_RE = re.compile(r"\b\d{4}[/-]\d{2}[/-]\d{2}\b")


_MORNINGSTAR_ORDINAL = {
    "very_low": 1.0,
    "low": 2.0,
    "below_average": 2.0,
    "negative": 1.0,
    "neutral": 3.0,
    "average": 3.0,
    "bronze": 3.0,
    "above_average": 4.0,
    "silver": 4.0,
    "high": 5.0,
    "very_high": 5.0,
    "gold": 5.0,
    "medalist_rating_negative": 1.0,
    "medalist_rating_neutral": 3.0,
    "medalist_rating_bronze": 3.0,
    "medalist_rating_silver": 4.0,
    "medalist_rating_gold": 5.0,
}


def _safe_list(value):
    return value if isinstance(value, list) else []


def _safe_dict(value):
    return value if isinstance(value, dict) else {}


def _slug(value):
    s = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return s or "metric"


def _looks_like_date_string(value):
    if value is None:
        return False
    s = str(value).strip()
    if not s:
        return False
    return bool(_DATE_LIKE_RE.search(s))


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

    if _looks_like_date_string(s):
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
    mult = {"": 1.0, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}.get(suffix, 1.0)
    out = num * mult
    if is_percent and percent_as_fraction:
        out /= 100.0
    if math.isfinite(out):
        return out
    return None


def _to_fraction_weight(value):
    parsed = _parse_number(value, percent_as_fraction=True)
    if parsed is None:
        return None
    if parsed > 1.5:
        return parsed / 100.0
    return parsed


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
    for key in ("search_id", "title_tag", "name_tag", "id", "title", "name"):
        if item.get(key):
            return _slug(item.get(key))
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


def _add_feature(rows, feature_name, feature_group, feature_value, feature_source):
    value = _parse_number(feature_value)
    if value is None or not math.isfinite(value):
        return
    rows.append(
        {
            "feature_name": _slug(feature_name),
            "feature_group": _slug(feature_group),
            "feature_value": float(value),
            "feature_source": str(feature_source),
        }
    )


def _dedupe_feature_rows(rows):
    grouped = defaultdict(list)
    meta = {}
    for row in rows:
        key = row["feature_name"]
        grouped[key].append(float(row["feature_value"]))
        meta[key] = (row["feature_group"], row["feature_source"])

    out = []
    for feature_name, values in grouped.items():
        group, source = meta[feature_name]
        out.append(
            {
                "feature_name": feature_name,
                "feature_group": group,
                "feature_value": float(sum(values) / len(values)),
                "feature_source": source,
            }
        )
    out.sort(key=lambda r: (r["feature_group"], r["feature_name"]))
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
    metric = _slug(metric_id)
    parsed = _parse_number(value, percent_as_fraction=metric in _PERCENT_FRACTION_METRICS)
    if parsed is not None:
        return parsed
    if metric in _KNOWN_DIVIDENDS_NUMERIC_METRICS:
        return None
    return value


# -------------------------
# Existing canonical helpers
# -------------------------
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
                "trade_date": event_date,
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
        "institutional_total_pct_num": _parse_number(inst_total.get("display_pct"), percent_as_fraction=True),
        "insider_total_value": insider_total.get("display_value"),
        "insider_total_shares": insider_total.get("display_shares"),
        "insider_total_pct": insider_total.get("display_pct"),
        "insider_total_pct_num": _parse_number(insider_total.get("display_pct"), percent_as_fraction=True),
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
            "shares": _parse_number(item.get("shares")),
            "value": _parse_number(item.get("value")),
            "holding": _parse_number(item.get("holding")),
            "party": item.get("party"),
            "source": item.get("source"),
            "insider": item.get("insider"),
        }

        if any(v is not None for v in row.values()):
            rows.append(row)

    return rows


# -------------------------
# Factor feature extractors
# -------------------------
def _extract_profile_and_fees_features(payload):
    payload = _safe_dict(payload)
    rows = []

    for item in _safe_list(payload.get("fund_and_profile")):
        item = _safe_dict(item)
        metric = _slug(item.get("name_tag") or item.get("name"))
        value = item.get("value")

        num = _parse_number(value, percent_as_fraction=True)
        if num is not None:
            _add_feature(rows, f"profile_{metric}", "profile", num, f"fund_and_profile:{metric}")

        if metric == "asset_type" and value:
            _add_feature(rows, f"asset_{_slug(value)}", "asset", 1.0, "fund_and_profile:asset_type")
        elif metric in {"fund_market_cap_focus", "market_cap_focus"} and value:
            _add_feature(rows, f"marketcap_{_slug(value)}", "marketcap", 1.0, "fund_and_profile:market_cap")
        elif metric == "domicile" and value:
            _add_feature(rows, f"domicile_{_slug(value)}", "domicile", 1.0, "fund_and_profile:domicile")
        elif metric in {"management_approach", "distribution_details", "fund_category"} and value:
            _add_feature(rows, f"profile_{metric}_{_slug(value)}", "profile", 1.0, f"fund_and_profile:{metric}")

    for report in _safe_list(payload.get("reports")):
        report = _safe_dict(report)
        report_name = _slug(report.get("name") or "report")
        for field in _safe_list(report.get("fields")):
            field = _safe_dict(field)
            field_name = _slug(field.get("name") or field.get("name_tag") or "field")
            num = _parse_number(field.get("value"), percent_as_fraction=True)
            if num is None:
                continue
            _add_feature(
                rows,
                f"profile_report_{report_name}_{field_name}",
                "profile",
                num,
                f"reports:{report_name}:{field_name}",
            )

    for theme in _safe_list(payload.get("themes")):
        theme_name = None
        if isinstance(theme, dict):
            theme_name = theme.get("name") or theme.get("title") or theme.get("theme")
        elif isinstance(theme, str):
            theme_name = theme
        if theme_name:
            _add_feature(rows, f"style_{_slug(theme_name)}", "style", 1.0, "themes")

    for exp in _safe_list(payload.get("expenses_allocation")):
        exp = _safe_dict(exp)
        name = _slug(exp.get("name") or exp.get("name_tag") or "expense")
        weight = exp.get("weight")
        if weight is None:
            weight = exp.get("value")
        if weight is None:
            weight = exp.get("formatted_weight")
        frac = _to_fraction_weight(weight)
        if frac is None:
            continue
        _add_feature(rows, f"profile_expense_{name}", "profile", frac, f"expenses_allocation:{name}")

    warning = payload.get("jap_fund_warning")
    if isinstance(warning, bool):
        _add_feature(rows, "profile_jap_fund_warning", "profile", 1.0 if warning else 0.0, "jap_fund_warning")

    return _dedupe_feature_rows(rows)


def _extract_holdings_features(payload):
    payload = _safe_dict(payload)
    rows = []

    sections = [
        ("allocation_self", "holding_types"),
        ("industry", "industries"),
        ("currency", "currencies"),
        ("investor_country", "countries"),
    ]

    for section_key, feature_prefix in sections:
        for item in _safe_list(payload.get(section_key)):
            item = _safe_dict(item)
            name = _slug(item.get("name") or item.get("type") or "item")
            weight = item.get("weight")
            if weight is None:
                weight = item.get("assets_pct")
            if weight is None:
                weight = item.get("formatted_weight")
            frac = _to_fraction_weight(weight)
            if frac is None:
                continue
            _add_feature(rows, f"{feature_prefix}_{name}", feature_prefix, frac, f"{section_key}:{name}")

    geo = _safe_dict(payload.get("geographic"))
    for key, value in geo.items():
        frac = _to_fraction_weight(value)
        if frac is None:
            continue
        _add_feature(rows, f"countries_{_slug(key)}", "countries", frac, f"geographic:{key}")

    top10_weight = _to_fraction_weight(payload.get("top_10_weight"))
    if top10_weight is not None:
        _add_feature(rows, "holding_top_10_weight", "holdings", top10_weight, "top_10_weight")

    return _dedupe_feature_rows(rows)


def _extract_ratios_features(payload):
    payload = _safe_dict(payload)
    rows = []
    seen = set()

    for section in ("ratios", "financials", "fixed_income", "dividend", "zscore"):
        for item in _safe_list(payload.get(section)):
            item = _safe_dict(item)
            metric = _metric_id(item)
            base = f"fundamentals_{metric}"
            if base in seen:
                base = f"fundamentals_{_slug(section)}_{metric}"
            seen.add(base)

            value = _parse_number(item.get("value"))
            if value is not None:
                _add_feature(rows, base, "fundamentals", value, f"{section}:{metric}:value")

            for field in ("vs", "min", "max", "avg", "percentile"):
                v = _parse_number(item.get(field))
                if v is None:
                    continue
                _add_feature(rows, f"{base}_{field}", "fundamentals", v, f"{section}:{metric}:{field}")

    return _dedupe_feature_rows(rows)


def _extract_lipper_ratings_features(payload):
    payload = _safe_dict(payload)
    rows = []

    for universe in _safe_list(payload.get("universes")):
        universe = _safe_dict(universe)
        for period_key, period_items in universe.items():
            if period_key in {"as_of_date", "asOfDate", "name", "title"}:
                continue
            if not isinstance(period_items, list):
                continue

            period = _slug(period_key)
            for item in period_items:
                item = _safe_dict(item)
                metric = _metric_id(item)
                rating = _safe_dict(item.get("rating")).get("value")
                value = _parse_number(rating)
                if value is None:
                    continue
                _add_feature(rows, f"lipper_{period}_{metric}", "lipper", value, f"universes:{period}:{metric}")

    return _dedupe_feature_rows(rows)


def _extract_dividends_features(payload):
    rows = []

    snapshot = normalize_dividends_snapshot(payload)
    for key, value in snapshot.items():
        if key in {"last_paid_currency", "response_type", "last_paid_date"}:
            continue
        num = _parse_number(value)
        if num is None:
            continue
        _add_feature(rows, f"dividends_{_slug(key)}", "dividends", num, f"snapshot:{key}")

    for item in extract_dividends_industry_metrics(payload):
        metric_id = _slug(item.get("metric_id"))
        value = _parse_number(item.get("value"))
        if value is None:
            continue
        _add_feature(rows, f"dividends_metric_{metric_id}", "dividends", value, f"industry:{metric_id}")

    return _dedupe_feature_rows(rows)


def _morningstar_numeric(value, metric_id):
    num = _parse_number(value)
    if num is not None:
        return num

    key = _slug(value)
    if key in _MORNINGSTAR_ORDINAL:
        return _MORNINGSTAR_ORDINAL[key]

    metric_key = f"{_slug(metric_id)}_{key}"
    if metric_key in _MORNINGSTAR_ORDINAL:
        return _MORNINGSTAR_ORDINAL[metric_key]

    return None


def _extract_morningstar_features(payload):
    payload = _safe_dict(payload)
    rows = []

    for item in _safe_list(payload.get("summary")):
        item = _safe_dict(item)
        metric = _slug(item.get("id") or item.get("title") or "metric")
        value = _morningstar_numeric(item.get("value"), metric)
        if value is None:
            continue
        _add_feature(rows, f"morningstar_{metric}", "morningstar", value, f"summary:{metric}")

    return _dedupe_feature_rows(rows)


def _extract_performance_features(payload):
    payload = _safe_dict(payload)
    rows = []

    for section in ("cumulative", "annualized", "yield", "risk", "statistic"):
        for item in _safe_list(payload.get(section)):
            item = _safe_dict(item)
            metric = _metric_id(item)
            base = f"performance_{_slug(section)}_{metric}"

            value = _parse_number(item.get("value"))
            if value is not None:
                _add_feature(rows, base, "performance", value, f"{section}:{metric}:value")

            for field in ("vs", "percentile", "min", "max", "avg"):
                extra = _parse_number(item.get(field))
                if extra is None:
                    continue
                _add_feature(rows, f"{base}_{field}", "performance", extra, f"{section}:{metric}:{field}")

    return _dedupe_feature_rows(rows)


def _extract_ownership_features(payload):
    payload = _safe_dict(payload)
    rows = []

    snapshot = normalize_ownership_snapshot(payload)
    for key, value in snapshot.items():
        if key in {
            "owners_types_json",
            "institutional_total_value",
            "institutional_total_shares",
            "institutional_total_pct",
            "insider_total_value",
            "insider_total_shares",
            "insider_total_pct",
        }:
            continue
        num = _parse_number(value)
        if num is None:
            continue
        _add_feature(rows, f"ownership_{_slug(key)}", "ownership", num, f"snapshot:{key}")

    for item in _safe_list(payload.get("owners_types")):
        item = _safe_dict(item)
        type_info = _safe_dict(item.get("type"))
        label = type_info.get("type") or type_info.get("display_type")
        if not label:
            continue
        value = item.get("float")
        if value is None:
            value = item.get("display_float")
        frac = _to_fraction_weight(value)
        if frac is None:
            continue
        _add_feature(rows, f"ownership_type_{_slug(label)}", "ownership", frac, f"owners_types:{label}")

    return _dedupe_feature_rows(rows)


def _extract_esg_features(payload):
    payload = _safe_dict(payload)
    rows = []

    def walk(nodes, prefix):
        for node in _safe_list(nodes):
            node = _safe_dict(node)
            name = _slug(node.get("name"))
            path = f"{prefix}_{name}" if prefix else name
            value = _parse_number(node.get("value"))
            if value is not None:
                _add_feature(rows, f"esg_{path}", "esg", value, f"content:{path}")
            walk(node.get("children"), path)

    walk(payload.get("content"), "")

    coverage = _parse_number(payload.get("coverage"))
    if coverage is not None:
        _add_feature(rows, "esg_coverage", "esg", coverage, "coverage")

    if isinstance(payload.get("no_settings"), bool):
        _add_feature(rows, "esg_no_settings", "esg", 1.0 if payload["no_settings"] else 0.0, "no_settings")

    return _dedupe_feature_rows(rows)


_FACTOR_EXTRACTORS = {
    "landing": lambda payload: [],
    "profile_and_fees": _extract_profile_and_fees_features,
    "holdings": _extract_holdings_features,
    "ratios": _extract_ratios_features,
    "lipper_ratings": _extract_lipper_ratings_features,
    "dividends": _extract_dividends_features,
    "morningstar": _extract_morningstar_features,
    "performance": _extract_performance_features,
    "ownership": _extract_ownership_features,
    "esg": _extract_esg_features,
    # Time-series endpoints are handled in dedicated series tables in v1.
    "price_chart": lambda payload: [],
    "sentiment_search": lambda payload: [],
}


def extract_factor_features(endpoint, payload, effective_at=None):
    extractor = _FACTOR_EXTRACTORS.get(str(endpoint or ""))
    if extractor is None:
        return []
    rows = extractor(payload)
    if not rows:
        return []

    # Keep deterministic output ordering and stable shape.
    out = []
    for row in rows:
        out.append(
            {
                "feature_name": row["feature_name"],
                "feature_group": row["feature_group"],
                "feature_value": float(row["feature_value"]),
                "feature_source": row["feature_source"],
            }
        )
    out.sort(key=lambda r: (r["feature_group"], r["feature_name"]))
    return out


__all__ = [
    "extract_factor_features",
    "normalize_dividends_snapshot",
    "extract_dividends_events",
    "extract_dividends_industry_metrics",
    "normalize_ownership_snapshot",
    "extract_ownership_trade_log",
]
