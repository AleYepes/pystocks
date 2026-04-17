from __future__ import annotations

import math
import re
from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import cast

DateLike = date | datetime | int | float | str | Mapping[str, object] | None

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


def parse_date_candidate(value: DateLike) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, bool):
        return None

    if isinstance(value, Mapping):
        text_date = value.get("t")
        if text_date is not None:
            return parse_date_candidate(cast(DateLike, text_date))

        raw_year = value.get("y")
        raw_month = value.get("m")
        raw_day = value.get("d")
        if raw_year is None or raw_month is None or raw_day is None:
            return None
        if not isinstance(raw_year, (int, float, str)):
            return None
        if not isinstance(raw_month, (int, float, str)):
            return None
        if not isinstance(raw_day, (int, float, str)):
            return None
        try:
            if isinstance(raw_month, str):
                month = _MONTH_MAP.get(raw_month.strip().upper())
                if month is None:
                    month = int(raw_month)
            else:
                month = int(raw_month)
            return datetime(int(raw_year), int(month), int(raw_day), tzinfo=UTC).date()
        except Exception:
            return None

    if isinstance(value, (int, float)):
        timestamp = float(value)
        if not math.isfinite(timestamp) or timestamp <= 0:
            return None
        integer_value = int(timestamp)
        if 19000101 <= integer_value <= 29991231:
            try:
                return datetime.strptime(str(integer_value), "%Y%m%d").date()
            except Exception:
                pass
        if timestamp > 1e12:
            timestamp = timestamp / 1000.0
        try:
            return datetime.fromtimestamp(timestamp, tz=UTC).date()
        except Exception:
            return None

    text = str(value).strip()
    if not text:
        return None
    if len(text) == 8 and text.isdigit():
        try:
            return datetime.strptime(text, "%Y%m%d").date()
        except Exception:
            pass
    if text.isdigit():
        return parse_date_candidate(int(text))
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except Exception:
        pass
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).date()
        except Exception:
            continue
    return None


def parse_ymd_text(value: object) -> date | None:
    if value is None:
        return None
    match = _DATE_IN_TEXT_RE.search(str(value))
    if match is None:
        return None
    return parse_date_candidate(match.group(1))


def to_iso_date(value: DateLike) -> str | None:
    parsed = parse_date_candidate(value)
    return parsed.isoformat() if parsed is not None else None
