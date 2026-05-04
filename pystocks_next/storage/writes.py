from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd

from .dates import DateLike, parse_ymd_text, to_iso_date
from .raw_capture import JsonValue, capture_raw_payload, new_capture_batch_id
from .series_rows import UnresolvedSeriesRowDateError, resolve_series_row_date
from .time import resolve_effective_at


@dataclass(frozen=True, slots=True)
class PriceSeriesWriteResult:
    payload_hash: str
    capture_batch_id: str
    raw_observation_inserted: bool
    rows_upserted: int


@dataclass(frozen=True, slots=True)
class DividendEventsWriteResult:
    payload_hash: str
    capture_batch_id: str
    raw_observation_inserted: bool
    rows_inserted: int


@dataclass(frozen=True, slots=True)
class SupplementaryWriteResult:
    table_name: str
    rows_written: int


@dataclass(frozen=True, slots=True)
class SupplementaryFetchLogWriteResult:
    rows_written: int


@dataclass(frozen=True, slots=True)
class SnapshotWriteResult:
    payload_hash: str
    capture_batch_id: str
    raw_observation_inserted: bool
    effective_at: str


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

_RATIOS_SECTION_TABLES = {
    "ratios": "ratios_key_ratios",
    "financials": "ratios_financials",
    "fixed_income": "ratios_fixed_income",
    "dividend": "ratios_dividend",
    "zscore": "ratios_zscore",
}


def _parse_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


_NUM_WITH_SUFFIX_RE = re.compile(r"^\s*([-+]?\d*\.?\d+)\s*([KMBT]?)\s*%?\s*$")


def _parse_number(value: object, *, percent_as_fraction: bool = False) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None

    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"na", "n/a", "none", "null", "unknown", "nan", "inf", "-inf"}:
        return None

    is_percent = "%" in text
    plain = text.replace(",", "").replace("%", "").strip()
    try:
        parsed = float(plain)
    except ValueError:
        match = _NUM_WITH_SUFFIX_RE.match(text.replace(",", ""))
        if match is None:
            return None
        parsed = float(match.group(1))
        suffix = (match.group(2) or "").upper()
        parsed *= {"": 1.0, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[suffix]
    if is_percent and percent_as_fraction:
        parsed /= 100.0
    return parsed if math.isfinite(parsed) else None


def _parse_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _to_int_bool(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in {0, 1}:
        return int(value)
    return None


def _parse_percent_fraction(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    parsed = _parse_number(value, percent_as_fraction=True)
    if parsed is None:
        return None
    if "%" in str(value):
        return parsed
    return parsed


def _to_fraction_weight(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed / 100.0 if abs(parsed) > 1.0 else parsed
    text = str(value).strip()
    if not text:
        return None
    if "%" in text:
        return _parse_number(text, percent_as_fraction=True)
    parsed = _parse_number(text)
    if parsed is None:
        return None
    return parsed / 100.0 if abs(parsed) > 1.0 else parsed


def _parse_holdings_weight(item: Mapping[str, object]) -> float | None:
    raw_weight = item.get("weight")
    raw_formatted = item.get("formatted_weight")
    if raw_weight is not None:
        weight_percent = _parse_number(raw_weight)
        if weight_percent is None:
            return None
        if raw_formatted is not None:
            formatted_fraction = _to_fraction_weight(raw_formatted)
            if formatted_fraction is not None and round(weight_percent, 2) != round(
                formatted_fraction * 100.0, 2
            ):
                msg = (
                    "holdings weight mismatch: "
                    f"weight={raw_weight!r}, formatted_weight={raw_formatted!r}"
                )
                raise ValueError(msg)
        return weight_percent / 100.0

    fallback = _pick_first_present(item, "assets_pct", "formatted_weight")
    return _to_fraction_weight(fallback)


def _parse_holdings_vs_peers(item: Mapping[str, object]) -> float | None:
    raw_vs = item.get("vs")
    parsed = _parse_number(raw_vs)
    return parsed / 100.0 if parsed is not None else None


_TOTAL_NET_ASSETS_DATE_BLOCK_RE = re.compile(
    r"\(\s*\d{4}[/-]\d{2}[/-]\d{2}\s*[\)\.]?\s*"
)


def _split_total_net_assets_value(value: object) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    raw_text = str(value).strip()
    if not raw_text:
        return None, None
    parsed_date = parse_ymd_text(raw_text)
    date_iso = parsed_date.isoformat() if parsed_date is not None else None
    clean_text = _TOTAL_NET_ASSETS_DATE_BLOCK_RE.sub("", raw_text)
    clean_text = re.sub(r"(?<!\d)\d{4}[/-]\d{2}[/-]\d{2}(?!\d)", "", clean_text)
    clean_text = clean_text.replace("()", "").strip()
    clean_text = re.sub(r"\s+", " ", clean_text).strip(" .,\t\r\n()")
    return (clean_text or None, date_iso)


def _require_frame_columns(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    contract_name: str,
) -> pd.DataFrame:
    missing_columns = [column for column in columns if column not in frame.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"{contract_name} is missing required columns: {missing}")
    return frame.loc[:, list(columns)].copy()


def _extract_price_chart_rows(payload: JsonValue | bytes) -> list[dict[str, object]]:
    if isinstance(payload, bytes) or not isinstance(payload, dict):
        return []

    history = payload.get("history")
    if not isinstance(history, dict):
        return []
    series_list = history.get("series")
    if not isinstance(series_list, list):
        return []

    selected_series: dict[str, Any] | None = None
    for series in series_list:
        if not isinstance(series, dict):
            continue
        name = str(series.get("name") or series.get("title") or "").strip().lower()
        if "price" in name:
            selected_series = series
            break
    if selected_series is None:
        for series in series_list:
            if isinstance(series, dict):
                selected_series = series
                break
    if selected_series is None:
        return []

    points = selected_series.get("plotData")
    if not isinstance(points, list):
        return []

    rows: list[dict[str, object]] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        resolution = None
        try:
            resolution = resolve_series_row_date("price_chart_series", point)
        except UnresolvedSeriesRowDateError:
            pass

        row = {
            "effective_at": resolution.row_date.isoformat() if resolution else None,
            "price": point.get("y"),
            "open": point.get("open"),
            "high": point.get("high"),
            "low": point.get("low"),
            "close": point.get("close"),
            "debug_mismatch": int(
                point.get("x") is not None
                and point.get("debugY") is not None
                and resolution is not None
                and to_iso_date(point.get("x")) != to_iso_date(point.get("debugY"))
            ),
        }
        if any(
            row.get(column) is not None
            for column in ("effective_at", "price", "open", "high", "low", "close")
        ):
            rows.append(row)
    return rows


def _extract_dividend_event_rows(payload: JsonValue | bytes) -> list[dict[str, object]]:
    if isinstance(payload, bytes) or not isinstance(payload, dict):
        return []

    history = payload.get("history")
    if not isinstance(history, dict):
        return []
    series = history.get("series")
    if not isinstance(series, list):
        return []

    fallback_currency = payload.get("last_payed_dividend_currency")
    rows: list[dict[str, object]] = []

    for node in series:
        if not isinstance(node, dict):
            continue
        name = str(node.get("name") or node.get("title") or "").strip().lower()
        if "dividend" not in name:
            continue

        points = node.get("plotData")
        if not isinstance(points, list):
            continue
        for point in points:
            if not isinstance(point, dict):
                continue

            trade_date = to_iso_date(point.get("x")) or to_iso_date(
                point.get("ex_dividend_date")
            )
            event_date = to_iso_date(point.get("ex_dividend_date"))
            amount = point.get("amount")
            if amount is None:
                amount = point.get("y")
            parsed_amount = _parse_float(amount)

            currency = None
            formatted_amount = point.get("formatted_amount")
            if formatted_amount is not None:
                text = str(formatted_amount)
                parts = [
                    part for part in text.split() if len(part) == 3 and part.isupper()
                ]
                currency = parts[0] if parts else None

            resolution = None
            try:
                resolution = resolve_series_row_date(
                    "dividends_events_series",
                    {
                        "trade_date": trade_date,
                        "event_date": event_date,
                        "x": point.get("x"),
                        "ex_dividend_date": point.get("ex_dividend_date"),
                    },
                )
            except UnresolvedSeriesRowDateError:
                pass

            row = {
                "effective_at": resolution.row_date.isoformat() if resolution else None,
                "event_date": event_date,
                "amount": parsed_amount if parsed_amount is not None else amount,
                "currency": currency or fallback_currency,
                "description": point.get("description"),
                "event_type": point.get("type"),
                "declaration_date": to_iso_date(point.get("declaration_date")),
                "record_date": to_iso_date(point.get("record_date")),
                "payment_date": to_iso_date(point.get("payment_date")),
            }
            if any(value is not None for value in row.values()):
                rows.append(row)
    return rows


def _dividend_event_signature(conid: str, row: dict[str, object]) -> str:
    signature_text = "|".join(
        [
            conid,
            str(row.get("effective_at") or ""),
            str(_parse_float(row.get("amount"))),
            str(row.get("currency") or ""),
            str(row.get("description") or ""),
            str(row.get("event_type") or ""),
            str(row.get("event_date") or ""),
            str(row.get("declaration_date") or ""),
            str(row.get("record_date") or ""),
            str(row.get("payment_date") or ""),
        ]
    )
    return hashlib.sha256(signature_text.encode("utf-8")).hexdigest()


_PROFILE_FIELD_MAP: dict[str, tuple[str, str]] = {
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

_PROFILE_DATE_FIELD_IDS = {
    "inception_date",
    "manager_tenure",
    "maturity_date",
    "total_net_assets_date",
}

_PROFILE_NUMERIC_FIELD_IDS = {
    "morningstar_stylebox_x_index",
    "morningstar_stylebox_y_index",
    "redemption_charge_actual",
    "redemption_charge_max",
    "total_expense_ratio",
}

_PROFILE_BOOL_FIELD_IDS = {
    "jap_fund_warning",
}

_DIVIDENDS_METRIC_ID_ALIASES = {
    "div_yield": "dividend_yield_ttm",
    "dividend_yield_ttm": "dividend_yield_ttm",
    "dividend_yield": "dividend_yield_ttm",
    "div_per_share": "dividend_ttm",
    "dividend_ttm": "dividend_ttm",
    "annual_dividend": "dividend_ttm",
}


def _sanitize_segment(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_") or "field"


def _normalize_profile_themes(payload: Mapping[str, object]) -> str | None:
    themes = payload.get("themes", [])
    if not isinstance(themes, list):
        return None

    names: list[str] = []
    for item in themes:
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, dict):
            raw_name = item.get("theme_name") or item.get("name")
            name = str(raw_name).strip() if raw_name is not None else ""
        else:
            name = ""
        if name:
            names.append(name)
    return " | ".join(dict.fromkeys(names)) if names else None


def _extract_profile_stylebox_fields(
    payload: Mapping[str, object],
) -> dict[str, object]:
    mstar = payload.get("mstar")
    if not isinstance(mstar, dict):
        return {}
    hist = mstar.get("hist")
    if not isinstance(hist, list) or not hist:
        return {}
    first_pair = hist[0]
    if not isinstance(first_pair, list) or len(first_pair) < 2:
        return {}

    x_index = _parse_int(first_pair[0])
    y_index = _parse_int(first_pair[1])
    if x_index is None or y_index is None:
        return {}

    x_axis = mstar.get("x_axis")
    y_axis = mstar.get("y_axis")
    x_axis_tag = mstar.get("x_axis_tag")
    y_axis_tag = mstar.get("y_axis_tag")
    x_label = (
        str(x_axis[x_index])
        if isinstance(x_axis, list) and 0 <= x_index < len(x_axis)
        else None
    )
    y_label = (
        str(y_axis[y_index])
        if isinstance(y_axis, list) and 0 <= y_index < len(y_axis)
        else None
    )
    x_tag = (
        _sanitize_segment(x_axis_tag[x_index])
        if isinstance(x_axis_tag, list) and 0 <= x_index < len(x_axis_tag)
        else _sanitize_segment(x_label)
        if x_label is not None
        else None
    )
    y_tag = (
        _sanitize_segment(y_axis_tag[y_index])
        if isinstance(y_axis_tag, list) and 0 <= y_index < len(y_axis_tag)
        else _sanitize_segment(y_label)
        if y_label is not None
        else None
    )

    return {
        "morningstar_stylebox": f"{x_tag}_{y_tag}" if x_tag and y_tag else None,
        "morningstar_stylebox_x": x_label,
        "morningstar_stylebox_y": y_label,
        "morningstar_stylebox_x_index": x_index,
        "morningstar_stylebox_y_index": y_index,
    }


def _extract_profile_and_fees_row(
    conid: str,
    effective_at: str,
    payload: JsonValue | bytes,
) -> dict[str, object]:
    profile_row: dict[str, object] = {
        "conid": conid,
        "effective_at": effective_at,
        "asset_type": None,
        "classification": None,
        "distribution_details": None,
        "domicile": None,
        "fiscal_date": None,
        "fund_category": None,
        "fund_management_company": None,
        "fund_manager_benchmark": None,
        "fund_market_cap_focus": None,
        "geographical_focus": None,
        "inception_date": None,
        "management_approach": None,
        "management_expenses": None,
        "manager_tenure": None,
        "maturity_date": None,
        "objective_type": None,
        "portfolio_manager": None,
        "redemption_charge_actual": None,
        "redemption_charge_max": None,
        "scheme": None,
        "total_expense_ratio": None,
        "total_net_assets_value": None,
        "total_net_assets_date": None,
        "objective": None,
        "jap_fund_warning": None,
        "theme_name": None,
        "morningstar_stylebox": None,
        "morningstar_stylebox_x": None,
        "morningstar_stylebox_y": None,
        "morningstar_stylebox_x_index": None,
        "morningstar_stylebox_y_index": None,
    }
    if isinstance(payload, bytes) or not isinstance(payload, dict):
        return profile_row

    profile_row["objective"] = (
        str(payload.get("objective")) if payload.get("objective") is not None else None
    )
    profile_row["jap_fund_warning"] = _to_int_bool(payload.get("jap_fund_warning"))
    profile_row["theme_name"] = _normalize_profile_themes(payload)
    profile_row.update(_extract_profile_stylebox_fields(payload))

    fund_and_profile = payload.get("fund_and_profile", [])
    if not isinstance(fund_and_profile, list):
        return profile_row

    for item in fund_and_profile:
        if not isinstance(item, dict):
            continue
        field_name = item.get("name")
        if field_name not in _PROFILE_FIELD_MAP:
            continue
        raw_value = item.get("value")
        column_name, value_type = _PROFILE_FIELD_MAP[field_name]
        if field_name == "Total Net Assets (Month End)":
            net_assets_value, net_assets_date = _split_total_net_assets_value(raw_value)
            profile_row["total_net_assets_value"] = net_assets_value
            profile_row["total_net_assets_date"] = net_assets_date
            continue
        if value_type == "text":
            profile_row[column_name] = str(raw_value) if raw_value is not None else None
        elif value_type == "percent":
            profile_row[column_name] = _parse_percent_fraction(raw_value)
        elif value_type == "date":
            parsed_date = to_iso_date(raw_value)
            if field_name == "Launch Opening Price":
                if profile_row["inception_date"] is None and parsed_date is not None:
                    profile_row["inception_date"] = parsed_date
            else:
                profile_row[column_name] = parsed_date

    return profile_row


def _snapshot_factor_row(
    *,
    conid: str,
    effective_at: str,
    factor_id: str,
    value_text: str | None = None,
    value_num: float | int | None = None,
    value_date: str | None = None,
    value_bool: int | None = None,
) -> dict[str, object]:
    return {
        "conid": conid,
        "effective_at": effective_at,
        "factor_id": factor_id,
        "value_text": value_text,
        "value_num": value_num,
        "value_date": value_date,
        "value_bool": value_bool,
    }


def _extract_profile_report_factor_rows(
    conid: str,
    effective_at: str,
    payload: JsonValue | bytes,
) -> list[dict[str, object]]:
    if isinstance(payload, bytes) or not isinstance(payload, dict):
        return []
    reports = payload.get("reports", [])
    if not isinstance(reports, list):
        return []

    rows: list[dict[str, object]] = []
    for report in reports:
        if not isinstance(report, dict):
            continue
        report_id = _sanitize_segment(report.get("name") or "report")
        report_as_of_date = to_iso_date(
            report.get("as_of_date") or report.get("asOfDate")
        )
        if report_as_of_date is not None:
            rows.append(
                _snapshot_factor_row(
                    conid=conid,
                    effective_at=effective_at,
                    factor_id=f"report_{report_id}_as_of_date",
                    value_date=report_as_of_date,
                )
            )

        fields = report.get("fields", [])
        if not isinstance(fields, list):
            continue
        for field in fields:
            if not isinstance(field, dict):
                continue
            name = field.get("name")
            if name is None:
                continue
            value = field.get("value")
            if value is None:
                continue
            factor_id = f"report_{report_id}_{_sanitize_segment(name)}"
            numeric_value = _parse_number(value, percent_as_fraction=True)
            if numeric_value is not None:
                rows.append(
                    _snapshot_factor_row(
                        conid=conid,
                        effective_at=effective_at,
                        factor_id=factor_id,
                        value_num=numeric_value,
                    )
                )
            else:
                rows.append(
                    _snapshot_factor_row(
                        conid=conid,
                        effective_at=effective_at,
                        factor_id=factor_id,
                        value_text=str(value),
                    )
                )
    return rows


def _extract_profile_and_fees_factor_rows(
    conid: str,
    effective_at: str,
    payload: JsonValue | bytes,
) -> list[dict[str, object]]:
    profile_row = _extract_profile_and_fees_row(conid, effective_at, payload)
    rows: list[dict[str, object]] = []
    for field_id, value in profile_row.items():
        if field_id in {"conid", "effective_at"} or value is None:
            continue
        if field_id in _PROFILE_DATE_FIELD_IDS:
            rows.append(
                _snapshot_factor_row(
                    conid=conid,
                    effective_at=effective_at,
                    factor_id=field_id,
                    value_date=str(value),
                )
            )
        elif field_id in _PROFILE_NUMERIC_FIELD_IDS:
            numeric_value = _parse_float(value)
            if numeric_value is None:
                continue
            rows.append(
                _snapshot_factor_row(
                    conid=conid,
                    effective_at=effective_at,
                    factor_id=field_id,
                    value_num=numeric_value,
                )
            )
        elif field_id in _PROFILE_BOOL_FIELD_IDS:
            bool_value = _to_int_bool(value)
            if bool_value is None:
                continue
            rows.append(
                _snapshot_factor_row(
                    conid=conid,
                    effective_at=effective_at,
                    factor_id=field_id,
                    value_bool=bool_value,
                )
            )
        else:
            rows.append(
                _snapshot_factor_row(
                    conid=conid,
                    effective_at=effective_at,
                    factor_id=field_id,
                    value_text=str(value),
                )
            )
    rows.extend(_extract_profile_report_factor_rows(conid, effective_at, payload))
    return rows


def _typed_profile_value(
    value: object,
    *,
    value_type: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "value_text": str(value).strip() if value is not None else None,
        "value_num": None,
        "value_date": None,
        "value_bool": _to_int_bool(value),
    }
    if value_type == "date":
        row["value_date"] = _profile_date_value(value)
    elif value_type == "percent":
        row["value_num"] = _parse_percent_fraction(value)
    else:
        row["value_date"] = _profile_date_value(value)
        row["value_num"] = _parse_number(value, percent_as_fraction=True)
    return row


def _profile_date_value(value: object) -> str | None:
    if isinstance(value, date | datetime | int | float | str | Mapping):
        return to_iso_date(value)
    return None


def _source_ordered_id(base_id: str, seen: dict[str, int]) -> str:
    count = seen.get(base_id, 0) + 1
    seen[base_id] = count
    return base_id if count == 1 else f"{base_id}_{count}"


def _resolve_profile_source_as_of_date(payload: JsonValue | bytes) -> str | None:
    if isinstance(payload, bytes) or not isinstance(payload, dict):
        return None

    explicit_as_of_date = to_iso_date(
        payload.get("as_of_date") or payload.get("asOfDate")
    )
    if explicit_as_of_date is not None:
        return explicit_as_of_date

    fund_and_profile = payload.get("fund_and_profile", [])
    if isinstance(fund_and_profile, list):
        for item in fund_and_profile:
            if not isinstance(item, dict):
                continue
            field_name = str(item.get("name") or item.get("name_tag") or "")
            field_id = _sanitize_segment(field_name)
            if field_id in {
                "total_net_assets_month_end",
                "total_net_assets_value",
            }:
                _, net_assets_date = _split_total_net_assets_value(item.get("value"))
                if net_assets_date is not None:
                    return net_assets_date

    report_dates: list[str] = []
    reports = payload.get("reports", [])
    if isinstance(reports, list):
        for report in reports:
            if not isinstance(report, dict):
                continue
            report_as_of_date = to_iso_date(
                report.get("as_of_date") or report.get("asOfDate")
            )
            if report_as_of_date is not None:
                report_dates.append(report_as_of_date)
    return max(report_dates) if report_dates else None


def _extract_profile_overview_row(
    conid: str,
    effective_at: str,
    payload: JsonValue | bytes,
) -> dict[str, object]:
    row: dict[str, object] = {
        "conid": conid,
        "effective_at": effective_at,
        "symbol": None,
        "objective": None,
        "jap_fund_warning": None,
        "management_expenses_ratio": None,
    }
    if isinstance(payload, bytes) or not isinstance(payload, dict):
        return row
    row["symbol"] = str(payload.get("symbol")) if payload.get("symbol") else None
    row["objective"] = (
        str(payload.get("objective")) if payload.get("objective") is not None else None
    )
    row["jap_fund_warning"] = _to_int_bool(payload.get("jap_fund_warning"))

    management_ratio: float | None = None
    non_management_ratio: float | None = None
    expenses_allocation = payload.get("expenses_allocation", [])
    if isinstance(expenses_allocation, list):
        for item in expenses_allocation:
            if not isinstance(item, dict):
                continue
            expense_id = _sanitize_segment(item.get("name"))
            ratio = _parse_float(item.get("ratio"))
            if ratio is None:
                ratio = _to_fraction_weight(item.get("value"))
            if expense_id == "management_expenses":
                management_ratio = ratio
            elif expense_id == "non_management_expenses":
                non_management_ratio = ratio

    if management_ratio is None and non_management_ratio is not None:
        management_ratio = 1.0 - non_management_ratio
    row["management_expenses_ratio"] = management_ratio

    return row


def _extract_profile_field_rows(
    conid: str,
    effective_at: str,
    payload: JsonValue | bytes,
) -> list[dict[str, object]]:
    if isinstance(payload, bytes) or not isinstance(payload, dict):
        return []
    values = payload.get("fund_and_profile", [])
    if not isinstance(values, list):
        return []

    rows: list[dict[str, object]] = []
    seen: dict[str, int] = {}
    for source_order, item in enumerate(values):
        if not isinstance(item, dict):
            continue
        field_name = item.get("name")
        name_tag = item.get("name_tag")
        base_id = _sanitize_segment(name_tag or field_name)
        if base_id == "management_expenses":
            continue
        field_id = _source_ordered_id(base_id, seen)
        raw_value = item.get("value")
        mapped = _PROFILE_FIELD_MAP.get(str(field_name))
        value_type = mapped[1] if mapped is not None else None
        typed = _typed_profile_value(raw_value, value_type=value_type)

        if str(field_name) == "Total Net Assets (Month End)":
            clean_value, date_value = _split_total_net_assets_value(raw_value)
            typed["value_text"] = clean_value
            typed["value_date"] = date_value
            typed["value_num"] = _parse_number(clean_value)
        elif str(field_name) == "Launch Opening Price":
            typed["value_date"] = _profile_date_value(raw_value)

        rows.append(
            {
                "conid": conid,
                "effective_at": effective_at,
                "field_id": field_id,
                **typed,
            }
        )
    return rows


def _extract_profile_report_rows(
    conid: str,
    payload: JsonValue | bytes,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if isinstance(payload, bytes) or not isinstance(payload, dict):
        return [], []
    reports = payload.get("reports", [])
    if not isinstance(reports, list):
        return [], []

    annual_rows: list[dict[str, object]] = []
    prospectus_rows: list[dict[str, object]] = []

    for report in reports:
        if not isinstance(report, dict):
            continue
        report_name = str(report.get("name") or "").strip().lower()
        report_as_of_date = to_iso_date(
            report.get("as_of_date") or report.get("asOfDate")
        )
        if not report_as_of_date:
            continue

        fields = report.get("fields", [])
        if not isinstance(fields, list) or not fields:
            continue

        target_list = None
        if "annual" in report_name:
            target_list = annual_rows
        elif "prospectus" in report_name:
            target_list = prospectus_rows

        if target_list is None:
            continue

        seen_fields: dict[str, int] = {}
        for field in fields:
            if not isinstance(field, dict):
                continue
            field_name = field.get("name")
            if field_name is None:
                continue
            field_id = _source_ordered_id(_sanitize_segment(field_name), seen_fields)
            typed = _typed_profile_value(field.get("value"))
            target_list.append(
                {
                    "conid": conid,
                    "effective_at": report_as_of_date,
                    "field_id": field_id,
                    "is_summary": _to_int_bool(field.get("is_summary")),
                    "value_num": typed.get("value_num"),
                }
            )
    return annual_rows, prospectus_rows


def _extract_profile_theme_rows(
    conid: str,
    effective_at: str,
    payload: JsonValue | bytes,
) -> list[dict[str, object]]:
    if isinstance(payload, bytes) or not isinstance(payload, dict):
        return []
    themes = payload.get("themes", [])
    if not isinstance(themes, list):
        return []

    rows: list[dict[str, object]] = []
    seen: dict[str, int] = {}
    for item in themes:
        if isinstance(item, str):
            theme_name = item.strip()
        elif isinstance(item, dict):
            raw_name = item.get("theme_name") or item.get("name") or item.get("title")
            theme_name = str(raw_name).strip() if raw_name is not None else ""
        else:
            theme_name = ""
        if not theme_name:
            continue
        theme_id = _source_ordered_id(_sanitize_segment(theme_name), seen)
        rows.append(
            {
                "conid": conid,
                "effective_at": effective_at,
                "theme_id": theme_id,
            }
        )
    return rows


def _extract_profile_stylebox_rows(
    conid: str,
    effective_at: str,
    payload: JsonValue | bytes,
) -> list[dict[str, object]]:
    if isinstance(payload, bytes) or not isinstance(payload, dict):
        return []
    fields = _extract_profile_stylebox_fields(payload)
    stylebox_id = fields.get("morningstar_stylebox")
    if stylebox_id is None:
        return []
    return [
        {
            "conid": conid,
            "effective_at": effective_at,
            "stylebox_id": str(stylebox_id),
            "x_index": _parse_int(fields.get("morningstar_stylebox_x_index")),
            "y_index": _parse_int(fields.get("morningstar_stylebox_y_index")),
            "x_label": fields.get("morningstar_stylebox_x"),
            "y_label": fields.get("morningstar_stylebox_y"),
            "x_tag": str(stylebox_id).split("_", 1)[0],
            "y_tag": str(stylebox_id).split("_", 1)[1]
            if "_" in str(stylebox_id)
            else None,
        }
    ]


_HOLDINGS_ASSET_TYPE_MAP = {
    "equity": "equity",
    "cash": "cash",
    "fixed_income": "fixed_income",
    "other": "other",
}


def _pick_first_present(mapping: Mapping[str, object], *keys: str) -> object:
    for key in keys:
        if key in mapping and mapping.get(key) is not None:
            return mapping.get(key)
    return None


def _delete_snapshot_child_rows(
    conn: sqlite3.Connection,
    *,
    tables: tuple[str, ...],
    conid: str,
    effective_at: str,
) -> None:
    for table_name in tables:
        conn.execute(
            f"DELETE FROM {table_name} WHERE conid = ? AND effective_at = ?",
            (conid, effective_at),
        )


def _extract_holdings_asset_type_row(
    conid: str,
    effective_at: str,
    payload: JsonValue | bytes,
) -> dict[str, object]:
    row: dict[str, object] = {
        "conid": conid,
        "effective_at": effective_at,
        "equity": None,
        "cash": None,
        "fixed_income": None,
        "other": None,
    }
    if isinstance(payload, bytes) or not isinstance(payload, dict):
        return row

    allocation_self = payload.get("allocation_self", [])
    if not isinstance(allocation_self, list):
        return row

    for item in allocation_self:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("type")
        if name is None:
            continue
        key = _sanitize_segment(name)
        column = _HOLDINGS_ASSET_TYPE_MAP.get(key, "other")
        parsed_weight = _parse_holdings_weight(item)
        if parsed_weight is None:
            continue
        parsed_vs = _parse_holdings_vs_peers(item)
        current_value = row[column]
        row[column] = (
            parsed_weight
            if current_value is None
            else (
                float(current_value) + parsed_weight
                if isinstance(current_value, (int, float))
                else parsed_weight
            )
        )
        current_vs = row.get(f"{column}_vs_peers")
        row[f"{column}_vs_peers"] = (
            parsed_vs
            if current_vs is None
            else (
                float(current_vs) + parsed_vs
                if parsed_vs is not None and isinstance(current_vs, (int, float))
                else current_vs
            )
        )
    return row


def _extract_holdings_bucket_rows(
    conid: str,
    effective_at: str,
    payload: JsonValue | bytes,
    *,
    section: str,
    name_column: str,
    extra_column: str | None = None,
    normalize_name: bool = False,
) -> list[dict[str, object]]:
    if isinstance(payload, bytes) or not isinstance(payload, dict):
        return []
    values = payload.get(section, [])
    if not isinstance(values, list):
        return []

    rows: list[dict[str, object]] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("type")
        if name is None:
            continue
        row: dict[str, object] = {
            "conid": conid,
            "effective_at": effective_at,
            name_column: _sanitize_segment(name) if normalize_name else str(name),
            "value_num": _parse_holdings_weight(item),
            "vs_peers": _parse_holdings_vs_peers(item),
        }
        if extra_column is not None:
            extra_value = item.get(extra_column)
            row[extra_column] = str(extra_value) if extra_value is not None else None
        rows.append(row)
    return rows


def _extract_holdings_debtor_quality_row(
    conid: str,
    effective_at: str,
    payload: JsonValue | bytes,
) -> dict[str, object] | None:
    row: dict[str, object] = {
        "conid": conid,
        "effective_at": effective_at,
        **{
            column: None
            for column in _HOLDINGS_DEBTOR_QUALITY_SOURCE_TO_COLUMN.values()
        },
    }
    if isinstance(payload, bytes) or not isinstance(payload, dict):
        return None

    values = payload.get("debtor", [])
    if not isinstance(values, list):
        return None
    for item in values:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("type")
        if name is None:
            continue
        column = _HOLDINGS_DEBTOR_QUALITY_SOURCE_TO_COLUMN.get(_sanitize_segment(name))
        if column is None:
            continue
        row[column] = _parse_holdings_weight(item)
        row[f"{column}_vs_peers"] = _parse_holdings_vs_peers(item)

    has_values = any(
        row[column] is not None
        for column in _HOLDINGS_DEBTOR_QUALITY_SOURCE_TO_COLUMN.values()
    )
    return row if has_values else None


def _extract_holdings_maturity_row(
    conid: str,
    effective_at: str,
    payload: JsonValue | bytes,
) -> dict[str, object] | None:
    row: dict[str, object] = {
        "conid": conid,
        "effective_at": effective_at,
        **{
            column: None for column in set(_HOLDINGS_MATURITY_SOURCE_TO_COLUMN.values())
        },
    }
    if isinstance(payload, bytes) or not isinstance(payload, dict):
        return None

    values = payload.get("maturity", [])
    if not isinstance(values, list):
        return None
    for item in values:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("type")
        if name is None:
            continue
        column = _HOLDINGS_MATURITY_SOURCE_TO_COLUMN.get(_sanitize_segment(name))
        if column is None:
            continue
        row[column] = _parse_holdings_weight(item)
        row[f"{column}_vs_peers"] = _parse_holdings_vs_peers(item)

    maturity_columns = tuple(
        {
            "maturity_less_than_1_year",
            "maturity_1_to_3_years",
            "maturity_3_to_5_years",
            "maturity_5_to_10_years",
            "maturity_10_to_20_years",
            "maturity_20_to_30_years",
            "maturity_greater_than_30_years",
            "maturity_other",
        }
    )
    has_values = any(row[column] is not None for column in maturity_columns)
    return row if has_values else None


def _extract_holdings_geographic_rows(
    conid: str,
    effective_at: str,
    payload: JsonValue | bytes,
) -> list[dict[str, object]]:
    if isinstance(payload, bytes) or not isinstance(payload, dict):
        return []
    geographic = payload.get("geographic")
    if not isinstance(geographic, dict):
        return []

    rows: list[dict[str, object]] = []
    for key, value in geographic.items():
        if isinstance(value, (dict, list)):
            continue
        rows.append(
            {
                "conid": conid,
                "effective_at": effective_at,
                "region_id": _sanitize_segment(key),
                "value_num": _to_fraction_weight(value),
                "vs_peers": None,
            }
        )
    return rows


def _extract_holdings_top10_rows(
    conid: str,
    effective_at: str,
    payload: JsonValue | bytes,
) -> list[dict[str, object]]:
    if isinstance(payload, bytes) or not isinstance(payload, dict):
        return []
    values = payload.get("top_10", [])
    if not isinstance(values, list):
        return []

    rows: list[dict[str, object]] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if name is None:
            continue
        rows.append(
            {
                "conid": conid,
                "effective_at": effective_at,
                "name": str(name),
                "ticker": str(item["ticker"]).strip() if item.get("ticker") else None,
                "rank": _parse_int(item.get("rank")),
                "holding_weight_num": _to_fraction_weight(item.get("assets_pct")),
                "vs_peers": _parse_holdings_vs_peers(item),
                "conids_json": json.dumps(item.get("conids"))
                if isinstance(item.get("conids"), list)
                else None,
            }
        )
    return rows


def _wide_bucket_row_to_factor_rows(
    row: dict[str, object] | None,
    *,
    bucket_keys: tuple[str, ...],
) -> list[dict[str, object]]:
    if row is None:
        return []
    conid = str(row["conid"])
    effective_at = str(row["effective_at"])
    rows: list[dict[str, object]] = []
    for bucket_id in bucket_keys:
        value = row.get(bucket_id)
        if value is None:
            continue
        numeric_value = _parse_float(value)
        if numeric_value is None:
            continue
        rows.append(
            {
                "conid": conid,
                "effective_at": effective_at,
                "bucket_id": bucket_id,
                "value_num": numeric_value,
                "vs_peers": _parse_float(row.get(f"{bucket_id}_vs_peers")),
            }
        )
    return rows


def _extract_ratios_metric_rows(
    conid: str,
    effective_at: str,
    payload: JsonValue | bytes,
    *,
    section: str,
) -> list[dict[str, object]]:
    if isinstance(payload, bytes) or not isinstance(payload, dict):
        return []
    values = payload.get(section, [])
    if not isinstance(values, list):
        return []

    rows: list[dict[str, object]] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        metric_id = _sanitize_segment(
            item.get("name_tag") or item.get("id") or item.get("name")
        )
        rows.append(
            {
                "conid": conid,
                "effective_at": effective_at,
                "metric_id": metric_id,
                "value_num": _parse_number(item.get("value")),
                "vs_peers": _parse_number(item.get("vs")),
            }
        )
    return rows


def _extract_dividends_industry_metric_rows(
    conid: str,
    effective_at: str,
    payload: JsonValue | bytes,
) -> list[dict[str, object]]:
    if isinstance(payload, bytes) or not isinstance(payload, dict):
        return []

    industry_average = payload.get("industry_average")
    industry_average = industry_average if isinstance(industry_average, dict) else {}
    industry_comparison = payload.get("industry_comparison")
    industry_comparison = (
        industry_comparison if isinstance(industry_comparison, dict) else {}
    )
    comparison_content = industry_comparison.get("content")
    comparison_content = (
        comparison_content if isinstance(comparison_content, list) else []
    )

    currency = payload.get("last_payed_dividend_currency")
    metric_values: dict[str, float | None] = {
        "dividend_yield": _parse_number(
            industry_average.get("dividend_yield"), percent_as_fraction=True
        ),
        "annual_dividend": _parse_number(industry_average.get("annual_dividend")),
        "dividend_ttm": None,
        "dividend_yield_ttm": None,
    }
    for item in comparison_content:
        if not isinstance(item, dict):
            continue
        metric_id = _sanitize_segment(
            item.get("search_id") or item.get("name_tag") or item.get("name")
        )
        canonical_metric_id = _DIVIDENDS_METRIC_ID_ALIASES.get(metric_id)
        if canonical_metric_id is None:
            continue
        value = item.get("value")
        metric_values[canonical_metric_id] = _parse_number(
            value,
            percent_as_fraction=canonical_metric_id == "dividend_yield_ttm",
        )

    if metric_values["dividend_ttm"] is None:
        metric_values["dividend_ttm"] = metric_values["annual_dividend"]
    if metric_values["dividend_yield_ttm"] is None:
        metric_values["dividend_yield_ttm"] = metric_values["dividend_yield"]

    rows: list[dict[str, object]] = []
    for metric_id, value in metric_values.items():
        if value is None:
            continue
        rows.append(
            {
                "conid": conid,
                "effective_at": effective_at,
                "metric_id": metric_id,
                "value_num": value,
                "currency": str(currency) if currency is not None else None,
            }
        )
    return rows


def _normalize_morningstar_summary_metric_id(item: Mapping[str, object]) -> str:
    metric_id = _sanitize_segment(item.get("id") or item.get("title") or "metric")
    derived_quantitatively = item.get("q") is True
    if derived_quantitatively and metric_id.startswith("q_"):
        return metric_id[2:]
    return metric_id


def _extract_morningstar_summary_rows(
    conid: str,
    effective_at: str,
    payload: JsonValue | bytes,
) -> list[dict[str, object]]:
    if isinstance(payload, bytes) or not isinstance(payload, dict):
        return []
    summary = payload.get("summary")
    if not isinstance(summary, list):
        return []

    rows: list[dict[str, object]] = []
    for item in summary:
        if not isinstance(item, dict):
            continue
        metric_id = _normalize_morningstar_summary_metric_id(item)
        value = item.get("value")
        rows.append(
            {
                "conid": conid,
                "effective_at": effective_at,
                "metric_id": metric_id,
                "title": str(item["title"]) if item.get("title") is not None else None,
                "derived_quantitatively": 1 if item.get("q") is True else 0,
                "publish_date": to_iso_date(
                    item.get("publish_date") or item.get("publishDate")
                ),
                "value_text": None
                if metric_id == "morningstar_rating"
                else (str(value) if value is not None else None),
                "value_num": _parse_number(value)
                if metric_id == "morningstar_rating"
                else None,
            }
        )
    return rows


def _extract_lipper_rating_rows(
    conid: str,
    effective_at: str,
    payload: JsonValue | bytes,
) -> list[dict[str, object]]:
    if isinstance(payload, bytes) or not isinstance(payload, dict):
        return []
    universes = payload.get("universes")
    if not isinstance(universes, list):
        return []

    rows: list[dict[str, object]] = []
    for universe in universes:
        if not isinstance(universe, dict):
            continue
        universe_name = (
            str(universe.get("name")) if universe.get("name") is not None else None
        )
        universe_as_of_date = to_iso_date(
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
                rating = item.get("rating")
                rating = rating if isinstance(rating, dict) else {}
                rows.append(
                    {
                        "conid": conid,
                        "effective_at": effective_at,
                        "period": period,
                        "metric_id": _sanitize_segment(
                            item.get("name_tag") or item.get("id") or item.get("name")
                        ),
                        "value_num": _parse_number(rating.get("value")),
                        "rating_label": rating.get("name"),
                        "universe_name": universe_name,
                        "universe_as_of_date": universe_as_of_date,
                    }
                )
    return rows


def _resolve_lipper_source_as_of_date(payload: JsonValue | bytes) -> DateLike:
    if isinstance(payload, bytes) or not isinstance(payload, dict):
        return None

    root_value = payload.get("as_of_date") or payload.get("asOfDate")
    if to_iso_date(root_value) is not None:
        return root_value

    universes = payload.get("universes")
    if not isinstance(universes, list):
        return None

    parsed_dates: dict[str, DateLike] = {}
    for universe in universes:
        if not isinstance(universe, dict):
            continue
        raw_value = universe.get("as_of_date") or universe.get("asOfDate")
        parsed = to_iso_date(raw_value)
        if parsed is None:
            continue
        parsed_dates.setdefault(parsed, raw_value)

    if not parsed_dates:
        return None
    return parsed_dates[max(parsed_dates)]


def _resolve_morningstar_source_as_of_date(payload: JsonValue | bytes) -> DateLike:
    if isinstance(payload, bytes) or not isinstance(payload, dict):
        return None

    root_value = payload.get("as_of_date") or payload.get("asOfDate")
    if to_iso_date(root_value) is not None:
        return root_value

    parsed_dates: dict[str, DateLike] = {}
    for section_name in ("summary", "commentary"):
        section = payload.get(section_name)
        if not isinstance(section, list):
            continue
        for item in section:
            if not isinstance(item, dict):
                continue
            raw_value = item.get("publish_date") or item.get("publishDate")
            parsed = to_iso_date(raw_value)
            if parsed is None:
                continue
            parsed_dates.setdefault(parsed, raw_value)

    if not parsed_dates:
        return None
    return parsed_dates[max(parsed_dates)]


def write_price_chart_series(
    conn: sqlite3.Connection,
    *,
    conid: str,
    payload: JsonValue | bytes,
    observed_at: str,
    source_family: str = "ibkr",
    capture_batch_id: str | None = None,
) -> PriceSeriesWriteResult:
    batch_id = capture_batch_id or new_capture_batch_id()
    capture = capture_raw_payload(
        conn,
        source_family=source_family,
        endpoint="price_chart",
        payload=payload,
        observed_at=observed_at,
        conid=conid,
        capture_batch_id=batch_id,
    )

    rows_upserted = 0
    for row in _extract_price_chart_rows(payload):
        effective_at = row.get("effective_at")
        if effective_at is None:
            continue
        conn.execute(
            """
            INSERT INTO price_chart_series (
                conid,
                effective_at,
                observed_at,
                payload_hash,
                capture_batch_id,
                price,
                open,
                high,
                low,
                close,
                debug_mismatch
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(conid, effective_at) DO UPDATE SET
                observed_at = excluded.observed_at,
                payload_hash = excluded.payload_hash,
                capture_batch_id = excluded.capture_batch_id,
                price = excluded.price,
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                debug_mismatch = excluded.debug_mismatch
            """,
            (
                conid,
                str(effective_at),
                observed_at,
                capture.payload_hash,
                batch_id,
                _parse_float(row.get("price")),
                _parse_float(row.get("open")),
                _parse_float(row.get("high")),
                _parse_float(row.get("low")),
                _parse_float(row.get("close")),
                _parse_int(row.get("debug_mismatch")) or 0,
            ),
        )
        rows_upserted += 1

    return PriceSeriesWriteResult(
        payload_hash=capture.payload_hash,
        capture_batch_id=batch_id,
        raw_observation_inserted=capture.observation_inserted,
        rows_upserted=rows_upserted,
    )


def write_dividend_events_series(
    conn: sqlite3.Connection,
    *,
    conid: str,
    payload: JsonValue | bytes,
    observed_at: str,
    source_family: str = "ibkr",
    source_as_of_date: DateLike = None,
    capture_batch_id: str | None = None,
) -> DividendEventsWriteResult:
    batch_id = capture_batch_id or new_capture_batch_id()
    capture = capture_raw_payload(
        conn,
        source_family=source_family,
        endpoint="dividends",
        payload=payload,
        observed_at=observed_at,
        conid=conid,
        source_as_of_date=source_as_of_date,
        capture_batch_id=batch_id,
    )

    rows_inserted = 0
    for row in _extract_dividend_event_rows(payload):
        effective_at = row.get("effective_at")
        if effective_at is None:
            continue
        signature = _dividend_event_signature(conid, row)
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO dividends_events_series (
                conid,
                event_signature,
                effective_at,
                observed_at,
                payload_hash,
                capture_batch_id,
                amount,
                currency,
                description,
                event_type,
                declaration_date,
                record_date,
                payment_date,
                event_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conid,
                signature,
                str(effective_at),
                observed_at,
                capture.payload_hash,
                batch_id,
                _parse_float(row.get("amount")),
                row.get("currency"),
                row.get("description"),
                row.get("event_type"),
                row.get("declaration_date"),
                row.get("record_date"),
                row.get("payment_date"),
                row.get("event_date"),
            ),
        )
        if cursor.rowcount > 0:
            rows_inserted += 1

    return DividendEventsWriteResult(
        payload_hash=capture.payload_hash,
        capture_batch_id=batch_id,
        raw_observation_inserted=capture.observation_inserted,
        rows_inserted=rows_inserted,
    )


def write_profile_and_fees_snapshot(
    conn: sqlite3.Connection,
    *,
    conid: str,
    payload: JsonValue | bytes,
    observed_at: str,
    source_family: str = "ibkr",
    capture_batch_id: str | None = None,
) -> SnapshotWriteResult:
    batch_id = capture_batch_id or new_capture_batch_id()
    source_as_of_date = _resolve_profile_source_as_of_date(payload)
    resolution = resolve_effective_at(
        "profile_and_fees_snapshot",
        observed_at=observed_at,
        source_as_of_date=source_as_of_date,
    )
    effective_at = resolution.effective_at.isoformat()
    capture = capture_raw_payload(
        conn,
        source_family=source_family,
        endpoint="profile_and_fees",
        payload=payload,
        observed_at=observed_at,
        conid=conid,
        source_as_of_date=source_as_of_date,
        capture_batch_id=batch_id,
    )

    overview_row = _extract_profile_overview_row(conid, effective_at, payload)
    field_rows = _extract_profile_field_rows(conid, effective_at, payload)
    annual_rows, prospectus_rows = _extract_profile_report_rows(conid, payload)
    theme_rows = _extract_profile_theme_rows(conid, effective_at, payload)
    stylebox_rows = _extract_profile_stylebox_rows(conid, effective_at, payload)
    conn.execute(
        """
        INSERT INTO profile_snapshots (
            conid,
            effective_at,
            observed_at,
            payload_hash,
            capture_batch_id,
            source_as_of_date
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(conid, effective_at) DO UPDATE SET
            observed_at = excluded.observed_at,
            payload_hash = excluded.payload_hash,
            capture_batch_id = excluded.capture_batch_id,
            source_as_of_date = excluded.source_as_of_date
        """,
        (
            conid,
            effective_at,
            observed_at,
            capture.payload_hash,
            batch_id,
            source_as_of_date,
        ),
    )
    _delete_snapshot_child_rows(
        conn,
        tables=(
            "profile_overview",
            "profile_fields",
            "profile_themes",
            "profile_stylebox",
        ),
        conid=conid,
        effective_at=effective_at,
    )
    conn.execute(
        """
        INSERT INTO profile_overview (
            conid,
            effective_at,
            symbol,
            objective,
            jap_fund_warning,
            management_expenses_ratio
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            overview_row["conid"],
            overview_row["effective_at"],
            overview_row["symbol"],
            overview_row["objective"],
            overview_row["jap_fund_warning"],
            overview_row["management_expenses_ratio"],
        ),
    )
    for row in field_rows:
        conn.execute(
            """
            INSERT INTO profile_fields (
                conid,
                effective_at,
                field_id,
                value_text,
                value_num,
                value_date
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row["conid"],
                row["effective_at"],
                row["field_id"],
                row["value_text"],
                row["value_num"],
                row["value_date"],
            ),
        )
    for row in annual_rows:
        conn.execute(
            """
            INSERT INTO profile_annual_report (
                conid,
                effective_at,
                field_id,
                value_num,
                is_summary
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(conid, effective_at, field_id) DO UPDATE SET
                value_num = excluded.value_num,
                is_summary = excluded.is_summary
            """,
            (
                row["conid"],
                row["effective_at"],
                row["field_id"],
                row["value_num"],
                row["is_summary"],
            ),
        )
    for row in prospectus_rows:
        conn.execute(
            """
            INSERT INTO profile_prospectus_report (
                conid,
                effective_at,
                field_id,
                value_num,
                is_summary
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(conid, effective_at, field_id) DO UPDATE SET
                value_num = excluded.value_num,
                is_summary = excluded.is_summary
            """,
            (
                row["conid"],
                row["effective_at"],
                row["field_id"],
                row["value_num"],
                row["is_summary"],
            ),
        )
    for row in theme_rows:
        conn.execute(
            """
            INSERT INTO profile_themes (
                conid,
                effective_at,
                theme_id
            ) VALUES (?, ?, ?)
            """,
            (
                row["conid"],
                row["effective_at"],
                row["theme_id"],
            ),
        )
    for row in stylebox_rows:
        conn.execute(
            """
            INSERT INTO profile_stylebox (
                conid,
                effective_at,
                stylebox_id,
                x_index,
                y_index,
                x_label,
                y_label,
                x_tag,
                y_tag
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["conid"],
                row["effective_at"],
                row["stylebox_id"],
                row["x_index"],
                row["y_index"],
                row["x_label"],
                row["y_label"],
                row["x_tag"],
                row["y_tag"],
            ),
        )
    return SnapshotWriteResult(
        payload_hash=capture.payload_hash,
        capture_batch_id=batch_id,
        raw_observation_inserted=capture.observation_inserted,
        effective_at=effective_at,
    )


def write_holdings_snapshot(
    conn: sqlite3.Connection,
    *,
    conid: str,
    payload: JsonValue | bytes,
    observed_at: str,
    source_family: str = "ibkr",
    capture_batch_id: str | None = None,
) -> SnapshotWriteResult:
    source_as_of_date = None
    if not isinstance(payload, bytes) and isinstance(payload, dict):
        source_as_of_date = payload.get("as_of_date") or payload.get("asOfDate")
    batch_id = capture_batch_id or new_capture_batch_id()
    resolution = resolve_effective_at(
        "holdings_snapshot",
        observed_at=observed_at,
        source_as_of_date=source_as_of_date,
    )
    effective_at = resolution.effective_at.isoformat()
    capture = capture_raw_payload(
        conn,
        source_family=source_family,
        endpoint="holdings",
        payload=payload,
        observed_at=observed_at,
        conid=conid,
        source_as_of_date=source_as_of_date,
        capture_batch_id=batch_id,
    )

    holdings_row = _extract_holdings_asset_type_row(conid, effective_at, payload)
    conn.execute(
        """
        INSERT INTO holdings_snapshots (
            conid,
            effective_at,
            observed_at,
            payload_hash,
            capture_batch_id,
            as_of_date
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(conid, effective_at) DO UPDATE SET
            observed_at = excluded.observed_at,
            payload_hash = excluded.payload_hash,
            capture_batch_id = excluded.capture_batch_id,
            as_of_date = excluded.as_of_date
        """,
        (
            conid,
            effective_at,
            observed_at,
            capture.payload_hash,
            batch_id,
            to_iso_date(source_as_of_date),
        ),
    )
    _delete_snapshot_child_rows(
        conn,
        tables=(
            "holdings_asset_type",
            "holdings_debtor_quality",
            "holdings_maturity",
            "holdings_industry",
            "holdings_currency",
            "holdings_investor_country",
            "holdings_geographic_weights",
            "holdings_debt_type",
            "holdings_top10",
        ),
        conid=conid,
        effective_at=effective_at,
    )
    for row in _wide_bucket_row_to_factor_rows(
        holdings_row,
        bucket_keys=("equity", "cash", "fixed_income", "other"),
    ):
        conn.execute(
            """
            INSERT INTO holdings_asset_type (
                conid,
                effective_at,
                bucket_id,
                value_num,
                vs_peers
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                row["conid"],
                row["effective_at"],
                row["bucket_id"],
                row["value_num"],
                row["vs_peers"],
            ),
        )
    debtor_quality_row = _extract_holdings_debtor_quality_row(
        conid, effective_at, payload
    )
    for row in _wide_bucket_row_to_factor_rows(
        debtor_quality_row,
        bucket_keys=(
            "quality_aaa",
            "quality_aa",
            "quality_a",
            "quality_bbb",
            "quality_bb",
            "quality_b",
            "quality_ccc",
            "quality_cc",
            "quality_c",
            "quality_d",
            "quality_not_rated",
            "quality_not_available",
        ),
    ):
        conn.execute(
            """
            INSERT INTO holdings_debtor_quality (
                conid,
                effective_at,
                bucket_id,
                value_num,
                vs_peers
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                row["conid"],
                row["effective_at"],
                row["bucket_id"],
                row["value_num"],
                row["vs_peers"],
            ),
        )

    maturity_row = _extract_holdings_maturity_row(conid, effective_at, payload)
    for row in _wide_bucket_row_to_factor_rows(
        maturity_row,
        bucket_keys=(
            "maturity_less_than_1_year",
            "maturity_1_to_3_years",
            "maturity_3_to_5_years",
            "maturity_5_to_10_years",
            "maturity_10_to_20_years",
            "maturity_20_to_30_years",
            "maturity_greater_than_30_years",
            "maturity_other",
        ),
    ):
        conn.execute(
            """
            INSERT INTO holdings_maturity (
                conid,
                effective_at,
                bucket_id,
                value_num,
                vs_peers
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                row["conid"],
                row["effective_at"],
                row["bucket_id"],
                row["value_num"],
                row["vs_peers"],
            ),
        )

    for row in _extract_holdings_bucket_rows(
        conid,
        effective_at,
        payload,
        section="industry",
        name_column="industry_id",
        normalize_name=True,
    ):
        conn.execute(
            """
            INSERT INTO holdings_industry (
                conid,
                effective_at,
                industry_id,
                value_num,
                vs_peers
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                row["conid"],
                row["effective_at"],
                row["industry_id"],
                row["value_num"],
                row["vs_peers"],
            ),
        )
    for row in _extract_holdings_bucket_rows(
        conid,
        effective_at,
        payload,
        section="currency",
        name_column="name",
        extra_column="code",
    ):
        conn.execute(
            """
            INSERT INTO holdings_currency (
                conid,
                effective_at,
                code,
                name,
                value_num,
                vs_peers
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row["conid"],
                row["effective_at"],
                row["code"],
                row["name"],
                row["value_num"],
                row["vs_peers"],
            ),
        )
    for row in _extract_holdings_bucket_rows(
        conid,
        effective_at,
        payload,
        section="investor_country",
        name_column="name",
        extra_column="country_code",
    ):
        conn.execute(
            """
            INSERT INTO holdings_investor_country (
                conid,
                effective_at,
                code,
                name,
                value_num,
                vs_peers
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row["conid"],
                row["effective_at"],
                row["country_code"],
                row["name"],
                row["value_num"],
                row["vs_peers"],
            ),
        )
    for row in _extract_holdings_bucket_rows(
        conid,
        effective_at,
        payload,
        section="debt_type",
        name_column="debt_type_id",
        normalize_name=True,
    ):
        conn.execute(
            """
            INSERT INTO holdings_debt_type (
                conid,
                effective_at,
                debt_type_id,
                value_num,
                vs_peers
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                row["conid"],
                row["effective_at"],
                row["debt_type_id"],
                row["value_num"],
                row["vs_peers"],
            ),
        )
    for row in _extract_holdings_geographic_rows(conid, effective_at, payload):
        conn.execute(
            """
            INSERT INTO holdings_geographic_weights (
                conid,
                effective_at,
                region_id,
                value_num,
                vs_peers
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                row["conid"],
                row["effective_at"],
                row["region_id"],
                row["value_num"],
                row["vs_peers"],
            ),
        )
    for row in _extract_holdings_top10_rows(conid, effective_at, payload):
        conn.execute(
            """
            INSERT INTO holdings_top10 (
                conid,
                effective_at,
                name,
                ticker,
                rank,
                holding_weight_num,
                vs_peers,
                conids_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["conid"],
                row["effective_at"],
                row["name"],
                row["ticker"],
                row["rank"],
                row["holding_weight_num"],
                row["vs_peers"],
                row["conids_json"],
            ),
        )
    return SnapshotWriteResult(
        payload_hash=capture.payload_hash,
        capture_batch_id=batch_id,
        raw_observation_inserted=capture.observation_inserted,
        effective_at=effective_at,
    )


def write_ratios_snapshot(
    conn: sqlite3.Connection,
    *,
    conid: str,
    payload: JsonValue | bytes,
    observed_at: str,
    source_family: str = "ibkr",
    capture_batch_id: str | None = None,
) -> SnapshotWriteResult:
    source_as_of_date = None
    if not isinstance(payload, bytes) and isinstance(payload, dict):
        source_as_of_date = payload.get("as_of_date") or payload.get("asOfDate")
    batch_id = capture_batch_id or new_capture_batch_id()
    resolution = resolve_effective_at(
        "ratios_snapshot",
        observed_at=observed_at,
        source_as_of_date=source_as_of_date,
    )
    effective_at = resolution.effective_at.isoformat()
    capture = capture_raw_payload(
        conn,
        source_family=source_family,
        endpoint="ratios",
        payload=payload,
        observed_at=observed_at,
        conid=conid,
        source_as_of_date=source_as_of_date,
        capture_batch_id=batch_id,
    )

    conn.execute(
        """
        INSERT INTO ratios_snapshots (
            conid,
            effective_at,
            observed_at,
            payload_hash,
            capture_batch_id,
            as_of_date
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(conid, effective_at) DO UPDATE SET
            observed_at = excluded.observed_at,
            payload_hash = excluded.payload_hash,
            capture_batch_id = excluded.capture_batch_id,
            as_of_date = excluded.as_of_date
        """,
        (
            conid,
            effective_at,
            observed_at,
            capture.payload_hash,
            batch_id,
            to_iso_date(source_as_of_date),
        ),
    )
    _delete_snapshot_child_rows(
        conn,
        tables=tuple(_RATIOS_SECTION_TABLES.values()),
        conid=conid,
        effective_at=effective_at,
    )
    for section, table_name in _RATIOS_SECTION_TABLES.items():
        for row in _extract_ratios_metric_rows(
            conid,
            effective_at,
            payload,
            section=section,
        ):
            conn.execute(
                f"""
                INSERT INTO {table_name} (
                    conid,
                    effective_at,
                    metric_id,
                    value_num,
                    vs_peers
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(conid, effective_at, metric_id) DO UPDATE SET
                    value_num = excluded.value_num,
                    vs_peers = excluded.vs_peers
                """,
                (
                    row["conid"],
                    row["effective_at"],
                    row["metric_id"],
                    row["value_num"],
                    row["vs_peers"],
                ),
            )
    return SnapshotWriteResult(
        payload_hash=capture.payload_hash,
        capture_batch_id=batch_id,
        raw_observation_inserted=capture.observation_inserted,
        effective_at=effective_at,
    )


def write_dividends_snapshot(
    conn: sqlite3.Connection,
    *,
    conid: str,
    payload: JsonValue | bytes,
    observed_at: str,
    source_family: str = "ibkr",
    capture_batch_id: str | None = None,
) -> SnapshotWriteResult:
    source_as_of_date = None
    if not isinstance(payload, bytes) and isinstance(payload, dict):
        source_as_of_date = payload.get("as_of_date") or payload.get("asOfDate")
    batch_id = capture_batch_id or new_capture_batch_id()
    resolution = resolve_effective_at(
        "dividends_snapshot",
        observed_at=observed_at,
        source_as_of_date=source_as_of_date,
    )
    effective_at = resolution.effective_at.isoformat()
    capture = capture_raw_payload(
        conn,
        source_family=source_family,
        endpoint="dividends",
        payload=payload,
        observed_at=observed_at,
        conid=conid,
        source_as_of_date=source_as_of_date,
        capture_batch_id=batch_id,
    )
    conn.execute(
        """
        INSERT INTO dividends_snapshots (
            conid,
            effective_at,
            observed_at,
            payload_hash,
            capture_batch_id,
            as_of_date
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(conid, effective_at) DO UPDATE SET
            observed_at = excluded.observed_at,
            payload_hash = excluded.payload_hash,
            capture_batch_id = excluded.capture_batch_id,
            as_of_date = excluded.as_of_date
        """,
        (
            conid,
            effective_at,
            observed_at,
            capture.payload_hash,
            batch_id,
            to_iso_date(source_as_of_date),
        ),
    )
    conn.execute(
        "DELETE FROM dividends_industry_metrics WHERE conid = ? AND effective_at = ?",
        (conid, effective_at),
    )
    for row in _extract_dividends_industry_metric_rows(conid, effective_at, payload):
        conn.execute(
            """
            INSERT INTO dividends_industry_metrics (
                conid,
                effective_at,
                metric_id,
                value_num,
                currency
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(conid, effective_at, metric_id) DO UPDATE SET
                value_num = excluded.value_num,
                currency = excluded.currency
            """,
            (
                row["conid"],
                row["effective_at"],
                row["metric_id"],
                row["value_num"],
                row["currency"],
            ),
        )
    return SnapshotWriteResult(
        payload_hash=capture.payload_hash,
        capture_batch_id=batch_id,
        raw_observation_inserted=capture.observation_inserted,
        effective_at=effective_at,
    )


def write_morningstar_snapshot(
    conn: sqlite3.Connection,
    *,
    conid: str,
    payload: JsonValue | bytes,
    observed_at: str,
    source_family: str = "ibkr",
    capture_batch_id: str | None = None,
) -> SnapshotWriteResult:
    source_as_of_date = None
    q_full_report_id = None
    if not isinstance(payload, bytes) and isinstance(payload, dict):
        source_as_of_date = _resolve_morningstar_source_as_of_date(payload)
        q_full_report_id = payload.get("q_full_report_id")
    batch_id = capture_batch_id or new_capture_batch_id()
    resolution = resolve_effective_at(
        "morningstar_snapshot",
        observed_at=observed_at,
        source_as_of_date=source_as_of_date,
    )
    effective_at = resolution.effective_at.isoformat()
    capture = capture_raw_payload(
        conn,
        source_family=source_family,
        endpoint="morningstar",
        payload=payload,
        observed_at=observed_at,
        conid=conid,
        source_as_of_date=source_as_of_date,
        capture_batch_id=batch_id,
    )
    conn.execute(
        """
        INSERT INTO morningstar_snapshots (
            conid,
            effective_at,
            observed_at,
            payload_hash,
            capture_batch_id,
            as_of_date,
            q_full_report_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(conid, effective_at) DO UPDATE SET
            observed_at = excluded.observed_at,
            payload_hash = excluded.payload_hash,
            capture_batch_id = excluded.capture_batch_id,
            as_of_date = excluded.as_of_date,
            q_full_report_id = excluded.q_full_report_id
        """,
        (
            conid,
            effective_at,
            observed_at,
            capture.payload_hash,
            batch_id,
            to_iso_date(source_as_of_date),
            q_full_report_id,
        ),
    )
    conn.execute(
        "DELETE FROM morningstar_summary WHERE conid = ? AND effective_at = ?",
        (conid, effective_at),
    )
    for row in _extract_morningstar_summary_rows(conid, effective_at, payload):
        conn.execute(
            """
            INSERT INTO morningstar_summary (
                conid,
                effective_at,
                metric_id,
                title,
                derived_quantitatively,
                publish_date,
                value_text,
                value_num
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(conid, effective_at, metric_id) DO UPDATE SET
                title = excluded.title,
                derived_quantitatively = excluded.derived_quantitatively,
                publish_date = excluded.publish_date,
                value_text = excluded.value_text,
                value_num = excluded.value_num
            """,
            (
                row["conid"],
                row["effective_at"],
                row["metric_id"],
                row["title"],
                row["derived_quantitatively"],
                row["publish_date"],
                row["value_text"],
                row["value_num"],
            ),
        )
    return SnapshotWriteResult(
        payload_hash=capture.payload_hash,
        capture_batch_id=batch_id,
        raw_observation_inserted=capture.observation_inserted,
        effective_at=effective_at,
    )


def write_lipper_ratings_snapshot(
    conn: sqlite3.Connection,
    *,
    conid: str,
    payload: JsonValue | bytes,
    observed_at: str,
    source_family: str = "ibkr",
    capture_batch_id: str | None = None,
) -> SnapshotWriteResult:
    source_as_of_date = None
    universes: list[object] = []
    if not isinstance(payload, bytes) and isinstance(payload, dict):
        source_as_of_date = _resolve_lipper_source_as_of_date(payload)
        raw_universes = payload.get("universes")
        universes = raw_universes if isinstance(raw_universes, list) else []
    batch_id = capture_batch_id or new_capture_batch_id()
    resolution = resolve_effective_at(
        "lipper_ratings_snapshot",
        observed_at=observed_at,
        source_as_of_date=source_as_of_date,
    )
    effective_at = resolution.effective_at.isoformat()
    capture = capture_raw_payload(
        conn,
        source_family=source_family,
        endpoint="lipper_ratings",
        payload=payload,
        observed_at=observed_at,
        conid=conid,
        source_as_of_date=source_as_of_date,
        capture_batch_id=batch_id,
    )
    conn.execute(
        """
        INSERT INTO lipper_ratings_snapshots (
            conid,
            effective_at,
            observed_at,
            payload_hash,
            capture_batch_id,
            universe_count
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(conid, effective_at) DO UPDATE SET
            observed_at = excluded.observed_at,
            payload_hash = excluded.payload_hash,
            capture_batch_id = excluded.capture_batch_id,
            universe_count = excluded.universe_count
        """,
        (
            conid,
            effective_at,
            observed_at,
            capture.payload_hash,
            batch_id,
            len(universes),
        ),
    )
    conn.execute(
        "DELETE FROM lipper_ratings WHERE conid = ? AND effective_at = ?",
        (conid, effective_at),
    )
    for row in _extract_lipper_rating_rows(conid, effective_at, payload):
        conn.execute(
            """
            INSERT INTO lipper_ratings (
                conid,
                effective_at,
                period,
                metric_id,
                value_num,
                rating_label,
                universe_name,
                universe_as_of_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(conid, effective_at, universe_name, period, metric_id) DO UPDATE SET
                value_num = excluded.value_num,
                rating_label = excluded.rating_label,
                universe_as_of_date = excluded.universe_as_of_date
            """,
            (
                row["conid"],
                row["effective_at"],
                row["period"],
                row["metric_id"],
                row["value_num"],
                row["rating_label"],
                row["universe_name"],
                row["universe_as_of_date"],
            ),
        )
    return SnapshotWriteResult(
        payload_hash=capture.payload_hash,
        capture_batch_id=batch_id,
        raw_observation_inserted=capture.observation_inserted,
        effective_at=effective_at,
    )


def write_supplementary_fetch_log(
    conn: sqlite3.Connection,
    *,
    dataset: str,
    observed_at: str,
    status: str,
    record_count: int,
    min_key: str | None = None,
    max_key: str | None = None,
    notes: str | None = None,
) -> SupplementaryFetchLogWriteResult:
    conn.execute(
        """
        INSERT INTO supplementary_fetch_log (
            dataset,
            observed_at,
            status,
            record_count,
            min_key,
            max_key,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (dataset, observed_at, status, int(record_count), min_key, max_key, notes),
    )
    return SupplementaryFetchLogWriteResult(rows_written=1)


def write_supplementary_risk_free_sources(
    conn: sqlite3.Connection,
    *,
    frame: pd.DataFrame,
    observed_at: str,
) -> SupplementaryWriteResult:
    required = ("series_id", "source_name", "trade_date", "nominal_rate")
    normalized = _require_frame_columns(
        frame,
        columns=required,
        contract_name="supplementary risk free sources",
    )
    normalized["economy_code"] = (
        frame["economy_code"].astype(str).str.upper()
        if "economy_code" in frame.columns
        else None
    )
    normalized["trade_date"] = pd.to_datetime(
        normalized["trade_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    normalized["nominal_rate"] = pd.to_numeric(
        normalized["nominal_rate"], errors="coerce"
    )
    normalized["observed_at"] = observed_at

    rows_written = 0
    for row in normalized.to_dict(orient="records"):
        cursor = conn.execute(
            """
            INSERT INTO supplementary_risk_free_sources (
                series_id,
                source_name,
                economy_code,
                trade_date,
                nominal_rate,
                observed_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(series_id, trade_date) DO UPDATE SET
                source_name = excluded.source_name,
                economy_code = excluded.economy_code,
                nominal_rate = excluded.nominal_rate,
                observed_at = excluded.observed_at
            """,
            (
                row["series_id"],
                row["source_name"],
                row["economy_code"],
                row["trade_date"],
                row["nominal_rate"],
                row["observed_at"],
            ),
        )
        rows_written += cursor.rowcount > 0

    return SupplementaryWriteResult(
        table_name="supplementary_risk_free_sources",
        rows_written=int(rows_written),
    )


def write_supplementary_world_bank_raw(
    conn: sqlite3.Connection,
    *,
    frame: pd.DataFrame,
    observed_at: str,
) -> SupplementaryWriteResult:
    required = ("economy_code", "indicator_id", "year", "value")
    normalized = _require_frame_columns(
        frame,
        columns=required,
        contract_name="supplementary world bank raw",
    )
    normalized["economy_code"] = normalized["economy_code"].astype(str).str.upper()
    normalized["indicator_id"] = normalized["indicator_id"].astype(str)
    normalized["year"] = pd.to_numeric(normalized["year"], errors="coerce")
    normalized["value"] = pd.to_numeric(normalized["value"], errors="coerce")
    normalized["observed_at"] = observed_at

    rows_written = 0
    for row in normalized.to_dict(orient="records"):
        cursor = conn.execute(
            """
            INSERT INTO supplementary_world_bank_raw (
                economy_code,
                indicator_id,
                year,
                value,
                observed_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(economy_code, indicator_id, year) DO UPDATE SET
                value = excluded.value,
                observed_at = excluded.observed_at
            """,
            (
                row["economy_code"],
                row["indicator_id"],
                int(row["year"]),
                row["value"],
                row["observed_at"],
            ),
        )
        rows_written += cursor.rowcount > 0

    return SupplementaryWriteResult(
        table_name="supplementary_world_bank_raw",
        rows_written=int(rows_written),
    )
