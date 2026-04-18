from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
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
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    is_percent = "%" in text
    text = text.replace("%", "").strip()
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed / 100.0 if is_percent else parsed


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
        return _parse_percent_fraction(text)
    parsed = _parse_float(text)
    if parsed is None:
        return None
    return parsed / 100.0 if abs(parsed) > 1.0 else parsed


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
    }
    if isinstance(payload, bytes) or not isinstance(payload, dict):
        return profile_row

    profile_row["objective"] = (
        str(payload.get("objective")) if payload.get("objective") is not None else None
    )
    profile_row["jap_fund_warning"] = _to_int_bool(payload.get("jap_fund_warning"))
    themes = [
        str(item.get("theme_name")).strip()
        for item in payload.get("themes", [])
        if isinstance(item, dict) and item.get("theme_name")
    ]
    profile_row["theme_name"] = " | ".join(dict.fromkeys(themes)) if themes else None

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


_HOLDINGS_ASSET_TYPE_MAP = {
    "equity": "equity",
    "cash": "cash",
    "fixed_income": "fixed_income",
    "other": "other",
}


def _sanitize_segment(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_") or "field"


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
        weight_value = (
            item.get("weight") or item.get("assets_pct") or item.get("formatted_weight")
        )
        parsed_weight = _to_fraction_weight(weight_value)
        if parsed_weight is None:
            continue
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
    return row


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
    resolution = resolve_effective_at(
        "profile_and_fees_snapshot",
        observed_at=observed_at,
    )
    effective_at = resolution.effective_at.isoformat()
    capture = capture_raw_payload(
        conn,
        source_family=source_family,
        endpoint="profile_and_fees",
        payload=payload,
        observed_at=observed_at,
        conid=conid,
        capture_batch_id=batch_id,
    )

    profile_row = _extract_profile_and_fees_row(conid, effective_at, payload)
    conn.execute(
        """
        INSERT INTO profile_and_fees_snapshots (
            conid,
            effective_at,
            observed_at,
            payload_hash,
            capture_batch_id
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(conid, effective_at) DO UPDATE SET
            observed_at = excluded.observed_at,
            payload_hash = excluded.payload_hash,
            capture_batch_id = excluded.capture_batch_id
        """,
        (conid, effective_at, observed_at, capture.payload_hash, batch_id),
    )
    conn.execute(
        """
        INSERT INTO profile_and_fees (
            conid,
            effective_at,
            asset_type,
            classification,
            distribution_details,
            domicile,
            fiscal_date,
            fund_category,
            fund_management_company,
            fund_manager_benchmark,
            fund_market_cap_focus,
            geographical_focus,
            inception_date,
            management_approach,
            management_expenses,
            manager_tenure,
            maturity_date,
            objective_type,
            portfolio_manager,
            redemption_charge_actual,
            redemption_charge_max,
            scheme,
            total_expense_ratio,
            total_net_assets_value,
            total_net_assets_date,
            objective,
            jap_fund_warning,
            theme_name
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(conid, effective_at) DO UPDATE SET
            asset_type = excluded.asset_type,
            classification = excluded.classification,
            distribution_details = excluded.distribution_details,
            domicile = excluded.domicile,
            fiscal_date = excluded.fiscal_date,
            fund_category = excluded.fund_category,
            fund_management_company = excluded.fund_management_company,
            fund_manager_benchmark = excluded.fund_manager_benchmark,
            fund_market_cap_focus = excluded.fund_market_cap_focus,
            geographical_focus = excluded.geographical_focus,
            inception_date = excluded.inception_date,
            management_approach = excluded.management_approach,
            management_expenses = excluded.management_expenses,
            manager_tenure = excluded.manager_tenure,
            maturity_date = excluded.maturity_date,
            objective_type = excluded.objective_type,
            portfolio_manager = excluded.portfolio_manager,
            redemption_charge_actual = excluded.redemption_charge_actual,
            redemption_charge_max = excluded.redemption_charge_max,
            scheme = excluded.scheme,
            total_expense_ratio = excluded.total_expense_ratio,
            total_net_assets_value = excluded.total_net_assets_value,
            total_net_assets_date = excluded.total_net_assets_date,
            objective = excluded.objective,
            jap_fund_warning = excluded.jap_fund_warning,
            theme_name = excluded.theme_name
        """,
        (
            profile_row["conid"],
            profile_row["effective_at"],
            profile_row["asset_type"],
            profile_row["classification"],
            profile_row["distribution_details"],
            profile_row["domicile"],
            profile_row["fiscal_date"],
            profile_row["fund_category"],
            profile_row["fund_management_company"],
            profile_row["fund_manager_benchmark"],
            profile_row["fund_market_cap_focus"],
            profile_row["geographical_focus"],
            profile_row["inception_date"],
            profile_row["management_approach"],
            profile_row["management_expenses"],
            profile_row["manager_tenure"],
            profile_row["maturity_date"],
            profile_row["objective_type"],
            profile_row["portfolio_manager"],
            profile_row["redemption_charge_actual"],
            profile_row["redemption_charge_max"],
            profile_row["scheme"],
            profile_row["total_expense_ratio"],
            profile_row["total_net_assets_value"],
            profile_row["total_net_assets_date"],
            profile_row["objective"],
            profile_row["jap_fund_warning"],
            profile_row["theme_name"],
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
    conn.execute(
        """
        INSERT INTO holdings_asset_type (
            conid,
            effective_at,
            equity,
            cash,
            fixed_income,
            other
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(conid, effective_at) DO UPDATE SET
            equity = excluded.equity,
            cash = excluded.cash,
            fixed_income = excluded.fixed_income,
            other = excluded.other
        """,
        (
            holdings_row["conid"],
            holdings_row["effective_at"],
            holdings_row["equity"],
            holdings_row["cash"],
            holdings_row["fixed_income"],
            holdings_row["other"],
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


def write_supplementary_risk_free_daily(
    conn: sqlite3.Connection,
    *,
    frame: pd.DataFrame,
) -> SupplementaryWriteResult:
    required = (
        "trade_date",
        "nominal_rate",
        "daily_nominal_rate",
        "source_count",
        "observed_at",
    )
    normalized = _require_frame_columns(
        frame,
        columns=required,
        contract_name="supplementary risk free daily",
    )
    normalized["trade_date"] = pd.to_datetime(
        normalized["trade_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    for column in ("nominal_rate", "daily_nominal_rate", "source_count"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    rows_written = 0
    for row in normalized.to_dict(orient="records"):
        cursor = conn.execute(
            """
            INSERT INTO supplementary_risk_free_daily (
                trade_date,
                nominal_rate,
                daily_nominal_rate,
                source_count,
                observed_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(trade_date) DO UPDATE SET
                nominal_rate = excluded.nominal_rate,
                daily_nominal_rate = excluded.daily_nominal_rate,
                source_count = excluded.source_count,
                observed_at = excluded.observed_at
            """,
            (
                row["trade_date"],
                row["nominal_rate"],
                row["daily_nominal_rate"],
                int(row["source_count"]),
                row["observed_at"],
            ),
        )
        rows_written += cursor.rowcount > 0

    return SupplementaryWriteResult(
        table_name="supplementary_risk_free_daily",
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


def write_supplementary_world_bank_country_features(
    conn: sqlite3.Connection,
    *,
    frame: pd.DataFrame,
) -> SupplementaryWriteResult:
    required = (
        "economy_code",
        "effective_at",
        "feature_year",
        "population_level",
        "population_growth",
        "population_acceleration",
        "gdp_pcap_level",
        "gdp_pcap_growth",
        "gdp_pcap_acceleration",
        "economic_output_gdp_level",
        "economic_output_gdp_growth",
        "economic_output_gdp_acceleration",
        "foreign_direct_investment_level",
        "foreign_direct_investment_growth",
        "foreign_direct_investment_acceleration",
        "share_trade_volume_level",
        "share_trade_volume_growth",
        "share_trade_volume_acceleration",
        "observed_at",
    )
    normalized = frame.reindex(columns=pd.Index(required)).copy()
    normalized = _require_frame_columns(
        normalized,
        columns=required,
        contract_name="supplementary world bank country features",
    )
    normalized["economy_code"] = normalized["economy_code"].astype(str).str.upper()
    normalized["effective_at"] = pd.to_datetime(
        normalized["effective_at"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    normalized["feature_year"] = pd.to_numeric(
        normalized["feature_year"], errors="coerce"
    )
    for column in required:
        if column in {"economy_code", "effective_at", "feature_year", "observed_at"}:
            continue
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    rows_written = 0
    for row in normalized.to_dict(orient="records"):
        cursor = conn.execute(
            """
            INSERT INTO supplementary_world_bank_country_features (
                economy_code,
                effective_at,
                feature_year,
                population_level,
                population_growth,
                population_acceleration,
                gdp_pcap_level,
                gdp_pcap_growth,
                gdp_pcap_acceleration,
                economic_output_gdp_level,
                economic_output_gdp_growth,
                economic_output_gdp_acceleration,
                foreign_direct_investment_level,
                foreign_direct_investment_growth,
                foreign_direct_investment_acceleration,
                share_trade_volume_level,
                share_trade_volume_growth,
                share_trade_volume_acceleration,
                observed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(economy_code, feature_year) DO UPDATE SET
                effective_at = excluded.effective_at,
                population_level = excluded.population_level,
                population_growth = excluded.population_growth,
                population_acceleration = excluded.population_acceleration,
                gdp_pcap_level = excluded.gdp_pcap_level,
                gdp_pcap_growth = excluded.gdp_pcap_growth,
                gdp_pcap_acceleration = excluded.gdp_pcap_acceleration,
                economic_output_gdp_level = excluded.economic_output_gdp_level,
                economic_output_gdp_growth = excluded.economic_output_gdp_growth,
                economic_output_gdp_acceleration = excluded.economic_output_gdp_acceleration,
                foreign_direct_investment_level = excluded.foreign_direct_investment_level,
                foreign_direct_investment_growth = excluded.foreign_direct_investment_growth,
                foreign_direct_investment_acceleration = excluded.foreign_direct_investment_acceleration,
                share_trade_volume_level = excluded.share_trade_volume_level,
                share_trade_volume_growth = excluded.share_trade_volume_growth,
                share_trade_volume_acceleration = excluded.share_trade_volume_acceleration,
                observed_at = excluded.observed_at
            """,
            (
                row["economy_code"],
                row["effective_at"],
                int(row["feature_year"]),
                row["population_level"],
                row["population_growth"],
                row["population_acceleration"],
                row["gdp_pcap_level"],
                row["gdp_pcap_growth"],
                row["gdp_pcap_acceleration"],
                row["economic_output_gdp_level"],
                row["economic_output_gdp_growth"],
                row["economic_output_gdp_acceleration"],
                row["foreign_direct_investment_level"],
                row["foreign_direct_investment_growth"],
                row["foreign_direct_investment_acceleration"],
                row["share_trade_volume_level"],
                row["share_trade_volume_growth"],
                row["share_trade_volume_acceleration"],
                row["observed_at"],
            ),
        )
        rows_written += cursor.rowcount > 0

    return SupplementaryWriteResult(
        table_name="supplementary_world_bank_country_features",
        rows_written=int(rows_written),
    )
