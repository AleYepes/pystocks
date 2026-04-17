from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .dates import DateLike, to_iso_date
from .raw_capture import JsonValue, capture_raw_payload, new_capture_batch_id
from .series_rows import UnresolvedSeriesRowDateError, resolve_series_row_date


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
