# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false, reportGeneralTypeIssues=false
from collections.abc import Iterable
from datetime import UTC, datetime
from importlib import import_module
from io import StringIO
from typing import cast

import pandas as pd
import requests

from ..config import SQLITE_DB_PATH
from ..preprocess.supplementary import (
    RISK_FREE_SERIES_BY_ECONOMY,
    WORLD_BANK_INDICATOR_MAP,
    _normalize_economy_code,
    build_risk_free_series_weights,
    derive_risk_free_daily,
    load_risk_free_country_weights,
    preprocess_world_bank_country_features,
)
from ..storage.txn import transaction

WORLD_BANK_ECONOMY_BATCH_SIZE = 40


def _utc_now():
    return datetime.now(UTC).isoformat()


def _fetch_fred_series_csv(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text))
    if frame.empty or len(frame.columns) < 2:
        return pd.DataFrame(
            columns=["series_id", "source_name", "trade_date", "nominal_rate"]
        )
    trade_date_column = (
        "DATE" if "DATE" in frame.columns else cast(str, frame.columns[0])
    )
    value_column = cast(str, list(frame.columns)[-1])
    out = frame.rename(
        columns={trade_date_column: "trade_date", value_column: "nominal_rate"}
    )
    if "trade_date" not in out.columns or "nominal_rate" not in out.columns:
        return pd.DataFrame(
            columns=["series_id", "source_name", "trade_date", "nominal_rate"]
        )
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    out["nominal_rate"] = pd.to_numeric(out["nominal_rate"], errors="coerce") / 100.0
    out = out.loc[out["nominal_rate"].notna()].copy()
    out["series_id"] = series_id
    out["source_name"] = "fred"
    return out[["series_id", "source_name", "trade_date", "nominal_rate"]]


def fetch_risk_free_sources(series_map=None):
    series_map = series_map or RISK_FREE_SERIES_BY_ECONOMY
    frames = []
    for economy_code, series_id in series_map.items():
        frame = _fetch_fred_series_csv(series_id)
        if not frame.empty:
            frame["economy_code"] = economy_code
            frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=[
                "series_id",
                "source_name",
                "trade_date",
                "nominal_rate",
                "economy_code",
            ]
        )
    return pd.concat(frames, ignore_index=True)


class _WorldBankBatchError(RuntimeError):
    pass


def _load_wbgapi_module():
    try:
        return import_module("wbgapi")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "wbgapi is required for World Bank refresh. Install it with "
            "./venv/bin/pip install wbgapi."
        ) from exc


def _normalize_world_bank_economies(economy_codes: Iterable[str]) -> list[str]:
    normalized_codes: list[str] = []
    seen: set[str] = set()
    for value in economy_codes:
        code = _normalize_economy_code(value)
        if code is None:
            continue
        upper = str(code).upper()
        if len(upper) != 3 or not upper.isalpha():
            continue
        if upper in seen:
            continue
        normalized_codes.append(upper)
        seen.add(upper)
    return normalized_codes


def _load_world_bank_supported_economies(wb_module) -> set[str]:
    supported: set[str] = set()
    for item in wb_module.economy.list():
        if not isinstance(item, dict):
            continue
        code = _normalize_economy_code(item.get("id"))
        if code is None:
            continue
        upper = str(code).upper()
        if len(upper) == 3 and upper.isalpha():
            supported.add(upper)
    return supported


def _chunked_economies(
    economy_codes: list[str], batch_size: int = WORLD_BANK_ECONOMY_BATCH_SIZE
) -> Iterable[list[str]]:
    size = max(int(batch_size), 1)
    for start in range(0, len(economy_codes), size):
        yield economy_codes[start : start + size]


def _coerce_world_bank_year(value):
    if value is None:
        raise ValueError("World Bank time value is missing.")
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        raise ValueError("World Bank time value is empty.")
    if text.upper().startswith("YR"):
        text = text[2:]
    return int(text)


def _extract_world_bank_scalar(value):
    if isinstance(value, dict):
        return value.get("id") or value.get("value")
    return value


def _iter_world_bank_rows(indicator_ids, economies: Iterable[str], wb_module=None):
    wb_module = wb_module or _load_wbgapi_module()
    normalized = _normalize_world_bank_economies(economy_codes=economies)
    if not normalized:
        return []

    supported = _load_world_bank_supported_economies(wb_module)
    recognized = [code for code in normalized if code in supported]
    if not recognized:
        return []

    rows: list[dict] = []
    series = list(indicator_ids)
    for chunk in _chunked_economies(recognized):
        try:
            rows.extend(
                wb_module.data.fetch(
                    series,
                    economy=chunk,
                    numericTimeKeys=True,
                )
            )
        except Exception as exc:
            raise _WorldBankBatchError(
                f"World Bank fetch failed for economies {chunk!r}."
            ) from exc
    return rows


def fetch_world_bank_raw(economy_codes, indicator_map=None):
    indicator_map = indicator_map or WORLD_BANK_INDICATOR_MAP
    indicator_ids = list(indicator_map)
    rows = []
    for item in _iter_world_bank_rows(indicator_ids, economy_codes):
        economy = _extract_world_bank_scalar(item.get("economy"))
        indicator_id = _extract_world_bank_scalar(item.get("series"))
        year = item.get("time")
        if economy is None or indicator_id is None or year is None:
            continue
        try:
            year_value = _coerce_world_bank_year(year)
        except ValueError:
            continue
        rows.append(
            {
                "economy_code": _normalize_economy_code(economy),
                "indicator_id": str(indicator_id),
                "year": year_value,
                "value": pd.to_numeric(item.get("value"), errors="coerce"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["economy_code", "indicator_id", "year", "value"])
    return (
        pd.DataFrame(rows)
        .dropna(subset=["economy_code", "year"])
        .loc[lambda df: df["indicator_id"].isin(indicator_ids)]
        .drop_duplicates(subset=["economy_code", "indicator_id", "year"], keep="last")
        .sort_values(["economy_code", "indicator_id", "year"])
        .reset_index(drop=True)
    )


def _write_fetch_log(
    tx, dataset, status, record_count, min_key=None, max_key=None, notes=None
):
    tx.execute(
        """
        INSERT INTO supplementary_fetch_log (
            dataset,
            fetched_at,
            status,
            record_count,
            min_key,
            max_key,
            notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [dataset, _utc_now(), status, int(record_count), min_key, max_key, notes],
    )


def refresh_supplementary_data(sqlite_path=SQLITE_DB_PATH):
    risk_free_sources = fetch_risk_free_sources()
    if risk_free_sources.empty:
        raise RuntimeError("Risk-free refresh returned no source rows.")

    country_weights = load_risk_free_country_weights(sqlite_path)
    economy_codes = []
    with transaction(sqlite_path) as tx:
        holdings_countries = tx.read_frame(
            """
            SELECT DISTINCT COALESCE(country_code, country) AS economy_code
            FROM holdings_investor_country
            WHERE COALESCE(country_code, country) IS NOT NULL
            """
        )
        if not holdings_countries.empty:
            economy_codes = [
                code
                for code in (
                    _normalize_economy_code(value)
                    for value in holdings_countries["economy_code"].tolist()
                )
                if code
            ]
    if not economy_codes:
        raise RuntimeError(
            "No holdings_investor_country rows were found for supplementary refresh."
        )

    world_bank_raw = fetch_world_bank_raw(economy_codes)
    if world_bank_raw.empty:
        raise RuntimeError("World Bank refresh returned no rows.")

    fetched_at = _utc_now()
    risk_free_sources = risk_free_sources.copy()
    risk_free_sources["fetched_at"] = fetched_at
    world_bank_raw = world_bank_raw.copy()
    world_bank_raw["fetched_at"] = fetched_at

    series_weights = build_risk_free_series_weights(
        country_weights, series_map=RISK_FREE_SERIES_BY_ECONOMY
    )
    risk_free_daily = derive_risk_free_daily(
        risk_free_sources,
        series_weights=series_weights,
    )
    world_bank_features = preprocess_world_bank_country_features(world_bank_raw)

    with transaction(sqlite_path) as tx:
        tx.write_frame(
            "supplementary_risk_free_sources",
            risk_free_sources,
            if_exists="replace",
            index=False,
        )
        tx.write_frame(
            "supplementary_risk_free_daily",
            risk_free_daily,
            if_exists="replace",
            index=False,
        )
        tx.write_frame(
            "supplementary_world_bank_raw",
            world_bank_raw,
            if_exists="replace",
            index=False,
        )
        tx.write_frame(
            "supplementary_world_bank_country_features",
            world_bank_features,
            if_exists="replace",
            index=False,
        )
        _write_fetch_log(
            tx,
            "risk_free_sources",
            "ok",
            len(risk_free_sources),
            min_key=str(risk_free_sources["trade_date"].min().date()),
            max_key=str(risk_free_sources["trade_date"].max().date()),
            notes="FRED short-rate series",
        )
        _write_fetch_log(
            tx,
            "world_bank_raw",
            "ok",
            len(world_bank_raw),
            min_key=str(int(world_bank_raw["year"].min())),
            max_key=str(int(world_bank_raw["year"].max())),
            notes="World Bank annual indicator rows",
        )
        _write_fetch_log(
            tx,
            "world_bank_country_features",
            "ok",
            len(world_bank_features),
            min_key=str(int(world_bank_features["feature_year"].min())),
            max_key=str(int(world_bank_features["feature_year"].max())),
            notes="Preprocessed annual country features",
        )

    return {
        "status": "ok",
        "risk_free_source_rows": int(len(risk_free_sources)),
        "risk_free_daily_rows": int(len(risk_free_daily)),
        "world_bank_raw_rows": int(len(world_bank_raw)),
        "world_bank_feature_rows": int(len(world_bank_features)),
    }
