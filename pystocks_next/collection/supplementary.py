from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from io import StringIO
from typing import Any

import pandas as pd
import requests

from ..progress import ProgressSink
from ..storage import (
    write_supplementary_fetch_log,
    write_supplementary_risk_free_sources,
    write_supplementary_world_bank_raw,
)
from ..supplementary_sources import (
    RISK_FREE_SERIES_BY_ECONOMY,
    WORLD_BANK_INDICATOR_MAP,
    normalize_economy_codes,
)

RequestsGet = Callable[..., requests.Response]
_RISK_FREE_SOURCE_FETCH_COLUMNS: tuple[str, ...] = (
    "series_id",
    "source_name",
    "trade_date",
    "nominal_rate",
)
_RISK_FREE_SOURCE_RESULT_COLUMNS: tuple[str, ...] = (
    "series_id",
    "source_name",
    "trade_date",
    "nominal_rate",
    "economy_code",
)
_WORLD_BANK_RAW_FETCH_COLUMNS: tuple[str, ...] = (
    "economy_code",
    "indicator_id",
    "year",
    "value",
)


@dataclass(frozen=True, slots=True)
class SupplementaryCollectionResult:
    status: str
    risk_free_source_rows: int
    world_bank_raw_rows: int
    fetch_log_rows: int
    economy_count: int


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _fetch_fred_series_csv(
    series_id: str,
    *,
    request_get: RequestsGet = requests.get,
) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    response = request_get(url, timeout=30)
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text))
    if frame.empty or len(frame.columns) < 2:
        return pd.DataFrame(columns=pd.Index(_RISK_FREE_SOURCE_FETCH_COLUMNS))

    trade_date_column = "DATE" if "DATE" in frame.columns else str(frame.columns[0])
    value_column = str(frame.columns[-1])
    out = frame.rename(
        columns={
            trade_date_column: "trade_date",
            value_column: "nominal_rate",
        }
    )
    if "trade_date" not in out.columns or "nominal_rate" not in out.columns:
        return pd.DataFrame(columns=pd.Index(_RISK_FREE_SOURCE_FETCH_COLUMNS))
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    nominal_rate = pd.to_numeric(out["nominal_rate"], errors="coerce")
    out["nominal_rate"] = pd.Series(nominal_rate, index=out.index).map(
        lambda value: float(value) / 100.0 if pd.notna(value) else pd.NA
    )
    out = out.loc[out["nominal_rate"].notna()].copy()
    out["series_id"] = series_id
    out["source_name"] = "fred"
    return out.loc[:, list(_RISK_FREE_SOURCE_FETCH_COLUMNS)]


def fetch_risk_free_sources(
    *,
    series_map: Mapping[str, str] | None = None,
    request_get: RequestsGet = requests.get,
) -> pd.DataFrame:
    mapping = series_map or RISK_FREE_SERIES_BY_ECONOMY
    frames: list[pd.DataFrame] = []
    for economy_code, series_id in mapping.items():
        frame = _fetch_fred_series_csv(series_id, request_get=request_get)
        if not frame.empty:
            frame["economy_code"] = str(economy_code).upper()
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=pd.Index(_RISK_FREE_SOURCE_RESULT_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def _load_wbgapi_module() -> Any:
    try:
        return import_module("wbgapi")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "wbgapi is required for World Bank refresh. Install it with "
            "./venv/bin/pip install wbgapi."
        ) from exc


def _iter_world_bank_rows(
    *,
    indicator_ids: Sequence[str],
    economy_codes: Sequence[str],
    wb_module: Any | None = None,
) -> list[Mapping[str, object]]:
    module = wb_module or _load_wbgapi_module()
    if not economy_codes:
        return []
    rows = module.data.fetch(
        list(indicator_ids),
        economy=list(dict.fromkeys(str(code).upper() for code in economy_codes)),
        numericTimeKeys=True,
    )
    return [row for row in rows if isinstance(row, Mapping)]


def fetch_world_bank_raw(
    *,
    economy_codes: Iterable[str],
    indicator_map: Mapping[str, str] | None = None,
    wb_module: Any | None = None,
) -> pd.DataFrame:
    indicator_ids = list((indicator_map or WORLD_BANK_INDICATOR_MAP).keys())
    rows: list[dict[str, object]] = []
    for item in _iter_world_bank_rows(
        indicator_ids=indicator_ids,
        economy_codes=list(economy_codes),
        wb_module=wb_module,
    ):
        economy = item.get("economy")
        indicator = item.get("series")
        year = item.get("time")
        if isinstance(economy, Mapping):
            economy = economy.get("id") or economy.get("value")
        if isinstance(indicator, Mapping):
            indicator = indicator.get("id") or indicator.get("value")
        if economy is None or indicator is None or year is None:
            continue
        try:
            year_value = int(str(year).removeprefix("YR"))
        except ValueError:
            continue
        rows.append(
            {
                "economy_code": str(economy).upper(),
                "indicator_id": str(indicator),
                "year": year_value,
                "value": pd.to_numeric(item.get("value"), errors="coerce"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=pd.Index(_WORLD_BANK_RAW_FETCH_COLUMNS))
    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["economy_code", "indicator_id", "year"], keep="last")
        .sort_values(["economy_code", "indicator_id", "year"])
        .reset_index(drop=True)
    )


def refresh_supplementary_sources(
    conn: sqlite3.Connection,
    *,
    economy_codes: Sequence[str] | None = None,
    risk_free_fetcher: Callable[[], pd.DataFrame] | None = None,
    world_bank_fetcher: Callable[[Sequence[str]], pd.DataFrame] | None = None,
    progress: ProgressSink | None = None,
) -> SupplementaryCollectionResult:
    observed_at = _utc_now()
    normalized_economies = normalize_economy_codes(economy_codes or ())
    if not normalized_economies:
        return SupplementaryCollectionResult(
            status="no_economies",
            risk_free_source_rows=0,
            world_bank_raw_rows=0,
            fetch_log_rows=0,
            economy_count=0,
        )
    tracker = (
        progress.stage("Refreshing supplementary data", total=4, unit="step")
        if progress is not None
        else None
    )

    risk_free_sources = (
        risk_free_fetcher()
        if risk_free_fetcher is not None
        else fetch_risk_free_sources()
    )
    if tracker is not None:
        tracker.advance(detail="Fetched risk-free source series")
    world_bank_raw = (
        world_bank_fetcher(normalized_economies)
        if world_bank_fetcher is not None
        else fetch_world_bank_raw(economy_codes=normalized_economies)
    )
    if tracker is not None:
        tracker.advance(detail="Fetched World Bank series")

    risk_free_result = write_supplementary_risk_free_sources(
        conn,
        frame=risk_free_sources,
        observed_at=observed_at,
    )
    if tracker is not None:
        tracker.advance(detail="Persisted risk-free source series")
    world_bank_result = write_supplementary_world_bank_raw(
        conn,
        frame=world_bank_raw,
        observed_at=observed_at,
    )
    write_supplementary_fetch_log(
        conn,
        dataset="risk_free_sources",
        observed_at=observed_at,
        status="ok",
        record_count=len(risk_free_sources),
        min_key=(
            str(risk_free_sources["trade_date"].min())
            if not risk_free_sources.empty
            else None
        ),
        max_key=(
            str(risk_free_sources["trade_date"].max())
            if not risk_free_sources.empty
            else None
        ),
    )
    write_supplementary_fetch_log(
        conn,
        dataset="world_bank_raw",
        observed_at=observed_at,
        status="ok",
        record_count=len(world_bank_raw),
        min_key=str(world_bank_raw["year"].min()) if not world_bank_raw.empty else None,
        max_key=str(world_bank_raw["year"].max()) if not world_bank_raw.empty else None,
    )
    if tracker is not None:
        tracker.advance(
            detail=(
                f"Persisted supplementary rows for {len(normalized_economies)} economies"
            )
        )
        tracker.close(
            detail=(
                f"{risk_free_result.rows_written + world_bank_result.rows_written} rows written"
            )
        )
    return SupplementaryCollectionResult(
        status="ok",
        risk_free_source_rows=risk_free_result.rows_written,
        world_bank_raw_rows=world_bank_result.rows_written,
        fetch_log_rows=2,
        economy_count=len(normalized_economies),
    )
