# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false, reportGeneralTypeIssues=false
from collections.abc import Iterable
from datetime import UTC, datetime
from io import StringIO
from typing import cast

import pandas as pd
import pycountry
import requests

from ..config import SQLITE_DB_PATH
from ..preprocess.supplementary import (
    WORLD_BANK_INDICATOR_MAP,
    derive_risk_free_daily,
    preprocess_world_bank_country_features,
)
from ..storage.txn import transaction

RISK_FREE_SERIES = {
    "DTB3": "fred",
    "IR3TIB01CAM156N": "fred",
    "IR3TIB01DEM156N": "fred",
    "IR3TIB01GBM156N": "fred",
    "IR3TIB01FRA156N": "fred",
}


def _utc_now():
    return datetime.now(UTC).isoformat()


def _normalize_economy_code(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    upper = text.upper()
    if len(upper) == 3 and upper.isalpha():
        return upper
    if len(upper) == 2 and upper.isalpha():
        country = pycountry.countries.get(alpha_2=upper)
        return country.alpha_3 if country else upper
    country = pycountry.countries.get(alpha_3=upper)
    if country:
        return country.alpha_3
    country = pycountry.countries.get(name=text)
    if country:
        return country.alpha_3
    try:
        matches = pycountry.countries.search_fuzzy(text)
    except LookupError:
        return upper
    return matches[0].alpha_3 if matches else upper


def _fetch_fred_series_csv(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text))
    if frame.empty:
        return frame
    value_column = cast(str, list(frame.columns)[-1])
    out = frame.rename(columns={"DATE": "trade_date", value_column: "nominal_rate"})
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    out["nominal_rate"] = pd.to_numeric(out["nominal_rate"], errors="coerce") / 100.0
    out = out.loc[out["nominal_rate"].notna()].copy()
    out["series_id"] = series_id
    out["source_name"] = "fred"
    return out[["series_id", "source_name", "trade_date", "nominal_rate"]]


def fetch_risk_free_sources(series_map=None):
    series_map = series_map or RISK_FREE_SERIES
    frames = []
    for series_id in series_map:
        frame = _fetch_fred_series_csv(series_id)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=["series_id", "source_name", "trade_date", "nominal_rate"]
        )
    return pd.concat(frames, ignore_index=True)


def _iter_world_bank_rows(indicator_id, economies: Iterable[str]):
    joined = ";".join(sorted(set(economies)))
    url = f"https://api.worldbank.org/v2/country/{joined}/indicator/{indicator_id}"
    params = {"format": "json", "per_page": 20000}
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2:
        return []
    return payload[1] or []


def fetch_world_bank_raw(economy_codes, indicator_map=None):
    indicator_map = indicator_map or WORLD_BANK_INDICATOR_MAP
    normalized = [
        code
        for code in (_normalize_economy_code(code) for code in economy_codes)
        if code
    ]
    rows = []
    for indicator_id in indicator_map:
        for item in _iter_world_bank_rows(indicator_id, normalized):
            economy = item.get("countryiso3code") or item.get("country", {}).get("id")
            year = item.get("date")
            if economy is None or year is None:
                continue
            rows.append(
                {
                    "economy_code": _normalize_economy_code(economy),
                    "indicator_id": indicator_id,
                    "year": int(year),
                    "value": pd.to_numeric(item.get("value"), errors="coerce"),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["economy_code", "indicator_id", "year", "value"])
    return pd.DataFrame(rows)


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

    economy_codes = []
    with transaction(sqlite_path) as tx:
        country_weights = tx.read_frame(
            """
            SELECT DISTINCT COALESCE(country_code, country) AS economy_code
            FROM holdings_investor_country
            WHERE COALESCE(country_code, country) IS NOT NULL
            """
        )
        if not country_weights.empty:
            economy_codes = [
                code
                for code in (
                    _normalize_economy_code(value)
                    for value in country_weights["economy_code"].tolist()
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

    risk_free_daily = derive_risk_free_daily(risk_free_sources)
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
