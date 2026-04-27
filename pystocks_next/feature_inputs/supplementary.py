from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pycountry

from ..storage import (
    HoldingsCountryWeightsRead,
    RiskFreeSourcesRead,
    WorldBankRawRead,
    load_latest_holdings_country_weights,
    load_risk_free_sources,
    load_world_bank_raw,
)
from .bundle import (
    MACRO_FEATURE_COLUMNS,
    RISK_FREE_DAILY_COLUMNS,
    AnalysisInputBundle,
)

WORLD_BANK_INDICATOR_MAP = {
    "SP.POP.TOTL": "population",
    "NY.GDP.PCAP.CD": "gdp_pcap",
    "NY.GDP.MKTP.CD": "economic_output_gdp",
    "BX.KLT.DINV.WD.GD.ZS": "foreign_direct_investment",
    "NE.IMP.GNFS.ZS": "imports_goods_services",
    "NE.EXP.GNFS.ZS": "exports_goods_services",
}

RISK_FREE_SERIES_BY_ECONOMY = {
    "USA": "DTB3",
    "CAN": "IR3TIB01CAM156N",
    "DEU": "IR3TIB01DEM156N",
    "GBR": "IR3TIB01GBM156N",
    "FRA": "IR3TIB01FRA156N",
}


@dataclass(frozen=True, slots=True)
class SupplementaryInputConfig:
    risk_free_trading_days: int = 252
    interpolation_limit: int = 5
    extrapolation_points: int = 3


def _to_float(value: object) -> float:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none", "null", "<na>"}:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _empty_frame(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        {column: pd.Series(dtype="object") for column in columns}
    ).reindex(columns=pd.Index(columns))


def _normalize_economy_code(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    upper = text.upper()
    if len(upper) == 3 and upper.isalpha():
        country = pycountry.countries.get(alpha_3=upper)
        return country.alpha_3 if country is not None else upper
    if len(upper) == 2 and upper.isalpha():
        country = pycountry.countries.get(alpha_2=upper)
        return country.alpha_3 if country is not None else upper
    country = pycountry.countries.get(name=text)
    if country is not None:
        return str(country.alpha_3)
    try:
        matches = pycountry.countries.search_fuzzy(text)
    except LookupError:
        return upper
    return str(matches[0].alpha_3) if matches else upper


def _resolve_risk_free_sources_frame(
    *,
    conn: sqlite3.Connection | None,
    risk_free_sources: pd.DataFrame | RiskFreeSourcesRead | None,
) -> pd.DataFrame:
    if risk_free_sources is None:
        if conn is None:
            raise ValueError("conn or risk_free_sources is required")
        return load_risk_free_sources(conn).frame.copy()
    if isinstance(risk_free_sources, RiskFreeSourcesRead):
        return risk_free_sources.frame.copy()
    return RiskFreeSourcesRead.from_frame(risk_free_sources).frame.copy()


def _resolve_world_bank_raw_frame(
    *,
    conn: sqlite3.Connection | None,
    world_bank_raw: pd.DataFrame | WorldBankRawRead | None,
) -> pd.DataFrame:
    if world_bank_raw is None:
        if conn is None:
            raise ValueError("conn or world_bank_raw is required")
        return load_world_bank_raw(conn).frame.copy()
    if isinstance(world_bank_raw, WorldBankRawRead):
        return world_bank_raw.frame.copy()
    return WorldBankRawRead.from_frame(world_bank_raw).frame.copy()


def _resolve_country_weights_frame(
    *,
    conn: sqlite3.Connection | None,
    country_weights: pd.DataFrame | HoldingsCountryWeightsRead | None,
) -> pd.DataFrame:
    if country_weights is None:
        if conn is None:
            raise ValueError("conn or country_weights is required")
        return load_latest_holdings_country_weights(conn).frame.copy()
    if isinstance(country_weights, HoldingsCountryWeightsRead):
        return country_weights.frame.copy()
    return HoldingsCountryWeightsRead.from_frame(country_weights).frame.copy()


def build_risk_free_series_weights(
    country_weights: pd.DataFrame,
    *,
    series_map: dict[str, str] | None = None,
) -> pd.Series:
    mapping = series_map or RISK_FREE_SERIES_BY_ECONOMY
    if country_weights.empty:
        return pd.Series(dtype=float)
    weights = country_weights.copy()
    weights["economy_code"] = [
        _normalize_economy_code(value) for value in weights["economy_code"].tolist()
    ]
    weights["weight"] = pd.to_numeric(weights["weight"], errors="coerce")
    weights = weights.loc[
        weights["economy_code"].notna()
        & weights["weight"].notna()
        & weights["weight"].gt(0.0)
    ].copy()
    if weights.empty:
        return pd.Series(dtype=float)
    weight_series = pd.Series(
        weights["weight"].to_numpy(dtype=float),
        index=weights["economy_code"].astype(str).str.upper(),
        dtype=float,
    )
    grouped = weight_series.groupby(level=0).sum()
    series_weights = pd.Series(
        {
            series_id: float(grouped.loc[economy_code])
            for economy_code, series_id in mapping.items()
            if economy_code in grouped.index
        },
        dtype=float,
    )
    if series_weights.empty:
        return pd.Series(dtype=float)
    return series_weights / float(series_weights.sum())


def derive_risk_free_daily(
    source_df: pd.DataFrame,
    *,
    config: SupplementaryInputConfig | None = None,
    series_weights: pd.Series | None = None,
) -> pd.DataFrame:
    config = config or SupplementaryInputConfig()
    if source_df.empty:
        return _empty_frame(RISK_FREE_DAILY_COLUMNS)
    work = source_df.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    work["nominal_rate"] = pd.to_numeric(work["nominal_rate"], errors="coerce")
    work["observed_at"] = pd.to_datetime(work["observed_at"], errors="coerce")
    per_series = (
        work.groupby(["trade_date", "series_id"], as_index=False)
        .agg(
            nominal_rate=("nominal_rate", "mean"),
            observed_at=("observed_at", "max"),
        )
        .reset_index(drop=True)
    )
    rates_wide = (
        per_series.pivot(index="trade_date", columns="series_id", values="nominal_rate")
        .sort_index()
        .sort_index(axis=1)
    )
    if rates_wide.empty:
        return _empty_frame(RISK_FREE_DAILY_COLUMNS)
    rates_wide = rates_wide.interpolate(
        method="linear",
        limit=int(config.interpolation_limit),
        limit_direction="both",
    )
    if series_weights is None or series_weights.empty:
        base_weights = pd.Series(1.0, index=rates_wide.columns, dtype=float)
    else:
        base_weights = pd.Series(series_weights, dtype=float).reindex(
            rates_wide.columns
        )
        base_weights = base_weights.fillna(0.0)
        if float(base_weights.sum()) <= 0.0:
            base_weights = pd.Series(1.0, index=rates_wide.columns, dtype=float)
    available = rates_wide.notna()
    weighted_presence = available.mul(base_weights, axis=1)
    weight_sum = weighted_presence.sum(axis=1).replace(0.0, np.nan)
    normalized_weights = weighted_presence.div(weight_sum, axis=0)
    nominal_rate = rates_wide.mul(normalized_weights).sum(axis=1, min_count=1)
    observed_at = (
        per_series.groupby("trade_date")["observed_at"].max().reindex(rates_wide.index)
    )
    source_count = available.mul(base_weights > 0.0, axis=1).sum(axis=1)
    return (
        pd.DataFrame(
            {
                "trade_date": rates_wide.index,
                "nominal_rate": nominal_rate.to_numpy(),
                "daily_nominal_rate": nominal_rate.to_numpy()
                / float(config.risk_free_trading_days),
                "source_count": source_count.to_numpy(),
                "observed_at": observed_at.to_numpy(),
            }
        )
        .reset_index(drop=True)
        .reindex(columns=pd.Index(RISK_FREE_DAILY_COLUMNS))
    )


def _linear_fill_and_extrapolate(
    row: pd.Series,
    extrapolation_points: int,
) -> pd.Series:
    series = pd.Series(row, dtype=float)
    if series.notna().sum() == 0:
        return series
    years = np.asarray(series.index, dtype=float)
    values = series.to_numpy(dtype=float)
    valid = np.isfinite(values)
    interp = np.interp(years, years[valid], values[valid])
    out = pd.Series(interp, index=series.index, dtype=float)
    valid_years = years[valid]
    valid_values = values[valid]
    n = min(int(extrapolation_points), len(valid_years))
    if n >= 2:
        left_slope, left_intercept = np.polyfit(valid_years[:n], valid_values[:n], 1)
        right_slope, right_intercept = np.polyfit(
            valid_years[-n:], valid_values[-n:], 1
        )
        for idx in series.loc[series.isna()].index.tolist():
            year_value = float(idx)
            if year_value < valid_years.min():
                out.loc[idx] = left_slope * year_value + left_intercept
            elif year_value > valid_years.max():
                out.loc[idx] = right_slope * year_value + right_intercept
    return out


def preprocess_world_bank_country_features(
    raw_df: pd.DataFrame,
    *,
    config: SupplementaryInputConfig | None = None,
) -> pd.DataFrame:
    config = config or SupplementaryInputConfig()
    if raw_df.empty:
        return _empty_frame(MACRO_FEATURE_COLUMNS)
    work = raw_df.copy()
    work["economy_code"] = work["economy_code"].astype(str).str.upper()
    year_values = pd.to_numeric(work["year"], errors="coerce")
    work["year"] = pd.Series(year_values, index=work.index).astype("Int64")
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work["observed_at"] = pd.to_datetime(work["observed_at"], errors="coerce")
    work = work.loc[work["year"].notna()].copy()
    if work.empty:
        return _empty_frame(MACRO_FEATURE_COLUMNS)
    pivot = work.pivot_table(
        index=["economy_code", "indicator_id"],
        columns="year",
        values="value",
        aggfunc="last",
    ).sort_index(axis=1)
    years = [int(year) for year in pivot.columns.tolist()]
    years.sort()
    if not years:
        return _empty_frame(MACRO_FEATURE_COLUMNS)
    pivot.columns = years
    pivot = pivot.groupby(level=[0, 1]).last()

    def indicator_frame(indicator_id: str) -> pd.DataFrame:
        if indicator_id not in pivot.index.get_level_values("indicator_id"):
            return pd.DataFrame(columns=pd.Index(years))
        frame = pivot.xs(indicator_id, level="indicator_id").copy()
        for year in years:
            if year not in frame.columns:
                frame[year] = np.nan
        return frame.loc[:, years]

    population = indicator_frame("SP.POP.TOTL")
    gdp = indicator_frame("NY.GDP.MKTP.CD")
    gdp_pcap = indicator_frame("NY.GDP.PCAP.CD")
    fdi = indicator_frame("BX.KLT.DINV.WD.GD.ZS")
    imports = indicator_frame("NE.IMP.GNFS.ZS")
    exports = indicator_frame("NE.EXP.GNFS.ZS")
    common_index = sorted(
        set(population.index)
        | set(gdp.index)
        | set(gdp_pcap.index)
        | set(fdi.index)
        | set(imports.index)
        | set(exports.index)
    )
    population = population.reindex(common_index).astype(float)
    gdp = gdp.reindex(common_index).astype(float)
    gdp_pcap = gdp_pcap.reindex(common_index).astype(float)
    fdi = fdi.reindex(common_index).astype(float)
    imports = imports.reindex(common_index).astype(float)
    exports = exports.reindex(common_index).astype(float)
    gdp_pcap = gdp_pcap.where(
        gdp_pcap.notna(), gdp.div(population.where(population != 0))
    )
    total_trade = imports.add(exports, fill_value=np.nan)
    share_trade = total_trade.div(total_trade.sum(axis=0).replace(0.0, np.nan), axis=1)
    gdp_share = gdp.div(gdp.sum(axis=0).replace(0.0, np.nan), axis=1)
    feature_frames = {
        "population": population,
        "gdp_pcap": gdp_pcap,
        "economic_output_gdp": gdp_share,
        "foreign_direct_investment": fdi,
        "share_trade_volume": share_trade,
    }
    processed: dict[str, pd.DataFrame] = {}
    for feature_name, frame in feature_frames.items():
        aligned = frame.copy()
        for economy_code in aligned.index:
            aligned.loc[economy_code] = _linear_fill_and_extrapolate(
                aligned.loc[economy_code],
                config.extrapolation_points,
            )
        aligned = aligned.fillna(aligned.mean())
        processed[feature_name] = aligned
    observed_at = work["observed_at"].max()
    rows: list[dict[str, object]] = []
    for economy_code in common_index:
        for year in years:
            row: dict[str, object] = {
                "economy_code": str(economy_code).upper(),
                "effective_at": pd.Timestamp(year=year, month=12, day=31),
                "feature_year": int(year),
                "observed_at": observed_at,
            }
            for feature_name, frame in processed.items():
                level_value = _to_float(frame.loc[economy_code, year])
                prev_year = year - 1
                prev_value = (
                    _to_float(frame.loc[economy_code, prev_year])
                    if prev_year in frame.columns
                    else float("nan")
                )
                growth_value = (
                    float(level_value - prev_value)
                    if np.isfinite(level_value) and np.isfinite(prev_value)
                    else float("nan")
                )
                prior_growth_source = (
                    _to_float(frame.loc[economy_code, prev_year])
                    if prev_year in frame.columns
                    else float("nan")
                )
                prior_growth_prev = (
                    _to_float(frame.loc[economy_code, prev_year - 1])
                    if (prev_year - 1) in frame.columns
                    else float("nan")
                )
                prev_growth_value = (
                    float(prior_growth_source - prior_growth_prev)
                    if prev_year in frame.columns
                    and (prev_year - 1) in frame.columns
                    and np.isfinite(prior_growth_source)
                    and np.isfinite(prior_growth_prev)
                    else float("nan")
                )
                acceleration_value = (
                    float(growth_value - prev_growth_value)
                    if np.isfinite(growth_value) and np.isfinite(prev_growth_value)
                    else float("nan")
                )
                row[f"{feature_name}_level"] = level_value
                row[f"{feature_name}_growth"] = growth_value
                row[f"{feature_name}_acceleration"] = acceleration_value
            rows.append(row)
    return (
        pd.DataFrame(rows)
        .reindex(columns=pd.Index(MACRO_FEATURE_COLUMNS))
        .sort_values(["economy_code", "effective_at"])
        .reset_index(drop=True)
    )


def build_supplementary_input_bundle(
    *,
    conn: sqlite3.Connection | None = None,
    risk_free_sources: pd.DataFrame | RiskFreeSourcesRead | None = None,
    world_bank_raw: pd.DataFrame | WorldBankRawRead | None = None,
    country_weights: pd.DataFrame | HoldingsCountryWeightsRead | None = None,
    config: SupplementaryInputConfig | None = None,
) -> AnalysisInputBundle:
    config = config or SupplementaryInputConfig()
    risk_free_source_frame = _resolve_risk_free_sources_frame(
        conn=conn,
        risk_free_sources=risk_free_sources,
    )
    world_bank_raw_frame = _resolve_world_bank_raw_frame(
        conn=conn,
        world_bank_raw=world_bank_raw,
    )
    country_weight_frame = _resolve_country_weights_frame(
        conn=conn,
        country_weights=country_weights,
    )
    series_weights = build_risk_free_series_weights(country_weight_frame)
    risk_free_daily = derive_risk_free_daily(
        risk_free_source_frame,
        config=config,
        series_weights=series_weights,
    )
    macro_features = preprocess_world_bank_country_features(
        world_bank_raw_frame,
        config=config,
    )
    return AnalysisInputBundle.from_frames(
        risk_free_daily=risk_free_daily,
        macro_features=macro_features,
    )
