# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false, reportGeneralTypeIssues=false
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DATA_DIR, SQLITE_DB_PATH
from ..progress import make_progress_bar
from ..storage.readers import (
    load_risk_free_daily as _load_risk_free_daily,
)
from ..storage.readers import (
    load_world_bank_country_features as _load_world_bank_country_features,
)
from ..storage.readers import (
    query_frame,
)
from ..storage.txn import transaction
from ..supplementary_contract import (
    RISK_FREE_SERIES_BY_ECONOMY,
    normalize_economy_code,
)


@dataclass
class SupplementaryPreprocessConfig:
    risk_free_trading_days: int = 252
    interpolation_limit: int = 5
    extrapolation_points: int = 3


def _empty_frame(columns):
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


def _normalize_economy_code(value):
    return normalize_economy_code(value)


def load_risk_free_sources(sqlite_path=SQLITE_DB_PATH):
    df = query_frame(
        """
        SELECT
            series_id,
            source_name,
            trade_date,
            nominal_rate,
            fetched_at
        FROM supplementary_risk_free_sources
        ORDER BY series_id, trade_date
        """,
        sqlite_path=sqlite_path,
    )
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["fetched_at"] = pd.to_datetime(df["fetched_at"])
    df["nominal_rate"] = pd.to_numeric(df["nominal_rate"], errors="coerce")
    return df


def load_world_bank_raw(sqlite_path=SQLITE_DB_PATH):
    df = query_frame(
        """
        SELECT
            economy_code,
            indicator_id,
            year,
            value,
            fetched_at
        FROM supplementary_world_bank_raw
        ORDER BY economy_code, indicator_id, year
        """,
        sqlite_path=sqlite_path,
    )
    if df.empty:
        return df
    df["economy_code"] = df["economy_code"].astype(str).str.upper()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["fetched_at"] = pd.to_datetime(df["fetched_at"])
    return df


def load_risk_free_country_weights(sqlite_path=SQLITE_DB_PATH):
    df = query_frame(
        """
        WITH latest_country_snapshot AS (
            SELECT
                conid,
                MAX(effective_at) AS effective_at
            FROM holdings_investor_country
            GROUP BY conid
        )
        SELECT
            COALESCE(h.country_code, h.country) AS economy_code,
            h.value_num
        FROM holdings_investor_country h
        INNER JOIN latest_country_snapshot latest
            ON latest.conid = h.conid
           AND latest.effective_at = h.effective_at
        WHERE COALESCE(h.country_code, h.country) IS NOT NULL
        """,
        sqlite_path=sqlite_path,
    )
    if df.empty:
        return pd.DataFrame(columns=["economy_code", "weight"])

    weights = df.copy()
    weights["economy_code"] = [
        _normalize_economy_code(value) for value in weights["economy_code"].tolist()
    ]
    weights["value_num"] = pd.to_numeric(weights["value_num"], errors="coerce")
    weights = weights.loc[
        weights["economy_code"].notna() & weights["value_num"].notna()
    ].copy()
    if weights.empty:
        return pd.DataFrame(columns=["economy_code", "weight"])

    grouped = (
        weights.groupby("economy_code", as_index=False)
        .agg(weight=("value_num", "sum"))
        .sort_values("economy_code")
        .reset_index(drop=True)
    )
    total = float(grouped["weight"].sum())
    if total <= 0.0:
        return pd.DataFrame(columns=["economy_code", "weight"])
    grouped["weight"] = grouped["weight"] / total
    return grouped


def build_risk_free_series_weights(country_weights, series_map=None):
    series_map = series_map or RISK_FREE_SERIES_BY_ECONOMY
    if country_weights is None:
        return pd.Series(dtype=float)

    if isinstance(country_weights, pd.DataFrame):
        if country_weights.empty:
            return pd.Series(dtype=float)
        weight_series = pd.Series(
            pd.to_numeric(country_weights["weight"], errors="coerce").to_numpy(),
            index=country_weights["economy_code"].astype(str).str.upper(),
            dtype=float,
        )
    else:
        weight_series = pd.Series(country_weights, dtype=float)
        weight_series.index = weight_series.index.astype(str).str.upper()

    weight_series = weight_series.loc[weight_series.notna() & (weight_series > 0.0)]
    if weight_series.empty:
        return pd.Series(dtype=float)

    series_weights = pd.Series(
        {
            series_id: float(weight_series.loc[economy_code])
            for economy_code, series_id in series_map.items()
            if economy_code in weight_series.index
        },
        dtype=float,
    )
    if series_weights.empty:
        return pd.Series(dtype=float)
    return series_weights / float(series_weights.sum())


def derive_risk_free_daily(source_df, config=None, series_weights=None):
    config = config or SupplementaryPreprocessConfig()
    if source_df.empty:
        return _empty_frame(
            [
                "trade_date",
                "nominal_rate",
                "daily_nominal_rate",
                "source_count",
                "observed_at",
            ]
        )

    df = source_df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["nominal_rate"] = pd.to_numeric(df["nominal_rate"], errors="coerce")
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], errors="coerce")
    per_series = (
        df.groupby(["trade_date", "series_id"], as_index=False)
        .agg(
            nominal_rate=("nominal_rate", "mean"),
            observed_at=("fetched_at", "max"),
        )
        .sort_values("trade_date")
        .reset_index(drop=True)
    )

    rates_wide = (
        per_series.pivot(index="trade_date", columns="series_id", values="nominal_rate")
        .sort_index()
        .sort_index(axis=1)
    )
    if rates_wide.empty:
        return _empty_frame(
            [
                "trade_date",
                "nominal_rate",
                "daily_nominal_rate",
                "source_count",
                "observed_at",
            ]
        )

    rates_wide = rates_wide.interpolate(
        method="linear",
        limit=int(config.interpolation_limit),
        limit_direction="both",
    )

    if series_weights is None or len(series_weights) == 0:
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

    grouped = pd.DataFrame(
        {
            "trade_date": rates_wide.index,
            "nominal_rate": nominal_rate.to_numpy(),
            "daily_nominal_rate": nominal_rate.to_numpy()
            / float(config.risk_free_trading_days),
            "source_count": source_count.to_numpy(),
            "observed_at": observed_at.to_numpy(),
        }
    ).reset_index(drop=True)
    return grouped


def _linear_fill_and_extrapolate(row, extrapolation_points):
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
        left_years = valid_years[:n]
        left_values = valid_values[:n]
        right_years = valid_years[-n:]
        right_values = valid_values[-n:]
        left_slope, left_intercept = np.polyfit(left_years, left_values, 1)
        right_slope, right_intercept = np.polyfit(right_years, right_values, 1)
        for idx in series.index[series.isna()]:
            year_value = float(idx)
            if year_value < valid_years.min():
                out.loc[idx] = left_slope * year_value + left_intercept
            elif year_value > valid_years.max():
                out.loc[idx] = right_slope * year_value + right_intercept
    return out


def preprocess_world_bank_country_features(raw_df, config=None):
    config = config or SupplementaryPreprocessConfig()
    if raw_df.empty:
        return _empty_frame(
            [
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
            ]
        )

    work = raw_df.copy()
    work["economy_code"] = work["economy_code"].astype(str).str.upper()
    work["year"] = pd.to_numeric(work["year"], errors="coerce").astype(int)
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work["fetched_at"] = pd.to_datetime(work["fetched_at"])

    pivot = work.pivot_table(
        index=["economy_code", "indicator_id"],
        columns="year",
        values="value",
        aggfunc="last",
    ).sort_index(axis=1)
    years = sorted(int(year) for year in pivot.columns)
    if not years:
        return _empty_frame(
            [
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
            ]
        )

    pivot.columns = years
    pivot = pivot.groupby(level=[0, 1]).last()

    def _indicator_frame(indicator_id):
        if indicator_id not in pivot.index.get_level_values("indicator_id"):
            return pd.DataFrame(columns=years)
        frame = pivot.xs(indicator_id, level="indicator_id").copy()
        for year in years:
            if year not in frame.columns:
                frame[year] = np.nan
        return frame.loc[:, years]

    population = _indicator_frame("SP.POP.TOTL")
    gdp = _indicator_frame("NY.GDP.MKTP.CD")
    gdp_pcap = _indicator_frame("NY.GDP.PCAP.CD")
    fdi = _indicator_frame("BX.KLT.DINV.WD.GD.ZS")
    imports = _indicator_frame("NE.IMP.GNFS.ZS")
    exports = _indicator_frame("NE.EXP.GNFS.ZS")

    common_index = sorted(
        set(population.index)
        | set(gdp.index)
        | set(gdp_pcap.index)
        | set(fdi.index)
        | set(imports.index)
        | set(exports.index)
    )
    for frame in [population, gdp, gdp_pcap, fdi, imports, exports]:
        frame = frame.reindex(common_index)

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

    processed = {}
    for feature_name, frame in feature_frames.items():
        aligned = frame.copy()
        for economy_code in aligned.index:
            aligned.loc[economy_code] = _linear_fill_and_extrapolate(
                aligned.loc[economy_code], config.extrapolation_points
            )
        aligned = aligned.fillna(aligned.mean())
        processed[feature_name] = aligned

    rows = []
    observed_at = work["fetched_at"].max()
    for economy_code in common_index:
        for year in years:
            row = {
                "economy_code": str(economy_code).upper(),
                "effective_at": pd.Timestamp(year=year, month=12, day=31),
                "feature_year": int(year),
                "observed_at": observed_at,
            }
            for feature_name, frame in processed.items():
                level_value = pd.to_numeric(
                    frame.loc[economy_code, year], errors="coerce"
                )
                prev_year = year - 1
                prev_value = (
                    pd.to_numeric(frame.loc[economy_code, prev_year], errors="coerce")
                    if prev_year in frame.columns
                    else np.nan
                )
                growth_value = (
                    float(level_value - prev_value)
                    if np.isfinite(level_value) and np.isfinite(prev_value)
                    else np.nan
                )
                prev_growth_value = (
                    float(
                        pd.to_numeric(
                            frame.loc[economy_code, prev_year], errors="coerce"
                        )
                        - pd.to_numeric(
                            frame.loc[economy_code, prev_year - 1], errors="coerce"
                        )
                    )
                    if prev_year in frame.columns
                    and (prev_year - 1) in frame.columns
                    and np.isfinite(
                        pd.to_numeric(
                            frame.loc[economy_code, prev_year], errors="coerce"
                        )
                    )
                    and np.isfinite(
                        pd.to_numeric(
                            frame.loc[economy_code, prev_year - 1], errors="coerce"
                        )
                    )
                    else np.nan
                )
                acceleration_value = (
                    float(growth_value - prev_growth_value)
                    if np.isfinite(growth_value) and np.isfinite(prev_growth_value)
                    else np.nan
                )
                row[f"{feature_name}_level"] = level_value
                row[f"{feature_name}_growth"] = growth_value
                row[f"{feature_name}_acceleration"] = acceleration_value
            rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["economy_code", "effective_at"])
        .reset_index(drop=True)
    )


def _utc_now():
    return datetime.now(UTC).isoformat()


def _write_preprocess_log(
    tx, dataset, row_count, min_key=None, max_key=None, notes=None
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
        [dataset, _utc_now(), "ok", int(row_count), min_key, max_key, notes],
    )


def persist_supplementary_preprocess_results(result, sqlite_path=SQLITE_DB_PATH):
    risk_free_daily = result["risk_free_daily"]
    world_bank_country_features = result["world_bank_country_features"]
    with transaction(sqlite_path) as tx:
        tx.write_frame(
            "supplementary_risk_free_daily",
            risk_free_daily,
            if_exists="replace",
            index=False,
        )
        tx.write_frame(
            "supplementary_world_bank_country_features",
            world_bank_country_features,
            if_exists="replace",
            index=False,
        )
        _write_preprocess_log(
            tx,
            "risk_free_daily",
            len(risk_free_daily),
            min_key=(
                str(risk_free_daily["trade_date"].min().date())
                if not risk_free_daily.empty
                else None
            ),
            max_key=(
                str(risk_free_daily["trade_date"].max().date())
                if not risk_free_daily.empty
                else None
            ),
            notes="Preprocessed weighted daily risk-free rates",
        )
        _write_preprocess_log(
            tx,
            "world_bank_country_features",
            len(world_bank_country_features),
            min_key=(
                str(int(world_bank_country_features["feature_year"].min()))
                if not world_bank_country_features.empty
                else None
            ),
            max_key=(
                str(int(world_bank_country_features["feature_year"].max()))
                if not world_bank_country_features.empty
                else None
            ),
            notes="Preprocessed annual country macro features",
        )


def _analysis_output_dir(output_dir=None):
    return Path(output_dir or (DATA_DIR / "analysis"))


def load_saved_risk_free_daily(output_dir=None):
    path = _analysis_output_dir(output_dir) / "analysis_risk_free_daily.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["observed_at"] = pd.to_datetime(df["observed_at"])
    for column in ["nominal_rate", "daily_nominal_rate", "source_count"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def load_saved_world_bank_country_features(output_dir=None):
    path = (
        _analysis_output_dir(output_dir)
        / "analysis_world_bank_country_features.parquet"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if df.empty:
        return df
    if "economy_code" in df.columns:
        df["economy_code"] = df["economy_code"].astype(str).str.upper()
    for column in ["effective_at", "observed_at"]:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    for column in df.columns:
        if column in {"economy_code", "effective_at", "observed_at"}:
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def save_supplementary_preprocess_results(result, output_dir=None):
    output_dir = Path(output_dir or (DATA_DIR / "analysis"))
    output_dir.mkdir(parents=True, exist_ok=True)
    risk_free_path = output_dir / "analysis_risk_free_daily.parquet"
    macro_path = output_dir / "analysis_world_bank_country_features.parquet"
    result["risk_free_daily"].to_parquet(risk_free_path, index=False)
    result["world_bank_country_features"].to_parquet(macro_path, index=False)
    return {
        "risk_free_daily_path": str(risk_free_path),
        "world_bank_country_features_path": str(macro_path),
    }


def run_supplementary_preprocess(
    sqlite_path=SQLITE_DB_PATH, output_dir=None, show_progress=False, **config_kwargs
):
    config = SupplementaryPreprocessConfig(**config_kwargs)
    with make_progress_bar(
        show_progress=show_progress,
        total=2,
        desc="Supplementary preprocess",
        unit="stage",
        leave=True,
    ) as progress:
        series_weights = build_risk_free_series_weights(
            load_risk_free_country_weights(sqlite_path)
        )
        risk_free = derive_risk_free_daily(
            load_risk_free_sources(sqlite_path),
            config=config,
            series_weights=series_weights,
        )
        progress.update(1)

        world_bank = preprocess_world_bank_country_features(
            load_world_bank_raw(sqlite_path), config=config
        )
        progress.update(1)
    result = {
        "risk_free_daily": risk_free,
        "world_bank_country_features": world_bank,
        "config": config,
    }
    persist_supplementary_preprocess_results(result, sqlite_path=sqlite_path)
    paths = save_supplementary_preprocess_results(result, output_dir=output_dir)
    return {
        "status": "ok",
        "risk_free_rows": int(len(risk_free)),
        "world_bank_rows": int(len(world_bank)),
        **paths,
    }


def load_risk_free_daily(sqlite_path=SQLITE_DB_PATH):
    return _load_risk_free_daily(sqlite_path=sqlite_path)


def load_world_bank_country_features(sqlite_path=SQLITE_DB_PATH):
    return _load_world_bank_country_features(sqlite_path=sqlite_path)
