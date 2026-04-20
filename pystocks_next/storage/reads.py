from __future__ import annotations

import sqlite3
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import date

import pandas as pd

PRICE_HISTORY_COLUMNS: tuple[str, ...] = (
    "conid",
    "trade_date",
    "price",
    "open",
    "high",
    "low",
    "close",
)

DIVIDEND_EVENTS_COLUMNS: tuple[str, ...] = (
    "conid",
    "symbol",
    "event_date",
    "amount",
    "dividend_currency",
    "product_currency",
    "description",
    "event_type",
    "declaration_date",
    "record_date",
    "payment_date",
)

RISK_FREE_DAILY_COLUMNS: tuple[str, ...] = (
    "trade_date",
    "nominal_rate",
    "daily_nominal_rate",
    "source_count",
    "observed_at",
)

WORLD_BANK_COUNTRY_FEATURE_COLUMNS: tuple[str, ...] = (
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

SNAPSHOT_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "profile_and_fees": (
        "conid",
        "effective_at",
        "field_id",
        "value_text",
        "value_num",
        "value_date",
        "value_bool",
    ),
    "holdings_asset_type": ("conid", "effective_at", "bucket_id", "value_num"),
    "holdings_debtor_quality": (
        "conid",
        "effective_at",
        "bucket_id",
        "value_num",
    ),
    "holdings_maturity": (
        "conid",
        "effective_at",
        "bucket_id",
        "value_num",
    ),
    "holdings_industry": ("conid", "effective_at", "industry", "value_num"),
    "holdings_currency": (
        "conid",
        "effective_at",
        "code",
        "currency",
        "value_num",
    ),
    "holdings_investor_country": (
        "conid",
        "effective_at",
        "country_code",
        "country",
        "value_num",
    ),
    "holdings_geographic_weights": ("conid", "effective_at", "region", "value_num"),
    "holdings_debt_type": ("conid", "effective_at", "debt_type", "value_num"),
    "holdings_top10": ("conid", "effective_at", "name", "holding_weight_num"),
    "ratios_key_ratios": (
        "conid",
        "effective_at",
        "metric_id",
        "value_num",
        "vs_num",
    ),
    "ratios_financials": (
        "conid",
        "effective_at",
        "metric_id",
        "value_num",
        "vs_num",
    ),
    "ratios_fixed_income": (
        "conid",
        "effective_at",
        "metric_id",
        "value_num",
        "vs_num",
    ),
    "ratios_dividend": (
        "conid",
        "effective_at",
        "metric_id",
        "value_num",
        "vs_num",
    ),
    "ratios_zscore": (
        "conid",
        "effective_at",
        "metric_id",
        "value_num",
        "vs_num",
    ),
    "dividends_industry_metrics": (
        "conid",
        "effective_at",
        "metric_id",
        "value_num",
        "currency",
    ),
    "morningstar_summary": (
        "conid",
        "effective_at",
        "metric_id",
        "value_text",
        "value_num",
    ),
    "lipper_ratings": (
        "conid",
        "effective_at",
        "period",
        "metric_id",
        "value_num",
        "rating_label",
        "universe_name",
        "universe_as_of_date",
    ),
}

SNAPSHOT_TABLE_DATE_COLUMNS: dict[str, tuple[str, ...]] = {
    "profile_and_fees": ("effective_at", "value_date"),
    "holdings_asset_type": ("effective_at",),
    "holdings_debtor_quality": ("effective_at",),
    "holdings_maturity": ("effective_at",),
    "holdings_industry": ("effective_at",),
    "holdings_currency": ("effective_at",),
    "holdings_investor_country": ("effective_at",),
    "holdings_geographic_weights": ("effective_at",),
    "holdings_debt_type": ("effective_at",),
    "holdings_top10": ("effective_at",),
    "ratios_key_ratios": ("effective_at",),
    "ratios_financials": ("effective_at",),
    "ratios_fixed_income": ("effective_at",),
    "ratios_dividend": ("effective_at",),
    "ratios_zscore": ("effective_at",),
    "dividends_industry_metrics": ("effective_at",),
    "morningstar_summary": ("effective_at",),
    "lipper_ratings": ("effective_at", "universe_as_of_date"),
}

SNAPSHOT_TABLE_NUMERIC_COLUMNS: dict[str, tuple[str, ...]] = {
    "profile_and_fees": ("value_num", "value_bool"),
    "holdings_asset_type": ("value_num",),
    "holdings_debtor_quality": ("value_num",),
    "holdings_maturity": ("value_num",),
    "holdings_industry": ("value_num",),
    "holdings_currency": ("value_num",),
    "holdings_investor_country": ("value_num",),
    "holdings_geographic_weights": ("value_num",),
    "holdings_debt_type": ("value_num",),
    "holdings_top10": ("holding_weight_num",),
    "ratios_key_ratios": ("value_num", "vs_num"),
    "ratios_financials": ("value_num", "vs_num"),
    "ratios_fixed_income": ("value_num", "vs_num"),
    "ratios_dividend": ("value_num", "vs_num"),
    "ratios_zscore": ("value_num", "vs_num"),
    "dividends_industry_metrics": ("value_num",),
    "morningstar_summary": ("value_num",),
    "lipper_ratings": ("value_num",),
}

SNAPSHOT_TABLE_STRING_COLUMNS: dict[str, tuple[str, ...]] = {
    "profile_and_fees": ("conid", "field_id", "value_text"),
    "holdings_asset_type": ("conid", "bucket_id"),
    "holdings_debtor_quality": ("conid", "bucket_id"),
    "holdings_maturity": ("conid", "bucket_id"),
    "holdings_industry": ("conid", "industry"),
    "holdings_currency": ("conid", "code", "currency"),
    "holdings_investor_country": ("conid", "country_code", "country"),
    "holdings_geographic_weights": ("conid", "region"),
    "holdings_debt_type": ("conid", "debt_type"),
    "holdings_top10": ("conid", "name"),
    "ratios_key_ratios": ("conid", "metric_id"),
    "ratios_financials": ("conid", "metric_id"),
    "ratios_fixed_income": ("conid", "metric_id"),
    "ratios_dividend": ("conid", "metric_id"),
    "ratios_zscore": ("conid", "metric_id"),
    "dividends_industry_metrics": ("conid", "metric_id", "currency"),
    "morningstar_summary": ("conid", "metric_id", "value_text"),
    "lipper_ratings": (
        "conid",
        "period",
        "metric_id",
        "rating_label",
        "universe_name",
    ),
}

SNAPSHOT_TABLE_SORT_COLUMNS: dict[str, tuple[str, ...]] = {
    "profile_and_fees": ("conid", "effective_at", "field_id"),
    "holdings_asset_type": ("conid", "effective_at", "bucket_id"),
    "holdings_debtor_quality": ("conid", "effective_at", "bucket_id"),
    "holdings_maturity": ("conid", "effective_at", "bucket_id"),
    "holdings_industry": ("conid", "effective_at", "industry"),
    "holdings_currency": ("conid", "effective_at", "code", "currency"),
    "holdings_investor_country": ("conid", "effective_at", "country_code", "country"),
    "holdings_geographic_weights": ("conid", "effective_at", "region"),
    "holdings_debt_type": ("conid", "effective_at", "debt_type"),
    "holdings_top10": ("conid", "effective_at", "name"),
    "ratios_key_ratios": ("conid", "effective_at", "metric_id"),
    "ratios_financials": ("conid", "effective_at", "metric_id"),
    "ratios_fixed_income": ("conid", "effective_at", "metric_id"),
    "ratios_dividend": ("conid", "effective_at", "metric_id"),
    "ratios_zscore": ("conid", "effective_at", "metric_id"),
    "dividends_industry_metrics": ("conid", "effective_at", "metric_id"),
    "morningstar_summary": ("conid", "effective_at", "metric_id"),
    "lipper_ratings": (
        "conid",
        "effective_at",
        "universe_name",
        "period",
        "metric_id",
    ),
}


def _empty_frame(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        {column: pd.Series(dtype="object") for column in columns}
    ).reindex(columns=pd.Index(columns))


def _require_columns(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    contract_name: str,
) -> None:
    missing_columns = [column for column in columns if column not in frame.columns]
    if missing_columns and not frame.empty:
        missing = ", ".join(missing_columns)
        raise ValueError(f"{contract_name} is missing required columns: {missing}")


def _normalize_frame(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    contract_name: str,
    date_columns: tuple[str, ...] = (),
    numeric_columns: tuple[str, ...] = (),
    string_columns: tuple[str, ...] = (),
    sort_by: tuple[str, ...] = (),
) -> pd.DataFrame:
    _require_columns(frame, columns=columns, contract_name=contract_name)
    normalized = frame.reindex(columns=pd.Index(columns)).copy()
    if normalized.empty:
        return _empty_frame(columns)

    for column in string_columns:
        if column in normalized.columns:
            normalized[column] = normalized[column].astype(str)
    for column in date_columns:
        if column in normalized.columns:
            normalized[column] = pd.to_datetime(normalized[column], errors="coerce")
    for column in numeric_columns:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    if sort_by:
        normalized = normalized.sort_values(list(sort_by)).reset_index(drop=True)
    return normalized


def _normalize_snapshot_table(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    columns = SNAPSHOT_TABLE_COLUMNS[name]
    return _normalize_frame(
        frame,
        columns=columns,
        contract_name=f"snapshot table {name}",
        date_columns=SNAPSHOT_TABLE_DATE_COLUMNS[name],
        numeric_columns=SNAPSHOT_TABLE_NUMERIC_COLUMNS[name],
        string_columns=SNAPSHOT_TABLE_STRING_COLUMNS[name],
        sort_by=SNAPSHOT_TABLE_SORT_COLUMNS[name],
    )


@dataclass(frozen=True, slots=True)
class PriceHistoryRead:
    """Consumer-oriented canonical price history contract."""

    frame: pd.DataFrame

    @classmethod
    def empty(cls) -> PriceHistoryRead:
        return cls(frame=_empty_frame(PRICE_HISTORY_COLUMNS))

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> PriceHistoryRead:
        return cls(
            frame=_normalize_frame(
                frame,
                columns=PRICE_HISTORY_COLUMNS,
                contract_name="price history",
                date_columns=("trade_date",),
                numeric_columns=("price", "open", "high", "low", "close"),
                string_columns=("conid",),
                sort_by=("conid", "trade_date"),
            )
        )


@dataclass(frozen=True, slots=True)
class DividendEventsRead:
    """Consumer-oriented canonical dividend history contract."""

    frame: pd.DataFrame

    @classmethod
    def empty(cls) -> DividendEventsRead:
        return cls(frame=_empty_frame(DIVIDEND_EVENTS_COLUMNS))

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> DividendEventsRead:
        return cls(
            frame=_normalize_frame(
                frame,
                columns=DIVIDEND_EVENTS_COLUMNS,
                contract_name="dividend events",
                date_columns=(
                    "event_date",
                    "declaration_date",
                    "record_date",
                    "payment_date",
                ),
                numeric_columns=("amount",),
                string_columns=("conid", "symbol"),
                sort_by=("conid", "event_date"),
            )
        )


@dataclass(frozen=True, slots=True)
class RiskFreeDailyRead:
    """Consumer-oriented daily risk-free contract."""

    frame: pd.DataFrame

    @classmethod
    def empty(cls) -> RiskFreeDailyRead:
        return cls(frame=_empty_frame(RISK_FREE_DAILY_COLUMNS))

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> RiskFreeDailyRead:
        return cls(
            frame=_normalize_frame(
                frame,
                columns=RISK_FREE_DAILY_COLUMNS,
                contract_name="risk free daily",
                date_columns=("trade_date", "observed_at"),
                numeric_columns=("nominal_rate", "daily_nominal_rate", "source_count"),
                sort_by=("trade_date",),
            )
        )


@dataclass(frozen=True, slots=True)
class WorldBankCountryFeaturesRead:
    """Consumer-oriented supplementary macro feature contract."""

    frame: pd.DataFrame

    @classmethod
    def empty(cls) -> WorldBankCountryFeaturesRead:
        return cls(frame=_empty_frame(WORLD_BANK_COUNTRY_FEATURE_COLUMNS))

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> WorldBankCountryFeaturesRead:
        numeric_columns = tuple(
            column
            for column in WORLD_BANK_COUNTRY_FEATURE_COLUMNS
            if column not in {"economy_code", "effective_at", "observed_at"}
        )
        normalized = _normalize_frame(
            frame,
            columns=WORLD_BANK_COUNTRY_FEATURE_COLUMNS,
            contract_name="world bank country features",
            date_columns=("effective_at", "observed_at"),
            numeric_columns=numeric_columns,
            string_columns=("economy_code",),
            sort_by=("economy_code", "effective_at"),
        )
        if not normalized.empty:
            normalized["economy_code"] = normalized["economy_code"].str.upper()
        return cls(frame=normalized)


@dataclass(frozen=True, slots=True)
class SnapshotFeatureTablesRead:
    """Consumer-oriented canonical snapshot table contract."""

    tables: dict[str, pd.DataFrame]

    @classmethod
    def empty(cls) -> SnapshotFeatureTablesRead:
        return cls(
            tables={
                name: _empty_frame(columns)
                for name, columns in SNAPSHOT_TABLE_COLUMNS.items()
            }
        )

    @classmethod
    def from_tables(
        cls, tables: Mapping[str, pd.DataFrame] | None
    ) -> SnapshotFeatureTablesRead:
        supplied_tables = {} if tables is None else dict(tables)
        unknown_tables = sorted(set(supplied_tables) - set(SNAPSHOT_TABLE_COLUMNS))
        if unknown_tables:
            unknown = ", ".join(unknown_tables)
            raise ValueError(f"unknown snapshot tables: {unknown}")

        normalized_tables: dict[str, pd.DataFrame] = {}
        for name in SNAPSHOT_TABLE_COLUMNS:
            frame = supplied_tables.get(
                name, _empty_frame(SNAPSHOT_TABLE_COLUMNS[name])
            )
            normalized_tables[name] = _normalize_snapshot_table(name, frame)
        return cls(tables=normalized_tables)


def _query_frame(
    conn: sqlite3.Connection,
    query: str,
    params: tuple[object, ...] = (),
) -> pd.DataFrame:
    return pd.read_sql_query(query, conn, params=list(params) if params else None)


def load_price_history(conn: sqlite3.Connection) -> PriceHistoryRead:
    frame = _query_frame(
        conn,
        """
        SELECT
            conid,
            effective_at AS trade_date,
            price,
            open,
            high,
            low,
            close
        FROM price_chart_series
        ORDER BY conid, effective_at
        """,
    )
    return PriceHistoryRead.from_frame(frame)


def load_latest_price_effective_at_by_conid(
    conn: sqlite3.Connection,
    *,
    conids: Collection[str] | None = None,
) -> dict[str, date | None]:
    query = """
        SELECT
            conid,
            MAX(effective_at) AS latest_effective_at
        FROM price_chart_series
    """
    params: tuple[object, ...] = ()
    requested = [str(conid) for conid in conids] if conids is not None else None
    if requested:
        placeholders = ", ".join("?" for _ in requested)
        query += f" WHERE conid IN ({placeholders})"
        params = tuple(requested)
    query += " GROUP BY conid ORDER BY conid"

    rows = conn.execute(query, params).fetchall()
    latest_by_conid: dict[str, date | None] = {}
    for row in rows:
        latest_effective_at = row["latest_effective_at"]
        if latest_effective_at is None:
            continue
        parsed = pd.to_datetime(latest_effective_at, errors="coerce")
        latest_by_conid[str(row["conid"])] = None if pd.isna(parsed) else parsed.date()
    if requested is None:
        return latest_by_conid
    return {conid: latest_by_conid.get(conid) for conid in requested}


def load_dividend_events(conn: sqlite3.Connection) -> DividendEventsRead:
    frame = _query_frame(
        conn,
        """
        SELECT
            d.conid,
            i.symbol,
            d.effective_at AS event_date,
            d.amount,
            d.currency AS dividend_currency,
            i.currency AS product_currency,
            d.description,
            d.event_type,
            d.declaration_date,
            d.record_date,
            d.payment_date
        FROM dividends_events_series AS d
        LEFT JOIN universe_instruments AS i
            ON i.conid = d.conid
        ORDER BY d.conid, d.effective_at
        """,
    )
    return DividendEventsRead.from_frame(frame)


def load_risk_free_daily(conn: sqlite3.Connection) -> RiskFreeDailyRead:
    frame = _query_frame(
        conn,
        """
        SELECT
            trade_date,
            nominal_rate,
            daily_nominal_rate,
            source_count,
            observed_at
        FROM supplementary_risk_free_daily
        ORDER BY trade_date
        """,
    )
    return RiskFreeDailyRead.from_frame(frame)


def load_world_bank_country_features(
    conn: sqlite3.Connection,
) -> WorldBankCountryFeaturesRead:
    frame = _query_frame(
        conn,
        """
        SELECT *
        FROM supplementary_world_bank_country_features
        ORDER BY economy_code, effective_at
        """,
    )
    normalized = frame.reindex(columns=pd.Index(WORLD_BANK_COUNTRY_FEATURE_COLUMNS))
    return WorldBankCountryFeaturesRead.from_frame(normalized)


def _load_snapshot_frame_from_db(
    conn: sqlite3.Connection,
    query: str,
    *,
    name: str,
) -> pd.DataFrame:
    try:
        frame = _query_frame(conn, query)
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return _empty_frame(SNAPSHOT_TABLE_COLUMNS[name])
    return _normalize_snapshot_table(name, frame)


def load_snapshot_feature_tables(conn: sqlite3.Connection) -> SnapshotFeatureTablesRead:
    tables: dict[str, pd.DataFrame] = {
        "profile_and_fees": _load_snapshot_frame_from_db(
            conn,
            """
            SELECT
                conid,
                effective_at,
                field_id,
                value_text,
                value_num,
                value_date,
                value_bool
            FROM profile_and_fees_factors
            ORDER BY conid, effective_at, field_id
            """,
            name="profile_and_fees",
        ),
        "holdings_asset_type": _load_snapshot_frame_from_db(
            conn,
            """
            SELECT
                conid,
                effective_at,
                bucket_id,
                value_num
            FROM holdings_asset_type_factors
            ORDER BY conid, effective_at, bucket_id
            """,
            name="holdings_asset_type",
        ),
        "holdings_debtor_quality": _load_snapshot_frame_from_db(
            conn,
            """
            SELECT
                conid,
                effective_at,
                bucket_id,
                value_num
            FROM holdings_debtor_quality_factors
            ORDER BY conid, effective_at, bucket_id
            """,
            name="holdings_debtor_quality",
        ),
        "holdings_maturity": _load_snapshot_frame_from_db(
            conn,
            """
            SELECT
                conid,
                effective_at,
                bucket_id,
                value_num
            FROM holdings_maturity_factors
            ORDER BY conid, effective_at, bucket_id
            """,
            name="holdings_maturity",
        ),
        "holdings_industry": _load_snapshot_frame_from_db(
            conn,
            """
            SELECT
                conid,
                effective_at,
                industry,
                value_num
            FROM holdings_industry
            ORDER BY conid, effective_at, industry
            """,
            name="holdings_industry",
        ),
        "holdings_currency": _load_snapshot_frame_from_db(
            conn,
            """
            SELECT
                conid,
                effective_at,
                COALESCE(code, currency) AS code,
                currency,
                value_num
            FROM holdings_currency
            ORDER BY conid, effective_at, code, currency
            """,
            name="holdings_currency",
        ),
        "holdings_investor_country": _load_snapshot_frame_from_db(
            conn,
            """
            SELECT
                conid,
                effective_at,
                COALESCE(country_code, country) AS country_code,
                country,
                value_num
            FROM holdings_investor_country
            ORDER BY conid, effective_at, country_code, country
            """,
            name="holdings_investor_country",
        ),
        "holdings_geographic_weights": _load_snapshot_frame_from_db(
            conn,
            """
            SELECT
                conid,
                effective_at,
                region,
                value_num
            FROM holdings_geographic_weights
            ORDER BY conid, effective_at, region
            """,
            name="holdings_geographic_weights",
        ),
        "holdings_debt_type": _load_snapshot_frame_from_db(
            conn,
            """
            SELECT
                conid,
                effective_at,
                debt_type,
                value_num
            FROM holdings_debt_type
            ORDER BY conid, effective_at, debt_type
            """,
            name="holdings_debt_type",
        ),
        "holdings_top10": _load_snapshot_frame_from_db(
            conn,
            """
            SELECT
                conid,
                effective_at,
                name,
                holding_weight_num
            FROM holdings_top10
            ORDER BY conid, effective_at, name
            """,
            name="holdings_top10",
        ),
        "ratios_key_ratios": _load_snapshot_frame_from_db(
            conn,
            """
            SELECT
                conid,
                effective_at,
                metric_id,
                value_num,
                vs_num
            FROM ratios_key_ratios
            ORDER BY conid, effective_at, metric_id
            """,
            name="ratios_key_ratios",
        ),
        "ratios_financials": _load_snapshot_frame_from_db(
            conn,
            """
            SELECT
                conid,
                effective_at,
                metric_id,
                value_num,
                vs_num
            FROM ratios_financials
            ORDER BY conid, effective_at, metric_id
            """,
            name="ratios_financials",
        ),
        "ratios_fixed_income": _load_snapshot_frame_from_db(
            conn,
            """
            SELECT
                conid,
                effective_at,
                metric_id,
                value_num,
                vs_num
            FROM ratios_fixed_income
            ORDER BY conid, effective_at, metric_id
            """,
            name="ratios_fixed_income",
        ),
        "ratios_dividend": _load_snapshot_frame_from_db(
            conn,
            """
            SELECT
                conid,
                effective_at,
                metric_id,
                value_num,
                vs_num
            FROM ratios_dividend
            ORDER BY conid, effective_at, metric_id
            """,
            name="ratios_dividend",
        ),
        "ratios_zscore": _load_snapshot_frame_from_db(
            conn,
            """
            SELECT
                conid,
                effective_at,
                metric_id,
                value_num,
                vs_num
            FROM ratios_zscore
            ORDER BY conid, effective_at, metric_id
            """,
            name="ratios_zscore",
        ),
        "dividends_industry_metrics": _load_snapshot_frame_from_db(
            conn,
            """
            SELECT
                conid,
                effective_at,
                metric_id,
                value_num,
                currency
            FROM dividends_industry_metrics_factors
            ORDER BY conid, effective_at, metric_id
            """,
            name="dividends_industry_metrics",
        ),
        "morningstar_summary": _load_snapshot_frame_from_db(
            conn,
            """
            SELECT
                conid,
                effective_at,
                metric_id,
                value_text,
                value_num
            FROM morningstar_summary_factors
            ORDER BY conid, effective_at, metric_id
            """,
            name="morningstar_summary",
        ),
        "lipper_ratings": _load_snapshot_frame_from_db(
            conn,
            """
            SELECT
                conid,
                effective_at,
                period,
                metric_id,
                value_num,
                rating_label,
                universe_name,
                universe_as_of_date
            FROM lipper_ratings
            ORDER BY conid, effective_at, universe_name, period, metric_id
            """,
            name="lipper_ratings",
        ),
    }
    return SnapshotFeatureTablesRead.from_tables(tables)
