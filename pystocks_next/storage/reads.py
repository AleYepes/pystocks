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

RISK_FREE_SOURCE_COLUMNS: tuple[str, ...] = (
    "series_id",
    "source_name",
    "economy_code",
    "trade_date",
    "nominal_rate",
    "observed_at",
)

WORLD_BANK_RAW_COLUMNS: tuple[str, ...] = (
    "economy_code",
    "indicator_id",
    "year",
    "value",
    "observed_at",
)

HOLDINGS_COUNTRY_WEIGHT_COLUMNS: tuple[str, ...] = (
    "economy_code",
    "weight",
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

SNAPSHOT_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "profile_overview": (
        "conid",
        "effective_at",
        "symbol",
        "objective",
        "jap_fund_warning",
    ),
    "profile_fields": (
        "conid",
        "effective_at",
        "field_id",
        "value_text",
        "value_num",
        "value_date",
    ),
    "profile_reports": (
        "conid",
        "effective_at",
        "report_id",
        "report_as_of_date",
    ),
    "profile_report_fields": (
        "conid",
        "effective_at",
        "report_id",
        "field_id",
        "value_text",
        "value_num",
        "value_date",
        "is_summary",
    ),
    "profile_themes": (
        "conid",
        "effective_at",
        "theme_id",
    ),
    "profile_expense_allocations": (
        "conid",
        "effective_at",
        "expense_id",
        "value_text",
        "ratio",
    ),
    "profile_stylebox": (
        "conid",
        "effective_at",
        "stylebox_id",
        "x_index",
        "y_index",
        "x_label",
        "y_label",
        "x_tag",
        "y_tag",
    ),
    "holdings_asset_type": (
        "conid",
        "effective_at",
        "bucket_id",
        "value_num",
        "vs_peers",
    ),
    "holdings_debtor_quality": (
        "conid",
        "effective_at",
        "bucket_id",
        "value_num",
        "vs_peers",
    ),
    "holdings_maturity": (
        "conid",
        "effective_at",
        "bucket_id",
        "value_num",
        "vs_peers",
    ),
    "holdings_industry": (
        "conid",
        "effective_at",
        "industry",
        "value_num",
        "vs_peers",
    ),
    "holdings_currency": (
        "conid",
        "effective_at",
        "code",
        "currency",
        "value_num",
        "vs_peers",
    ),
    "holdings_investor_country": (
        "conid",
        "effective_at",
        "country_code",
        "country",
        "value_num",
        "vs_peers",
    ),
    "holdings_geographic_weights": (
        "conid",
        "effective_at",
        "region",
        "value_num",
        "vs_peers",
    ),
    "holdings_debt_type": (
        "conid",
        "effective_at",
        "debt_type",
        "value_num",
        "vs_peers",
    ),
    "holdings_top10": (
        "conid",
        "effective_at",
        "name",
        "ticker",
        "rank",
        "holding_weight_num",
        "vs_peers",
        "conids_json",
    ),
    "ratios_key_ratios": (
        "conid",
        "effective_at",
        "metric_id",
        "value_num",
        "vs_peers",
    ),
    "ratios_financials": (
        "conid",
        "effective_at",
        "metric_id",
        "value_num",
        "vs_peers",
    ),
    "ratios_fixed_income": (
        "conid",
        "effective_at",
        "metric_id",
        "value_num",
        "vs_peers",
    ),
    "ratios_dividend": (
        "conid",
        "effective_at",
        "metric_id",
        "value_num",
        "vs_peers",
    ),
    "ratios_zscore": (
        "conid",
        "effective_at",
        "metric_id",
        "value_num",
        "vs_peers",
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
        "title",
        "derived_quantitatively",
        "publish_date",
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
    "profile_overview": ("effective_at",),
    "profile_fields": ("effective_at", "value_date"),
    "profile_reports": ("effective_at", "report_as_of_date"),
    "profile_report_fields": ("effective_at", "value_date"),
    "profile_themes": ("effective_at",),
    "profile_expense_allocations": ("effective_at",),
    "profile_stylebox": ("effective_at",),
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
    "morningstar_summary": ("effective_at", "publish_date"),
    "lipper_ratings": ("effective_at", "universe_as_of_date"),
}

SNAPSHOT_TABLE_NUMERIC_COLUMNS: dict[str, tuple[str, ...]] = {
    "profile_overview": ("jap_fund_warning",),
    "profile_fields": ("value_num",),
    "profile_reports": (),
    "profile_report_fields": (
        "value_num",
        "is_summary",
    ),
    "profile_themes": (),
    "profile_expense_allocations": ("ratio",),
    "profile_stylebox": ("x_index", "y_index"),
    "holdings_asset_type": ("value_num", "vs_peers"),
    "holdings_debtor_quality": ("value_num", "vs_peers"),
    "holdings_maturity": ("value_num", "vs_peers"),
    "holdings_industry": ("value_num", "vs_peers"),
    "holdings_currency": ("value_num", "vs_peers"),
    "holdings_investor_country": ("value_num", "vs_peers"),
    "holdings_geographic_weights": ("value_num", "vs_peers"),
    "holdings_debt_type": ("value_num", "vs_peers"),
    "holdings_top10": ("rank", "holding_weight_num", "vs_peers"),
    "ratios_key_ratios": ("value_num", "vs_peers"),
    "ratios_financials": ("value_num", "vs_peers"),
    "ratios_fixed_income": ("value_num", "vs_peers"),
    "ratios_dividend": ("value_num", "vs_peers"),
    "ratios_zscore": ("value_num", "vs_peers"),
    "dividends_industry_metrics": ("value_num",),
    "morningstar_summary": ("derived_quantitatively", "value_num"),
    "lipper_ratings": ("value_num",),
}

SNAPSHOT_TABLE_STRING_COLUMNS: dict[str, tuple[str, ...]] = {
    "profile_overview": ("conid", "symbol", "objective"),
    "profile_fields": (
        "conid",
        "field_id",
        "value_text",
    ),
    "profile_reports": ("conid", "report_id"),
    "profile_report_fields": (
        "conid",
        "report_id",
        "field_id",
        "value_text",
    ),
    "profile_themes": ("conid", "theme_id"),
    "profile_expense_allocations": (
        "conid",
        "expense_id",
        "value_text",
    ),
    "profile_stylebox": (
        "conid",
        "stylebox_id",
        "x_label",
        "y_label",
        "x_tag",
        "y_tag",
    ),
    "holdings_asset_type": ("conid", "bucket_id"),
    "holdings_debtor_quality": ("conid", "bucket_id"),
    "holdings_maturity": ("conid", "bucket_id"),
    "holdings_industry": ("conid", "industry"),
    "holdings_currency": ("conid", "code", "currency"),
    "holdings_investor_country": ("conid", "country_code", "country"),
    "holdings_geographic_weights": ("conid", "region"),
    "holdings_debt_type": ("conid", "debt_type"),
    "holdings_top10": ("conid", "name", "ticker", "conids_json"),
    "ratios_key_ratios": ("conid", "metric_id"),
    "ratios_financials": ("conid", "metric_id"),
    "ratios_fixed_income": ("conid", "metric_id"),
    "ratios_dividend": ("conid", "metric_id"),
    "ratios_zscore": ("conid", "metric_id"),
    "dividends_industry_metrics": ("conid", "metric_id", "currency"),
    "morningstar_summary": ("conid", "metric_id", "title", "value_text"),
    "lipper_ratings": (
        "conid",
        "period",
        "metric_id",
        "rating_label",
        "universe_name",
    ),
}

SNAPSHOT_TABLE_SORT_COLUMNS: dict[str, tuple[str, ...]] = {
    "profile_overview": ("conid", "effective_at"),
    "profile_fields": ("conid", "effective_at", "field_id"),
    "profile_reports": ("conid", "effective_at", "report_id"),
    "profile_report_fields": (
        "conid",
        "effective_at",
        "report_id",
        "field_id",
    ),
    "profile_themes": ("conid", "effective_at", "theme_id"),
    "profile_expense_allocations": (
        "conid",
        "effective_at",
        "expense_id",
    ),
    "profile_stylebox": ("conid", "effective_at", "stylebox_id"),
    "holdings_asset_type": ("conid", "effective_at", "bucket_id"),
    "holdings_debtor_quality": ("conid", "effective_at", "bucket_id"),
    "holdings_maturity": ("conid", "effective_at", "bucket_id"),
    "holdings_industry": ("conid", "effective_at", "industry"),
    "holdings_currency": ("conid", "effective_at", "code", "currency"),
    "holdings_investor_country": ("conid", "effective_at", "country_code", "country"),
    "holdings_geographic_weights": ("conid", "effective_at", "region"),
    "holdings_debt_type": ("conid", "effective_at", "debt_type"),
    "holdings_top10": ("conid", "effective_at", "rank", "name"),
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
class RiskFreeSourcesRead:
    """Consumer-oriented raw supplementary risk-free source contract."""

    frame: pd.DataFrame

    @classmethod
    def empty(cls) -> RiskFreeSourcesRead:
        return cls(frame=_empty_frame(RISK_FREE_SOURCE_COLUMNS))

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> RiskFreeSourcesRead:
        normalized = _normalize_frame(
            frame,
            columns=RISK_FREE_SOURCE_COLUMNS,
            contract_name="risk free sources",
            date_columns=("trade_date", "observed_at"),
            numeric_columns=("nominal_rate",),
            string_columns=("series_id", "source_name", "economy_code"),
            sort_by=("series_id", "trade_date"),
        )
        if not normalized.empty:
            normalized["economy_code"] = normalized["economy_code"].str.upper()
        return cls(frame=normalized)


@dataclass(frozen=True, slots=True)
class WorldBankRawRead:
    """Consumer-oriented raw supplementary World Bank contract."""

    frame: pd.DataFrame

    @classmethod
    def empty(cls) -> WorldBankRawRead:
        return cls(frame=_empty_frame(WORLD_BANK_RAW_COLUMNS))

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> WorldBankRawRead:
        normalized = _normalize_frame(
            frame,
            columns=WORLD_BANK_RAW_COLUMNS,
            contract_name="world bank raw",
            date_columns=("observed_at",),
            numeric_columns=("year", "value"),
            string_columns=("economy_code", "indicator_id"),
            sort_by=("economy_code", "indicator_id", "year"),
        )
        if not normalized.empty:
            normalized["economy_code"] = normalized["economy_code"].str.upper()
            normalized["year"] = normalized["year"].astype("Int64")
        return cls(frame=normalized)


@dataclass(frozen=True, slots=True)
class HoldingsCountryWeightsRead:
    """Consumer-oriented latest holdings-country weight contract."""

    frame: pd.DataFrame

    @classmethod
    def empty(cls) -> HoldingsCountryWeightsRead:
        return cls(frame=_empty_frame(HOLDINGS_COUNTRY_WEIGHT_COLUMNS))

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> HoldingsCountryWeightsRead:
        normalized = _normalize_frame(
            frame,
            columns=HOLDINGS_COUNTRY_WEIGHT_COLUMNS,
            contract_name="holdings country weights",
            numeric_columns=("weight",),
            string_columns=("economy_code",),
            sort_by=("economy_code",),
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


def load_risk_free_sources(conn: sqlite3.Connection) -> RiskFreeSourcesRead:
    frame = _query_frame(
        conn,
        """
        SELECT
            series_id,
            source_name,
            economy_code,
            trade_date,
            nominal_rate,
            observed_at
        FROM supplementary_risk_free_sources
        ORDER BY series_id, trade_date
        """,
    )
    return RiskFreeSourcesRead.from_frame(frame)


def load_world_bank_raw(conn: sqlite3.Connection) -> WorldBankRawRead:
    frame = _query_frame(
        conn,
        """
        SELECT
            economy_code,
            indicator_id,
            year,
            value,
            observed_at
        FROM supplementary_world_bank_raw
        ORDER BY economy_code, indicator_id, year
        """,
    )
    return WorldBankRawRead.from_frame(frame)


def load_latest_holdings_country_weights(
    conn: sqlite3.Connection,
) -> HoldingsCountryWeightsRead:
    frame = _query_frame(
        conn,
        """
        WITH latest_snapshot_by_conid AS (
            SELECT
                conid,
                MAX(effective_at) AS effective_at
            FROM holdings_investor_country
            GROUP BY conid
        )
        SELECT
            UPPER(COALESCE(country_code, country)) AS economy_code,
            SUM(value_num) AS weight
        FROM holdings_investor_country AS h
        INNER JOIN latest_snapshot_by_conid AS latest
            ON latest.conid = h.conid
           AND latest.effective_at = h.effective_at
        WHERE COALESCE(country_code, country) IS NOT NULL
        GROUP BY UPPER(COALESCE(country_code, country))
        ORDER BY UPPER(COALESCE(country_code, country))
        """,
    )
    return HoldingsCountryWeightsRead.from_frame(frame)


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
        "profile_overview": _load_snapshot_frame_from_db(
            conn,
            """
            SELECT
                conid,
                effective_at,
                symbol,
                objective,
                jap_fund_warning
            FROM profile_overview
            ORDER BY conid, effective_at
            """,
            name="profile_overview",
        ),
        "profile_fields": _load_snapshot_frame_from_db(
            conn,
            """
            SELECT
                conid,
                effective_at,
                field_id,
                value_text,
                value_num,
                value_date
            FROM profile_fields
            ORDER BY conid, effective_at, field_id
            """,
            name="profile_fields",
        ),
        "profile_reports": _load_snapshot_frame_from_db(
            conn,
            """
            SELECT
                conid,
                effective_at,
                report_id,
                report_as_of_date
            FROM profile_reports
            ORDER BY conid, effective_at, report_id
            """,
            name="profile_reports",
        ),
        "profile_report_fields": _load_snapshot_frame_from_db(
            conn,
            """
            SELECT
                conid,
                effective_at,
                report_id,
                field_id,
                value_text,
                value_num,
                value_date,
                is_summary
            FROM profile_report_fields
            ORDER BY conid, effective_at, report_id, field_id
            """,
            name="profile_report_fields",
        ),
        "profile_themes": _load_snapshot_frame_from_db(
            conn,
            """
            SELECT
                conid,
                effective_at,
                theme_id
            FROM profile_themes
            ORDER BY conid, effective_at, theme_id
            """,
            name="profile_themes",
        ),
        "profile_expense_allocations": _load_snapshot_frame_from_db(
            conn,
            """
            SELECT
                conid,
                effective_at,
                expense_id,
                value_text,
                ratio
            FROM profile_expense_allocations
            ORDER BY conid, effective_at, expense_id
            """,
            name="profile_expense_allocations",
        ),
        "profile_stylebox": _load_snapshot_frame_from_db(
            conn,
            """
            SELECT
                conid,
                effective_at,
                stylebox_id,
                x_index,
                y_index,
                x_label,
                y_label,
                x_tag,
                y_tag
            FROM profile_stylebox
            ORDER BY conid, effective_at, stylebox_id
            """,
            name="profile_stylebox",
        ),
        "holdings_asset_type": _load_snapshot_frame_from_db(
            conn,
            """
            SELECT
                conid,
                effective_at,
                bucket_id,
                value_num,
                vs_peers
            FROM holdings_asset_type
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
                value_num,
                vs_peers
            FROM holdings_debtor_quality
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
                value_num,
                vs_peers
            FROM holdings_maturity
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
                value_num,
                vs_peers
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
                value_num,
                vs_peers
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
                value_num,
                vs_peers
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
                value_num,
                vs_peers
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
                value_num,
                vs_peers
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
                ticker,
                rank,
                holding_weight_num,
                vs_peers,
                conids_json
            FROM holdings_top10
            ORDER BY conid, effective_at, rank, name
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
                vs_peers
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
                vs_peers
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
                vs_peers
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
                vs_peers
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
                vs_peers
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
            FROM dividends_industry_metrics
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
                title,
                derived_quantitatively,
                publish_date,
                value_text,
                value_num
            FROM morningstar_summary
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
