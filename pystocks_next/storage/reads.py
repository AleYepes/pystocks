from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass

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
        "asset_type",
        "classification",
        "distribution_details",
        "domicile",
        "fiscal_date",
        "fund_category",
        "fund_management_company",
        "fund_manager_benchmark",
        "fund_market_cap_focus",
        "geographical_focus",
        "inception_date",
        "management_approach",
        "management_expenses",
        "manager_tenure",
        "maturity_date",
        "objective_type",
        "portfolio_manager",
        "redemption_charge_actual",
        "redemption_charge_max",
        "scheme",
        "total_expense_ratio",
        "total_net_assets_value",
        "total_net_assets_date",
        "objective",
        "jap_fund_warning",
        "theme_name",
    ),
    "holdings_asset_type": (
        "conid",
        "effective_at",
        "equity",
        "cash",
        "fixed_income",
        "other",
    ),
    "holdings_debtor_quality": (
        "conid",
        "effective_at",
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
    "holdings_maturity": (
        "conid",
        "effective_at",
        "maturity_less_than_1_year",
        "maturity_1_to_3_years",
        "maturity_3_to_5_years",
        "maturity_5_to_10_years",
        "maturity_10_to_20_years",
        "maturity_20_to_30_years",
        "maturity_greater_than_30_years",
        "maturity_other",
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
        "dividend_yield",
        "annual_dividend",
        "dividend_ttm",
        "dividend_yield_ttm",
        "currency",
    ),
    "morningstar_summary": (
        "conid",
        "effective_at",
        "medalist_rating",
        "process",
        "people",
        "parent",
        "morningstar_rating",
        "sustainability_rating",
        "category",
        "category_index",
    ),
    "lipper_ratings": ("conid", "effective_at", "period", "metric_id", "value_num"),
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
    numeric_columns = tuple(
        column
        for column in columns
        if column
        not in {
            "conid",
            "effective_at",
            "asset_type",
            "classification",
            "distribution_details",
            "domicile",
            "fiscal_date",
            "fund_category",
            "fund_management_company",
            "fund_manager_benchmark",
            "fund_market_cap_focus",
            "geographical_focus",
            "inception_date",
            "management_approach",
            "maturity_date",
            "objective_type",
            "portfolio_manager",
            "scheme",
            "total_net_assets_value",
            "total_net_assets_date",
            "objective",
            "jap_fund_warning",
            "theme_name",
            "industry",
            "code",
            "currency",
            "country_code",
            "country",
            "region",
            "debt_type",
            "name",
            "metric_id",
            "period",
            "medalist_rating",
            "process",
            "people",
            "parent",
            "morningstar_rating",
            "sustainability_rating",
            "category",
        }
    )
    return _normalize_frame(
        frame,
        columns=columns,
        contract_name=f"snapshot table {name}",
        date_columns=("effective_at",),
        numeric_columns=numeric_columns,
        string_columns=("conid",),
        sort_by=("conid", "effective_at"),
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
            FROM profile_and_fees
            ORDER BY conid, effective_at
            """,
            name="profile_and_fees",
        ),
        "holdings_asset_type": _load_snapshot_frame_from_db(
            conn,
            """
            SELECT
                conid,
                effective_at,
                equity,
                cash,
                fixed_income,
                other
            FROM holdings_asset_type
            ORDER BY conid, effective_at
            """,
            name="holdings_asset_type",
        ),
    }
    return SnapshotFeatureTablesRead.from_tables(tables)
