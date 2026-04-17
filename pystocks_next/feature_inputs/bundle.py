from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

PRICE_INPUT_COLUMNS: tuple[str, ...] = (
    "conid",
    "trade_date",
    "price_value",
    "clean_price",
    "raw_return",
    "clean_return",
    "is_valid_price",
    "is_stale_price",
    "is_outlier_return",
    "is_price_level_anomaly",
    "is_clean_price",
)

PRICE_ELIGIBILITY_COLUMNS: tuple[str, ...] = (
    "conid",
    "total_rows",
    "valid_rows",
    "min_date",
    "max_date",
    "expected_business_days",
    "missing_ratio",
    "max_internal_gap_days",
    "eligible",
    "eligibility_reason",
)

DIVIDEND_EVENT_COLUMNS: tuple[str, ...] = (
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
    "previous_price_date",
    "previous_clean_price",
    "price_reference_age_days",
    "implied_yield_vs_previous_price",
    "trailing_dividend_sum_365d",
    "is_missing_amount",
    "is_nonpositive_amount",
    "is_missing_currency",
    "is_currency_mismatch",
    "is_duplicate_event_signature",
    "is_missing_price_reference",
    "is_stale_price_reference",
    "is_suspicious_implied_yield",
    "usable_for_total_return_adjustment",
)

DIVIDEND_SUMMARY_COLUMNS: tuple[str, ...] = (
    "conid",
    "symbol",
    "product_currency",
    "event_rows",
    "usable_rows",
    "duplicate_rows",
    "currency_mismatch_rows",
    "missing_currency_rows",
    "suspicious_yield_rows",
    "missing_price_reference_rows",
    "min_event_date",
    "max_event_date",
    "usable_ratio",
)

SNAPSHOT_FEATURE_REQUIRED_COLUMNS: tuple[str, ...] = ("conid", "effective_at", "sleeve")

SNAPSHOT_HOLDINGS_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "conid",
    "effective_at",
    "table_name",
    "value_sum",
    "category_count",
    "max_value",
    "is_sum_near_one",
    "is_sum_over_one",
    "is_sparse_category_coverage",
)

SNAPSHOT_RATIO_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "conid",
    "effective_at",
    "table_name",
    "metric_rows",
    "distinct_metric_keys",
    "duplicate_metric_keys",
    "duplicate_row_count",
    "nonnull_value_rows",
    "null_value_rows",
    "all_values_null",
)

SNAPSHOT_TABLE_SUMMARY_COLUMNS: tuple[str, ...] = (
    "table_name",
    "row_count",
    "key_count",
    "conid_count",
    "min_effective_at",
    "max_effective_at",
)

RISK_FREE_DAILY_COLUMNS: tuple[str, ...] = (
    "trade_date",
    "nominal_rate",
    "daily_nominal_rate",
    "source_count",
    "observed_at",
)

MACRO_FEATURE_COLUMNS: tuple[str, ...] = (
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


def _empty_frame(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        {column: pd.Series(dtype="object") for column in columns}
    ).reindex(columns=pd.Index(columns))


def _normalize_exact_frame(
    frame: pd.DataFrame | None,
    *,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    source = _empty_frame(columns) if frame is None else frame.copy()
    missing_columns = [column for column in columns if column not in source.columns]
    if missing_columns and not source.empty:
        missing = ", ".join(missing_columns)
        raise ValueError(f"frame is missing required columns: {missing}")
    return source.reindex(columns=pd.Index(columns)).copy()


def _normalize_minimum_frame(
    frame: pd.DataFrame | None,
    *,
    required_columns: tuple[str, ...],
) -> pd.DataFrame:
    if frame is None:
        return _empty_frame(required_columns)

    source = frame.copy()
    missing_columns = [
        column for column in required_columns if column not in source.columns
    ]
    if missing_columns and not source.empty:
        missing = ", ".join(missing_columns)
        raise ValueError(f"frame is missing required columns: {missing}")
    if source.empty:
        return _empty_frame(required_columns)

    ordered_columns = list(required_columns) + [
        column for column in source.columns if column not in required_columns
    ]
    return source.loc[:, ordered_columns].copy()


@dataclass(frozen=True, slots=True)
class AnalysisInputBundle:
    """Stable analysis-facing inputs owned by the feature-input stage."""

    prices: pd.DataFrame = field(
        default_factory=lambda: _empty_frame(PRICE_INPUT_COLUMNS)
    )
    price_eligibility: pd.DataFrame = field(
        default_factory=lambda: _empty_frame(PRICE_ELIGIBILITY_COLUMNS)
    )
    dividends: pd.DataFrame = field(
        default_factory=lambda: _empty_frame(DIVIDEND_EVENT_COLUMNS)
    )
    dividend_summary: pd.DataFrame = field(
        default_factory=lambda: _empty_frame(DIVIDEND_SUMMARY_COLUMNS)
    )
    snapshot_features: pd.DataFrame = field(
        default_factory=lambda: _empty_frame(SNAPSHOT_FEATURE_REQUIRED_COLUMNS)
    )
    snapshot_holdings_diagnostics: pd.DataFrame = field(
        default_factory=lambda: _empty_frame(SNAPSHOT_HOLDINGS_DIAGNOSTIC_COLUMNS)
    )
    snapshot_ratio_diagnostics: pd.DataFrame = field(
        default_factory=lambda: _empty_frame(SNAPSHOT_RATIO_DIAGNOSTIC_COLUMNS)
    )
    snapshot_table_summary: pd.DataFrame = field(
        default_factory=lambda: _empty_frame(SNAPSHOT_TABLE_SUMMARY_COLUMNS)
    )
    risk_free_daily: pd.DataFrame = field(
        default_factory=lambda: _empty_frame(RISK_FREE_DAILY_COLUMNS)
    )
    macro_features: pd.DataFrame = field(
        default_factory=lambda: _empty_frame(MACRO_FEATURE_COLUMNS)
    )

    @classmethod
    def empty(cls) -> AnalysisInputBundle:
        return cls()

    @classmethod
    def from_frames(
        cls,
        *,
        prices: pd.DataFrame | None = None,
        price_eligibility: pd.DataFrame | None = None,
        dividends: pd.DataFrame | None = None,
        dividend_summary: pd.DataFrame | None = None,
        snapshot_features: pd.DataFrame | None = None,
        snapshot_holdings_diagnostics: pd.DataFrame | None = None,
        snapshot_ratio_diagnostics: pd.DataFrame | None = None,
        snapshot_table_summary: pd.DataFrame | None = None,
        risk_free_daily: pd.DataFrame | None = None,
        macro_features: pd.DataFrame | None = None,
    ) -> AnalysisInputBundle:
        return cls(
            prices=_normalize_exact_frame(prices, columns=PRICE_INPUT_COLUMNS),
            price_eligibility=_normalize_exact_frame(
                price_eligibility, columns=PRICE_ELIGIBILITY_COLUMNS
            ),
            dividends=_normalize_exact_frame(dividends, columns=DIVIDEND_EVENT_COLUMNS),
            dividend_summary=_normalize_exact_frame(
                dividend_summary, columns=DIVIDEND_SUMMARY_COLUMNS
            ),
            snapshot_features=_normalize_minimum_frame(
                snapshot_features,
                required_columns=SNAPSHOT_FEATURE_REQUIRED_COLUMNS,
            ),
            snapshot_holdings_diagnostics=_normalize_exact_frame(
                snapshot_holdings_diagnostics,
                columns=SNAPSHOT_HOLDINGS_DIAGNOSTIC_COLUMNS,
            ),
            snapshot_ratio_diagnostics=_normalize_exact_frame(
                snapshot_ratio_diagnostics,
                columns=SNAPSHOT_RATIO_DIAGNOSTIC_COLUMNS,
            ),
            snapshot_table_summary=_normalize_exact_frame(
                snapshot_table_summary,
                columns=SNAPSHOT_TABLE_SUMMARY_COLUMNS,
            ),
            risk_free_daily=_normalize_exact_frame(
                risk_free_daily, columns=RISK_FREE_DAILY_COLUMNS
            ),
            macro_features=_normalize_exact_frame(
                macro_features, columns=MACRO_FEATURE_COLUMNS
            ),
        )
