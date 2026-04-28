from dataclasses import dataclass
from typing import Final, Literal

SnapshotOutputKind = Literal["prefixed", "wide_holdings", "series", "metric", "top10"]


@dataclass(frozen=True)
class SnapshotTableContract:
    name: str
    columns: tuple[str, ...]
    output_kind: SnapshotOutputKind
    prefix: str | None = None
    key_col: str | None = None
    value_col: str | None = None
    key_cols: tuple[str, ...] = ()


SNAPSHOT_SOURCE_CONTRACTS: Final[dict[str, SnapshotTableContract]] = {
    "profile_and_fees": SnapshotTableContract(
        name="profile_and_fees",
        columns=(
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
        output_kind="prefixed",
        prefix="profile",
    ),
    "holdings_asset_type": SnapshotTableContract(
        name="holdings_asset_type",
        columns=("conid", "effective_at", "equity", "cash", "fixed_income", "other"),
        output_kind="wide_holdings",
        prefix="holding_asset",
    ),
    "holdings_debtor_quality": SnapshotTableContract(
        name="holdings_debtor_quality",
        columns=(
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
        output_kind="wide_holdings",
        prefix="holding_quality",
    ),
    "holdings_maturity": SnapshotTableContract(
        name="holdings_maturity",
        columns=(
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
        output_kind="wide_holdings",
        prefix="holding_maturity",
    ),
    "holdings_industry": SnapshotTableContract(
        name="holdings_industry",
        columns=("conid", "effective_at", "industry", "value_num"),
        output_kind="series",
        prefix="industry",
        key_col="industry",
        value_col="value_num",
    ),
    "holdings_currency": SnapshotTableContract(
        name="holdings_currency",
        columns=("conid", "effective_at", "code", "currency", "value_num"),
        output_kind="series",
        prefix="currency",
        key_col="code",
        value_col="value_num",
    ),
    "holdings_investor_country": SnapshotTableContract(
        name="holdings_investor_country",
        columns=("conid", "effective_at", "country_code", "country", "value_num"),
        output_kind="series",
        prefix="country",
        key_col="country_code",
        value_col="value_num",
    ),
    "holdings_geographic_weights": SnapshotTableContract(
        name="holdings_geographic_weights",
        columns=("conid", "effective_at", "region", "value_num"),
        output_kind="series",
        prefix="region",
        key_col="region",
        value_col="value_num",
    ),
    "holdings_debt_type": SnapshotTableContract(
        name="holdings_debt_type",
        columns=("conid", "effective_at", "debt_type", "value_num"),
        output_kind="series",
        prefix="debt_type",
        key_col="debt_type",
        value_col="value_num",
    ),
    "holdings_top10": SnapshotTableContract(
        name="holdings_top10",
        columns=("conid", "effective_at", "name", "holding_weight_num"),
        output_kind="top10",
        prefix="top10",
    ),
    "ratios_key_ratios": SnapshotTableContract(
        name="ratios_key_ratios",
        columns=("conid", "effective_at", "metric_id", "value_num", "vs_num"),
        output_kind="metric",
        prefix="ratio_key",
        key_cols=("metric_id",),
    ),
    "ratios_financials": SnapshotTableContract(
        name="ratios_financials",
        columns=("conid", "effective_at", "metric_id", "value_num", "vs_num"),
        output_kind="metric",
        prefix="ratio_financial",
        key_cols=("metric_id",),
    ),
    "ratios_fixed_income": SnapshotTableContract(
        name="ratios_fixed_income",
        columns=("conid", "effective_at", "metric_id", "value_num", "vs_num"),
        output_kind="metric",
        prefix="ratio_fixed_income",
        key_cols=("metric_id",),
    ),
    "ratios_dividend": SnapshotTableContract(
        name="ratios_dividend",
        columns=("conid", "effective_at", "metric_id", "value_num", "vs_num"),
        output_kind="metric",
        prefix="ratio_dividend",
        key_cols=("metric_id",),
    ),
    "ratios_zscore": SnapshotTableContract(
        name="ratios_zscore",
        columns=("conid", "effective_at", "metric_id", "value_num", "vs_num"),
        output_kind="metric",
        prefix="ratio_zscore",
        key_cols=("metric_id",),
    ),
    "dividends_industry_metrics": SnapshotTableContract(
        name="dividends_industry_metrics",
        columns=(
            "conid",
            "effective_at",
            "dividend_yield",
            "annual_dividend",
            "dividend_ttm",
            "dividend_yield_ttm",
            "currency",
        ),
        output_kind="prefixed",
        prefix="dividend_metric",
    ),
    "morningstar_summary": SnapshotTableContract(
        name="morningstar_summary",
        columns=(
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
        output_kind="prefixed",
        prefix="morningstar",
    ),
    "lipper_ratings": SnapshotTableContract(
        name="lipper_ratings",
        columns=("conid", "effective_at", "period", "metric_id", "value_num"),
        output_kind="metric",
        prefix="lipper",
        key_cols=("period", "metric_id"),
    ),
}

SNAPSHOT_TABLE_COLUMNS: Final[dict[str, list[str]]] = {
    name: list(contract.columns) for name, contract in SNAPSHOT_SOURCE_CONTRACTS.items()
}

SNAPSHOT_FEATURE_NAMESPACE_PREFIXES: Final[tuple[str, ...]] = tuple(
    sorted(
        {
            f"{contract.prefix}__"
            for contract in SNAPSHOT_SOURCE_CONTRACTS.values()
            if contract.prefix is not None
        }
    )
)

SNAPSHOT_TIME_CONTRACT: Final[dict[str, str]] = {
    "source_observed_at": "Preserved in canonical storage where the endpoint exposes it.",
    "source_as_of_date": "Endpoint-specific source date; not collapsed globally in preprocess.",
    "storage_effective_at": "Point-in-time join key emitted with every snapshot feature row.",
    "analysis_join_date": "Analysis rebalance or join date chosen downstream from preprocess outputs.",
}


def expand_snapshot_feature_sources(
    columns: list[str] | tuple[str, ...],
    sources: list[str] | tuple[str, ...],
) -> list[str]:
    resolved: list[str] = []
    for source in sources:
        if source.endswith("::*"):
            prefix = f"{source[:-3]}__"
            resolved.extend(
                sorted(column for column in columns if column.startswith(prefix))
            )
            continue
        if source in columns:
            resolved.append(source)
    return resolved
