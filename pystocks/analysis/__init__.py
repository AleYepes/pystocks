# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false, reportGeneralTypeIssues=false
from dataclasses import dataclass
from functools import cache
from itertools import combinations
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import pycountry
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from babel.numbers import get_territory_currencies
except ImportError:  # pragma: no cover - exercised in environments without Babel
    get_territory_currencies = None

from ..config import DATA_DIR, SQLITE_DB_PATH
from ..preprocess.price import (
    PricePreprocessConfig,
    load_price_history,
    preprocess_price_history,
    save_price_preprocess_results,
)
from ..preprocess.snapshots import (
    load_snapshot_features as load_preprocessed_snapshot_features,
)
from ..preprocess.supplementary import (
    load_risk_free_daily as load_preprocessed_risk_free_daily,
)
from ..preprocess.supplementary import (
    load_world_bank_country_features as load_preprocessed_world_bank_country_features,
)
from ..progress import make_progress_bar, track_progress
from ..storage.txn import transaction


@dataclass
class AnalysisConfig:
    sqlite_path: Path = SQLITE_DB_PATH
    output_dir: Path = DATA_DIR / "analysis"
    rebalance_freq: str = "ME"
    min_assets_per_factor: int = 12
    quantile: float = 0.20
    factor_corr_threshold: float = 0.90
    min_factor_coverage: float = 0.60
    min_train_days: int = 126
    min_test_days: int = 21
    trailing_beta_days: int = 252
    selection_frequency_threshold: float = 0.15
    min_selection_count: int = 2
    outlier_z_threshold: float = 50.0
    training_window_years: tuple[int, ...] = (3, 4)
    walk_forward_step_months: int = 12
    use_risk_free_excess: bool = True
    require_supplementary_data: bool = True
    include_macro_features: bool = True
    include_dynamic_fundamental_trends: bool = True
    return_alignment_max_gap_days: int = 0
    sparse_feature_max_ratio: float = 0.995
    persist_progress_bars: bool = True
    max_final_vif: float = 10.0


SUPERSECTOR_MAP = {
    "defensive": [
        "industry__consumer_non_cyclicals",
        "industry__utilities",
        "industry__healthcare",
        "industry__telecommunication_services",
        "industry__academic_educational_services",
    ],
    "cyclical": [
        "industry__technology",
        "industry__consumer_cyclicals",
        "industry__industrials",
        "industry__financials",
        "industry__real_estate",
    ],
    "commodities": [
        "industry__basic_materials",
        "industry__energy",
    ],
}

MACRO_FEATURE_COLUMNS = [
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
]

MACRO_THEME_COMPONENTS = {
    "demographic_scale": ["population_level"],
    "demographic_momentum": ["population_growth", "population_acceleration"],
    "development": [
        "gdp_pcap_level",
        "gdp_pcap_growth",
        "gdp_pcap_acceleration",
    ],
    "external_investment_intensity": [
        "foreign_direct_investment_level",
        "foreign_direct_investment_growth",
        "foreign_direct_investment_acceleration",
    ],
    "global_output_share": [
        "economic_output_gdp_level",
        "economic_output_gdp_growth",
        "economic_output_gdp_acceleration",
    ],
    "trade_centrality": [
        "share_trade_volume_level",
        "share_trade_volume_growth",
        "share_trade_volume_acceleration",
    ],
}

COMPOSITE_SOURCE_COLUMNS = {
    "composite__value": [
        "ratio_key__price_book",
        "ratio_key__price_cash",
        "ratio_key__price_earnings",
        "ratio_key__price_sales",
    ],
    "composite__profitability": [
        "ratio_key__return_on_assets_1yr",
        "ratio_key__return_on_assets_3yr",
        "ratio_key__return_on_capital",
        "ratio_key__return_on_capital_3yr",
        "ratio_key__return_on_equity_1yr",
        "ratio_key__return_on_equity_3yr",
        "ratio_key__return_on_investment_1yr",
        "ratio_key__return_on_investment_3yr",
    ],
    "composite__leverage": [
        "ratio_key__lt_debt_shareholders_equity",
        "ratio_key__total_debt_total_capital",
        "ratio_key__total_debt_total_equity",
        "ratio_key__total_assets_total_equity",
    ],
    "composite__momentum": [
        "price_feature__momentum_3mo",
        "price_feature__momentum_6mo",
        "price_feature__momentum_1y",
        "price_feature__rs_3mo",
        "price_feature__rs_6mo",
        "price_feature__rs_1y",
    ],
    "composite__income": [
        "dividend_metric__dividend_yield",
        "dividend_metric__dividend_yield_ttm",
        "ratio_dividend__dividend_yield",
    ],
    "composite__duration": [
        "holding_maturity__maturity_10_to_20_years",
        "holding_maturity__maturity_20_to_30_years",
        "holding_maturity__maturity_greater_than_30_years",
        "holding_maturity__maturity_less_than_1_year",
        "holding_maturity__maturity_1_to_3_years",
        "holding_maturity__maturity_3_to_5_years",
    ],
    "composite__credit": [
        "holding_quality__quality_aaa",
        "holding_quality__quality_aa",
        "holding_quality__quality_a",
        "holding_quality__quality_bbb",
        "holding_quality__quality_bb",
        "holding_quality__quality_b",
        "holding_quality__quality_ccc",
        "holding_quality__quality_cc",
        "holding_quality__quality_c",
        "holding_quality__quality_d",
    ],
    "composite__concentration": [
        "top10__top10_weight_sum",
        "top10__top10_weight_max",
        "industry__*",
        "country__*",
    ],
}

COUNTRY_CONTINENT_MAP = {
    "arg": "america",
    "aus": "oceania",
    "aut": "europe",
    "bel": "europe",
    "bra": "america",
    "can": "america",
    "che": "europe",
    "chl": "america",
    "chn": "asia",
    "col": "america",
    "deu": "europe",
    "dnk": "europe",
    "egy": "africa",
    "esp": "europe",
    "fin": "europe",
    "fra": "europe",
    "gbr": "europe",
    "hkg": "asia",
    "idn": "asia",
    "ind": "asia",
    "irl": "europe",
    "isr": "asia",
    "ita": "europe",
    "jpn": "asia",
    "kor": "asia",
    "mex": "america",
    "mys": "asia",
    "nld": "europe",
    "nor": "europe",
    "nzl": "oceania",
    "per": "america",
    "phl": "asia",
    "pol": "europe",
    "prt": "europe",
    "qat": "asia",
    "sau": "asia",
    "sgp": "asia",
    "swe": "europe",
    "tha": "asia",
    "tur": "asia",
    "twn": "asia",
    "usa": "america",
    "zaf": "africa",
}

COUNTRY_CURRENCY_OVERRIDE_MAP = {
    "ggy": "gbp",
    "imn": "gbp",
    "jey": "gbp",
}

# Compatibility fallback while Babel is not guaranteed to be installed in every
# environment yet. Production logic resolves currencies with Babel first.
COUNTRY_CURRENCY_MAP = {
    "aus": "aud",
    "can": "cad",
    "che": "chf",
    "chn": "cny",
    "deu": "eur",
    "esp": "eur",
    "fra": "eur",
    "gbr": "gbp",
    "hkg": "hkd",
    "ind": "inr",
    "ita": "eur",
    "jpn": "jpy",
    "kor": "krw",
    "mex": "mxn",
    "nld": "eur",
    "nor": "nok",
    "nzl": "nzd",
    "sgp": "sgd",
    "swe": "sek",
    "twn": "twd",
    "usa": "usd",
    "zaf": "zar",
}

COUNTRY_BLOC_MAP = {
    "asia_pacific": [
        "aus",
        "chn",
        "hkg",
        "idn",
        "ind",
        "jpn",
        "kor",
        "mys",
        "nzl",
        "phl",
        "sgp",
        "tha",
        "twn",
    ],
    "developed_markets": [
        "aus",
        "aut",
        "bel",
        "can",
        "che",
        "deu",
        "dnk",
        "esp",
        "fin",
        "fra",
        "gbr",
        "hkg",
        "irl",
        "ita",
        "jpn",
        "nld",
        "nor",
        "nzl",
        "sgp",
        "swe",
        "usa",
    ],
    "emerging_markets": [
        "arg",
        "bra",
        "chl",
        "chn",
        "col",
        "egy",
        "idn",
        "ind",
        "kor",
        "mex",
        "mys",
        "per",
        "phl",
        "pol",
        "qat",
        "sau",
        "tha",
        "tur",
        "twn",
        "zaf",
    ],
    "euro_area": [
        "aut",
        "bel",
        "deu",
        "esp",
        "fin",
        "fra",
        "irl",
        "ita",
        "nld",
        "prt",
    ],
    "north_america": ["can", "mex", "usa"],
}

CURRENCY_BLOC_MAP = {
    "asia_pacific": ["aud", "cny", "hkd", "jpy", "krw", "nzd", "sgd", "twd"],
    "commodity": ["aud", "cad", "nok", "nzd"],
    "european": ["chf", "eur", "gbp", "nok", "sek"],
    "reserve": ["chf", "eur", "gbp", "jpy", "usd"],
}

INDUSTRY_TO_SUPERSECTOR = {
    industry: supersector
    for supersector, industries in SUPERSECTOR_MAP.items()
    for industry in industries
}

EXPECTED_DIRECTION_LABELS = {
    -1.0: "prefer_lower",
    1.0: "prefer_higher",
}


def _empty_frame(columns):
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


def _to_timestamp(value):
    return pd.Timestamp(value)


def _write_output(name, df, output_dir, sqlite_path, long_sql_df=None, tx=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / f"{name}.parquet"
    parquet_df = (
        df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df.copy()
    )
    parquet_df.to_parquet(parquet_path, index=False)

    sql_df = long_sql_df if long_sql_df is not None else parquet_df
    if tx is not None:
        if len(sql_df.columns) == 0:
            tx.execute(f"DROP TABLE IF EXISTS {name}")
        else:
            tx.write_frame(name, sql_df, if_exists="replace", index=False)
    else:
        with transaction(sqlite_path) as managed_tx:
            if len(sql_df.columns) == 0:
                managed_tx.execute(f"DROP TABLE IF EXISTS {name}")
            else:
                managed_tx.write_frame(name, sql_df, if_exists="replace", index=False)

    return str(parquet_path)


def _series_or_zero(df, column):
    if column in df.columns:
        return pd.Series(
            pd.to_numeric(df[column], errors="coerce"), index=df.index
        ).fillna(0.0)
    return pd.Series(0.0, index=df.index, dtype=float)


def _text_series(df, column):
    if column in df.columns:
        return pd.Series(df[column], index=df.index).fillna("").astype(str)
    return pd.Series("", index=df.index, dtype=object)


def _safe_numeric(df, column):
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _mean_if_present(df, columns):
    present = [col for col in columns if col in df.columns]
    if not present:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return df[present].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)


def _sum_if_present(df, columns):
    present = [col for col in columns if col in df.columns]
    if not present:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return df[present].apply(pd.to_numeric, errors="coerce").sum(axis=1, skipna=True)


def _join_source_columns(columns):
    present = [str(column) for column in columns if column]
    return "|".join(sorted(dict.fromkeys(present)))


def _kind_priority(kind):
    return {
        "benchmark": 0,
        "grouped": 1,
        "composite": 1,
        "macro_derived": 1,
        "raw": 2,
    }.get(str(kind), 3)


def _is_macro_column(column):
    return str(column).startswith(
        (
            "macro__",
            "macro_bloc__",
            "macro_theme__",
            "macro_bloc_theme__",
        )
    )


def _macro_theme_name_for_feature(macro_feature):
    normalized = str(macro_feature).strip()
    if normalized.startswith("population_"):
        if normalized.endswith("_level"):
            return "demographic_scale"
        return "demographic_momentum"
    if normalized.startswith("gdp_pcap_"):
        return "development"
    if normalized.startswith("economic_output_gdp_"):
        return "global_output_share"
    if normalized.startswith("share_trade_volume_"):
        return "trade_centrality"
    if normalized.startswith("foreign_direct_investment_"):
        return "external_investment_intensity"
    return None


def _macro_theme_source_columns(theme_name, bloc=None):
    components = MACRO_THEME_COMPONENTS.get(str(theme_name), [])
    if bloc:
        return [f"macro_bloc__{bloc}__{component}" for component in components]
    return [f"macro__{component}" for component in components]


@cache
def _resolve_country_currency(country_code):
    if country_code is None:
        return None
    normalized = str(country_code).strip().lower()
    if not normalized:
        return None
    if normalized in COUNTRY_CURRENCY_OVERRIDE_MAP:
        return COUNTRY_CURRENCY_OVERRIDE_MAP[normalized]

    country = pycountry.countries.get(alpha_3=normalized.upper())
    if (
        country is not None
        and get_territory_currencies is not None
        and getattr(country, "alpha_2", None)
    ):
        currencies = get_territory_currencies(country.alpha_2, tender=True)
        if currencies:
            return str(currencies[-1]).strip().lower()

    return COUNTRY_CURRENCY_MAP.get(normalized)


def _countries_for_currency(currency_code, panel_slice=None):
    normalized = str(currency_code).strip().lower()
    if not normalized:
        return []

    candidates: set[str] = set(COUNTRY_CURRENCY_OVERRIDE_MAP) | set(
        COUNTRY_CURRENCY_MAP
    )
    if panel_slice is not None:
        candidates.update(
            col.replace("country__", "")
            for col in panel_slice.columns
            if col.startswith("country__")
        )

    matches = [
        country_code
        for country_code in sorted(candidates)
        if _resolve_country_currency(country_code) == normalized
    ]
    return matches


def _country_columns_for_continent(df, continent):
    return [
        f"country__{code}"
        for code, mapped_continent in COUNTRY_CONTINENT_MAP.items()
        if mapped_continent == continent and f"country__{code}" in df.columns
    ]


def _country_columns_for_bloc(df, bloc):
    return [
        f"country__{code}"
        for code in COUNTRY_BLOC_MAP.get(bloc, [])
        if f"country__{code}" in df.columns
    ]


def _currency_columns_for_bloc(df, bloc):
    return [
        f"currency__{code}"
        for code in CURRENCY_BLOC_MAP.get(bloc, [])
        if f"currency__{code}" in df.columns
    ]


def _macro_bloc_source_columns(panel_slice, bloc, macro_column):
    return _country_columns_for_bloc(panel_slice, bloc) + [
        f"world_bank::{macro_column}"
    ]


def _add_continent_features(df):
    out = df.copy()
    country_columns = [col for col in out.columns if col.startswith("country__")]
    if not country_columns:
        return out
    for continent in sorted(set(COUNTRY_CONTINENT_MAP.values())):
        present = _country_columns_for_continent(out, continent)
        if not present:
            continue
        out[f"continent__{continent}"] = (
            out[present].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        )
    return out


def _add_geo_bloc_features(df):
    out = df.copy()
    country_columns = [col for col in out.columns if col.startswith("country__")]
    if not country_columns:
        return out
    for bloc in sorted(COUNTRY_BLOC_MAP):
        present = _country_columns_for_bloc(out, bloc)
        if not present:
            continue
        out[f"bloc__{bloc}"] = (
            out[present].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        )
    return out


def _add_currency_bloc_features(df):
    out = df.copy()
    currency_columns = [col for col in out.columns if col.startswith("currency__")]
    if not currency_columns:
        return out
    for bloc in sorted(CURRENCY_BLOC_MAP):
        present = _currency_columns_for_bloc(out, bloc)
        if not present:
            continue
        out[f"currency_bloc__{bloc}"] = (
            out[present].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        )
    return out


def _source_columns_for_feature(column, panel_slice):
    if column in COMPOSITE_SOURCE_COLUMNS:
        sources = []
        for source in COMPOSITE_SOURCE_COLUMNS[column]:
            if source == "industry__*":
                sources.extend(
                    sorted(
                        col
                        for col in panel_slice.columns
                        if col.startswith("industry__")
                    )
                )
            elif source == "country__*":
                sources.extend(
                    sorted(
                        col
                        for col in panel_slice.columns
                        if col.startswith("country__")
                    )
                )
            elif source in panel_slice.columns:
                sources.append(source)
        return sources
    if column.startswith("supersector__"):
        return [
            source
            for source in SUPERSECTOR_MAP.get(column.split("__", 1)[1], [])
            if source in panel_slice.columns
        ]
    if column.startswith("continent__"):
        return _country_columns_for_continent(panel_slice, column.split("__", 1)[1])
    if column.startswith("bloc__"):
        return _country_columns_for_bloc(panel_slice, column.split("__", 1)[1])
    if column.startswith("currency_bloc__"):
        return _currency_columns_for_bloc(panel_slice, column.split("__", 1)[1])
    if column.startswith("macro_bloc__"):
        _, bloc, macro_column = column.split("__", 2)
        return _macro_bloc_source_columns(panel_slice, bloc, macro_column)
    if column.startswith("macro_bloc_theme__"):
        _, bloc, theme_name = column.split("__", 2)
        return [
            source
            for source in _macro_theme_source_columns(theme_name, bloc=bloc)
            if source in panel_slice.columns
        ]
    if column.startswith("macro__"):
        return [
            f"country_weights::{column.replace('macro__', '')}",
            f"world_bank::{column.replace('macro__', '')}",
        ]
    if column.startswith("macro_theme__"):
        return [
            source
            for source in _macro_theme_source_columns(column.split("__", 1)[1])
            if source in panel_slice.columns
        ]
    return [column]


def _semantic_group_for_column(column, panel_slice=None):
    if column.startswith("price_feature__momentum_") or column.startswith(
        "price_feature__rs_"
    ):
        return "momentum_relative_strength"
    if column.startswith("price_feature__volatility_") or column.startswith(
        "price_feature__downside_volatility_"
    ):
        return "risk_volatility"
    if column.startswith("price_feature__max_drawdown_"):
        return "risk_drawdown"
    if column.startswith("supersector__"):
        return f"sector_theme__{column.split('__', 1)[1]}"
    if column.startswith("bloc__"):
        return f"country_bloc__{column.split('__', 1)[1]}"
    if column.startswith("currency_bloc__"):
        return f"currency_bloc__{column.split('__', 1)[1]}"
    if column.startswith("macro_bloc_theme__"):
        _, bloc, theme_name = column.split("__", 2)
        return f"macro_bloc_theme__{bloc}__{theme_name}"
    if column.startswith("macro_bloc__"):
        _, bloc, suffix = column.split("__", 2)
        theme_name = _macro_theme_name_for_feature(suffix)
        if theme_name:
            return f"macro_bloc_theme__{bloc}__{theme_name}"
        return f"macro_bloc_theme__{bloc}__{suffix}"
    if column.startswith("industry__"):
        supersector = INDUSTRY_TO_SUPERSECTOR.get(column)
        if supersector:
            return f"sector_theme__{supersector}"
    if column.startswith("holding_maturity__") or column == "composite__duration":
        return "fixed_income_duration"
    if column.startswith("holding_quality__") or column == "composite__credit":
        return "fixed_income_credit"
    if column.startswith("country__"):
        country = column.split("__", 1)[1]
        currency = _resolve_country_currency(country)
        if currency:
            return f"country_currency_pair__{country}__{currency}"
        continent = COUNTRY_CONTINENT_MAP.get(country)
        if continent:
            return f"continent_theme__{continent}"
    if column.startswith("currency__"):
        currency = column.split("__", 1)[1]
        countries = _countries_for_currency(currency, panel_slice=panel_slice)
        if len(countries) == 1:
            return f"country_currency_pair__{countries[0]}__{currency}"
        if countries:
            return f"currency_family__{currency}"
    if column.startswith("continent__"):
        return f"continent_theme__{column.split('__', 1)[1]}"
    if column.startswith("macro_theme__"):
        return column
    if column.startswith("macro__"):
        suffix = column.replace("macro__", "", 1)
        theme_name = _macro_theme_name_for_feature(suffix)
        if theme_name:
            return f"macro_theme__{theme_name}"
    return column


def _economic_rationale_for_column(column):
    if column.startswith("composite__value"):
        return "Preserves V1 valuation compression across price multiple ratios."
    if column.startswith("composite__leverage"):
        return "Preserves V1 balance-sheet leverage grouping."
    if column.startswith("composite__profitability"):
        return "Preserves V1 profitability grouping across return metrics."
    if column.startswith("composite__momentum"):
        return "Preserves V1 momentum and relative-strength aggregation."
    if column.startswith("supersector__"):
        return "Preserves V1 industry-to-supersector grouping."
    if column.startswith("continent__"):
        return "Preserves V1 continent-level geographic grouping."
    if column.startswith("bloc__"):
        return "Extends V1 geographic compression into deterministic regional and development blocs."
    if column.startswith("currency_bloc__"):
        return "Prefers reserve and bloc-level currency exposures over single-currency leaves."
    if column.startswith("macro_bloc_theme__"):
        return "Compresses bloc-level World Bank leaves into curated macro themes."
    if column.startswith("macro_bloc__"):
        return "Groups World Bank macro exposures into deterministic geographic blocs."
    if column.startswith("macro_theme__"):
        return "Compresses World Bank leaves into curated macro themes inspired by legacy research."
    if column.startswith("macro__"):
        return "Country weights collapsed into stored World Bank macro leaves."
    if column.startswith("benchmark__"):
        return "Baseline benchmark factor retained explicitly."
    return "Leaf-level exposure retained for candidate evaluation."


def _construction_type_for_kind(kind):
    if kind == "benchmark":
        return "benchmark_series"
    return "long_short_size_weighted"


def _rolling_compound(values: pd.Series, window: int, min_periods: int) -> pd.Series:
    return (1.0 + values.fillna(0.0)).rolling(window, min_periods=min_periods).apply(
        np.prod, raw=True
    ) - 1.0


def _bounded_align_return_frame(returns_wide, max_gap_days):
    if returns_wide.empty or int(max_gap_days) <= 0:
        return returns_wide
    aligned = returns_wide.copy()
    for column in aligned.columns:
        aligned[column] = aligned[column].interpolate(
            method="linear",
            limit=int(max_gap_days),
            limit_area="inside",
        )
    return aligned


def load_snapshot_features(sqlite_path=SQLITE_DB_PATH):
    return load_preprocessed_snapshot_features(sqlite_path=sqlite_path)


def load_risk_free_daily(sqlite_path=SQLITE_DB_PATH):
    return load_preprocessed_risk_free_daily(sqlite_path=sqlite_path)


def load_world_bank_country_features(sqlite_path=SQLITE_DB_PATH):
    return load_preprocessed_world_bank_country_features(sqlite_path=sqlite_path)


def _prepare_analysis_inputs(config, show_progress=False):
    price_config = PricePreprocessConfig(outlier_z_threshold=config.outlier_z_threshold)
    price_result = preprocess_price_history(
        load_price_history(config.sqlite_path),
        config=price_config,
        show_progress=show_progress,
    )
    save_price_preprocess_results(price_result, output_dir=config.output_dir)
    snapshot_features = load_snapshot_features(config.sqlite_path)
    risk_free_daily = load_risk_free_daily(config.sqlite_path)
    world_bank_country_features = load_world_bank_country_features(config.sqlite_path)

    if config.require_supplementary_data:
        if config.use_risk_free_excess and risk_free_daily.empty:
            raise RuntimeError(
                "Missing supplementary risk-free data. Run refresh_supplementary_data first."
            )
        if config.include_macro_features and world_bank_country_features.empty:
            raise RuntimeError(
                "Missing supplementary World Bank features. Run refresh_supplementary_data first."
            )

    return snapshot_features, price_result, risk_free_daily, world_bank_country_features


def _empty_cluster_frame():
    return _empty_frame(
        [
            "factor_id",
            "sleeve",
            "cluster_id",
            "cluster_representative",
            "cluster_size",
            "keep_factor",
        ]
    )


def _build_price_features(
    prices, show_progress=False, persist_progress_bars: bool = True
):
    clean = prices.loc[prices["is_clean_price"]].copy()
    if clean.empty:
        return _empty_frame(["conid", "trade_date"])

    clean = clean.sort_values(["conid", "trade_date"])
    frames = []
    group_count = int(clean["conid"].nunique())
    for _, group in track_progress(
        clean.groupby("conid"),
        show_progress=show_progress,
        total=group_count,
        desc="Price features",
        unit="conid",
        leave=persist_progress_bars,
    ):
        g = group.copy()
        returns = pd.to_numeric(g["clean_return"], errors="coerce")
        g["price_feature__momentum_21"] = g["clean_price"].pct_change(21)
        g["price_feature__momentum_63"] = g["clean_price"].pct_change(63)
        g["price_feature__momentum_126"] = g["clean_price"].pct_change(126)
        g["price_feature__momentum_252"] = g["clean_price"].pct_change(252)
        g["price_feature__momentum_3mo"] = returns.rolling(63, min_periods=21).mean()
        g["price_feature__momentum_6mo"] = returns.rolling(126, min_periods=42).mean()
        g["price_feature__momentum_1y"] = returns.rolling(252, min_periods=84).mean()
        g["price_feature__rs_3mo"] = _rolling_compound(returns, 63, 21)
        g["price_feature__rs_6mo"] = _rolling_compound(returns, 126, 42)
        g["price_feature__rs_1y"] = _rolling_compound(returns, 252, 84)
        g["price_feature__volatility_21"] = returns.rolling(
            21, min_periods=10
        ).std() * np.sqrt(252.0)
        g["price_feature__volatility_63"] = returns.rolling(
            63, min_periods=21
        ).std() * np.sqrt(252.0)
        g["price_feature__downside_volatility_63"] = returns.where(
            returns < 0.0
        ).rolling(63, min_periods=21).std() * np.sqrt(252.0)
        rolling_peak = g["clean_price"].rolling(126, min_periods=21).max()
        drawdown = g["clean_price"] / rolling_peak - 1.0
        g["price_feature__max_drawdown_126"] = drawdown.rolling(
            126, min_periods=21
        ).min()
        frames.append(g)

    price_feature_columns = [
        "conid",
        "trade_date",
        "price_feature__momentum_21",
        "price_feature__momentum_63",
        "price_feature__momentum_126",
        "price_feature__momentum_252",
        "price_feature__momentum_3mo",
        "price_feature__momentum_6mo",
        "price_feature__momentum_1y",
        "price_feature__rs_3mo",
        "price_feature__rs_6mo",
        "price_feature__rs_1y",
        "price_feature__volatility_21",
        "price_feature__volatility_63",
        "price_feature__downside_volatility_63",
        "price_feature__max_drawdown_126",
    ]
    return pd.concat(frames, ignore_index=True)[price_feature_columns]


def _build_rebalance_dates(snapshot_features, prices, freq):
    if snapshot_features.empty or prices.empty:
        return pd.DatetimeIndex([])
    if freq == "M":
        freq = "ME"
    start = snapshot_features["effective_at"].min().normalize()
    end = prices["trade_date"].max().normalize()
    dates = pd.date_range(start=start, end=end, freq=freq)
    if len(dates) == 0 or dates[-1] != end:
        dates = dates.append(pd.DatetimeIndex([end]))
    return dates.unique().sort_values()


def _merge_price_features(latest, price_features):
    if latest.empty or price_features.empty:
        latest["feature_trade_date"] = pd.NaT
        return latest

    left = latest.sort_values(["rebalance_date", "conid"]).copy()
    right = price_features.sort_values(["trade_date", "conid"]).copy()
    merged = pd.merge_asof(
        left,
        right,
        by="conid",
        left_on="rebalance_date",
        right_on="trade_date",
        direction="backward",
    )
    return merged.rename(columns={"trade_date": "feature_trade_date"})


def _add_dynamic_fundamental_features(df):
    out = df.copy()

    def _slope(col_a, col_b, time_a, time_b):
        if col_a not in out.columns or col_b not in out.columns:
            return pd.Series(np.nan, index=out.index, dtype=float)
        return (
            pd.to_numeric(out[col_a], errors="coerce")
            - pd.to_numeric(out[col_b], errors="coerce")
        ) / float(time_a - time_b)

    slope_specs = [
        (
            "trend__eps_growth_slope",
            "ratio_key__eps_growth_1yr",
            "ratio_key__eps_growth_5yr",
            1,
            5,
        ),
        (
            "trend__return_on_assets_slope",
            "ratio_key__return_on_assets_1yr",
            "ratio_key__return_on_assets_3yr",
            1,
            3,
        ),
        (
            "trend__return_on_capital_slope",
            "ratio_key__return_on_capital",
            "ratio_key__return_on_capital_3yr",
            1,
            3,
        ),
        (
            "trend__return_on_equity_slope",
            "ratio_key__return_on_equity_1yr",
            "ratio_key__return_on_equity_3yr",
            1,
            3,
        ),
        (
            "trend__return_on_investment_slope",
            "ratio_key__return_on_investment_1yr",
            "ratio_key__return_on_investment_3yr",
            1,
            3,
        ),
    ]
    for new_col, col_a, col_b, time_a, time_b in slope_specs:
        out[new_col] = _slope(col_a, col_b, time_a, time_b)

    if {
        "ratio_key__eps_growth_1yr",
        "ratio_key__eps_growth_3yr",
        "ratio_key__eps_growth_5yr",
    }.issubset(out.columns):
        slope_1_3 = _slope(
            "ratio_key__eps_growth_1yr", "ratio_key__eps_growth_3yr", 1, 3
        )
        slope_3_5 = _slope(
            "ratio_key__eps_growth_3yr", "ratio_key__eps_growth_5yr", 3, 5
        )
        out["trend__eps_growth_second_derivative"] = (slope_1_3 - slope_3_5) / -2.0

    return out


def _add_supersector_features(df):
    out = df.copy()
    for name, columns in SUPERSECTOR_MAP.items():
        present = [column for column in columns if column in out.columns]
        if present:
            out[f"supersector__{name}"] = (
                out[present].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            )
    return out


def _add_macro_features(panel, world_bank_country_features):
    if panel.empty:
        return panel.copy()
    if world_bank_country_features is None or world_bank_country_features.empty:
        return panel.copy()

    out = panel.copy()
    country_columns = [col for col in out.columns if col.startswith("country__")]
    if not country_columns:
        return out

    world_bank = world_bank_country_features.copy()
    world_bank["economy_code"] = world_bank["economy_code"].astype(str).str.upper()
    world_bank["effective_at"] = pd.to_datetime(world_bank["effective_at"])

    enriched_parts = []
    for rebalance_date, panel_slice in out.groupby("rebalance_date", sort=True):
        latest = (
            world_bank.loc[world_bank["effective_at"] <= rebalance_date]
            .sort_values(["economy_code", "effective_at"])
            .groupby("economy_code", as_index=False)
            .tail(1)
        )
        if latest.empty:
            enriched_parts.append(panel_slice)
            continue

        latest = latest.set_index("economy_code")
        work = panel_slice.copy()
        for macro_column in MACRO_FEATURE_COLUMNS:
            weighted = pd.Series(0.0, index=work.index, dtype=float)
            bloc_weighted = {
                bloc: pd.Series(0.0, index=work.index, dtype=float)
                for bloc in COUNTRY_BLOC_MAP
            }
            for country_column in country_columns:
                code = country_column.replace("country__", "").upper()
                if code not in latest.index or macro_column not in latest.columns:
                    continue
                weight = pd.to_numeric(work[country_column], errors="coerce").fillna(
                    0.0
                )
                value = pd.to_numeric(latest.loc[code, macro_column], errors="coerce")
                if np.isscalar(value):
                    contribution = weight * float(value)
                    weighted = weighted.add(contribution, fill_value=0.0)
                    country_code = code.lower()
                    for bloc, members in COUNTRY_BLOC_MAP.items():
                        if country_code in members:
                            bloc_weighted[bloc] = bloc_weighted[bloc].add(
                                contribution, fill_value=0.0
                            )
            work[f"macro__{macro_column}"] = weighted
            for bloc, bloc_values in bloc_weighted.items():
                work[f"macro_bloc__{bloc}__{macro_column}"] = bloc_values
        enriched_parts.append(work)

    return pd.concat(enriched_parts, ignore_index=True)


def _cross_sectional_zscore(values):
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    std = float(numeric.std(skipna=True, ddof=0))
    if not np.isfinite(std) or std <= 0.0:
        return pd.Series(0.0, index=values.index, dtype=float)
    mean = float(numeric.mean(skipna=True))
    return (numeric - mean) / std


def _add_macro_theme_features(panel):
    if panel.empty:
        return panel.copy()
    if not any(column.startswith("macro__") for column in panel.columns):
        return panel.copy()

    enriched_parts = []
    for _, panel_slice in panel.groupby("rebalance_date", sort=True):
        work = panel_slice.copy()
        for theme_name, components in MACRO_THEME_COMPONENTS.items():
            global_sources = [
                f"macro__{component}"
                for component in components
                if f"macro__{component}" in work.columns
            ]
            if global_sources:
                standardized = pd.DataFrame(
                    {
                        column: _cross_sectional_zscore(work[column])
                        for column in global_sources
                    }
                )
                work[f"macro_theme__{theme_name}"] = standardized.mean(
                    axis=1, skipna=True
                )

            for bloc in sorted(COUNTRY_BLOC_MAP):
                bloc_sources = [
                    f"macro_bloc__{bloc}__{component}"
                    for component in components
                    if f"macro_bloc__{bloc}__{component}" in work.columns
                ]
                if not bloc_sources:
                    continue
                standardized = pd.DataFrame(
                    {
                        column: _cross_sectional_zscore(work[column])
                        for column in bloc_sources
                    }
                )
                work[f"macro_bloc_theme__{bloc}__{theme_name}"] = standardized.mean(
                    axis=1, skipna=True
                )
        enriched_parts.append(work)

    return pd.concat(enriched_parts, ignore_index=True)


def _add_composite_features(panel):
    df = panel.copy()
    df["composite__value"] = -_mean_if_present(
        df,
        [
            "ratio_key__price_book",
            "ratio_key__price_cash",
            "ratio_key__price_earnings",
            "ratio_key__price_sales",
        ],
    )
    df["composite__profitability"] = _mean_if_present(
        df,
        [
            "ratio_key__return_on_assets_1yr",
            "ratio_key__return_on_assets_3yr",
            "ratio_key__return_on_capital",
            "ratio_key__return_on_capital_3yr",
            "ratio_key__return_on_equity_1yr",
            "ratio_key__return_on_equity_3yr",
            "ratio_key__return_on_investment_1yr",
            "ratio_key__return_on_investment_3yr",
        ],
    )
    df["composite__leverage"] = _mean_if_present(
        df,
        [
            "ratio_key__lt_debt_shareholders_equity",
            "ratio_key__total_debt_total_capital",
            "ratio_key__total_debt_total_equity",
            "ratio_key__total_assets_total_equity",
        ],
    )
    df["composite__momentum"] = _mean_if_present(
        df,
        [
            "price_feature__momentum_3mo",
            "price_feature__momentum_6mo",
            "price_feature__momentum_1y",
            "price_feature__rs_3mo",
            "price_feature__rs_6mo",
            "price_feature__rs_1y",
        ],
    )
    df["composite__income"] = _mean_if_present(
        df,
        [
            "dividend_metric__dividend_yield",
            "dividend_metric__dividend_yield_ttm",
            "ratio_dividend__dividend_yield",
        ],
    )
    df["composite__duration"] = _sum_if_present(
        df,
        [
            "holding_maturity__maturity_10_to_20_years",
            "holding_maturity__maturity_20_to_30_years",
            "holding_maturity__maturity_greater_than_30_years",
        ],
    ) - _sum_if_present(
        df,
        [
            "holding_maturity__maturity_less_than_1_year",
            "holding_maturity__maturity_1_to_3_years",
            "holding_maturity__maturity_3_to_5_years",
        ],
    )
    df["composite__credit"] = _sum_if_present(
        df,
        [
            "holding_quality__quality_aaa",
            "holding_quality__quality_aa",
            "holding_quality__quality_a",
            "holding_quality__quality_bbb",
        ],
    ) - _sum_if_present(
        df,
        [
            "holding_quality__quality_bb",
            "holding_quality__quality_b",
            "holding_quality__quality_ccc",
            "holding_quality__quality_cc",
            "holding_quality__quality_c",
            "holding_quality__quality_d",
        ],
    )
    industry_cols = [col for col in df.columns if col.startswith("industry__")]
    country_cols = [col for col in df.columns if col.startswith("country__")]
    df["composite__concentration"] = _mean_if_present(
        df,
        ["top10__top10_weight_sum", "top10__top10_weight_max"]
        + ([industry_cols[0]] if industry_cols else [])
        + ([country_cols[0]] if country_cols else []),
    )
    return df


def build_analysis_panel_data(
    snapshot_features,
    price_result,
    config,
    world_bank_country_features=None,
    show_progress=False,
):
    prices = price_result["prices"]
    eligibility = price_result["eligibility"]
    price_features = _build_price_features(
        prices,
        show_progress=show_progress,
        persist_progress_bars=config.persist_progress_bars,
    )
    rebalance_dates = _build_rebalance_dates(
        snapshot_features, prices, config.rebalance_freq
    )
    eligible_conids = set(eligibility.loc[eligibility["eligible"], "conid"].astype(str))

    if (
        config.require_supplementary_data
        and config.include_macro_features
        and (world_bank_country_features is None or world_bank_country_features.empty)
    ):
        raise RuntimeError(
            "Missing supplementary World Bank features. Run refresh_supplementary_data first."
        )

    panels = []
    for rebalance_date in track_progress(
        rebalance_dates,
        show_progress=show_progress,
        total=len(rebalance_dates),
        desc="Analysis rebalance dates",
        unit="date",
        leave=config.persist_progress_bars,
    ):
        eligible_snapshots = snapshot_features.loc[
            snapshot_features["effective_at"] <= rebalance_date
        ]
        if eligible_snapshots.empty:
            continue
        latest = (
            eligible_snapshots.sort_values(["conid", "effective_at"])
            .groupby("conid", as_index=False)
            .tail(1)
            .copy()
        )
        latest = latest.loc[latest["conid"].isin(eligible_conids)].copy()
        if latest.empty:
            continue
        latest["rebalance_date"] = rebalance_date
        latest["snapshot_age_days"] = (rebalance_date - latest["effective_at"]).dt.days
        latest = latest.merge(eligibility, on="conid", how="left")
        latest = _merge_price_features(latest, price_features)
        panels.append(latest)

    if not panels:
        return pd.DataFrame()

    panel = pd.concat(panels, ignore_index=True)
    panel = _add_composite_features(panel)
    if config.include_dynamic_fundamental_trends:
        panel = _add_dynamic_fundamental_features(panel)
    panel = _add_supersector_features(panel)
    panel = _add_continent_features(panel)
    panel = _add_geo_bloc_features(panel)
    panel = _add_currency_bloc_features(panel)
    if config.include_macro_features:
        panel = _add_macro_features(panel, world_bank_country_features)
        panel = _add_macro_theme_features(panel)
    return panel.sort_values(["rebalance_date", "conid"]).reset_index(drop=True)


def _build_returns_wide(prices, max_gap_days=0):
    clean = prices.loc[
        prices["is_clean_price"], ["conid", "trade_date", "clean_return"]
    ].copy()
    if clean.empty:
        return pd.DataFrame()
    wide = (
        clean.pivot(index="trade_date", columns="conid", values="clean_return")
        .sort_index()
        .sort_index(axis=1)
    )
    return _bounded_align_return_frame(wide, max_gap_days)


def _build_risk_free_series(risk_free_daily):
    if risk_free_daily is None or risk_free_daily.empty:
        return pd.Series(dtype=float, name="daily_nominal_rate")
    rf = risk_free_daily.copy()
    rf["trade_date"] = pd.to_datetime(rf["trade_date"])
    rf["daily_nominal_rate"] = pd.to_numeric(rf["daily_nominal_rate"], errors="coerce")
    return rf.set_index("trade_date")["daily_nominal_rate"].sort_index()


def _normalized_weights(values):
    series = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan)
    series = series.where(series > 0.0)
    if series.notna().sum() == 0:
        series = pd.Series(1.0, index=series.index)
    series = series.fillna(0.0)
    total = float(series.sum())
    if total <= 0:
        return pd.Series(1.0 / len(series), index=series.index)
    return series / total


def _drop_uninformative_factor_columns(panel_slice, candidate_columns, config):
    keep = []
    for column in candidate_columns:
        values = pd.to_numeric(panel_slice[column], errors="coerce")
        if values.notna().sum() == 0:
            continue
        if values.nunique(dropna=True) <= 1:
            continue
        nonzero_ratio = float(values.fillna(0.0).eq(0.0).mean())
        if nonzero_ratio >= config.sparse_feature_max_ratio:
            continue
        keep.append(column)
    return keep


def _list_factor_columns(panel_slice, sleeve):
    numeric_cols = panel_slice.select_dtypes(include=[np.number, bool]).columns.tolist()
    excluded = {
        "valid_rows",
        "total_rows",
        "expected_business_days",
        "eligible",
        "snapshot_age_days",
        "max_internal_gap_days",
        "feature_year",
        "source_count",
    }
    candidates = []
    for col in numeric_cols:
        if col in excluded or col.startswith("beta__"):
            continue
        if (
            col.startswith("holding_maturity__")
            or col.startswith("holding_quality__")
            or col.startswith("debt_type__")
            or col.startswith("ratio_fixed_income__")
        ) and sleeve != "bond":
            continue
        candidates.append(col)
    return candidates


def _select_factor_columns(panel_slice, sleeve, config):
    return _drop_uninformative_factor_columns(
        panel_slice, _list_factor_columns(panel_slice, sleeve), config
    )


def _factor_direction(column):
    inverse_tokens = [
        "expense",
        "price_book",
        "price_cash",
        "price_earnings",
        "price_sales",
        "leverage",
        "volatility",
        "drawdown",
        "missing_ratio",
        "snapshot_age",
    ]
    return -1.0 if any(token in column for token in inverse_tokens) else 1.0


def _factor_family(column):
    return column.split("__", 1)[0]


def _factor_kind(column):
    if column.startswith("benchmark__"):
        return "benchmark"
    if column.startswith("composite__"):
        return "composite"
    if column.startswith("macro_theme__") or column.startswith("macro_bloc_theme__"):
        return "macro_derived"
    if (
        column.startswith("supersector__")
        or column.startswith("continent__")
        or column.startswith("bloc__")
        or column.startswith("currency_bloc__")
    ):
        return "grouped"
    return "raw"


def _factor_id_for_column(sleeve, column):
    return f"{sleeve}__{_factor_kind(column)}__{column}"


def _build_long_short_series(
    values, returns_frame, size_weights, direction, quantile, min_assets
):
    valid = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < min_assets or valid.nunique() < 4:
        return None

    scores = valid * direction
    bucket_size = max(2, int(np.floor(len(scores) * quantile)))
    if bucket_size * 2 > len(scores):
        return None

    ranked = scores.sort_values()
    short_ids = ranked.index[:bucket_size]
    long_ids = ranked.index[-bucket_size:]
    if len(set(long_ids) & set(short_ids)) > 0:
        return None

    long_weights = _normalized_weights(size_weights.reindex(long_ids))
    short_weights = _normalized_weights(size_weights.reindex(short_ids))
    long_returns = (
        returns_frame.reindex(columns=list(long_weights.index))
        .fillna(0.0)
        .dot(long_weights)
    )
    short_returns = (
        returns_frame.reindex(columns=list(short_weights.index))
        .fillna(0.0)
        .dot(short_weights)
    )
    return long_returns - short_returns


def _build_benchmark_factors(
    sleeve_slice,
    interval_returns,
    risk_free_series,
    config,
):
    size_weights = sleeve_slice.set_index("conid")["profile__total_net_assets_num"]
    benchmark_map = {}

    market_weights = _normalized_weights(size_weights.reindex(sleeve_slice["conid"]))
    market = (
        interval_returns.reindex(columns=list(market_weights.index))
        .fillna(0.0)
        .dot(market_weights)
    )
    if not risk_free_series.empty and config.use_risk_free_excess:
        market = market.subtract(risk_free_series.reindex(market.index).fillna(0.0))
    benchmark_map["benchmark__market_excess"] = market

    size_signal = -pd.to_numeric(
        sleeve_slice.set_index("conid")["profile__total_net_assets_num"],
        errors="coerce",
    )
    smb = _build_long_short_series(
        size_signal,
        interval_returns,
        size_weights,
        1.0,
        config.quantile,
        config.min_assets_per_factor,
    )
    if smb is not None:
        benchmark_map["benchmark__smb"] = smb

    if "composite__value" in sleeve_slice.columns:
        hml = _build_long_short_series(
            pd.to_numeric(
                sleeve_slice.set_index("conid")["composite__value"], errors="coerce"
            ),
            interval_returns,
            size_weights,
            1.0,
            config.quantile,
            config.min_assets_per_factor,
        )
        if hml is not None:
            benchmark_map["benchmark__hml"] = hml

    return benchmark_map


def _select_baseline_bond_members(panel_slice):
    bond_slice = panel_slice.loc[panel_slice["sleeve"] == "bond"].copy()
    if bond_slice.empty:
        return bond_slice

    short_score = (
        _series_or_zero(bond_slice, "holding_maturity__maturity_less_than_1_year")
        + 0.75 * _series_or_zero(bond_slice, "holding_maturity__maturity_1_to_3_years")
        - _series_or_zero(bond_slice, "holding_maturity__maturity_10_to_20_years")
        - _series_or_zero(bond_slice, "holding_maturity__maturity_20_to_30_years")
        - _series_or_zero(
            bond_slice, "holding_maturity__maturity_greater_than_30_years"
        )
    )
    quality_score = (
        _series_or_zero(bond_slice, "holding_quality__quality_aaa")
        + _series_or_zero(bond_slice, "holding_quality__quality_aa")
        + _series_or_zero(bond_slice, "holding_quality__quality_a")
        - _series_or_zero(bond_slice, "holding_quality__quality_bb")
        - _series_or_zero(bond_slice, "holding_quality__quality_b")
        - _series_or_zero(bond_slice, "holding_quality__quality_ccc")
    )
    sovereign_cols = [
        col
        for col in bond_slice.columns
        if col.startswith("debt_type__")
        and any(token in col for token in ["sovereign", "government", "treasury"])
    ]
    sovereign_score = (
        bond_slice[sovereign_cols].sum(axis=1)
        if sovereign_cols
        else pd.Series(0.0, index=bond_slice.index)
    )
    text_bonus = _text_series(bond_slice, "profile__classification").str.contains(
        "treasury|government|short", case=False, regex=True
    ).astype(float) + _text_series(bond_slice, "profile__objective").str.contains(
        "treasury|government|short", case=False, regex=True
    ).astype(float)
    bond_slice["baseline_score"] = (
        short_score + quality_score + sovereign_score + text_bonus
    )
    bond_slice = bond_slice.sort_values(
        ["baseline_score", "profile__total_net_assets_num"],
        ascending=[False, False],
        na_position="last",
    )
    target_count = min(10, max(3, int(np.ceil(len(bond_slice) * 0.10))))
    return bond_slice.head(target_count)


def _build_baseline_returns(
    panel,
    returns_wide,
    show_progress=False,
    persist_progress_bars: bool = True,
):
    if panel.empty or returns_wide.empty:
        return pd.Series(dtype=float), pd.DataFrame()

    rebalance_dates = sorted(pd.to_datetime(panel["rebalance_date"]).unique())
    baseline = pd.Series(0.0, index=returns_wide.index, name="bond_baseline_return")
    membership_rows = []

    intervals = list(zip(rebalance_dates[:-1], rebalance_dates[1:]))
    for start, end in track_progress(
        intervals,
        show_progress=show_progress,
        total=len(intervals),
        desc="Baseline windows",
        unit="window",
        leave=persist_progress_bars,
    ):
        panel_slice = panel.loc[panel["rebalance_date"] == start]
        selected = _select_baseline_bond_members(panel_slice)
        if selected.empty:
            continue
        interval_returns = returns_wide.loc[
            (returns_wide.index > start) & (returns_wide.index <= end)
        ]
        if interval_returns.empty:
            continue
        conids = selected["conid"].astype(str).tolist()
        weights = _normalized_weights(
            selected.set_index("conid")["profile__total_net_assets_num"].reindex(conids)
        )
        portfolio = interval_returns.reindex(columns=conids).fillna(0.0).dot(weights)
        baseline.loc[portfolio.index] = portfolio

        for conid, weight in weights.items():
            membership_rows.append(
                {
                    "rebalance_date": pd.Timestamp(start),
                    "conid": str(conid),
                    "weight": float(weight),
                }
            )

    return baseline, pd.DataFrame(membership_rows)


def _candidate_score_components(kind, coverage_ratio, zero_ratio, unique_count):
    kind_component = float(max(0, 4 - _kind_priority(kind)))
    coverage_component = float(coverage_ratio)
    density_component = float(1.0 - zero_ratio)
    uniqueness_component = float(np.log1p(max(int(unique_count), 0)))
    return (
        kind_component,
        coverage_component,
        density_component,
        uniqueness_component,
    )


def _build_candidate_context(panel, config):
    registry_rows = []
    diagnostics_rows = []
    decision_rows = []
    score_rows = []
    distinctness_rows = []
    admitted_columns: dict[str, list[str]] = {}

    if panel.empty or "sleeve" not in panel.columns:
        return {
            "factor_registry": pd.DataFrame(),
            "candidate_diagnostics": pd.DataFrame(),
            "screening_decisions": pd.DataFrame(),
            "selection_scores": pd.DataFrame(),
            "distinctness": pd.DataFrame(),
            "admitted_columns": admitted_columns,
        }

    for sleeve, panel_slice in panel.groupby("sleeve", sort=True):
        candidate_rows = []
        for column in _list_factor_columns(panel_slice, sleeve):
            values = pd.to_numeric(panel_slice[column], errors="coerce")
            coverage_ratio = float(values.notna().mean())
            zero_ratio = float(values.fillna(0.0).eq(0.0).mean())
            unique_count = int(values.nunique(dropna=True))
            spread = (
                float(values.quantile(0.90) - values.quantile(0.10))
                if values.notna().any()
                else 0.0
            )
            kind = _factor_kind(column)
            factor_id = _factor_id_for_column(str(sleeve), column)
            (
                kind_component,
                coverage_component,
                density_component,
                uniqueness_component,
            ) = _candidate_score_components(
                kind, coverage_ratio, zero_ratio, unique_count
            )
            score = (
                kind_component * 100.0
                + coverage_component * 10.0
                + density_component * 5.0
                + uniqueness_component
            )
            expected_direction = _factor_direction(column)
            rejection_reason = ""
            admission_status = "candidate"
            if values.notna().sum() == 0:
                rejection_reason = "empty"
                admission_status = "rejected"
            elif unique_count <= 1:
                rejection_reason = "constant"
                admission_status = "rejected"
            elif zero_ratio >= config.sparse_feature_max_ratio:
                rejection_reason = "sparse"
                admission_status = "rejected"
            elif coverage_ratio < config.min_factor_coverage:
                rejection_reason = "low_coverage"
                admission_status = "rejected"
            elif not np.isfinite(spread) or abs(spread) <= 0.0:
                rejection_reason = "zero_spread"
                admission_status = "rejected"

            source_columns = _source_columns_for_feature(column, panel_slice)
            row = {
                "factor_id": factor_id,
                "sleeve": str(sleeve),
                "family": _factor_family(column),
                "semantic_group": _semantic_group_for_column(column, panel_slice),
                "kind": kind,
                "source_columns": _join_source_columns(source_columns),
                "construction_type": _construction_type_for_kind(kind),
                "economic_rationale": _economic_rationale_for_column(column),
                "expected_direction": EXPECTED_DIRECTION_LABELS.get(
                    expected_direction, "prefer_higher"
                ),
                "is_benchmark": False,
                "is_macro": bool(_is_macro_column(column)),
                "is_composite": bool(column.startswith("composite__")),
                "admission_status": admission_status,
                "rejection_reason": rejection_reason,
                "coverage_ratio": coverage_ratio,
                "zero_ratio": zero_ratio,
                "unique_count": unique_count,
                "cross_sectional_spread": float(spread),
                "kind_priority": int(_kind_priority(kind)),
                "selection_score": float(score),
                "source_column": column,
            }
            candidate_rows.append(row)
            registry_rows.append(
                {key: row[key] for key in row if key != "source_column"}
            )
            diagnostics_rows.append(
                {
                    "factor_id": factor_id,
                    "sleeve": str(sleeve),
                    "source_column": column,
                    "family": row["family"],
                    "semantic_group": row["semantic_group"],
                    "kind": kind,
                    "coverage_ratio": coverage_ratio,
                    "zero_ratio": zero_ratio,
                    "unique_count": unique_count,
                    "cross_sectional_spread": float(spread),
                }
            )
            score_rows.append(
                {
                    "factor_id": factor_id,
                    "sleeve": str(sleeve),
                    "stage": "semantic_compression",
                    "kind_priority_component": kind_component,
                    "coverage_component": coverage_component,
                    "density_component": density_component,
                    "uniqueness_component": uniqueness_component,
                    "selection_score": float(score),
                }
            )
            decision_rows.append(
                {
                    "factor_id": factor_id,
                    "sleeve": str(sleeve),
                    "stage": "structural_screen",
                    "decision": "drop" if rejection_reason else "keep",
                    "reason": rejection_reason or "passed_structural_screen",
                    "reference_factor_id": "",
                }
            )

        admitted_rows = [
            row for row in candidate_rows if row["admission_status"] != "rejected"
        ]
        admitted_set = {row["factor_id"] for row in admitted_rows}
        grouped_rows = (
            pd.DataFrame(admitted_rows).groupby("semantic_group", sort=True)
            if admitted_rows
            else []
        )
        for semantic_group, group_rows_iter in grouped_rows:
            group_rows = group_rows_iter.to_dict("records")
            if len(group_rows) <= 1:
                continue

            by_column = {
                row["source_column"]: pd.to_numeric(
                    panel_slice[row["source_column"]], errors="coerce"
                )
                for row in group_rows
            }
            for left, right in combinations(group_rows, 2):
                corr = by_column[left["source_column"]].corr(
                    by_column[right["source_column"]]
                )
                distinctness_rows.append(
                    {
                        "sleeve": str(sleeve),
                        "left_factor_id": left["factor_id"],
                        "right_factor_id": right["factor_id"],
                        "comparison_stage": "semantic_screen",
                        "comparison_type": "panel_value_correlation",
                        "semantic_group": semantic_group,
                        "abs_correlation": float(abs(corr))
                        if pd.notna(corr)
                        else np.nan,
                    }
                )

            nonraw_rows = [row for row in group_rows if row["kind"] != "raw"]
            if nonraw_rows:
                representative = sorted(
                    nonraw_rows,
                    key=lambda row: (-row["selection_score"], row["factor_id"]),
                )[0]
                for row in group_rows:
                    if (
                        row["factor_id"] == representative["factor_id"]
                        or row["kind"] != "raw"
                    ):
                        continue
                    admitted_set.discard(row["factor_id"])
                    decision_rows.append(
                        {
                            "factor_id": row["factor_id"],
                            "sleeve": str(sleeve),
                            "stage": "semantic_screen",
                            "decision": "drop",
                            "reason": "semantic_duplicate_of_grouped_or_composite",
                            "reference_factor_id": representative["factor_id"],
                        }
                    )
                continue

            if semantic_group.startswith("country_currency_pair__"):
                representative = sorted(
                    group_rows,
                    key=lambda row: (
                        row["source_column"].startswith("currency__"),
                        -row["selection_score"],
                        row["factor_id"],
                    ),
                )[0]
                for row in group_rows:
                    if row["factor_id"] == representative["factor_id"]:
                        continue
                    sibling_corr = next(
                        (
                            distinct["abs_correlation"]
                            for distinct in distinctness_rows
                            if (
                                distinct["left_factor_id"]
                                == representative["factor_id"]
                                and distinct["right_factor_id"] == row["factor_id"]
                            )
                            or (
                                distinct["left_factor_id"] == row["factor_id"]
                                and distinct["right_factor_id"]
                                == representative["factor_id"]
                            )
                        ),
                        np.nan,
                    )
                    if pd.notna(sibling_corr) and float(sibling_corr) >= 0.8:
                        admitted_set.discard(row["factor_id"])
                        decision_rows.append(
                            {
                                "factor_id": row["factor_id"],
                                "sleeve": str(sleeve),
                                "stage": "semantic_screen",
                                "decision": "drop",
                                "reason": "country_currency_near_duplicate",
                                "reference_factor_id": representative["factor_id"],
                            }
                        )

        grouped_candidates = [
            row
            for row in candidate_rows
            if row["factor_id"] in admitted_set
            and row["kind"] in {"grouped", "composite", "macro_derived"}
        ]
        raw_candidates = [
            row
            for row in candidate_rows
            if row["factor_id"] in admitted_set and row["kind"] == "raw"
        ]
        for raw_row in raw_candidates:
            raw_values = pd.to_numeric(
                panel_slice[raw_row["source_column"]], errors="coerce"
            )
            overlap_candidates = []
            for grouped_row in grouped_candidates:
                grouped_sources = set(grouped_row["source_columns"].split("|"))
                if raw_row["source_column"] not in grouped_sources:
                    continue
                grouped_values = pd.to_numeric(
                    panel_slice[grouped_row["source_column"]], errors="coerce"
                )
                corr = raw_values.corr(grouped_values)
                distinctness_rows.append(
                    {
                        "sleeve": str(sleeve),
                        "left_factor_id": raw_row["factor_id"],
                        "right_factor_id": grouped_row["factor_id"],
                        "comparison_stage": "semantic_screen",
                        "comparison_type": "source_overlap_correlation",
                        "semantic_group": grouped_row["semantic_group"],
                        "abs_correlation": float(abs(corr))
                        if pd.notna(corr)
                        else np.nan,
                    }
                )
                overlap_candidates.append((corr, grouped_row))
            if not overlap_candidates:
                continue

            representative_corr, representative = sorted(
                overlap_candidates,
                key=lambda item: (
                    pd.isna(item[0]),
                    -(abs(float(item[0])) if pd.notna(item[0]) else -1.0),
                    -item[1]["selection_score"],
                    item[1]["factor_id"],
                ),
            )[0]
            if pd.notna(representative_corr) and float(abs(representative_corr)) >= 0.8:
                admitted_set.discard(raw_row["factor_id"])
                decision_rows.append(
                    {
                        "factor_id": raw_row["factor_id"],
                        "sleeve": str(sleeve),
                        "stage": "semantic_screen",
                        "decision": "drop",
                        "reason": "semantic_duplicate_of_grouped_source_overlap",
                        "reference_factor_id": representative["factor_id"],
                    }
                )

        admitted_columns[str(sleeve)] = sorted(
            row["source_column"]
            for row in candidate_rows
            if row["factor_id"] in admitted_set
        )
        admitted_lookup = {
            _factor_id_for_column(str(sleeve), column)
            for column in admitted_columns[str(sleeve)]
        }
        for row in registry_rows:
            if row["sleeve"] != str(sleeve):
                continue
            if (
                row["factor_id"] in admitted_lookup
                and row["admission_status"] == "candidate"
            ):
                row["admission_status"] = "admitted"
                row["rejection_reason"] = ""
            elif row["admission_status"] == "candidate":
                row["admission_status"] = "rejected"
                if not row["rejection_reason"]:
                    row["rejection_reason"] = "semantic_screen"

        pre_count = len(candidate_rows)
        post_count = len(admitted_columns[str(sleeve)])
        compression_removed = max(pre_count - post_count, 0)
        compression_ratio = (
            float(compression_removed / pre_count) if pre_count > 0 else 0.0
        )
        admitted_factor_ids = {
            _factor_id_for_column(str(sleeve), column)
            for column in admitted_columns[str(sleeve)]
        }
        for row in diagnostics_rows:
            if row["sleeve"] != str(sleeve):
                continue
            row["admitted_for_construction"] = row["factor_id"] in admitted_factor_ids
            row["pre_compression_candidate_count"] = pre_count
            row["post_compression_candidate_count"] = post_count
            row["compression_removed_count"] = compression_removed
            row["compression_removed_ratio"] = compression_ratio

    return {
        "factor_registry": pd.DataFrame(registry_rows).sort_values(
            ["sleeve", "factor_id"]
        )
        if registry_rows
        else pd.DataFrame(),
        "candidate_diagnostics": pd.DataFrame(diagnostics_rows).sort_values(
            ["sleeve", "factor_id"]
        )
        if diagnostics_rows
        else pd.DataFrame(),
        "screening_decisions": pd.DataFrame(decision_rows).sort_values(
            ["sleeve", "factor_id", "stage", "decision"]
        )
        if decision_rows
        else pd.DataFrame(),
        "selection_scores": pd.DataFrame(score_rows).sort_values(
            ["sleeve", "factor_id", "stage"]
        )
        if score_rows
        else pd.DataFrame(),
        "distinctness": pd.DataFrame(distinctness_rows).sort_values(
            ["sleeve", "semantic_group", "left_factor_id", "right_factor_id"]
        )
        if distinctness_rows
        else pd.DataFrame(),
        "admitted_columns": admitted_columns,
    }


def build_factor_returns(
    panel,
    prices,
    risk_free_daily,
    config,
    show_progress=False,
    candidate_context=None,
):
    candidate_context = candidate_context or _build_candidate_context(panel, config)
    registry_lookup = (
        candidate_context["factor_registry"].set_index("factor_id").to_dict("index")
        if not candidate_context["factor_registry"].empty
        else {}
    )
    returns_wide = _build_returns_wide(prices, config.return_alignment_max_gap_days)
    risk_free_series = _build_risk_free_series(risk_free_daily)
    baseline_returns, baseline_members = _build_baseline_returns(
        panel,
        returns_wide,
        show_progress=show_progress,
        persist_progress_bars=config.persist_progress_bars,
    )
    if panel.empty or returns_wide.empty:
        return pd.DataFrame(), pd.DataFrame(), baseline_returns, baseline_members

    factor_map: dict[str, list[pd.Series]] = {}
    metadata = {}
    rebalance_dates = sorted(pd.to_datetime(panel["rebalance_date"]).unique())

    intervals = list(zip(rebalance_dates[:-1], rebalance_dates[1:]))
    for start, end in track_progress(
        intervals,
        show_progress=show_progress,
        total=len(intervals),
        desc="Factor return windows",
        unit="window",
        leave=config.persist_progress_bars,
    ):
        interval_returns = returns_wide.loc[
            (returns_wide.index > start) & (returns_wide.index <= end)
        ]
        if interval_returns.empty:
            continue

        interval_risk_free = risk_free_series.loc[
            (risk_free_series.index > start) & (risk_free_series.index <= end)
        ]
        panel_slice = panel.loc[panel["rebalance_date"] == start].copy()
        for sleeve in sorted(panel_slice["sleeve"].dropna().unique()):
            sleeve_slice = panel_slice.loc[panel_slice["sleeve"] == sleeve].copy()
            if len(sleeve_slice) < config.min_assets_per_factor:
                continue

            benchmark_factors = _build_benchmark_factors(
                sleeve_slice=sleeve_slice,
                interval_returns=interval_returns,
                risk_free_series=interval_risk_free,
                config=config,
            )
            for factor_key, series in benchmark_factors.items():
                factor_id = f"{sleeve}__{factor_key}"
                factor_map.setdefault(factor_id, []).append(series.rename(factor_id))
                metadata[factor_id] = {
                    "factor_id": factor_id,
                    "sleeve": sleeve,
                    "family": factor_key.split("__", 1)[-1],
                    "kind": "benchmark",
                    "source_column": None,
                    "semantic_group": factor_key,
                    "source_columns": "",
                    "construction_type": "benchmark_series",
                    "economic_rationale": _economic_rationale_for_column(factor_key),
                    "expected_direction": "prefer_higher",
                    "is_benchmark": True,
                    "is_macro": False,
                    "is_composite": False,
                    "admission_status": "constructed",
                    "rejection_reason": "",
                }

            size_weights = sleeve_slice.set_index("conid")[
                "profile__total_net_assets_num"
            ]
            for column in candidate_context["admitted_columns"].get(str(sleeve), []):
                coverage = sleeve_slice[column].notna().mean()
                if coverage < config.min_factor_coverage:
                    continue

                factor_id = _factor_id_for_column(str(sleeve), column)
                series = _build_long_short_series(
                    pd.to_numeric(
                        sleeve_slice.set_index("conid")[column], errors="coerce"
                    ),
                    interval_returns,
                    size_weights,
                    _factor_direction(column),
                    config.quantile,
                    config.min_assets_per_factor,
                )
                if series is None or series.abs().sum() == 0.0:
                    continue

                factor_map.setdefault(factor_id, []).append(series.rename(factor_id))
                metadata[factor_id] = {
                    "factor_id": factor_id,
                    "sleeve": sleeve,
                    "family": _factor_family(column),
                    "kind": _factor_kind(column),
                    "source_column": column,
                    "semantic_group": _semantic_group_for_column(column, sleeve_slice),
                    "source_columns": _join_source_columns(
                        _source_columns_for_feature(column, sleeve_slice)
                    ),
                    "construction_type": _construction_type_for_kind(
                        _factor_kind(column)
                    ),
                    "economic_rationale": _economic_rationale_for_column(column),
                    "expected_direction": EXPECTED_DIRECTION_LABELS.get(
                        _factor_direction(column), "prefer_higher"
                    ),
                    "is_benchmark": False,
                    "is_macro": bool(_is_macro_column(column)),
                    "is_composite": bool(column.startswith("composite__")),
                    "admission_status": "constructed",
                    "rejection_reason": "",
                }
                if factor_id in registry_lookup:
                    metadata[factor_id].update(
                        {
                            key: value
                            for key, value in registry_lookup[factor_id].items()
                            if key != "factor_id"
                        }
                    )

    if not factor_map:
        return pd.DataFrame(), pd.DataFrame(), baseline_returns, baseline_members

    factor_returns = pd.concat(
        [
            pd.concat(parts).groupby(level=0).sum().rename(factor_id)
            for factor_id, parts in factor_map.items()
        ],
        axis=1,
    ).sort_index()
    factor_meta = pd.DataFrame(
        sorted(metadata.values(), key=lambda row: row["factor_id"])
    )
    return factor_returns, factor_meta, baseline_returns, baseline_members


def cluster_factor_returns(factor_returns, factor_meta, config, show_progress=False):
    if factor_returns.empty or factor_meta.empty:
        return _empty_cluster_frame(), pd.DataFrame()

    cluster_rows = []
    keepers = []

    sleeve_count = int(factor_meta["sleeve"].nunique())
    for sleeve, meta_slice in track_progress(
        factor_meta.groupby("sleeve"),
        show_progress=show_progress,
        total=sleeve_count,
        desc="Factor clustering",
        unit="sleeve",
        leave=config.persist_progress_bars,
    ):
        factor_ids = meta_slice["factor_id"].tolist()
        sleeve_returns = factor_returns.reindex(columns=factor_ids)
        sleeve_returns = sleeve_returns.loc[
            :, sleeve_returns.notna().sum() >= config.min_train_days
        ]
        if sleeve_returns.empty:
            continue

        corr = sleeve_returns.corr().abs().fillna(0.0)
        graph = nx.Graph()
        graph.add_nodes_from(corr.columns.tolist())
        for left in corr.columns:
            for right in corr.columns:
                if left >= right:
                    continue
                if corr.loc[left, right] >= config.factor_corr_threshold:
                    graph.add_edge(left, right)

        for cluster_id, component in enumerate(nx.connected_components(graph), start=1):
            members = sorted(component)
            member_meta = meta_slice.set_index("factor_id").loc[members].reset_index()
            coverage = sleeve_returns[members].notna().sum().rename("coverage")
            member_meta = member_meta.merge(
                coverage, left_on="factor_id", right_index=True, how="left"
            )
            member_meta["kind_priority"] = (
                member_meta["kind"].map(_kind_priority).fillna(3)
            )
            representative = member_meta.sort_values(
                ["kind_priority", "coverage", "factor_id"],
                ascending=[True, False, True],
            ).iloc[0]["factor_id"]
            keepers.append(representative)
            for factor_id in members:
                cluster_rows.append(
                    {
                        "factor_id": factor_id,
                        "sleeve": sleeve,
                        "cluster_id": f"{sleeve}_{cluster_id}",
                        "cluster_representative": representative,
                        "cluster_size": len(members),
                        "keep_factor": bool(factor_id == representative),
                    }
                )

    if not cluster_rows:
        return _empty_cluster_frame(), pd.DataFrame()

    cluster_df = pd.DataFrame(cluster_rows).sort_values(
        ["sleeve", "cluster_id", "factor_id"]
    )
    reduced = factor_returns.reindex(columns=sorted(set(keepers)))
    return cluster_df, reduced


def _fit_elastic_net(X_train, y_train):
    elastic_net_params: Any = {
        "alphas": [float(alpha) for alpha in np.logspace(-4, 0, 20)],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        "cv": 5,
        "random_state": 42,
        "max_iter": 20000,
    }
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("enet", ElasticNetCV(**elastic_net_params)),
        ]
    )
    pipeline.fit(X_train, y_train)
    scaler: Any = pipeline.named_steps["scaler"]
    model: Any = pipeline.named_steps["enet"]
    coefs = model.coef_ / scaler.scale_
    intercept = model.intercept_ - np.dot(coefs, scaler.mean_)
    return pipeline, model, intercept, coefs


def _build_research_windows(
    panel, reduced_factors, config
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, int]]:
    if panel.empty or reduced_factors.empty:
        return []

    rebalance_dates = sorted(pd.to_datetime(panel["rebalance_date"].dropna().unique()))
    if len(rebalance_dates) < 2:
        return []
    first_factor_date = pd.to_datetime(reduced_factors.index.min())

    windows = []
    last_step_end = None
    for train_end, test_end in zip(rebalance_dates[:-1], rebalance_dates[1:]):
        if test_end <= first_factor_date:
            continue
        if last_step_end is not None:
            month_gap = (test_end.year - last_step_end.year) * 12 + (
                test_end.month - last_step_end.month
            )
            if month_gap < config.walk_forward_step_months:
                continue
        for years in sorted(set(int(year) for year in config.training_window_years)):
            train_start = pd.Timestamp(train_end) - pd.DateOffset(years=years)
            windows.append(
                (
                    pd.Timestamp(train_start),
                    pd.Timestamp(train_end),
                    pd.Timestamp(test_end),
                    int(years),
                )
            )
        last_step_end = pd.Timestamp(test_end)
    return windows


def _factor_diagnostics(factor_returns, reduced_factors, factor_meta):
    if factor_returns.empty:
        return pd.DataFrame()
    diagnostics = pd.DataFrame(
        {
            "factor_id": factor_returns.columns.astype(str),
            "coverage_ratio": factor_returns.notna().mean().to_numpy(dtype=float),
            "mean_return": factor_returns.mean().to_numpy(dtype=float),
            "volatility": factor_returns.std().to_numpy(dtype=float),
            "selected_for_model": factor_returns.columns.isin(reduced_factors.columns),
        }
    )
    if not factor_meta.empty:
        diagnostics = diagnostics.merge(factor_meta, on="factor_id", how="left")
    return diagnostics.sort_values("factor_id").reset_index(drop=True)


def _finalize_factor_registry(candidate_context, factor_meta):
    registry = candidate_context["factor_registry"].copy()
    if factor_meta.empty:
        return registry

    built_ids = set(factor_meta["factor_id"].astype(str))
    if not registry.empty:
        registry.loc[registry["factor_id"].isin(built_ids), "admission_status"] = (
            "constructed"
        )
        registry.loc[registry["factor_id"].isin(built_ids), "rejection_reason"] = ""
        admitted_mask = registry["admission_status"].eq("admitted") & ~registry[
            "factor_id"
        ].isin(built_ids)
        registry.loc[admitted_mask, "admission_status"] = "rejected"
        registry.loc[admitted_mask, "rejection_reason"] = "construction_filtered"

    benchmark_meta = factor_meta.loc[
        ~factor_meta["factor_id"].isin(registry["factor_id"])
        if not registry.empty
        else slice(None)
    ]
    benchmark_rows = benchmark_meta.to_dict("records")
    if benchmark_rows:
        registry = pd.concat(
            [registry, pd.DataFrame(benchmark_rows)], ignore_index=True
        )
    if registry.empty:
        return registry
    columns = [
        "factor_id",
        "sleeve",
        "family",
        "semantic_group",
        "kind",
        "source_columns",
        "construction_type",
        "economic_rationale",
        "expected_direction",
        "is_benchmark",
        "is_macro",
        "is_composite",
        "admission_status",
        "rejection_reason",
    ]
    for column in columns:
        if column not in registry.columns:
            registry[column] = ""
    return registry[columns].sort_values(["sleeve", "factor_id"]).reset_index(drop=True)


def _build_cluster_selection_scores(cluster_df, factor_returns, factor_meta):
    if cluster_df.empty or factor_meta.empty:
        return pd.DataFrame()
    coverage = factor_returns.notna().mean().rename("coverage_component")
    volatility = factor_returns.std().rename("volatility_component")
    score_df = cluster_df.merge(factor_meta, on=["factor_id", "sleeve"], how="left")
    score_df = score_df.merge(
        coverage, left_on="factor_id", right_index=True, how="left"
    )
    score_df = score_df.merge(
        volatility, left_on="factor_id", right_index=True, how="left"
    )
    score_df["kind_priority_component"] = score_df["kind"].map(
        lambda kind: float(max(0, 4 - _kind_priority(kind)))
    )
    score_df["selection_score"] = (
        score_df["kind_priority_component"] * 100.0
        + score_df["coverage_component"].fillna(0.0) * 10.0
        + score_df["volatility_component"].fillna(0.0)
    )
    score_df["stage"] = "cluster_representative"
    return (
        score_df[
            [
                "factor_id",
                "sleeve",
                "stage",
                "kind_priority_component",
                "coverage_component",
                "volatility_component",
                "selection_score",
                "cluster_id",
                "cluster_representative",
                "keep_factor",
            ]
        ]
        .sort_values(["sleeve", "cluster_id", "factor_id"])
        .reset_index(drop=True)
    )


def _build_factor_return_distinctness(factor_returns, factor_meta, config):
    if factor_returns.empty or factor_meta.empty:
        return pd.DataFrame()
    rows = []
    for sleeve, meta_slice in factor_meta.groupby("sleeve", sort=True):
        factor_ids = meta_slice["factor_id"].tolist()
        sleeve_returns = factor_returns.reindex(columns=factor_ids)
        corr = sleeve_returns.corr().abs()
        for left, right in combinations(sorted(factor_ids), 2):
            value = corr.loc[left, right]
            if not np.isfinite(value) or float(value) < config.factor_corr_threshold:
                continue
            rows.append(
                {
                    "sleeve": str(sleeve),
                    "left_factor_id": left,
                    "right_factor_id": right,
                    "comparison_stage": "statistical_duplication",
                    "comparison_type": "factor_return_correlation",
                    "semantic_group": "",
                    "abs_correlation": float(value),
                }
            )
    return (
        pd.DataFrame(rows).sort_values(["sleeve", "left_factor_id", "right_factor_id"])
        if rows
        else pd.DataFrame()
    )


def _build_model_telemetry(model_results, selections, persistence, factor_meta):
    if model_results.empty and selections.empty and persistence.empty:
        return pd.DataFrame()

    fit_counts = (
        model_results.groupby("sleeve").size().rename("model_fit_count")
        if not model_results.empty
        else pd.Series(dtype=float)
    )
    telemetry = (
        factor_meta[["factor_id", "sleeve", "family", "kind"]].drop_duplicates().copy()
        if not factor_meta.empty
        else pd.DataFrame(columns=["factor_id", "sleeve", "family", "kind"])
    )
    if not selections.empty:
        selection_stats = selections.groupby(
            ["sleeve", "factor_id"], as_index=False
        ).agg(
            selection_count=("factor_id", "size"),
            mean_abs_beta=("abs_beta", "mean"),
            median_abs_beta=("abs_beta", "median"),
            sign_consistency=("sign", lambda s: float(abs(np.nanmean(s)))),
        )
        telemetry = telemetry.merge(
            selection_stats, on=["sleeve", "factor_id"], how="outer"
        )
        if not model_results.empty:
            selection_with_results = selections.merge(
                model_results[
                    [
                        "sleeve",
                        "conid",
                        "train_start",
                        "train_end",
                        "test_end",
                        "training_window_years",
                        "mse_test",
                        "r2_test",
                    ]
                ],
                on=[
                    "sleeve",
                    "conid",
                    "train_start",
                    "train_end",
                    "test_end",
                    "training_window_years",
                ],
                how="left",
            )
            usefulness_stats = selection_with_results.groupby(
                ["sleeve", "factor_id"], as_index=False
            ).agg(
                mean_selected_r2_test=("r2_test", "mean"),
                mean_selected_mse_test=("mse_test", "mean"),
            )
            telemetry = telemetry.merge(
                usefulness_stats, on=["sleeve", "factor_id"], how="left"
            )
    if not persistence.empty:
        telemetry = telemetry.merge(
            persistence[
                [
                    "sleeve",
                    "factor_id",
                    "selection_frequency",
                    "is_persistent",
                    "model_fit_count",
                ]
            ],
            on=["sleeve", "factor_id"],
            how="outer",
        )
    if not fit_counts.empty:
        telemetry = telemetry.merge(
            fit_counts.rename_axis("sleeve").reset_index(),
            on="sleeve",
            how="left",
            suffixes=("", "_fallback"),
        )
        if "model_fit_count_fallback" in telemetry.columns:
            telemetry["model_fit_count"] = telemetry["model_fit_count"].fillna(
                telemetry["model_fit_count_fallback"]
            )
            telemetry = telemetry.drop(columns=["model_fit_count_fallback"])
    numeric_defaults = [
        "selection_count",
        "mean_abs_beta",
        "median_abs_beta",
        "sign_consistency",
        "selection_frequency",
        "model_fit_count",
        "mean_selected_r2_test",
        "mean_selected_mse_test",
    ]
    for column in numeric_defaults:
        if column not in telemetry.columns:
            telemetry[column] = np.nan
    if "is_persistent" not in telemetry.columns:
        telemetry["is_persistent"] = False
    else:
        telemetry["is_persistent"] = (
            telemetry["is_persistent"]
            .where(telemetry["is_persistent"].notna(), False)
            .astype(bool)
        )
    return telemetry.sort_values(["sleeve", "factor_id"]).reset_index(drop=True)


def _compute_factor_vif(factor_frame):
    if factor_frame.empty:
        return pd.DataFrame(columns=["factor_id", "vif"])

    clean = factor_frame.dropna()
    if clean.empty:
        return pd.DataFrame(
            {"factor_id": factor_frame.columns.astype(str), "vif": np.nan}
        )

    rows = []
    for factor_id in clean.columns.astype(str):
        remaining = [col for col in clean.columns.astype(str) if col != factor_id]
        if not remaining:
            rows.append({"factor_id": factor_id, "vif": 1.0})
            continue
        X = clean[remaining].to_numpy(dtype=float)
        y = clean[factor_id].to_numpy(dtype=float)
        if X.shape[0] <= X.shape[1]:
            rows.append({"factor_id": factor_id, "vif": np.nan})
            continue
        model = LinearRegression()
        model.fit(X, y)
        r2 = float(model.score(X, y))
        vif = np.inf if r2 >= 1.0 - 1e-10 else 1.0 / max(1.0 - r2, 1e-10)
        rows.append({"factor_id": factor_id, "vif": float(vif)})
    return pd.DataFrame(rows)


def _build_final_vif_diagnostics(
    final_factors, factor_meta, factor_model_telemetry, model_usefulness_scores, config
):
    columns = [
        "factor_id",
        "sleeve",
        "family",
        "semantic_group",
        "kind",
        "source_columns",
        "is_benchmark",
        "selected_for_model",
        "final_vif",
        "max_final_vif",
        "exceeds_max_final_vif",
        "vif_rank_within_sleeve",
        "next_nonbenchmark_vif_drop_candidate",
        "final_factor_count",
        "max_abs_corr_with_final_set",
        "max_abs_corr_factor_id",
        "selection_score",
        "selection_count",
        "selection_frequency",
        "sign_consistency",
        "mean_abs_beta",
        "mean_selected_r2_test",
        "model_fit_count",
    ]
    if final_factors.empty or factor_meta.empty:
        return pd.DataFrame(columns=columns)

    meta_columns = [
        "factor_id",
        "sleeve",
        "family",
        "semantic_group",
        "kind",
        "source_columns",
        "is_benchmark",
    ]
    meta = factor_meta.copy()
    meta_defaults = {
        "factor_id": "",
        "sleeve": "",
        "family": "",
        "semantic_group": "",
        "kind": "",
        "source_columns": "",
        "is_benchmark": False,
    }
    for column, default in meta_defaults.items():
        if column not in meta.columns:
            meta[column] = default
    meta = meta.loc[meta["factor_id"].isin(final_factors.columns), meta_columns]
    meta = meta.drop_duplicates(subset=["factor_id", "sleeve"])
    if meta.empty:
        return pd.DataFrame(columns=columns)

    telemetry_columns = [
        "factor_id",
        "sleeve",
        "selection_count",
        "selection_frequency",
        "sign_consistency",
        "mean_abs_beta",
        "mean_selected_r2_test",
        "model_fit_count",
    ]
    telemetry = (
        factor_model_telemetry[
            [column for column in telemetry_columns if column in factor_model_telemetry]
        ]
        .drop_duplicates(subset=["factor_id", "sleeve"])
        .copy()
        if not factor_model_telemetry.empty
        else pd.DataFrame(columns=telemetry_columns)
    )
    score_df = (
        model_usefulness_scores[
            ["factor_id", "sleeve", "selection_score"]
        ].drop_duplicates(subset=["factor_id", "sleeve"])
        if not model_usefulness_scores.empty
        else pd.DataFrame(columns=["factor_id", "sleeve", "selection_score"])
    )

    rows = []
    max_final_vif = float(config.max_final_vif)
    for sleeve, meta_slice in meta.groupby("sleeve", sort=True):
        sleeve_factor_ids = [
            factor_id
            for factor_id in meta_slice["factor_id"].astype(str).tolist()
            if factor_id in final_factors.columns
        ]
        if not sleeve_factor_ids:
            continue

        sleeve_returns = final_factors.reindex(columns=sleeve_factor_ids)
        vif_df = _compute_factor_vif(sleeve_returns).rename(
            columns={"vif": "final_vif"}
        )
        if vif_df.empty:
            vif_df = pd.DataFrame(
                {
                    "factor_id": sleeve_factor_ids,
                    "final_vif": np.nan,
                }
            )

        corr = sleeve_returns.corr().abs()
        corr_rows = []
        for factor_id in sleeve_factor_ids:
            max_corr = np.nan
            max_corr_factor_id = ""
            if factor_id in corr.index:
                factor_corr = corr.loc[factor_id].drop(
                    labels=[factor_id], errors="ignore"
                )
                factor_corr = factor_corr.dropna()
                if not factor_corr.empty:
                    factor_corr = factor_corr.sort_values(ascending=False)
                    max_corr_factor_id = str(factor_corr.index[0])
                    max_corr = float(factor_corr.iloc[0])
            corr_rows.append(
                {
                    "factor_id": factor_id,
                    "max_abs_corr_with_final_set": max_corr,
                    "max_abs_corr_factor_id": max_corr_factor_id,
                }
            )
        corr_df = pd.DataFrame(corr_rows)

        sleeve_df = (
            meta_slice.merge(vif_df, on="factor_id", how="left")
            .merge(corr_df, on="factor_id", how="left")
            .merge(telemetry, on=["factor_id", "sleeve"], how="left")
            .merge(score_df, on=["factor_id", "sleeve"], how="left")
        )
        if "is_benchmark" not in sleeve_df.columns:
            sleeve_df["is_benchmark"] = False
        sleeve_df["is_benchmark"] = (
            sleeve_df["is_benchmark"]
            .where(sleeve_df["is_benchmark"].notna(), False)
            .astype(bool)
        )
        sleeve_df["selected_for_model"] = True
        sleeve_df["max_final_vif"] = max_final_vif
        sleeve_df["exceeds_max_final_vif"] = sleeve_df["final_vif"] > max_final_vif
        sleeve_df["final_factor_count"] = int(len(sleeve_factor_ids))
        ordered = sleeve_df.sort_values(
            ["final_vif", "selection_score", "factor_id"],
            ascending=[False, True, True],
            na_position="last",
        )
        rank_lookup = {
            factor_id: rank
            for rank, factor_id in enumerate(ordered["factor_id"].astype(str), start=1)
        }
        sleeve_df["vif_rank_within_sleeve"] = (
            sleeve_df["factor_id"].astype(str).map(rank_lookup).astype("Int64")
        )

        non_bench = ordered.loc[~ordered["is_benchmark"]]
        next_drop_factor_id = (
            str(non_bench.iloc[0]["factor_id"]) if not non_bench.empty else None
        )
        sleeve_df["next_nonbenchmark_vif_drop_candidate"] = (
            sleeve_df["factor_id"].astype(str).eq(next_drop_factor_id)
        )
        rows.extend(sleeve_df[columns].to_dict("records"))

    if not rows:
        return pd.DataFrame(columns=columns)
    diagnostics = pd.DataFrame(rows)
    return diagnostics.sort_values(["sleeve", "vif_rank_within_sleeve", "factor_id"])


def _build_model_usefulness_scores(factor_model_telemetry, factor_meta):
    if factor_model_telemetry.empty:
        return pd.DataFrame()

    telemetry = factor_model_telemetry.copy()
    if not factor_meta.empty:
        telemetry = telemetry.merge(
            factor_meta[["factor_id", "sleeve", "kind"]].drop_duplicates(),
            on=["factor_id", "sleeve"],
            how="left",
            suffixes=("", "_meta"),
        )
        if "kind_meta" in telemetry.columns:
            telemetry["kind"] = telemetry["kind"].fillna(telemetry["kind_meta"])
            telemetry = telemetry.drop(columns=["kind_meta"])

    telemetry["kind_priority_component"] = telemetry["kind"].map(
        lambda kind: float(max(0, 4 - _kind_priority(kind)))
    )
    telemetry["nonzero_frequency_component"] = telemetry["selection_frequency"].fillna(
        0.0
    )
    telemetry["sign_consistency_component"] = telemetry["sign_consistency"].fillna(0.0)
    telemetry["abs_beta_component"] = np.log1p(telemetry["mean_abs_beta"].fillna(0.0))
    telemetry["oos_usefulness_component"] = (
        telemetry["mean_selected_r2_test"].fillna(0.0).clip(lower=0.0)
    )
    telemetry["selection_score"] = (
        telemetry["kind_priority_component"] * 100.0
        + telemetry["nonzero_frequency_component"] * 50.0
        + telemetry["sign_consistency_component"] * 20.0
        + telemetry["abs_beta_component"] * 10.0
        + telemetry["oos_usefulness_component"] * 10.0
    )
    telemetry["stage"] = "model_usefulness"
    return (
        telemetry[
            [
                "factor_id",
                "sleeve",
                "stage",
                "kind_priority_component",
                "nonzero_frequency_component",
                "sign_consistency_component",
                "abs_beta_component",
                "oos_usefulness_component",
                "selection_score",
            ]
        ]
        .sort_values(["sleeve", "factor_id"])
        .reset_index(drop=True)
    )


def _select_final_factor_set(
    reduced_factors, factor_meta, factor_model_telemetry, config
):
    if reduced_factors.empty or factor_meta.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    score_df = _build_model_usefulness_scores(factor_model_telemetry, factor_meta)
    score_lookup = (
        score_df.set_index(["sleeve", "factor_id"])["selection_score"].to_dict()
        if not score_df.empty
        else {}
    )
    telemetry_lookup = (
        factor_model_telemetry.set_index(["sleeve", "factor_id"]).to_dict("index")
        if not factor_model_telemetry.empty
        else {}
    )

    kept_factor_ids = []
    decision_rows = []
    vif_rows = []

    for sleeve, meta_slice in factor_meta.groupby("sleeve", sort=True):
        sleeve_factor_ids = [
            factor_id
            for factor_id in meta_slice["factor_id"].astype(str).tolist()
            if factor_id in reduced_factors.columns
        ]
        if not sleeve_factor_ids:
            continue

        benchmarks = sorted(
            meta_slice.loc[
                meta_slice["kind"].astype(str).eq("benchmark"), "factor_id"
            ].astype(str)
        )
        non_bench = sorted(set(sleeve_factor_ids) - set(benchmarks))

        persistent = [
            factor_id
            for factor_id in non_bench
            if bool(
                telemetry_lookup.get((str(sleeve), factor_id), {}).get(
                    "is_persistent", False
                )
            )
        ]
        if persistent:
            shortlisted = persistent
        else:
            useful = [
                factor_id
                for factor_id in non_bench
                if float(
                    telemetry_lookup.get((str(sleeve), factor_id), {}).get(
                        "selection_count", 0.0
                    )
                )
                > 0.0
            ]
            useful = sorted(
                useful,
                key=lambda factor_id: (
                    -score_lookup.get((str(sleeve), factor_id), 0.0),
                    factor_id,
                ),
            )
            shortlisted = useful[: min(3, len(useful))]

        chosen = sorted(set(benchmarks) | set(shortlisted))
        chosen_set = set(chosen)
        for factor_id in sleeve_factor_ids:
            decision_rows.append(
                {
                    "factor_id": factor_id,
                    "sleeve": str(sleeve),
                    "stage": "final_model_selection",
                    "decision": "keep" if factor_id in chosen_set else "drop",
                    "reason": (
                        "benchmark_retained"
                        if factor_id in benchmarks
                        else "model_usefulness_selected"
                        if factor_id in chosen_set
                        else "not_selected_by_model_usefulness"
                    ),
                    "reference_factor_id": "",
                }
            )

        current = chosen.copy()
        while len(current) > 2:
            vif_df = _compute_factor_vif(reduced_factors.reindex(columns=current))
            if vif_df.empty or vif_df["vif"].dropna().empty:
                break
            max_vif = float(vif_df["vif"].dropna().max())
            if max_vif <= float(config.max_final_vif):
                break
            droppable = [
                factor_id
                for factor_id in vif_df.loc[
                    vif_df["vif"] > float(config.max_final_vif), "factor_id"
                ]
                .astype(str)
                .tolist()
                if factor_id not in benchmarks
            ]
            if not droppable:
                break
            drop_factor = sorted(
                droppable,
                key=lambda factor_id: (
                    score_lookup.get((str(sleeve), factor_id), 0.0),
                    factor_id,
                ),
            )[0]
            current.remove(drop_factor)
            decision_rows.append(
                {
                    "factor_id": drop_factor,
                    "sleeve": str(sleeve),
                    "stage": "final_vif",
                    "decision": "drop",
                    "reason": "max_final_vif_exceeded",
                    "reference_factor_id": "",
                }
            )

        final_vif = _compute_factor_vif(reduced_factors.reindex(columns=current))
        if not final_vif.empty:
            final_vif["sleeve"] = str(sleeve)
            vif_rows.extend(final_vif.to_dict("records"))
        kept_factor_ids.extend(current)

    final_factors = reduced_factors.reindex(
        columns=sorted(dict.fromkeys(kept_factor_ids))
    )
    final_vif_df = (
        pd.DataFrame(vif_rows)
        .sort_values(["sleeve", "factor_id"])
        .reset_index(drop=True)
        if vif_rows
        else pd.DataFrame(columns=["factor_id", "vif", "sleeve"])
    )
    final_screening = (
        pd.DataFrame(decision_rows)
        .sort_values(["sleeve", "factor_id", "stage", "decision"])
        .reset_index(drop=True)
        if decision_rows
        else pd.DataFrame()
    )
    vif_score_rows = (
        final_vif_df.assign(
            stage="final_vif",
            vif_component=final_vif_df["vif"],
            selection_score=-final_vif_df["vif"].fillna(np.inf),
        )[["factor_id", "sleeve", "stage", "vif_component", "selection_score"]]
        if not final_vif_df.empty
        else pd.DataFrame()
    )
    return final_factors, final_screening, score_df, vif_score_rows


def _build_expected_return_outputs(current_betas, reduced_factors):
    if current_betas.empty or reduced_factors.empty:
        return pd.DataFrame(), pd.DataFrame()

    factor_premia = reduced_factors.mean()
    beta_columns = [col for col in current_betas.columns if col.startswith("beta__")]
    long_rows = []
    expected_rows = []
    for _, row in current_betas.iterrows():
        expected_return = float(row.get("alpha", 0.0))
        for beta_column in beta_columns:
            factor_id = beta_column.replace("beta__", "", 1)
            beta_value = pd.to_numeric(row.get(beta_column), errors="coerce")
            if not np.isfinite(beta_value):
                continue
            premium = float(
                pd.to_numeric(factor_premia.get(factor_id), errors="coerce")
            )
            if np.isfinite(premium):
                expected_return += float(beta_value) * premium
            long_rows.append(
                {
                    "conid": row["conid"],
                    "sleeve": row["sleeve"],
                    "factor_id": factor_id,
                    "beta": float(beta_value),
                }
            )
        expected_rows.append(
            {
                "conid": row["conid"],
                "sleeve": row["sleeve"],
                "alpha": float(pd.to_numeric(row.get("alpha"), errors="coerce")),
                "expected_return": float(expected_return),
            }
        )
    return pd.DataFrame(expected_rows), pd.DataFrame(long_rows)


def run_factor_research_data(
    panel,
    prices,
    risk_free_daily,
    config,
    show_progress=False,
):
    candidate_context = _build_candidate_context(panel, config)
    factor_returns, factor_meta, baseline_returns, baseline_members = (
        build_factor_returns(
            panel,
            prices,
            risk_free_daily,
            config,
            show_progress=show_progress,
            candidate_context=candidate_context,
        )
    )
    cluster_df, reduced_factors = cluster_factor_returns(
        factor_returns, factor_meta, config, show_progress=show_progress
    )
    factor_registry = _finalize_factor_registry(candidate_context, factor_meta)
    distinctness_frames = [
        df
        for df in [
            candidate_context["distinctness"],
            _build_factor_return_distinctness(factor_returns, factor_meta, config),
        ]
        if not df.empty
    ]
    factor_distinctness = (
        pd.concat(distinctness_frames, ignore_index=True)
        if distinctness_frames
        else pd.DataFrame()
    )
    returns_wide = _build_returns_wide(prices, config.return_alignment_max_gap_days)
    risk_free_series = _build_risk_free_series(risk_free_daily)

    if factor_returns.empty or reduced_factors.empty or returns_wide.empty:
        empty = pd.DataFrame()
        return {
            "factor_returns": factor_returns,
            "factor_meta": factor_meta,
            "factor_registry": factor_registry,
            "factor_candidate_diagnostics": candidate_context["candidate_diagnostics"],
            "factor_distinctness": factor_distinctness,
            "factor_selection_scores": candidate_context["selection_scores"],
            "factor_screening_decisions": candidate_context["screening_decisions"],
            "factor_clusters": cluster_df,
            "factor_cluster_membership": cluster_df,
            "factor_diagnostics": _factor_diagnostics(
                factor_returns, reduced_factors, factor_meta
            ),
            "factor_final_vif_diagnostics": empty,
            "baseline_returns": baseline_returns,
            "baseline_members": baseline_members,
            "model_results": empty,
            "factor_model_telemetry": empty,
            "factor_persistence": empty,
            "current_betas": empty,
            "asset_expected_returns": empty,
            "asset_factor_betas": empty,
        }

    research_rows = []
    selection_rows = []
    train_test_windows = _build_research_windows(panel, reduced_factors, config)
    sleeve_specs = []
    for sleeve in sorted(panel["sleeve"].dropna().unique()):
        sleeve_conids = sorted(
            panel.loc[panel["sleeve"] == sleeve, "conid"].astype(str).unique()
        )
        sleeve_factors = sorted(
            [col for col in reduced_factors.columns if col.startswith(f"{sleeve}__")]
        )
        if sleeve_conids and sleeve_factors:
            sleeve_specs.append((sleeve, sleeve_conids, sleeve_factors))

    total_model_fits = sum(
        len(sleeve_conids) * len(train_test_windows)
        for _, sleeve_conids, _ in sleeve_specs
    )
    with make_progress_bar(
        show_progress=show_progress,
        total=total_model_fits,
        desc="Research model fits",
        unit="fit",
        leave=config.persist_progress_bars,
    ) as progress_bar:
        for sleeve, sleeve_conids, sleeve_factors in sleeve_specs:
            progress_bar.set_postfix_str(str(sleeve), refresh=False)
            for train_start, train_end, test_end, window_years in train_test_windows:
                X_train = reduced_factors.loc[
                    (reduced_factors.index > train_start)
                    & (reduced_factors.index <= train_end),
                    sleeve_factors,
                ].dropna(how="all")
                X_test = reduced_factors.loc[
                    (reduced_factors.index > train_end)
                    & (reduced_factors.index <= test_end),
                    sleeve_factors,
                ].dropna(how="all")
                if (
                    len(X_train) < config.min_train_days
                    or len(X_test) < config.min_test_days
                ):
                    progress_bar.update(len(sleeve_conids))
                    continue

                for conid in sleeve_conids:
                    y = returns_wide.get(conid)
                    if y is None:
                        progress_bar.update(1)
                        continue
                    if config.use_risk_free_excess:
                        y_target = y.subtract(risk_free_series, fill_value=0.0)
                    else:
                        y_target = y.copy()

                    train = pd.concat(
                        [X_train, y_target.rename("target")], axis=1
                    ).dropna()
                    test = pd.concat(
                        [X_test, y_target.rename("target")], axis=1
                    ).dropna()
                    if (
                        len(train) < config.min_train_days
                        or len(test) < config.min_test_days
                    ):
                        progress_bar.update(1)
                        continue

                    X_train_fit = np.asarray(
                        train[sleeve_factors].to_numpy(), dtype=float
                    )
                    y_train_fit = np.asarray(train["target"].to_numpy(), dtype=float)
                    X_test_fit = np.asarray(
                        test[sleeve_factors].to_numpy(), dtype=float
                    )
                    y_test_fit = np.asarray(test["target"].to_numpy(), dtype=float)

                    try:
                        pipeline, model, intercept, coefs = _fit_elastic_net(
                            X_train_fit, y_train_fit
                        )
                    except ValueError:
                        progress_bar.update(1)
                        continue

                    pred_train = pipeline.predict(X_train_fit)
                    pred_test = pipeline.predict(X_test_fit)
                    selected = []
                    nonzero = int(np.sum(np.abs(coefs) > 1e-8))
                    for factor_id, beta in zip(sleeve_factors, coefs):
                        if abs(beta) <= 1e-8:
                            continue
                        selected.append(factor_id)
                        selection_rows.append(
                            {
                                "sleeve": sleeve,
                                "conid": conid,
                                "train_start": train_start,
                                "train_end": train_end,
                                "test_end": test_end,
                                "training_window_years": window_years,
                                "factor_id": factor_id,
                                "beta": float(beta),
                                "abs_beta": float(abs(beta)),
                                "sign": float(np.sign(beta)),
                            }
                        )

                    research_row = {
                        "sleeve": sleeve,
                        "conid": conid,
                        "train_start": train_start,
                        "train_end": train_end,
                        "test_end": test_end,
                        "training_window_years": window_years,
                        "alpha": float(intercept),
                        "enet_alpha": float(model.alpha_),
                        "l1_ratio": float(model.l1_ratio_),
                        "n_iter": int(model.n_iter_),
                        "dual_gap": float(model.dual_gap_),
                        "n_nonzero": nonzero,
                        "selected_factor_count": int(len(selected)),
                        "selected_factors": "|".join(sorted(selected)),
                        "mse_train": float(mean_squared_error(y_train_fit, pred_train)),
                        "mse_test": float(mean_squared_error(y_test_fit, pred_test)),
                        "r2_train": float(r2_score(y_train_fit, pred_train)),
                        "r2_test": float(r2_score(y_test_fit, pred_test)),
                        "cv_mse_best": float(np.min(model.mse_path_.mean(axis=1))),
                        "cv_mse_average": float(np.mean(model.mse_path_.mean(axis=1))),
                        "cv_mse_worst": float(np.max(model.mse_path_.mean(axis=1))),
                    }
                    for factor_id, beta in zip(sleeve_factors, coefs):
                        research_row[f"beta__{factor_id}"] = float(beta)
                    research_rows.append(research_row)
                    progress_bar.update(1)

    model_results = pd.DataFrame(research_rows)
    selections = pd.DataFrame(selection_rows)
    persistence = pd.DataFrame()
    if not selections.empty and not model_results.empty:
        sleeve_fit_counts = model_results.groupby("sleeve").size()
        fit_counts = pd.DataFrame(
            {
                "sleeve": sleeve_fit_counts.index.astype(str),
                "model_fit_count": sleeve_fit_counts.to_numpy(),
            }
        )
        persistence = selections.groupby(["sleeve", "factor_id"], as_index=False).agg(
            selection_count=("factor_id", "size"),
            median_abs_beta=("abs_beta", "median"),
            sign_consistency=("sign", lambda s: float(abs(np.nanmean(s)))),
        )
        persistence = persistence.merge(fit_counts, on="sleeve", how="left")
        persistence["selection_frequency"] = (
            persistence["selection_count"] / persistence["model_fit_count"]
        )
        persistence["is_persistent"] = (
            persistence["selection_count"] >= config.min_selection_count
        ) & (persistence["selection_frequency"] >= config.selection_frequency_threshold)
    factor_model_telemetry = _build_model_telemetry(
        model_results, selections, persistence, factor_meta
    )
    (
        final_factors,
        final_selection_decisions,
        model_usefulness_scores,
        final_vif_scores,
    ) = _select_final_factor_set(
        reduced_factors, factor_meta, factor_model_telemetry, config
    )
    if final_factors.empty:
        final_factors = reduced_factors.copy()
    if not persistence.empty:
        persistence = persistence.loc[
            persistence["factor_id"].isin(final_factors.columns)
        ].reset_index(drop=True)
    final_selected_df = (
        factor_meta.loc[
            factor_meta["factor_id"].isin(final_factors.columns),
            ["factor_id", "sleeve"],
        ]
        .drop_duplicates()
        .assign(final_selected=True)
        if not final_factors.empty
        else pd.DataFrame(columns=["factor_id", "sleeve", "final_selected"])
    )
    factor_model_telemetry = (
        factor_model_telemetry.merge(
            final_selected_df,
            on=["factor_id", "sleeve"],
            how="left",
        )
        if not factor_model_telemetry.empty
        else factor_model_telemetry.assign(final_selected=False)
    )
    if "final_selected" in factor_model_telemetry.columns:
        factor_model_telemetry["final_selected"] = (
            factor_model_telemetry["final_selected"]
            .where(factor_model_telemetry["final_selected"].notna(), False)
            .astype(bool)
        )
    if not final_vif_scores.empty:
        factor_model_telemetry = factor_model_telemetry.merge(
            final_vif_scores[["factor_id", "sleeve", "vif_component"]].rename(
                columns={"vif_component": "final_vif"}
            ),
            on=["factor_id", "sleeve"],
            how="left",
        )
    elif "final_vif" not in factor_model_telemetry.columns:
        factor_model_telemetry["final_vif"] = np.nan
    factor_diagnostics = _factor_diagnostics(factor_returns, final_factors, factor_meta)
    factor_final_vif_diagnostics = _build_final_vif_diagnostics(
        final_factors,
        factor_meta,
        factor_model_telemetry,
        model_usefulness_scores,
        config,
    )

    current_betas = compute_current_betas_data(
        panel=panel,
        prices=prices,
        reduced_factors=final_factors,
        risk_free_daily=risk_free_daily,
        persistence=persistence,
        config=config,
        show_progress=show_progress,
    )
    asset_expected_returns, asset_factor_betas = _build_expected_return_outputs(
        current_betas, final_factors
    )
    selection_score_frames = [
        df
        for df in [
            candidate_context["selection_scores"],
            _build_cluster_selection_scores(cluster_df, factor_returns, factor_meta),
            model_usefulness_scores,
            final_vif_scores,
        ]
        if not df.empty
    ]
    factor_selection_scores = (
        pd.concat(selection_score_frames, ignore_index=True)
        if selection_score_frames
        else pd.DataFrame()
    )
    screening_decision_frames = [
        df
        for df in [
            candidate_context["screening_decisions"],
            cluster_df.assign(
                stage="statistical_duplication",
                decision=np.where(cluster_df["keep_factor"], "keep", "drop"),
                reason=np.where(
                    cluster_df["keep_factor"],
                    "cluster_representative",
                    "correlated_cluster_member",
                ),
                reference_factor_id=cluster_df["cluster_representative"],
            )[
                [
                    "factor_id",
                    "sleeve",
                    "stage",
                    "decision",
                    "reason",
                    "reference_factor_id",
                ]
            ],
            final_selection_decisions,
        ]
        if not df.empty
    ]
    factor_screening_decisions = (
        pd.concat(screening_decision_frames, ignore_index=True)
        if screening_decision_frames
        else pd.DataFrame()
    )

    return {
        "factor_returns": factor_returns,
        "factor_meta": factor_meta,
        "factor_registry": factor_registry,
        "factor_candidate_diagnostics": candidate_context["candidate_diagnostics"],
        "factor_distinctness": factor_distinctness,
        "factor_selection_scores": factor_selection_scores,
        "factor_screening_decisions": factor_screening_decisions,
        "factor_clusters": cluster_df,
        "factor_cluster_membership": cluster_df,
        "factor_diagnostics": factor_diagnostics,
        "factor_final_vif_diagnostics": factor_final_vif_diagnostics,
        "baseline_returns": baseline_returns,
        "baseline_members": baseline_members,
        "model_results": model_results,
        "factor_model_telemetry": factor_model_telemetry,
        "factor_persistence": persistence,
        "current_betas": current_betas,
        "asset_expected_returns": asset_expected_returns,
        "asset_factor_betas": asset_factor_betas,
    }


def compute_current_betas_data(
    panel,
    prices,
    reduced_factors,
    risk_free_daily,
    persistence,
    config,
    show_progress=False,
):
    returns_wide = _build_returns_wide(prices, config.return_alignment_max_gap_days)
    risk_free_series = _build_risk_free_series(risk_free_daily)
    if returns_wide.empty or reduced_factors.empty or persistence.empty:
        return pd.DataFrame()

    latest_rebalance = _to_timestamp(panel["rebalance_date"].max())
    latest_panel = panel.loc[
        panel["rebalance_date"] == latest_rebalance, ["conid", "sleeve"]
    ].drop_duplicates()
    last_return_date = _to_timestamp(returns_wide.index.max())
    start_date = last_return_date - pd.Timedelta(
        days=int(config.trailing_beta_days * 1.5)
    )

    rows = []
    beta_fit_total = sum(
        len(sleeve_panel)
        for sleeve, sleeve_panel in latest_panel.groupby("sleeve")
        if not persistence.loc[
            (persistence["sleeve"] == sleeve) & (persistence["is_persistent"]),
            "factor_id",
        ].empty
    )
    with make_progress_bar(
        show_progress=show_progress,
        total=beta_fit_total,
        desc="Current beta fits",
        unit="fit",
        leave=config.persist_progress_bars,
    ) as progress_bar:
        for sleeve, sleeve_panel in latest_panel.groupby("sleeve"):
            progress_bar.set_postfix_str(str(sleeve), refresh=False)
            persistent_factors = persistence.loc[
                (persistence["sleeve"] == sleeve) & (persistence["is_persistent"]),
                "factor_id",
            ].tolist()
            if not persistent_factors:
                continue

            X = reduced_factors.loc[
                reduced_factors.index >= start_date, persistent_factors
            ].dropna()
            if len(X) < config.min_test_days:
                progress_bar.update(len(sleeve_panel))
                continue

            for conid in sleeve_panel["conid"].astype(str).tolist():
                y = returns_wide.get(conid)
                if y is None:
                    progress_bar.update(1)
                    continue
                if config.use_risk_free_excess:
                    y_target = y.subtract(risk_free_series, fill_value=0.0)
                else:
                    y_target = y.copy()
                data = pd.concat([X, y_target.rename("target")], axis=1).dropna()
                if len(data) < config.min_test_days:
                    progress_bar.update(1)
                    continue

                model = LinearRegression()
                X_fit = data[persistent_factors].to_numpy(dtype=float)
                y_fit = data["target"].to_numpy(dtype=float)
                model.fit(X_fit, y_fit)
                rows.append(
                    {
                        "conid": conid,
                        "sleeve": sleeve,
                        "window_start": _to_timestamp(data.index.min()),
                        "window_end": _to_timestamp(data.index.max()),
                        "n_obs": int(len(data)),
                        "alpha": float(model.intercept_),
                        "r2": float(model.score(X_fit, y_fit)),
                        **{
                            f"beta__{factor_id}": float(beta)
                            for factor_id, beta in zip(persistent_factors, model.coef_)
                        },
                    }
                )
                progress_bar.update(1)

    return pd.DataFrame(rows)


def build_analysis_panel(
    sqlite_path=SQLITE_DB_PATH,
    output_dir=None,
    show_progress=False,
    **config_kwargs,
):
    config = AnalysisConfig(
        sqlite_path=Path(sqlite_path),
        output_dir=Path(output_dir or (DATA_DIR / "analysis")),
        **config_kwargs,
    )
    (
        snapshot_features,
        price_result,
        _risk_free_daily,
        world_bank_country_features,
    ) = _prepare_analysis_inputs(config, show_progress=show_progress)
    panel = build_analysis_panel_data(
        snapshot_features,
        price_result,
        config,
        world_bank_country_features=world_bank_country_features,
        show_progress=show_progress,
    )

    with transaction(config.sqlite_path) as tx:
        panel_path = _write_output(
            "analysis_snapshot_panel",
            panel,
            config.output_dir,
            config.sqlite_path,
            tx=tx,
        )
    return {
        "status": "ok",
        "rows": int(len(panel)),
        "rebalance_dates": int(panel["rebalance_date"].nunique())
        if not panel.empty
        else 0,
        "snapshot_panel_path": panel_path,
    }


def run_factor_research(
    sqlite_path=SQLITE_DB_PATH,
    output_dir=None,
    show_progress=False,
    **config_kwargs,
):
    config = AnalysisConfig(
        sqlite_path=Path(sqlite_path),
        output_dir=Path(output_dir or (DATA_DIR / "analysis")),
        **config_kwargs,
    )
    (
        snapshot_features,
        price_result,
        risk_free_daily,
        world_bank_country_features,
    ) = _prepare_analysis_inputs(config, show_progress=show_progress)
    panel = build_analysis_panel_data(
        snapshot_features,
        price_result,
        config,
        world_bank_country_features=world_bank_country_features,
        show_progress=show_progress,
    )
    research = run_factor_research_data(
        panel,
        price_result["prices"],
        risk_free_daily,
        config,
        show_progress=show_progress,
    )
    for key in [
        "factor_registry",
        "factor_candidate_diagnostics",
        "factor_distinctness",
        "factor_selection_scores",
        "factor_screening_decisions",
        "factor_cluster_membership",
        "factor_model_telemetry",
        "factor_clusters",
        "factor_diagnostics",
        "factor_final_vif_diagnostics",
        "factor_persistence",
        "model_results",
        "current_betas",
        "asset_expected_returns",
        "asset_factor_betas",
        "baseline_members",
    ]:
        research.setdefault(key, pd.DataFrame())

    factor_returns_wide = research["factor_returns"].copy()
    factor_returns_long = pd.DataFrame()
    if not factor_returns_wide.empty:
        factor_returns_long = (
            factor_returns_wide.reset_index()
            .rename(columns={"index": "trade_date"})
            .melt(
                id_vars=["trade_date"], var_name="factor_id", value_name="factor_return"
            )
            .dropna(subset=["factor_return"])
        )

    with transaction(config.sqlite_path) as tx:
        paths = {
            "snapshot_panel_path": _write_output(
                "analysis_snapshot_panel",
                panel,
                config.output_dir,
                config.sqlite_path,
                tx=tx,
            ),
            "factor_returns_path": _write_output(
                "analysis_factor_returns",
                factor_returns_wide,
                config.output_dir,
                config.sqlite_path,
                long_sql_df=factor_returns_long,
                tx=tx,
            )
            if not factor_returns_wide.empty
            else None,
            "factor_clusters_path": _write_output(
                "analysis_factor_clusters",
                research["factor_clusters"],
                config.output_dir,
                config.sqlite_path,
                tx=tx,
            ),
            "factor_registry_path": _write_output(
                "analysis_factor_registry",
                research["factor_registry"],
                config.output_dir,
                config.sqlite_path,
                tx=tx,
            ),
            "factor_candidate_diagnostics_path": _write_output(
                "analysis_factor_candidate_diagnostics",
                research["factor_candidate_diagnostics"],
                config.output_dir,
                config.sqlite_path,
                tx=tx,
            ),
            "factor_distinctness_path": _write_output(
                "analysis_factor_distinctness",
                research["factor_distinctness"],
                config.output_dir,
                config.sqlite_path,
                tx=tx,
            ),
            "factor_selection_scores_path": _write_output(
                "analysis_factor_selection_scores",
                research["factor_selection_scores"],
                config.output_dir,
                config.sqlite_path,
                tx=tx,
            ),
            "factor_screening_decisions_path": _write_output(
                "analysis_factor_screening_decisions",
                research["factor_screening_decisions"],
                config.output_dir,
                config.sqlite_path,
                tx=tx,
            ),
            "factor_cluster_membership_path": _write_output(
                "analysis_factor_cluster_membership",
                research["factor_cluster_membership"],
                config.output_dir,
                config.sqlite_path,
                tx=tx,
            ),
            "factor_diagnostics_path": _write_output(
                "analysis_factor_diagnostics",
                research["factor_diagnostics"],
                config.output_dir,
                config.sqlite_path,
                tx=tx,
            ),
            "factor_final_vif_diagnostics_path": _write_output(
                "analysis_factor_final_vif_diagnostics",
                research["factor_final_vif_diagnostics"],
                config.output_dir,
                config.sqlite_path,
                tx=tx,
            ),
            "factor_persistence_path": _write_output(
                "analysis_factor_persistence",
                research["factor_persistence"],
                config.output_dir,
                config.sqlite_path,
                tx=tx,
            ),
            "model_results_path": _write_output(
                "analysis_model_results",
                research["model_results"],
                config.output_dir,
                config.sqlite_path,
                tx=tx,
            ),
            "factor_model_telemetry_path": _write_output(
                "analysis_factor_model_telemetry",
                research["factor_model_telemetry"],
                config.output_dir,
                config.sqlite_path,
                tx=tx,
            ),
            "current_betas_path": _write_output(
                "analysis_current_betas",
                research["current_betas"],
                config.output_dir,
                config.sqlite_path,
                tx=tx,
            ),
            "asset_expected_returns_path": _write_output(
                "analysis_asset_expected_returns",
                research["asset_expected_returns"],
                config.output_dir,
                config.sqlite_path,
                tx=tx,
            ),
            "asset_factor_betas_path": _write_output(
                "analysis_asset_factor_betas",
                research["asset_factor_betas"],
                config.output_dir,
                config.sqlite_path,
                tx=tx,
            ),
            "baseline_members_path": _write_output(
                "analysis_bond_baseline_members",
                research["baseline_members"],
                config.output_dir,
                config.sqlite_path,
                tx=tx,
            ),
        }

    persistent = research["factor_persistence"]
    return {
        "status": "ok",
        "snapshot_rows": int(len(panel)),
        "factor_count": int(factor_returns_wide.shape[1])
        if not factor_returns_wide.empty
        else 0,
        "persistent_factor_count": int(persistent["is_persistent"].sum())
        if not persistent.empty
        else 0,
        **paths,
    }


def compute_current_betas(
    sqlite_path=SQLITE_DB_PATH,
    output_dir=None,
    show_progress=False,
    **config_kwargs,
):
    result = run_factor_research(
        sqlite_path=sqlite_path,
        output_dir=output_dir,
        show_progress=show_progress,
        **config_kwargs,
    )
    return {
        "status": result["status"],
        "current_betas_path": result["current_betas_path"],
        "persistent_factor_count": result["persistent_factor_count"],
    }


def run_analysis_pipeline(
    sqlite_path=SQLITE_DB_PATH,
    output_dir=None,
    show_progress=False,
    **config_kwargs,
):
    return run_factor_research(
        sqlite_path=sqlite_path,
        output_dir=output_dir,
        show_progress=show_progress,
        **config_kwargs,
    )
