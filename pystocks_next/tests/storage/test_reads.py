from __future__ import annotations

import pandas as pd
import pytest

from pystocks_next.storage.reads import (
    DIVIDEND_EVENTS_COLUMNS,
    PRICE_HISTORY_COLUMNS,
    RISK_FREE_DAILY_COLUMNS,
    SNAPSHOT_TABLE_COLUMNS,
    WORLD_BANK_COUNTRY_FEATURE_COLUMNS,
    DividendEventsRead,
    PriceHistoryRead,
    RiskFreeDailyRead,
    SnapshotFeatureTablesRead,
    WorldBankCountryFeaturesRead,
    load_dividend_events,
    load_latest_price_effective_at_by_conid,
    load_price_history,
    load_risk_free_daily,
    load_snapshot_feature_tables,
    load_world_bank_country_features,
)
from pystocks_next.storage.writes import (
    write_dividend_events_series,
    write_dividends_snapshot,
    write_holdings_snapshot,
    write_lipper_ratings_snapshot,
    write_morningstar_snapshot,
    write_price_chart_series,
    write_profile_and_fees_snapshot,
    write_ratios_snapshot,
    write_supplementary_risk_free_daily,
    write_supplementary_world_bank_country_features,
)
from pystocks_next.universe.products import UniverseInstrument, upsert_instruments


def test_price_history_read_normalizes_types_and_order(
    sample_price_history_frame: pd.DataFrame,
) -> None:
    result = PriceHistoryRead.from_frame(sample_price_history_frame).frame

    assert tuple(result.columns) == PRICE_HISTORY_COLUMNS
    assert result["conid"].tolist() == ["100", "200"]
    assert pd.api.types.is_datetime64_any_dtype(result["trade_date"])
    assert pd.api.types.is_float_dtype(result["close"])


def test_dividend_events_read_normalizes_types_and_order(
    sample_dividend_events_frame: pd.DataFrame,
) -> None:
    result = DividendEventsRead.from_frame(sample_dividend_events_frame).frame

    assert tuple(result.columns) == DIVIDEND_EVENTS_COLUMNS
    assert result["conid"].tolist() == ["100", "200"]
    assert pd.api.types.is_datetime64_any_dtype(result["event_date"])
    assert pd.api.types.is_float_dtype(result["amount"])


def test_snapshot_feature_tables_read_normalizes_known_tables(
    sample_snapshot_tables: dict[str, pd.DataFrame],
) -> None:
    result = SnapshotFeatureTablesRead.from_tables(sample_snapshot_tables).tables

    assert set(result) == set(SNAPSHOT_TABLE_COLUMNS)
    assert (
        tuple(result["profile_and_fees"].columns)
        == SNAPSHOT_TABLE_COLUMNS["profile_and_fees"]
    )
    assert pd.api.types.is_datetime64_any_dtype(
        result["profile_and_fees"]["effective_at"]
    )
    assert pd.api.types.is_float_dtype(result["holdings_asset_type"]["value_num"])
    assert result["profile_and_fees"]["field_id"].tolist() == ["asset_type"]
    assert result["holdings_top10"].empty


def test_snapshot_feature_tables_read_rejects_unknown_table_names() -> None:
    with pytest.raises(ValueError, match="unknown snapshot tables"):
        SnapshotFeatureTablesRead.from_tables({"unknown_table": pd.DataFrame()})


def test_risk_free_daily_read_normalizes_types_and_order(
    sample_risk_free_daily_frame: pd.DataFrame,
) -> None:
    result = RiskFreeDailyRead.from_frame(sample_risk_free_daily_frame).frame

    assert tuple(result.columns) == RISK_FREE_DAILY_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(result["trade_date"])
    assert pd.api.types.is_float_dtype(result["nominal_rate"])


def test_world_bank_country_features_read_normalizes_types_and_order(
    sample_world_bank_country_features_frame: pd.DataFrame,
) -> None:
    result = WorldBankCountryFeaturesRead.from_frame(
        sample_world_bank_country_features_frame
    ).frame

    assert tuple(result.columns) == WORLD_BANK_COUNTRY_FEATURE_COLUMNS
    assert result["economy_code"].tolist() == ["USA"]
    assert pd.api.types.is_datetime64_any_dtype(result["effective_at"])
    assert pd.api.types.is_float_dtype(result["population_acceleration"])


def test_load_price_history_reads_from_canonical_storage(
    temp_store,
    sample_price_chart_payload: dict[str, object],
) -> None:
    upsert_instruments(temp_store, [UniverseInstrument(conid="100", symbol="AAA")])
    write_price_chart_series(
        temp_store,
        conid="100",
        payload=sample_price_chart_payload,
        observed_at="2026-01-05T10:00:00+00:00",
    )

    result = load_price_history(temp_store).frame

    assert tuple(result.columns) == PRICE_HISTORY_COLUMNS
    assert result["conid"].tolist() == ["100", "100"]
    assert result["trade_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2025-12-31",
        "2026-01-02",
    ]


def test_load_latest_price_effective_at_by_conid_reads_grouped_dates(
    temp_store,
    sample_price_chart_payload: dict[str, object],
) -> None:
    upsert_instruments(
        temp_store,
        [
            UniverseInstrument(conid="100", symbol="AAA"),
            UniverseInstrument(conid="200", symbol="BBB"),
        ],
    )
    write_price_chart_series(
        temp_store,
        conid="100",
        payload=sample_price_chart_payload,
        observed_at="2026-01-05T10:00:00+00:00",
    )

    result = load_latest_price_effective_at_by_conid(temp_store, conids=["100", "200"])

    assert result == {
        "100": pd.Timestamp("2026-01-02").date(),
        "200": None,
    }


def test_load_dividend_events_reads_from_canonical_storage(
    temp_store,
    sample_dividends_payload: dict[str, object],
) -> None:
    upsert_instruments(
        temp_store,
        [UniverseInstrument(conid="100", symbol="AAA", currency="USD")],
    )
    write_dividend_events_series(
        temp_store,
        conid="100",
        payload=sample_dividends_payload,
        observed_at="2026-01-05T10:00:00+00:00",
        source_as_of_date="2026-01-03",
    )

    result = load_dividend_events(temp_store).frame

    assert tuple(result.columns) == DIVIDEND_EVENTS_COLUMNS
    assert result["symbol"].tolist() == ["AAA", "AAA"]
    assert result["product_currency"].tolist() == ["USD", "USD"]
    assert result["event_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-01-03",
        "2026-01-10",
    ]


def test_load_risk_free_daily_reads_from_canonical_storage(
    temp_store,
    sample_risk_free_daily_frame: pd.DataFrame,
) -> None:
    write_supplementary_risk_free_daily(temp_store, frame=sample_risk_free_daily_frame)

    result = load_risk_free_daily(temp_store).frame

    assert tuple(result.columns) == RISK_FREE_DAILY_COLUMNS
    assert result["trade_date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-02"]
    assert result["nominal_rate"].iloc[0] == pytest.approx(0.022)


def test_load_world_bank_country_features_reads_from_canonical_storage(
    temp_store,
    sample_world_bank_country_features_frame: pd.DataFrame,
) -> None:
    write_supplementary_world_bank_country_features(
        temp_store,
        frame=sample_world_bank_country_features_frame,
    )

    result = load_world_bank_country_features(temp_store).frame

    assert tuple(result.columns) == WORLD_BANK_COUNTRY_FEATURE_COLUMNS
    assert result["economy_code"].tolist() == ["USA"]
    assert result["feature_year"].tolist() == [2025]


def test_load_world_bank_country_features_tolerates_old_schema(temp_store) -> None:
    temp_store.execute("DROP TABLE supplementary_world_bank_country_features")
    temp_store.execute(
        """
        CREATE TABLE supplementary_world_bank_country_features (
            economy_code TEXT NOT NULL,
            effective_at TEXT NOT NULL,
            feature_year INTEGER NOT NULL,
            population_level REAL,
            population_growth REAL,
            gdp_pcap_level REAL,
            gdp_pcap_growth REAL,
            economic_output_gdp_level REAL,
            economic_output_gdp_growth REAL,
            foreign_direct_investment_level REAL,
            foreign_direct_investment_growth REAL,
            share_trade_volume_level REAL,
            share_trade_volume_growth REAL,
            observed_at TEXT NOT NULL,
            PRIMARY KEY (economy_code, feature_year)
        )
        """
    )
    temp_store.execute(
        """
        INSERT INTO supplementary_world_bank_country_features (
            economy_code,
            effective_at,
            feature_year,
            population_level,
            population_growth,
            gdp_pcap_level,
            gdp_pcap_growth,
            economic_output_gdp_level,
            economic_output_gdp_growth,
            foreign_direct_investment_level,
            foreign_direct_investment_growth,
            share_trade_volume_level,
            share_trade_volume_growth,
            observed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "USA",
            "2025-12-31",
            2025,
            100.0,
            10.0,
            10.0,
            1.0,
            1000.0,
            5.0,
            1.0,
            0.1,
            0.5,
            0.05,
            "2026-01-05T10:00:00+00:00",
        ),
    )

    result = load_world_bank_country_features(temp_store).frame
    row = result.iloc[0]

    assert pd.isna(row["population_acceleration"])
    assert pd.isna(row["gdp_pcap_acceleration"])
    assert pd.isna(row["economic_output_gdp_acceleration"])
    assert pd.isna(row["foreign_direct_investment_acceleration"])
    assert pd.isna(row["share_trade_volume_acceleration"])


def test_load_snapshot_feature_tables_reads_supported_snapshot_tables(
    temp_store,
    sample_profile_and_fees_payload: dict[str, object],
    sample_holdings_payload: dict[str, object],
    sample_ratios_payload: dict[str, object],
    sample_dividends_snapshot_payload: dict[str, object],
    sample_morningstar_payload: dict[str, object],
    sample_lipper_payload: dict[str, object],
) -> None:
    upsert_instruments(
        temp_store,
        [UniverseInstrument(conid="100", symbol="AAA", currency="USD")],
    )
    write_profile_and_fees_snapshot(
        temp_store,
        conid="100",
        payload=sample_profile_and_fees_payload,
        observed_at="2026-01-05T10:00:00+00:00",
    )
    write_holdings_snapshot(
        temp_store,
        conid="100",
        payload=sample_holdings_payload,
        observed_at="2026-01-05T10:00:00+00:00",
    )
    write_ratios_snapshot(
        temp_store,
        conid="100",
        payload=sample_ratios_payload,
        observed_at="2026-01-05T10:00:00+00:00",
    )
    write_dividends_snapshot(
        temp_store,
        conid="100",
        payload=sample_dividends_snapshot_payload,
        observed_at="2026-01-05T10:00:00+00:00",
    )
    write_morningstar_snapshot(
        temp_store,
        conid="100",
        payload=sample_morningstar_payload,
        observed_at="2026-01-05T10:00:00+00:00",
    )
    write_lipper_ratings_snapshot(
        temp_store,
        conid="100",
        payload=sample_lipper_payload,
        observed_at="2026-01-05T10:00:00+00:00",
    )

    result = load_snapshot_feature_tables(temp_store).tables

    assert result["profile_and_fees"]["conid"].nunique() == 1
    assert result["profile_and_fees"]["conid"].iloc[0] == "100"
    assert result["profile_and_fees"]["field_id"].tolist() == [
        "asset_type",
        "classification",
        "distribution_details",
        "domicile",
        "fund_category",
        "fund_management_company",
        "fund_manager_benchmark",
        "fund_market_cap_focus",
        "geographical_focus",
        "inception_date",
        "jap_fund_warning",
        "management_approach",
        "management_expenses",
        "manager_tenure",
        "objective",
        "objective_type",
        "portfolio_manager",
        "redemption_charge_actual",
        "redemption_charge_max",
        "scheme",
        "theme_name",
        "total_expense_ratio",
        "total_net_assets_date",
        "total_net_assets_value",
    ]
    assert result["holdings_asset_type"]["bucket_id"].tolist() == [
        "cash",
        "equity",
        "fixed_income",
    ]
    assert result["holdings_asset_type"]["value_num"].tolist() == [0.10, 0.85, 0.05]
    assert result["holdings_industry"]["industry"].tolist() == ["Technology"]
    assert result["holdings_currency"]["code"].tolist() == ["USD"]
    assert result["holdings_debtor_quality"]["bucket_id"].tolist() == [
        "quality_aa",
        "quality_bbb",
        "quality_not_rated",
    ]
    assert result["holdings_maturity"]["bucket_id"].tolist() == [
        "maturity_1_to_3_years",
        "maturity_less_than_1_year",
    ]
    assert result["holdings_geographic_weights"]["region"].tolist() == ["eu", "us"]
    assert result["holdings_top10"]["name"].tolist() == ["NVIDIA CORPORATION"]
    assert result["ratios_key_ratios"]["metric_id"].tolist() == ["price_sales"]
    assert result["ratios_financials"]["metric_id"].tolist() == ["sales_growth_1_year"]
    assert result["ratios_fixed_income"]["value_num"].tolist() == [3.17]
    assert result["ratios_dividend"]["metric_id"].tolist() == ["dividend_yield"]
    assert result["ratios_zscore"]["metric_id"].tolist() == ["1_month"]
    assert result["dividends_industry_metrics"]["metric_id"].tolist() == [
        "annual_dividend",
        "dividend_ttm",
        "dividend_yield",
        "dividend_yield_ttm",
    ]
    assert result["morningstar_summary"]["metric_id"].tolist() == [
        "category",
        "category_index",
        "medalist_rating",
        "morningstar_rating",
        "parent",
        "people",
        "process",
        "sustainability_rating",
    ]
    assert result["lipper_ratings"]["metric_id"].tolist() == [
        "total_return",
        "total_return",
    ]
