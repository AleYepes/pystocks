from __future__ import annotations

import pytest

from pystocks_next.storage.writes import (
    write_dividends_snapshot,
    write_holdings_snapshot,
    write_lipper_ratings_snapshot,
    write_morningstar_snapshot,
    write_profile_and_fees_snapshot,
    write_ratios_snapshot,
)
from pystocks_next.universe.products import UniverseInstrument, upsert_instruments


def test_write_profile_and_fees_snapshot_persists_tall_factor_rows(
    temp_store,
    sample_profile_and_fees_payload: dict[str, object],
) -> None:
    upsert_instruments(temp_store, [UniverseInstrument(conid="100", symbol="AAA")])

    result = write_profile_and_fees_snapshot(
        temp_store,
        conid="100",
        payload=sample_profile_and_fees_payload,
        observed_at="2026-01-05T10:00:00+00:00",
        capture_batch_id="batch-profile-001",
    )

    snapshot_row = temp_store.execute(
        """
        SELECT effective_at, capture_batch_id
        FROM profile_and_fees_snapshots
        WHERE conid = '100'
        """
    ).fetchone()
    factor_rows = temp_store.execute(
        """
        SELECT field_id, value_text, value_num, value_date, value_bool
        FROM profile_and_fees
        WHERE conid = '100'
        ORDER BY field_id
        """
    ).fetchall()
    by_field = {row["field_id"]: row for row in factor_rows}

    assert result.effective_at == "2026-01-05"
    assert snapshot_row["capture_batch_id"] == "batch-profile-001"
    assert by_field["asset_type"]["value_text"] == "Equity"
    assert by_field["management_expenses"]["value_num"] == pytest.approx(0.0012)
    assert by_field["total_net_assets_value"]["value_text"] == "$1.2B"
    assert by_field["total_net_assets_date"]["value_date"] == "2026-01-02"
    assert by_field["jap_fund_warning"]["value_bool"] == 0


def test_write_profile_and_fees_snapshot_persists_documented_nested_sections(
    temp_store,
) -> None:
    upsert_instruments(temp_store, [UniverseInstrument(conid="100", symbol="AAA")])
    payload = {
        "objective": "Track an index.",
        "symbol": "SPY",
        "fund_and_profile": [
            {"name": "Asset Type", "value": "Equity"},
            {"name": "Launch Opening Price", "value": "1993-01-22"},
            {"name": "Total Net Assets (Month End)", "value": "$708.92B (2026/01/30)"},
        ],
        "mstar": {
            "x_axis": ["Value", "Core", "Growth"],
            "y_axis": ["Large", "Multi", "Mid", "Small"],
            "x_axis_tag": ["value", "core", "growth"],
            "y_axis_tag": ["large", "multi", "mid", "small"],
            "selected": [],
            "hist": [[2, 1]],
        },
        "reports": [
            {
                "name": "Annual Report",
                "as_of_date": 1759204800000,
                "fields": [
                    {"name": "Total Expense", "value": "0.0893%", "is_summary": True},
                    {"name": "Management Fees", "value": "0.0469%"},
                ],
            },
            {"as_of_date": 0},
        ],
        "themes": ["Index Tracking"],
        "expenses_allocation": [],
        "jap_fund_warning": False,
    }

    write_profile_and_fees_snapshot(
        temp_store,
        conid="100",
        payload=payload,
        observed_at="2026-01-05T10:00:00+00:00",
    )

    rows = temp_store.execute(
        """
        SELECT field_id, value_text, value_num, value_date, value_bool
        FROM profile_and_fees
        WHERE conid = '100'
        ORDER BY field_id
        """
    ).fetchall()
    by_field = {row["field_id"]: row for row in rows}

    assert by_field["theme_name"]["value_text"] == "Index Tracking"
    assert by_field["morningstar_stylebox"]["value_text"] == "growth_multi"
    assert by_field["morningstar_stylebox_x"]["value_text"] == "Growth"
    assert by_field["morningstar_stylebox_y"]["value_text"] == "Multi"
    assert by_field["morningstar_stylebox_x_index"]["value_num"] == pytest.approx(2.0)
    assert by_field["morningstar_stylebox_y_index"]["value_num"] == pytest.approx(1.0)
    assert by_field["report_annual_report_as_of_date"]["value_date"] == "2025-09-30"
    assert by_field["report_annual_report_total_expense"]["value_num"] == pytest.approx(
        0.000893
    )
    assert by_field["report_annual_report_management_fees"][
        "value_num"
    ] == pytest.approx(0.000469)


def test_write_holdings_snapshot_persists_tall_factor_rows(
    temp_store,
    sample_holdings_payload: dict[str, object],
) -> None:
    upsert_instruments(temp_store, [UniverseInstrument(conid="100", symbol="AAA")])

    result = write_holdings_snapshot(
        temp_store,
        conid="100",
        payload=sample_holdings_payload,
        observed_at="2026-01-05T10:00:00+00:00",
        capture_batch_id="batch-holdings-001",
    )

    snapshot_row = temp_store.execute(
        """
        SELECT effective_at, as_of_date, capture_batch_id
        FROM holdings_snapshots
        WHERE conid = '100'
        """
    ).fetchone()
    asset_type_rows = temp_store.execute(
        """
        SELECT bucket_id, value_num
        FROM holdings_asset_type
        WHERE conid = '100'
        ORDER BY bucket_id
        """
    ).fetchall()
    debtor_rows = temp_store.execute(
        """
        SELECT bucket_id, value_num
        FROM holdings_debtor_quality
        WHERE conid = '100'
        ORDER BY bucket_id
        """
    ).fetchall()
    maturity_rows = temp_store.execute(
        """
        SELECT bucket_id, value_num
        FROM holdings_maturity
        WHERE conid = '100'
        ORDER BY bucket_id
        """
    ).fetchall()

    asset_by_bucket = {row["bucket_id"]: row["value_num"] for row in asset_type_rows}
    debtor_by_bucket = {row["bucket_id"]: row["value_num"] for row in debtor_rows}
    maturity_by_bucket = {row["bucket_id"]: row["value_num"] for row in maturity_rows}

    assert result.effective_at == "2026-01-03"
    assert snapshot_row["as_of_date"] == "2026-01-03"
    assert snapshot_row["capture_batch_id"] == "batch-holdings-001"
    assert asset_by_bucket["equity"] == pytest.approx(0.85)
    assert asset_by_bucket["cash"] == pytest.approx(0.10)
    assert asset_by_bucket["fixed_income"] == pytest.approx(0.05)
    assert "other" not in asset_by_bucket
    assert debtor_by_bucket["quality_aa"] == pytest.approx(0.15)
    assert debtor_by_bucket["quality_bbb"] == pytest.approx(0.08)
    assert debtor_by_bucket["quality_not_rated"] == pytest.approx(0.02)
    assert maturity_by_bucket["maturity_1_to_3_years"] == pytest.approx(0.125)
    assert maturity_by_bucket["maturity_less_than_1_year"] == pytest.approx(0.054)


def test_write_holdings_snapshot_persists_supported_long_child_tables(
    temp_store,
    sample_holdings_payload: dict[str, object],
) -> None:
    upsert_instruments(temp_store, [UniverseInstrument(conid="100", symbol="AAA")])

    write_holdings_snapshot(
        temp_store,
        conid="100",
        payload=sample_holdings_payload,
        observed_at="2026-01-05T10:00:00+00:00",
    )

    industry_row = temp_store.execute(
        """
        SELECT industry, value_num
        FROM holdings_industry
        WHERE conid = '100'
        """
    ).fetchone()
    currency_row = temp_store.execute(
        """
        SELECT code, currency, value_num
        FROM holdings_currency
        WHERE conid = '100'
        """
    ).fetchone()
    country_row = temp_store.execute(
        """
        SELECT country_code, country, value_num
        FROM holdings_investor_country
        WHERE conid = '100'
        """
    ).fetchone()
    geographic_rows = temp_store.execute(
        """
        SELECT region, value_num
        FROM holdings_geographic_weights
        WHERE conid = '100'
        ORDER BY region
        """
    ).fetchall()
    debt_type_row = temp_store.execute(
        """
        SELECT debt_type, value_num
        FROM holdings_debt_type
        WHERE conid = '100'
        """
    ).fetchone()
    top10_row = temp_store.execute(
        """
        SELECT name, holding_weight_num
        FROM holdings_top10
        WHERE conid = '100'
        """
    ).fetchone()

    assert industry_row["industry"] == "Technology"
    assert industry_row["value_num"] == pytest.approx(0.448681)
    assert currency_row["code"] == "USD"
    assert currency_row["currency"] == "US Dollar"
    assert currency_row["value_num"] == pytest.approx(0.999604)
    assert country_row["country_code"] == "US"
    assert country_row["country"] == "United States"
    assert country_row["value_num"] == pytest.approx(0.973418)
    assert [(row["region"], row["value_num"]) for row in geographic_rows] == [
        ("eu", pytest.approx(0.0189)),
        ("us", pytest.approx(0.9734)),
    ]
    assert debt_type_row["debt_type"] == "Sovereign Bond"
    assert debt_type_row["value_num"] == pytest.approx(0.20)
    assert top10_row["name"] == "NVIDIA CORPORATION"
    assert top10_row["holding_weight_num"] == pytest.approx(0.0783)


def test_write_holdings_snapshot_persists_documented_top10_identifiers(
    temp_store,
) -> None:
    upsert_instruments(temp_store, [UniverseInstrument(conid="100", symbol="AAA")])
    payload = {
        "as_of_date": 1769835600000,
        "allocation_self": [
            {
                "name": "Equity",
                "formatted_weight": "99.91%",
                "weight": 99.9088,
                "rank": 1,
                "vs": 107.198704166667,
            },
            {
                "name": "Other",
                "formatted_weight": "0.04%",
                "weight": 0.0396,
                "rank": 3,
                "vs": 0.45558125,
            },
        ],
        "top_10": [
            {
                "name": "NVIDIA CORPORATION",
                "ticker": "NVDA",
                "rank": 1,
                "assets_pct": "7.83%",
                "conids": [4815747, 13104788],
            }
        ],
        "currency": [
            {
                "name": "<No Currency>",
                "formatted_weight": "0.04%",
                "weight": 0.0396,
                "rank": 2,
            }
        ],
        "investor_country": [
            {
                "name": "Unidentified",
                "formatted_weight": "0.04%",
                "weight": 0.0396,
                "rank": 8,
            }
        ],
        "geographic": {
            "eu": "1.89%",
            "uk": "0.46%",
            "us": "97.34%",
            "others": "0.04%",
        },
    }

    result = write_holdings_snapshot(
        temp_store,
        conid="100",
        payload=payload,
        observed_at="2026-02-01T10:00:00+00:00",
    )

    asset_rows = temp_store.execute(
        """
        SELECT bucket_id, value_num
        FROM holdings_asset_type
        WHERE conid = '100'
        ORDER BY bucket_id
        """
    ).fetchall()
    top10_row = temp_store.execute(
        """
        SELECT name, ticker, rank, holding_weight_num, conids_json
        FROM holdings_top10
        WHERE conid = '100'
        """
    ).fetchone()
    currency_row = temp_store.execute(
        """
        SELECT code, currency, value_num
        FROM holdings_currency
        WHERE conid = '100'
        """
    ).fetchone()
    country_row = temp_store.execute(
        """
        SELECT country_code, country, value_num
        FROM holdings_investor_country
        WHERE conid = '100'
        """
    ).fetchone()

    asset_by_bucket = {row["bucket_id"]: row["value_num"] for row in asset_rows}
    assert result.effective_at == "2026-01-31"
    assert asset_by_bucket["equity"] == pytest.approx(0.999088)
    assert asset_by_bucket["other"] == pytest.approx(0.000396)
    assert top10_row["name"] == "NVIDIA CORPORATION"
    assert top10_row["ticker"] == "NVDA"
    assert top10_row["rank"] == 1
    assert top10_row["holding_weight_num"] == pytest.approx(0.0783)
    assert top10_row["conids_json"] == "[4815747, 13104788]"
    assert currency_row["code"] is None
    assert currency_row["currency"] == "<No Currency>"
    assert currency_row["value_num"] == pytest.approx(0.000396)
    assert country_row["country_code"] is None
    assert country_row["country"] == "Unidentified"
    assert country_row["value_num"] == pytest.approx(0.000396)


def test_write_holdings_snapshot_raises_when_weight_fields_disagree(
    temp_store,
) -> None:
    upsert_instruments(temp_store, [UniverseInstrument(conid="100", symbol="AAA")])
    payload = {
        "as_of_date": "2026-01-31",
        "allocation_self": [
            {
                "name": "Equity",
                "formatted_weight": "99.91%",
                "weight": 99.0,
            }
        ],
    }

    with pytest.raises(ValueError, match="holdings weight mismatch"):
        write_holdings_snapshot(
            temp_store,
            conid="100",
            payload=payload,
            observed_at="2026-02-01T10:00:00+00:00",
        )


def test_write_ratios_snapshot_persists_supported_sections(
    temp_store,
    sample_ratios_payload: dict[str, object],
) -> None:
    upsert_instruments(temp_store, [UniverseInstrument(conid="100", symbol="AAA")])

    result = write_ratios_snapshot(
        temp_store,
        conid="100",
        payload=sample_ratios_payload,
        observed_at="2026-01-05T10:00:00+00:00",
        capture_batch_id="batch-ratios-001",
    )

    snapshot_row = temp_store.execute(
        """
        SELECT effective_at, as_of_date, capture_batch_id
        FROM ratios_snapshots
        WHERE conid = '100'
        """
    ).fetchone()
    ratios_row = temp_store.execute(
        """
        SELECT metric_id, value_num, vs_num
        FROM ratios_key_ratios
        WHERE conid = '100'
        """
    ).fetchone()
    financials_row = temp_store.execute(
        """
        SELECT metric_id, value_num, vs_num
        FROM ratios_financials
        WHERE conid = '100'
        """
    ).fetchone()
    fixed_income_row = temp_store.execute(
        """
        SELECT metric_id, value_num, vs_num
        FROM ratios_fixed_income
        WHERE conid = '100'
        """
    ).fetchone()
    dividend_row = temp_store.execute(
        """
        SELECT metric_id, value_num, vs_num
        FROM ratios_dividend
        WHERE conid = '100'
        """
    ).fetchone()
    zscore_row = temp_store.execute(
        """
        SELECT metric_id, value_num, vs_num
        FROM ratios_zscore
        WHERE conid = '100'
        """
    ).fetchone()

    assert result.effective_at == "2026-01-03"
    assert snapshot_row["as_of_date"] == "2026-01-03"
    assert snapshot_row["capture_batch_id"] == "batch-ratios-001"
    assert ratios_row["metric_id"] == "price_sales"
    assert ratios_row["value_num"] == pytest.approx(3.63)
    assert ratios_row["vs_num"] == pytest.approx(0.0146)
    assert financials_row["metric_id"] == "sales_growth_1_year"
    assert financials_row["value_num"] == pytest.approx(5.04)
    assert financials_row["vs_num"] == pytest.approx(-0.15)
    assert fixed_income_row["metric_id"] == "current_yield"
    assert fixed_income_row["value_num"] == pytest.approx(3.17)
    assert fixed_income_row["vs_num"] == pytest.approx(0.05)
    assert dividend_row["metric_id"] == "dividend_yield"
    assert dividend_row["value_num"] == pytest.approx(2.35)
    assert dividend_row["vs_num"] == pytest.approx(-0.08)
    assert zscore_row["metric_id"] == "1_month"
    assert zscore_row["value_num"] == pytest.approx(-0.04)
    assert zscore_row["vs_num"] is None


def test_write_dividends_snapshot_persists_tall_metric_rows(
    temp_store,
    sample_dividends_snapshot_payload: dict[str, object],
) -> None:
    upsert_instruments(temp_store, [UniverseInstrument(conid="100", symbol="AAA")])

    result = write_dividends_snapshot(
        temp_store,
        conid="100",
        payload=sample_dividends_snapshot_payload,
        observed_at="2026-01-05T10:00:00+00:00",
    )

    rows = temp_store.execute(
        """
        SELECT metric_id, value_num, currency
        FROM dividends_industry_metrics
        WHERE conid = '100'
        ORDER BY metric_id
        """
    ).fetchall()
    by_metric = {row["metric_id"]: row for row in rows}

    assert result.effective_at == "2026-01-03"
    assert by_metric["annual_dividend"]["value_num"] == pytest.approx(25.65)
    assert by_metric["dividend_ttm"]["value_num"] == pytest.approx(5.585599)
    assert by_metric["dividend_yield"]["value_num"] == pytest.approx(0.0122)
    assert by_metric["dividend_yield_ttm"]["value_num"] == pytest.approx(0.0085)
    assert by_metric["dividend_yield"]["currency"] == "USD"


def test_write_morningstar_snapshot_persists_tall_summary_rows(
    temp_store,
    sample_morningstar_payload: dict[str, object],
) -> None:
    upsert_instruments(temp_store, [UniverseInstrument(conid="100", symbol="AAA")])

    result = write_morningstar_snapshot(
        temp_store,
        conid="100",
        payload=sample_morningstar_payload,
        observed_at="2026-01-05T10:00:00+00:00",
    )

    rows = temp_store.execute(
        """
        SELECT metric_id, value_text, value_num
        FROM morningstar_summary
        WHERE conid = '100'
        ORDER BY metric_id
        """
    ).fetchall()
    by_metric = {row["metric_id"]: row for row in rows}

    assert result.effective_at == "2026-01-31"
    assert by_metric["medalist_rating"]["value_text"] == "Silver"
    assert by_metric["process"]["value_text"] == "High"
    assert by_metric["morningstar_rating"]["value_num"] == pytest.approx(4.0)


def test_write_morningstar_snapshot_falls_back_to_latest_summary_publish_date(
    temp_store,
    sample_morningstar_payload: dict[str, object],
) -> None:
    upsert_instruments(temp_store, [UniverseInstrument(conid="100", symbol="AAA")])
    payload = dict(sample_morningstar_payload)
    payload.pop("as_of_date", None)

    result = write_morningstar_snapshot(
        temp_store,
        conid="100",
        payload=payload,
        observed_at="2026-01-05T10:00:00+00:00",
    )

    snapshot_row = temp_store.execute(
        """
        SELECT effective_at, as_of_date
        FROM morningstar_snapshots
        WHERE conid = '100'
        """
    ).fetchone()

    assert result.effective_at == "2026-01-31"
    assert snapshot_row["effective_at"] == "2026-01-31"
    assert snapshot_row["as_of_date"] == "2026-01-31"


def test_write_morningstar_snapshot_falls_back_to_commentary_publish_date(
    temp_store,
) -> None:
    upsert_instruments(temp_store, [UniverseInstrument(conid="100", symbol="AAA")])
    payload = {
        "summary": [
            {
                "id": "category",
                "title": "Category",
                "value": "Trading - Leveraged/Inverse Equity",
                "q": False,
            }
        ],
        "commentary": [
            {
                "id": "sustainability",
                "title": "Sustainability",
                "q": True,
                "publish_date": "20250531",
                "text": "Insufficient data.",
            }
        ],
    }

    result = write_morningstar_snapshot(
        temp_store,
        conid="100",
        payload=payload,
        observed_at="2026-01-05T10:00:00+00:00",
    )

    snapshot_row = temp_store.execute(
        """
        SELECT effective_at, as_of_date
        FROM morningstar_snapshots
        WHERE conid = '100'
        """
    ).fetchone()
    summary_row = temp_store.execute(
        """
        SELECT metric_id, value_text
        FROM morningstar_summary
        WHERE conid = '100'
        """
    ).fetchone()

    assert result.effective_at == "2025-05-31"
    assert snapshot_row["effective_at"] == "2025-05-31"
    assert snapshot_row["as_of_date"] == "2025-05-31"
    assert summary_row["metric_id"] == "category"
    assert summary_row["value_text"] == "Trading - Leveraged/Inverse Equity"


def test_write_lipper_ratings_snapshot_persists_rows(
    temp_store,
    sample_lipper_payload: dict[str, object],
) -> None:
    upsert_instruments(temp_store, [UniverseInstrument(conid="100", symbol="AAA")])

    result = write_lipper_ratings_snapshot(
        temp_store,
        conid="100",
        payload=sample_lipper_payload,
        observed_at="2026-01-05T10:00:00+00:00",
    )

    snapshot_row = temp_store.execute(
        """
        SELECT universe_count
        FROM lipper_ratings_snapshots
        WHERE conid = '100'
        """
    ).fetchone()
    rows = temp_store.execute(
        """
        SELECT period, metric_id, value_num, rating_label, universe_name, universe_as_of_date
        FROM lipper_ratings
        WHERE conid = '100'
        ORDER BY universe_name, period
        """
    ).fetchall()

    assert result.effective_at == "2026-01-30"
    assert snapshot_row["universe_count"] == 1
    assert [
        (
            row["period"],
            row["metric_id"],
            row["value_num"],
            row["rating_label"],
            row["universe_name"],
            row["universe_as_of_date"],
        )
        for row in rows
    ] == [
        (
            "3_year",
            "total_return",
            pytest.approx(4.0),
            "236 funds",
            "Sweden",
            "2026-01-30",
        ),
        (
            "overall",
            "total_return",
            pytest.approx(5.0),
            "236 funds",
            "Sweden",
            "2026-01-30",
        ),
    ]
