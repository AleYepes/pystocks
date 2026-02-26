import sqlite3
import tempfile
from pathlib import Path

import pytest

from pystocks.fundamentals_store import FundamentalsStore


def _make_store():
    tmp = tempfile.TemporaryDirectory()
    db_path = Path(tmp.name) / "pystocks.sqlite"
    store = FundamentalsStore(sqlite_path=db_path)
    return tmp, db_path, store


def _table_columns(con, table):
    return [r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()]


def test_profile_fees_merges_launch_opening_price_and_splits_total_net_assets():
    tmp, db_path, store = _make_store()
    try:
        snapshot = {
            "conid": "profile_merge_1",
            "scraped_at": "2026-02-23T12:00:00+00:00",
            "profile_and_fees": {
                "objective": "Objective text",
                "jap_fund_warning": False,
                "themes": ["Index Tracking"],
                "fund_and_profile": [
                    {"name": "Launch Opening Price", "value": "2018/07/15"},
                    {"name": "Total Net Assets (Month End)", "value": "CAD289.15M (2026/01/30)"},
                ]
            },
        }

        result = store.persist_combined_snapshot(snapshot)
        assert result["status"] == "ok"
        assert result["inserted_events"] == 1

        con = sqlite3.connect(db_path)
        try:
            row = con.execute(
                """
                SELECT inception_date, total_net_assets_value, total_net_assets_date, objective, jap_fund_warning, theme_name
                FROM profile_and_fees
                WHERE conid = ?
                """,
                ["profile_merge_1"],
            ).fetchone()
            assert row == ("2018-07-15", "CAD289.15M", "2026-01-30", "Objective text", 0, "Index Tracking")

            cols = _table_columns(con, "profile_and_fees")
            assert "total_net_assets_value" in cols
            assert "total_net_assets_date" in cols
            assert "launch_opening_price" not in cols
            assert "total_net_assets_month_end_text" not in cols
            assert "total_net_assets_month_end_num" not in cols
            assert "total_net_assets_month_end_as_of_date" not in cols
        finally:
            con.close()
    finally:
        tmp.cleanup()


def test_profile_fees_prefers_inception_date_over_launch_opening_price():
    tmp, db_path, store = _make_store()
    try:
        snapshot = {
            "conid": "profile_merge_2",
            "scraped_at": "2026-02-23T12:00:00+00:00",
            "profile_and_fees": {
                "fund_and_profile": [
                    {"name": "Inception Date", "value": "2010-01-02"},
                    {"name": "Launch Opening Price", "value": "2018/07/15"},
                    {"name": "Total Net Assets (Month End)", "value": "€20.86M (2026/01/30."},
                ]
            },
        }

        result = store.persist_combined_snapshot(snapshot)
        assert result["status"] == "ok"

        con = sqlite3.connect(db_path)
        try:
            row = con.execute(
                """
                SELECT inception_date, total_net_assets_value, total_net_assets_date
                FROM profile_and_fees
                WHERE conid = ?
                """,
                ["profile_merge_2"],
            ).fetchone()
            assert row == ("2010-01-02", "€20.86M", "2026-01-30")
        finally:
            con.close()
    finally:
        tmp.cleanup()


def test_profile_fees_reports_are_pivoted_to_numeric_columns():
    tmp, db_path, store = _make_store()
    try:
        snapshot = {
            "conid": "profile_report_1",
            "scraped_at": "2026-02-23T12:00:00+00:00",
            "profile_and_fees": {
                "reports": [
                    {
                        "name": "Annual Report",
                        "as_of_date": "2025-09-30",
                        "fields": [
                            {"name": "Total Expense", "value": "0.0893%"},
                            {"name": "Management Fees", "value": "1.25%"},
                            {"name": "Misc. Expenses", "value": "0%"},
                        ],
                    },
                    {
                        "name": "Prospectus Report",
                        "as_of_date": "2025-01-27",
                        "fields": [
                            {"name": "Prospectus Net Expense Ratio", "value": "0.40%"},
                            {"name": "Prospectus Net Management Fee Ratio", "value": "0.15%"},
                            {"name": "Unknown Field", "value": "99%"},
                        ],
                    },
                    {
                        "name": "Empty Report",
                        "as_of_date": "2025-01-01",
                        "fields": [],
                    },
                ]
            },
        }

        result = store.persist_combined_snapshot(snapshot)
        assert result["status"] == "ok"

        con = sqlite3.connect(db_path)
        try:
            rows = con.execute(
                """
                SELECT report_name, report_as_of_date, total_expense, management_fees, misc_expenses,
                       prospectus_net_expense_ratio, prospectus_net_management_fee_ratio
                FROM profile_and_fees_reports
                WHERE conid = ?
                ORDER BY report_name
                """,
                ["profile_report_1"],
            ).fetchall()
            assert len(rows) == 2

            annual = rows[0]
            assert annual[0] == "Annual Report"
            assert annual[1] == "2025-09-30"
            assert annual[2] == pytest.approx(0.000893)
            assert annual[3] == pytest.approx(0.0125)
            assert annual[4] == pytest.approx(0.0)
            assert annual[5] is None
            assert annual[6] is None

            prospectus = rows[1]
            assert prospectus[0] == "Prospectus Report"
            assert prospectus[1] == "2025-01-27"
            assert prospectus[2] is None
            assert prospectus[3] is None
            assert prospectus[4] is None
            assert prospectus[5] == pytest.approx(0.004)
            assert prospectus[6] == pytest.approx(0.0015)
        finally:
            con.close()
    finally:
        tmp.cleanup()


def test_profile_fees_snapshots_schema_and_legacy_tables_removed():
    tmp, db_path, store = _make_store()
    try:
        snapshot = {
            "conid": "profile_snapshot_1",
            "scraped_at": "2026-02-23T12:00:00+00:00",
            "profile_and_fees": {
                "objective": "Objective text",
                "symbol": "ABC",
                "jap_fund_warning": True,
                "themes": ["Index Tracking", "Ethical"],
            },
        }

        result = store.persist_combined_snapshot(snapshot)
        assert result["status"] == "ok"

        con = sqlite3.connect(db_path)
        try:
            snapshot_cols = _table_columns(con, "profile_and_fees_snapshots")
            assert "source_file" not in snapshot_cols
            assert "symbol" not in snapshot_cols
            assert "objective" not in snapshot_cols
            assert "jap_fund_warning" not in snapshot_cols

            profile_row = con.execute(
                """
                SELECT objective, jap_fund_warning, theme_name
                FROM profile_and_fees
                WHERE conid = ?
                """,
                ["profile_snapshot_1"],
            ).fetchone()
            assert profile_row == ("Objective text", 1, "Index Tracking | Ethical")

            table_names = {
                row[0]
                for row in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            }
            assert "profile_and_fees" in table_names
            assert "profile_and_fees_snapshots" in table_names
            assert "profile_and_fees_reports" in table_names
            assert "profile_and_fees_stylebox" in table_names
            assert "profile_fees_fund_profile_fields" not in table_names
            assert "profile_fees_snapshots" not in table_names
            assert "profile_fees_report_fields" not in table_names
            assert "profile_fees_expense_allocations" not in table_names
            assert "profile_fees_themes" not in table_names
        finally:
            con.close()
    finally:
        tmp.cleanup()


def test_profile_fees_stylebox_maps_mstar_hist_to_boolean_cells():
    tmp, db_path, store = _make_store()
    try:
        snapshot = {
            "conid": "profile_stylebox_1",
            "scraped_at": "2026-02-23T12:00:00+00:00",
            "profile_and_fees": {
                "mstar": {
                    "hist": [
                        [0, 1],
                        [2, 0],
                        [2, 3],
                        ["1", "2"],
                        [99, 99],
                    ]
                }
            },
        }

        result = store.persist_combined_snapshot(snapshot)
        assert result["status"] == "ok"

        con = sqlite3.connect(db_path)
        try:
            cols = _table_columns(con, "profile_and_fees_stylebox")
            expected_cols = {
                "value_large",
                "value_multi",
                "value_mid",
                "value_small",
                "core_large",
                "core_multi",
                "core_mid",
                "core_small",
                "growth_large",
                "growth_multi",
                "growth_mid",
                "growth_small",
            }
            assert expected_cols.issubset(set(cols))

            row = con.execute(
                """
                SELECT value_large, value_multi, value_mid, value_small,
                       core_large, core_multi, core_mid, core_small,
                       growth_large, growth_multi, growth_mid, growth_small
                FROM profile_and_fees_stylebox
                WHERE conid = ?
                """,
                ["profile_stylebox_1"],
            ).fetchone()

            assert row == (
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                1,
            )
        finally:
            con.close()
    finally:
        tmp.cleanup()
