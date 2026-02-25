import sqlite3
import tempfile
from pathlib import Path

from pystocks.fundamentals_store import FundamentalsStore


def _make_store():
    tmp = tempfile.TemporaryDirectory()
    db_path = Path(tmp.name) / "pystocks.sqlite"
    store = FundamentalsStore(sqlite_path=db_path)
    return tmp, db_path, store


def test_profile_fees_merges_launch_opening_price_and_splits_total_net_assets():
    tmp, db_path, store = _make_store()
    try:
        snapshot = {
            "conid": "profile_merge_1",
            "scraped_at": "2026-02-23T12:00:00+00:00",
            "profile_and_fees": {
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
                SELECT inception_date, total_net_assets_value, total_net_assets_date
                FROM profile_fees_fund_profile_fields
                WHERE conid = ?
                """,
                ["profile_merge_1"],
            ).fetchone()
            assert row == ("2018-07-15", "CAD289.15M", "2026-01-30")

            cols = [
                r[1]
                for r in con.execute("PRAGMA table_info(profile_fees_fund_profile_fields)").fetchall()
            ]
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
                FROM profile_fees_fund_profile_fields
                WHERE conid = ?
                """,
                ["profile_merge_2"],
            ).fetchone()
            assert row == ("2010-01-02", "€20.86M", "2026-01-30")
        finally:
            con.close()
    finally:
        tmp.cleanup()
