import sqlite3
import tempfile
from pathlib import Path

from pystocks.fundamentals_store import FundamentalsStore


def _make_store():
    tmp = tempfile.TemporaryDirectory()
    db_path = Path(tmp.name) / "pystocks.sqlite"
    store = FundamentalsStore(sqlite_path=db_path)
    return tmp, db_path, store


def _table_columns(con, table):
    return [r[1] for r in con.execute(f"PRAGMA table_info({table})").fetchall()]


def test_morningstar_summary_is_wide_and_commentary_is_filtered():
    tmp, db_path, store = _make_store()
    try:
        snapshot = {
            "conid": "mstar_1",
            "scraped_at": "2026-02-23T12:00:00+00:00",
            "ratios": {"as_of_date": "2026-02-23"},
            "morningstar": {
                "as_of_date": "20260131",
                "q_full_report_id": "report_123",
                "summary": [
                    {"id": "medalist_rating", "value": "Silver"},
                    {"id": "process", "value": "High"},
                    {"id": "people", "value": "Above_Average"},
                    {"id": "parent", "value": "Above_Average"},
                    {"id": "morningstar_rating", "value": "4"},
                    {"id": "sustainability_rating", "value": "Average"},
                    {"id": "category", "value": "Large Blend"},
                    {"id": "category_index", "value": "Morningstar US Large-Mid TR USD"},
                    {"id": "ignored_metric", "value": "ignore me"},
                ],
                "commentary": [
                    {
                        "id": "summary",
                        "subsection_id": "summary_title",
                        "publish_date": "20260128",
                        "text": "Best-in-class option for large-cap US stocks.",
                        "title": "Summary",
                        "subtitle": "Title",
                        "q": False,
                        "author": {"name": "Brendan McCann"},
                    },
                    {
                        "id": "summary",
                        "subsection_id": "summary_body",
                        "publish_date": "20260128",
                        "text": "   ",
                    },
                ],
            },
        }

        result = store.persist_combined_snapshot(snapshot)
        assert result["status"] == "ok"

        con = sqlite3.connect(db_path)
        try:
            summary = con.execute(
                """
                SELECT medalist_rating, process, people, parent, morningstar_rating,
                       sustainability_rating, category, category_index
                FROM morningstar_summary
                WHERE conid = ?
                """,
                ["mstar_1"],
            ).fetchone()
            assert summary == (
                "Silver",
                "High",
                "Above_Average",
                "Above_Average",
                4.0,
                "Average",
                "Large Blend",
                "Morningstar US Large-Mid TR USD",
            )

            snapshot_cols = _table_columns(con, "morningstar_snapshots")
            assert "source_file" not in snapshot_cols

            summary_cols = _table_columns(con, "morningstar_summary")
            assert "metric_id" not in summary_cols
            assert "title" not in summary_cols
            assert "value_text" not in summary_cols
            assert "value_num" not in summary_cols
            assert "publish_date" not in summary_cols
            assert "q" not in summary_cols
            assert "id" not in summary_cols

            commentary_cols = _table_columns(con, "morningstar_commentary")
            assert "title" not in commentary_cols
            assert "subtitle" not in commentary_cols
            assert "q" not in commentary_cols
            assert "id" not in commentary_cols

            commentary_rows = con.execute(
                """
                SELECT item_id, subsection_id, publish_date, text, author_name
                FROM morningstar_commentary
                WHERE conid = ?
                """,
                ["mstar_1"],
            ).fetchall()
            assert commentary_rows == [
                (
                    "summary",
                    "summary_title",
                    "2026-01-28",
                    "Best-in-class option for large-cap US stocks.",
                    "Brendan McCann",
                )
            ]
        finally:
            con.close()
    finally:
        tmp.cleanup()


def test_child_and_series_raw_tables_do_not_use_synthetic_id_columns():
    tmp, db_path, _ = _make_store()
    try:
        con = sqlite3.connect(db_path)
        try:
            tables = [
                "profile_and_fees_reports",
                "ratios_ratios",
                "ratios_financials",
                "ratios_fixed_income",
                "ratios_dividend",
                "ratios_zscore",
                "lipper_ratings",
                "dividends_industry_metrics",
                "morningstar_summary",
                "morningstar_commentary",
                "performance_metrics",
                "ownership_owners_types",
                "ownership_holders",
                "esg_nodes",
                "price_chart_series_raw",
                "sentiment_search_series_raw",
                "ownership_trade_log_series_raw",
                "dividends_events_series_raw",
            ]
            for table in tables:
                cols = _table_columns(con, table)
                assert cols
                assert "id" not in cols
        finally:
            con.close()
    finally:
        tmp.cleanup()
