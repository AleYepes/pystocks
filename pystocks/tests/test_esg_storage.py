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


def test_esg_scores_are_pivoted_to_wide_table():
    tmp, db_path, store = _make_store()
    try:
        snapshot = {
            "conid": "esg_1",
            "scraped_at": "2026-02-24T12:00:00+00:00",
            "ratios": {"as_of_date": "2026-02-24"},
            "esg": {
                "asOfDate": "20260131",
                "coverage": 0.997393582014,
                "source": "REFINITIV_LIPPER",
                "symbol": "SPY",
                "no_settings": True,
                "content": [
                    {"name": "TRESGS", "value": 7},
                    {"name": "TRESGCS", "value": 4},
                    {"name": "TRESGCCS", "value": 2},
                    {
                        "name": "TRESGENS",
                        "value": 6,
                        "children": [
                            {"name": "TRESGENRRS", "value": 8},
                            {"name": "TRESGENERS", "value": 7},
                            {"name": "TRESGENPIS", "value": 4},
                        ],
                    },
                    {
                        "name": "TRESGSOS",
                        "value": 7,
                        "children": [
                            {"name": "TRESGSOWOS", "value": 7},
                            {"name": "TRESGSOHRS", "value": 7},
                            {"name": "TRESGSOCOS", "value": 7},
                            {"name": "TRESGSOPRS", "value": 6},
                        ],
                    },
                    {
                        "name": "TRESGCGS",
                        "value": 6,
                        "children": [
                            {"name": "TRESGCGBDS", "value": 7},
                            {"name": "TRESGCGSRS", "value": 5},
                            {"name": "TRESGCGVSS", "value": 7},
                        ],
                    },
                ],
            },
        }

        result = store.persist_combined_snapshot(snapshot)
        assert result["status"] == "ok"

        con = sqlite3.connect(db_path)
        try:
            table_names = {
                row[0]
                for row in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            }
            assert "esg" in table_names
            assert "esg_nodes" not in table_names

            esg_row = con.execute(
                """
                SELECT coverage, source, esg_overall_score, esg_combined_score, esg_controversies_score,
                       environmental_overall_score, environmental_resource_use_score,
                       environmental_emissions_score, environmental_innovation_score,
                       social_overall_score, social_workforce_score, social_human_rights_score,
                       social_community_score, social_product_responsibility_score,
                       governance_overall_score, governance_management_score,
                       governance_shareholders_score, governance_csr_strategy_score
                FROM esg
                WHERE conid = ?
                """,
                ["esg_1"],
            ).fetchone()
            assert esg_row == (
                0.997393582014,
                "REFINITIV_LIPPER",
                7.0,
                4.0,
                2.0,
                6.0,
                8.0,
                7.0,
                4.0,
                7.0,
                7.0,
                7.0,
                7.0,
                6.0,
                6.0,
                7.0,
                5.0,
                7.0,
            )

            snapshot_cols = _table_columns(con, "esg_snapshots")
            assert "as_of_date" in snapshot_cols
            assert "coverage" not in snapshot_cols
            assert "source" not in snapshot_cols
            assert "symbol" not in snapshot_cols
            assert "no_settings" not in snapshot_cols

            esg_cols = _table_columns(con, "esg")
            assert "node_path" not in esg_cols
            assert "parent_path" not in esg_cols
            assert "depth" not in esg_cols
            assert "node_name" not in esg_cols
            assert "node_value" not in esg_cols

            as_of_date = con.execute(
                """
                SELECT as_of_date
                FROM esg_snapshots
                WHERE conid = ?
                """,
                ["esg_1"],
            ).fetchone()
            assert as_of_date == ("2026-01-31",)
        finally:
            con.close()
    finally:
        tmp.cleanup()
