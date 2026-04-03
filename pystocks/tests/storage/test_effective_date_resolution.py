import sqlite3
import tempfile
from pathlib import Path

from pystocks.storage.fundamentals_store import FundamentalsStore


def _make_store():
    tmp = tempfile.TemporaryDirectory()
    db_path = Path(tmp.name) / "pystocks.sqlite"
    store = FundamentalsStore(sqlite_path=db_path)
    return tmp, db_path, store


def test_effective_date_uses_ratios_as_of_date_for_all_endpoints():
    tmp, db_path, store = _make_store()
    try:
        snapshot = {
            "conid": "effective_anchor_1",
            "scraped_at": "2026-02-24T12:00:00+00:00",
            "holdings": {
                "as_of_date": "2026-02-20",
                "currency": [
                    {"name": "US Dollar", "weight": "100%", "vs": "95%", "code": "USD"}
                ],
            },
            "ratios": {
                "as_of_date": "2026-02-19",
            },
            "profile_and_fees": {
                "fund_and_profile": [
                    {"name": "Maturity Date", "value": "2030-12-31"},
                    {
                        "name": "Total Net Assets (Month End)",
                        "value": "$1.2B (2026/01/30)",
                    },
                ],
            },
        }
        result = store.persist_combined_snapshot(snapshot)
        assert result["status"] == "ok"

        con = sqlite3.connect(db_path)
        try:
            holdings_row = con.execute(
                "SELECT effective_at FROM holdings_snapshots WHERE conid = ?",
                ["effective_anchor_1"],
            ).fetchone()
            ratios_row = con.execute(
                "SELECT effective_at FROM ratios_snapshots WHERE conid = ?",
                ["effective_anchor_1"],
            ).fetchone()
            profile_row = con.execute(
                """
                SELECT s.effective_at, p.maturity_date, p.total_net_assets_date
                FROM profile_and_fees_snapshots s
                JOIN profile_and_fees p USING (conid, effective_at)
                WHERE s.conid = ?
                """,
                ["effective_anchor_1"],
            ).fetchone()

            assert holdings_row[0] == "2026-02-19"
            assert ratios_row[0] == "2026-02-19"
            assert profile_row[0] == "2026-02-19"
            assert profile_row[1] == "2030-12-31"
            assert profile_row[2] == "2026-01-30"
        finally:
            con.close()
    finally:
        tmp.cleanup()


def test_effective_date_uses_ratios_when_holdings_missing():
    tmp, db_path, store = _make_store()
    try:
        snapshot = {
            "conid": "effective_anchor_2",
            "scraped_at": "2026-02-24T12:00:00+00:00",
            "ratios": {
                "as_of_date": "2026-02-19",
            },
            "profile_and_fees": {
                "fund_and_profile": [
                    {"name": "Maturity Date", "value": "2030-12-31"},
                ],
            },
        }
        result = store.persist_combined_snapshot(snapshot)
        assert result["status"] == "ok"

        con = sqlite3.connect(db_path)
        try:
            ratios_row = con.execute(
                "SELECT effective_at FROM ratios_snapshots WHERE conid = ?",
                ["effective_anchor_2"],
            ).fetchone()
            profile_row = con.execute(
                "SELECT effective_at FROM profile_and_fees_snapshots WHERE conid = ?",
                ["effective_anchor_2"],
            ).fetchone()
            assert ratios_row[0] == "2026-02-19"
            assert profile_row[0] == "2026-02-19"
        finally:
            con.close()
    finally:
        tmp.cleanup()


def test_persist_skips_snapshot_when_ratios_as_of_date_is_missing():
    tmp, db_path, store = _make_store()
    try:
        snapshot = {
            "conid": "effective_anchor_3",
            "scraped_at": "2026-02-24T12:34:56+00:00",
            "profile_and_fees": {
                "objective": "Only profile endpoint in snapshot",
            },
        }
        result = store.persist_combined_snapshot(snapshot)
        assert result["status"] == "missing_ratios_effective_at"

        con = sqlite3.connect(db_path)
        try:
            row = con.execute(
                "SELECT effective_at FROM profile_and_fees_snapshots WHERE conid = ?",
                ["effective_anchor_3"],
            ).fetchone()
            assert row is None
        finally:
            con.close()
    finally:
        tmp.cleanup()
