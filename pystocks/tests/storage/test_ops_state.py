import tempfile
from pathlib import Path

import pystocks.storage.ops_state as ops_state


def test_get_scraped_conids_returns_last_seven_days():
    tmp = tempfile.TemporaryDirectory()
    try:
        db_path = Path(tmp.name) / "ops_state.sqlite"
        original_db_path = ops_state.DB_PATH
        original_initialized_db_path = ops_state._INITIALIZED_DB_PATH
        ops_state.DB_PATH = db_path
        ops_state._INITIALIZED_DB_PATH = None

        try:
            ops_state.init_db()
            conn = ops_state.get_connection()
            try:
                conn.executemany(
                    """
                    INSERT INTO products (conid, last_scraped_fundamentals, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    [
                        ("inside_window", "2026-03-27", "2026-03-27T00:00:00+00:00"),
                        ("window_start", "2026-03-21", "2026-03-21T00:00:00+00:00"),
                        ("outside_window", "2026-03-20", "2026-03-20T00:00:00+00:00"),
                    ],
                )
                conn.commit()
            finally:
                conn.close()

            assert ops_state.get_scraped_conids(today="2026-03-27") == [
                "inside_window",
                "window_start",
            ]
        finally:
            ops_state.DB_PATH = original_db_path
            ops_state._INITIALIZED_DB_PATH = original_initialized_db_path
    finally:
        tmp.cleanup()
