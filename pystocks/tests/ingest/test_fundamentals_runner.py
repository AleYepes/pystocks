import asyncio

import pystocks.ingest.fundamentals as fundamentals_module


class _DummyClientContext:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _DummySession:
    async def validate_auth_state(self):
        return True

    async def login(self, headless=False, force_browser=False):
        return True

    async def reauthenticate(self, headless=False):
        return True

    def get_primary_account_id(self):
        return "U19746488"

    def get_client(self):
        return _DummyClientContext()


class _DummyStore:
    def get_latest_price_series_effective_at_map(self, conids):
        return {str(conid): None for conid in conids}

    def persist_combined_snapshot(self, data):
        assert data["conid"] == "ok1"
        return {
            "status": "ok",
            "inserted_events": 2,
            "overwritten_events": 1,
            "unchanged_events": 0,
            "series_raw_rows_written": 3,
            "series_latest_rows_upserted": 1,
        }


class _DummyProgressBar:
    def __init__(self, total=0, desc=None):
        self.total = total
        self.desc = desc
        self.updated = 0

    def update(self, count):
        self.updated += count

    def close(self):
        return None


def test_main_marks_skipped_instruments_scraped_and_returns_structured_result(
    tmp_path, monkeypatch
):
    statuses = []

    monkeypatch.setattr(fundamentals_module, "RESEARCH_DIR", tmp_path)
    monkeypatch.setattr(fundamentals_module, "init_db", lambda: None)
    monkeypatch.setattr(fundamentals_module, "get_scraped_conids", lambda: [])
    monkeypatch.setattr(
        fundamentals_module, "get_all_instrument_conids", lambda: ["skip1", "ok1"]
    )
    monkeypatch.setattr(
        fundamentals_module,
        "update_instrument_fundamentals_status",
        lambda conid, status, mark_scraped: statuses.append(
            (conid, status, mark_scraped)
        ),
    )
    monkeypatch.setattr(fundamentals_module, "IBKRSession", _DummySession)
    monkeypatch.setattr(fundamentals_module, "FundamentalsStore", _DummyStore)
    monkeypatch.setattr(fundamentals_module, "tqdm", _DummyProgressBar)

    async def fake_scrape_conid(self, client, conid):
        if conid == "skip1":
            return {
                "conid": conid,
                "scraped_at": "2026-04-05T12:00:00",
                "_skip_fanout": True,
                "_skip_status": "skip_missing_total_net_assets",
            }
        return {
            "conid": conid,
            "scraped_at": "2026-04-05T12:00:00",
            "profile_and_fees": {"fund_and_profile": {"name": "Demo Fund"}},
        }

    monkeypatch.setattr(
        fundamentals_module.FundamentalScraper,
        "scrape_conid",
        fake_scrape_conid,
    )

    result = asyncio.run(
        fundamentals_module.main(
            telemetry_output=tmp_path / "fundamentals_run_telemetry.json",
        )
    )

    assert statuses == [
        ("skip1", "skip_missing_total_net_assets", True),
        ("ok1", "success", True),
    ]
    assert result["status"] == "ok"
    assert result["processed_conids"] == 2
    assert result["saved_snapshots"] == 1
    assert result["inserted_events"] == 2
    assert result["overwritten_events"] == 1
    assert result["series_raw_rows_written"] == 3
    assert result["series_latest_rows_upserted"] == 1
    assert (tmp_path / "fundamentals_run_telemetry.json").exists()
    assert (tmp_path / "fundamentals_run_telemetry_latest.json").exists()


def test_run_fundamentals_update_returns_main_result(monkeypatch):
    async def fake_main(**kwargs):
        return {"status": "ok", "processed_conids": kwargs["limit"]}

    monkeypatch.setattr(fundamentals_module, "main", fake_main)

    result = asyncio.run(
        fundamentals_module.run_fundamentals_update(limit=7, verbose=True)
    )

    assert result == {"status": "ok", "processed_conids": 7}
