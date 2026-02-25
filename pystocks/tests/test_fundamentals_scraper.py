import unittest
from collections import Counter, defaultdict
from pathlib import Path
import tempfile
from unittest.mock import AsyncMock

from pystocks.fundamentals import FundamentalScraper, _load_conids_from_file


class FundamentalScraperLandingFilterTests(unittest.IsolatedAsyncioTestCase):
    def _build_scraper(self):
        scraper = FundamentalScraper.__new__(FundamentalScraper)
        scraper.esg_account_id = None
        scraper.telemetry = {
            "run_started_at": "2026-02-24T00:00:00+00:00",
            "endpoint_calls": Counter(),
            "endpoint_useful_payloads": Counter(),
            "status_codes": defaultdict(Counter),
        }
        return scraper

    def test_landing_has_total_net_assets(self):
        scraper = self._build_scraper()
        landing = {
            "key_profile": {
                "data": {
                    "total_net_assets": "$1.66B (2026/01/30)",
                }
            }
        }
        self.assertTrue(scraper._landing_has_total_net_assets(landing))

    def test_landing_without_total_net_assets(self):
        scraper = self._build_scraper()
        self.assertFalse(
            scraper._landing_has_total_net_assets(
                {"key_profile": {"data": {"market_geo_focus": "Global"}}}
            )
        )
        self.assertFalse(
            scraper._landing_has_total_net_assets(
                {"key_profile": {"data": {"total_net_assets": None}}}
            )
        )
        self.assertFalse(
            scraper._landing_has_total_net_assets(
                {"key_profile": {"data": {"total_net_assets": "   "}}}
            )
        )

    async def test_scrape_conid_skips_followups_when_total_net_assets_missing(self):
        scraper = self._build_scraper()
        landing_payload = {
            "key_profile": {
                "meta": {"name_tag": "key_profile"},
                "data": {"market_geo_focus": "Global"},
            }
        }
        scraper.fetch_endpoint = AsyncMock(return_value=landing_payload)

        data = await scraper.scrape_conid(client=object(), conid="625493486")

        self.assertEqual(scraper.fetch_endpoint.await_count, 1)
        first_call_args = scraper.fetch_endpoint.await_args_list[0].args
        self.assertTrue(first_call_args[1].startswith("landing/625493486?widgets="))
        self.assertEqual(set(data.keys()), {"conid", "scraped_at", "landing"})
        self.assertEqual(data["landing"], landing_payload)
        self.assertEqual(scraper.telemetry["endpoint_useful_payloads"]["landing"], 1)

    async def test_scrape_conid_fetches_followups_when_total_net_assets_present(self):
        scraper = self._build_scraper()
        landing_payload = {
            "key_profile": {
                "meta": {"name_tag": "key_profile"},
                "data": {"total_net_assets": "$1.66B (2026/01/30)"},
            }
        }

        async def fake_fetch_endpoint(_client, endpoint):
            if endpoint.startswith("landing/"):
                return landing_payload
            return {}

        scraper.fetch_endpoint = AsyncMock(side_effect=fake_fetch_endpoint)
        scraper.fetch_performance_with_period_fallback = AsyncMock(return_value=(None, None))

        data = await scraper.scrape_conid(client=object(), conid="89384980")

        self.assertEqual(scraper.fetch_endpoint.await_count, 11)
        self.assertEqual(scraper.fetch_performance_with_period_fallback.await_count, 1)
        self.assertEqual(data["landing"], landing_payload)


if __name__ == "__main__":
    unittest.main()


class ConidFileParsingTests(unittest.TestCase):
    def test_load_conids_from_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "conids.txt"
            path.write_text("123\n456\n123\n\n0051\n")
            self.assertEqual(_load_conids_from_file(path), ["123", "456", "0051"])
