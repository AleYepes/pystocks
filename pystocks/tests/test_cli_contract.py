import unittest
from unittest.mock import MagicMock

from pystocks.cli import PyStocksCLI


class CliContractTests(unittest.TestCase):
    def test_postprocessing_commands_are_deferred(self):
        cli = PyStocksCLI()

        preprocess = cli.preprocess_prices()
        analysis = cli.run_analysis()

        self.assertEqual(preprocess["status"], "deferred")
        self.assertEqual(analysis["status"], "deferred")

    def test_run_pipeline_stops_after_fundamentals(self):
        cli = PyStocksCLI()
        cli.scrape_products = MagicMock(return_value={"status": "ok"})
        cli.scrape_fundamentals = MagicMock(return_value={"status": "ok"})
        cli.refresh_fundamentals_views = MagicMock(return_value={"status": "should_not_call"})
        cli.preprocess_prices = MagicMock(return_value={"status": "should_not_call"})
        cli.run_analysis = MagicMock(return_value={"status": "should_not_call"})

        result = cli.run_pipeline(limit=5, verbose=True, force=True)

        self.assertEqual(set(result.keys()), {"products", "fundamentals"})
        self.assertEqual(cli.scrape_products.call_count, 1)
        self.assertEqual(cli.scrape_fundamentals.call_count, 1)
        self.assertEqual(cli.refresh_fundamentals_views.call_count, 0)
        self.assertEqual(cli.preprocess_prices.call_count, 0)
        self.assertEqual(cli.run_analysis.call_count, 0)


if __name__ == "__main__":
    unittest.main()
