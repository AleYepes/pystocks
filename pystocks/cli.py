import fire
import asyncio
from .fundamentals_store import FundamentalsStore

class PyStocksCLI:
    def scrape_products(self):
        """Scrape the list of available ETF products from IBKR website."""
        from .product_scraper import scrape_ibkr_products
        return asyncio.run(scrape_ibkr_products())

    def scrape_fundamentals(
        self,
        limit=None,
        verbose=False,
        force=False,
        refresh_views_at_end=True,
    ):
        """Scrape fundamental data for ETFs using the web portal proxy."""
        from .fundamentals import run_fundamentals_update
        return asyncio.run(
            run_fundamentals_update(
                limit=limit,
                verbose=verbose,
                force=force,
                refresh_duckdb_at_end=refresh_views_at_end,
            )
        )

    def refresh_fundamentals_views(self):
        """Refresh DuckDB views over normalized fundamentals stores."""
        store = FundamentalsStore()
        result = store.refresh_duckdb_views()
        print(result)
        return result

    def preprocess_prices(self):
        """Run the price preprocessing pipeline (clean, dedup, quality check)."""
        from .price_preprocess import run
        result = run()
        print(result)
        return result

    def run_analysis(self):
        """Run the daily factor analysis using cleaned prices and factors."""
        from .analysis import run
        result = run()
        print(result)
        return result

    def run_pipeline(self, limit=100, verbose=False, force=False):
        """Run full pipeline: products -> fundamentals -> views -> preprocess -> analysis."""
        print("Starting full pipeline...")
        result = {}

        print("1. Scraping products...")
        result["products"] = self.scrape_products()

        print("2. Scraping fundamentals...")
        result["fundamentals"] = self.scrape_fundamentals(
            limit=limit,
            verbose=verbose,
            force=force,
            refresh_views_at_end=False,
        )

        print("3. Refreshing fundamentals views...")
        result["refresh_views"] = self.refresh_fundamentals_views()

        print("4. Preprocessing prices...")
        result["preprocess_prices"] = self.preprocess_prices()

        print("5. Running analysis...")
        result["analysis"] = self.run_analysis()

        print("Pipeline complete.")
        print(result)
        return result

if __name__ == "__main__":
    fire.Fire(PyStocksCLI)
