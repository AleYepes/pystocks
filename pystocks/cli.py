import fire
import asyncio
from .fundamentals_store import FundamentalsStore

class PyStocksCLI:
    def scrape_products(self):
        """Scrape the list of available ETF products from IBKR website."""
        from .product_scraper import scrape_ibkr_products
        asyncio.run(scrape_ibkr_products())

    def scrape_fundamentals(self, limit=None, verbose=False):
        """Scrape fundamental data for ETFs using the web portal proxy."""
        from .fundamentals import run_fundamentals_update
        asyncio.run(run_fundamentals_update(limit=limit, verbose=verbose))

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

    def run_pipeline(self):
        """Run the full tail-end pipeline: preprocess prices -> analysis."""
        print("Starting pipeline...")
        print("1. Preprocessing prices...")
        self.preprocess_prices()
        print("2. Running analysis...")
        self.run_analysis()
        print("Pipeline complete.")

if __name__ == "__main__":
    fire.Fire(PyStocksCLI)
