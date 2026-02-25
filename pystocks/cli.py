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
        conids_file=None,
        refresh_views_at_end=True,
    ):
        """Scrape fundamental data for ETFs using the web portal proxy."""
        from .fundamentals import run_fundamentals_update
        return asyncio.run(
            run_fundamentals_update(
                limit=limit,
                verbose=verbose,
                force=force,
                conids_file=conids_file,
                refresh_views_at_end=refresh_views_at_end,
            )
        )

    def refresh_fundamentals_views(self):
        """Run SQLite maintenance and return table counts."""
        store = FundamentalsStore()
        result = store.refresh_sqlite_views()
        print(result)
        return result


    def run_pipeline(self, limit=100, verbose=False, force=False, conids_file=None):
        """Run ingestion pipeline: products -> fundamentals."""
        print("Starting full pipeline...")
        result = {}

        print("1. Scraping products...")
        result["products"] = self.scrape_products()

        print("2. Scraping fundamentals...")
        result["fundamentals"] = self.scrape_fundamentals(
            limit=limit,
            verbose=verbose,
            force=force,
            conids_file=conids_file,
            refresh_views_at_end=True,
        )

        print("Pipeline complete.")
        print(result)
        return result

if __name__ == "__main__":
    fire.Fire(PyStocksCLI)
