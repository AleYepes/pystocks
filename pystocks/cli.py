import fire
import asyncio

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
    ):
        """Scrape fundamental data for ETFs using the web portal proxy."""
        from .fundamentals import run_fundamentals_update
        return asyncio.run(
            run_fundamentals_update(
                limit=limit,
                verbose=verbose,
                force=force,
                conids_file=conids_file,
            )
        )

    def preprocess_prices(self):
        """Deferred until price ingestion/materialization is finalized."""
        return {"status": "deferred", "step": "price_preprocess"}

    def run_analysis(self):
        """Deferred until factor analysis pipeline is finalized."""
        return {"status": "deferred", "step": "analysis"}

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
        )

        print("Pipeline complete.")
        print(result)
        return result

if __name__ == "__main__":
    fire.Fire(PyStocksCLI)
