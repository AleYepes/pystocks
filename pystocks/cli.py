import fire
import asyncio
from .product_scraper import scrape_ibkr_products
from .fundamentals import run_fundamentals_update
from .preprocess import Preprocessor
from .analysis import PortfolioAnalyzer

class PyStocksCLI:
    def scrape_products(self):
        """Scrape the list of available ETF products from IBKR website."""
        asyncio.run(scrape_ibkr_products())

    def scrape_fundamentals(self, limit=100):
        """Scrape fundamental data for ETFs using the web portal proxy."""
        asyncio.run(run_fundamentals_update(limit=limit))

    def preprocess(self):
        """Clean and prepare raw data for analysis."""
        pp = Preprocessor()
        pp.run_full_pipeline()

    def analyze(self):
        """Run factor analysis and portfolio optimization."""
        analyzer = PortfolioAnalyzer()
        analyzer.load_latest_fundamentals()
        analyzer.load_historical_series()
        analyzer.run_factor_analysis()

    def full_pipeline(self):
        """Run the entire pipeline from contract discovery to analysis."""
        print("Starting full pipeline...")
        self.scrape_products()
        self.scrape_fundamentals()
        self.preprocess()
        self.analyze()

if __name__ == "__main__":
    fire.Fire(PyStocksCLI)
