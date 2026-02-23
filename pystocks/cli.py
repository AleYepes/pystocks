import fire
import asyncio
from .fundamentals_store import FundamentalsStore

class PyStocksCLI:
    def scrape_products(self, verbose=False):
        """Scrape the list of available ETF products from IBKR website."""
        from .product_scraper import scrape_ibkr_products
        asyncio.run(scrape_ibkr_products(verbose=verbose))

    def scrape_fundamentals(self, limit=None, verbose=False):
        """Scrape fundamental data for ETFs using the web portal proxy."""
        from .fundamentals import run_fundamentals_update
        asyncio.run(run_fundamentals_update(limit=limit, verbose=verbose))

    def preprocess(self):
        """Clean and prepare raw data for analysis."""
        from .preprocess import Preprocessor
        pp = Preprocessor()
        pp.run_full_pipeline()

    def refresh_fundamentals_views(self):
        """Refresh DuckDB views over normalized fundamentals stores."""
        store = FundamentalsStore()
        result = store.refresh_duckdb_views()
        print(result)
        return result

    def backfill_fundamentals_normalized(self):
        """Rebuild normalized factor + series parquet stores from CAS endpoint events."""
        store = FundamentalsStore()
        result = {
            "factor_features": store.backfill_factor_features(refresh_duckdb=False, purge_existing=True),
            "price_chart_series": store.backfill_price_chart_series(refresh_duckdb=False, purge_existing=True),
            "sentiment_search_series": store.backfill_sentiment_search_series(refresh_duckdb=False, purge_existing=True),
            "ownership_trade_log_series": store.backfill_ownership_trade_log_series(refresh_duckdb=False, purge_existing=True),
            "dividends_events_series": store.backfill_dividends_events_series(refresh_duckdb=False, purge_existing=True),
        }
        result["refresh_views"] = store.refresh_duckdb_views()
        print(result)
        return result

    def analyze(self):
        """Run factor analysis and portfolio optimization."""
        from .analysis import PortfolioAnalyzer
        analyzer = PortfolioAnalyzer()
        analyzer.load_latest_fundamentals()
        analyzer.load_historical_series()
        analyzer.run_factor_analysis()

    def full_pipeline(self, verbose=False):
        """Run the entire pipeline from contract discovery to analysis."""
        if verbose:
            print("Starting full pipeline...")
        self.scrape_products()
        self.scrape_fundamentals(verbose=verbose)
        self.preprocess()
        self.analyze()

    def preprocess_prices(self):
        """Run the price preprocessing pipeline (clean, dedup, quality check)."""
        from .price_preprocess import run
        result = run()
        print(result)
        return result

    def run_analysis_v1(self):
        """Run the daily factor analysis (v1) using cleaned prices and factors."""
        from .analysis_v1 import run
        result = run()
        print(result)
        return result

    def run_tail_pipeline(self):
        """Run the full tail-end pipeline: preprocess prices -> analysis v1."""
        print("Starting tail pipeline...")
        print("1. Preprocessing prices...")
        self.preprocess_prices()
        print("2. Running Analysis V1...")
        self.run_analysis_v1()
        print("Tail pipeline complete.")

if __name__ == "__main__":
    fire.Fire(PyStocksCLI)
