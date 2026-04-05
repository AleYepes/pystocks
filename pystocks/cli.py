import asyncio
from typing import Any

import fire

from .config import SQLITE_DB_PATH
from .storage._sqlite import open_connection


class PyStocksCLI:
    def scrape_products(self) -> Any:
        """Scrape the list of available ETF products from IBKR website."""
        from .ingest.product_scraper import scrape_ibkr_products

        return asyncio.run(scrape_ibkr_products())

    def scrape_fundamentals(
        self,
        limit: int | None = None,
        verbose: bool = False,
        force: bool = False,
        conids_file: str | None = None,
    ) -> Any:
        """Scrape fundamental data for ETFs using the web portal proxy."""
        from .ingest.fundamentals import run_fundamentals_update

        return asyncio.run(
            run_fundamentals_update(
                limit=limit,
                verbose=verbose,
                force=force,
                conids_file=conids_file,
            )
        )

    def preprocess_prices(self, show_progress: bool = True) -> Any:
        """Build clean daily return artifacts and price eligibility tables."""
        from .preprocess.price import run_price_preprocess

        return run_price_preprocess(show_progress=show_progress)

    def preprocess_dividends(self) -> Any:
        """Build cleaned dividend-event artifacts for total-return analysis."""
        from .preprocess.dividends import run_dividend_preprocess

        return run_dividend_preprocess()

    def preprocess_snapshots(self) -> Any:
        """Build cleaned snapshot-feature artifacts and diagnostics."""
        from .preprocess.snapshots import run_snapshot_preprocess

        return run_snapshot_preprocess()

    def build_analysis_panel(self, show_progress: bool = True) -> Any:
        """Build the point-in-time analysis snapshot panel."""
        from .analysis import build_analysis_panel

        return build_analysis_panel(show_progress=show_progress)

    def run_factor_research(self, show_progress: bool = True) -> Any:
        """Build factor returns, run sleeve research, and persist outputs."""
        from .analysis import run_factor_research

        return run_factor_research(show_progress=show_progress)

    def compute_factor_betas(self, show_progress: bool = True) -> Any:
        """Compute current ETF factor betas from persistent factors."""
        from .analysis import compute_current_betas

        return compute_current_betas(show_progress=show_progress)

    def run_analysis(self, show_progress: bool = True) -> Any:
        """Run the full analysis pipeline."""
        from .analysis import run_analysis_pipeline

        return run_analysis_pipeline(show_progress=show_progress)

    def refresh_fundamentals_views(self) -> dict[str, str]:
        """Run lightweight SQLite maintenance for the fundamentals store."""
        with open_connection(SQLITE_DB_PATH) as conn:
            conn.execute("ANALYZE;")
        return {"status": "ok", "sqlite_path": str(SQLITE_DB_PATH)}

    def run_pipeline(
        self,
        limit: int | None = 100,
        verbose: bool = False,
        force: bool = False,
        conids_file: str | None = None,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """Run ingestion and analysis pipeline: products -> fundamentals -> analysis."""
        print("Starting full pipeline...")
        result: dict[str, Any] = {}

        print("1. Scraping products...")
        result["products"] = self.scrape_products()

        print("2. Scraping fundamentals...")
        result["fundamentals"] = self.scrape_fundamentals(
            limit=limit,
            verbose=verbose,
            force=force,
            conids_file=conids_file,
        )

        print("3. Running analysis...")
        result["analysis"] = self.run_analysis(show_progress=show_progress)

        print("Pipeline complete.")
        print(result)
        return result


if __name__ == "__main__":
    fire.Fire(PyStocksCLI)
