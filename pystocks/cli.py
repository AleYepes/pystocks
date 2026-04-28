import asyncio
from copy import deepcopy
from typing import Any

import fire

from .config import SQLITE_DB_PATH
from .storage._sqlite import open_connection

ANALYSIS_INPUT_CONTRACT = {
    "stage_order": [
        "scrape_products",
        "scrape_fundamentals",
        "fetch_supplementary_data",
        "preprocess_supplementary_data",
        "preprocess_prices",
        "preprocess_snapshots",
        "run_analysis",
    ],
    "required_analysis_artifacts": {
        "price_history": {
            "producer": "preprocess_prices",
            "artifacts": [
                "analysis_daily_returns.parquet",
                "analysis_price_eligibility.parquet",
            ],
        },
        "snapshot_features": {
            "producer": "preprocess_snapshots",
            "artifacts": [
                "analysis_snapshot_features.parquet",
            ],
        },
        "supplementary_features": {
            "producers": [
                "fetch_supplementary_data",
                "preprocess_supplementary_data",
            ],
            "alternative": "refresh_supplementary_data",
            "artifacts": [
                "analysis_risk_free_daily.parquet",
                "analysis_world_bank_country_features.parquet",
            ],
        },
    },
    "optional_preprocess_artifacts": {
        "dividend_events": {
            "producer": "preprocess_dividends",
            "artifacts": [
                "analysis_dividend_events.parquet",
                "analysis_dividend_summary.parquet",
            ],
        }
    },
}


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

    def preprocess_snapshots(self, show_progress: bool = True) -> Any:
        """Build cleaned snapshot-feature artifacts and diagnostics."""
        from .preprocess.snapshots import run_snapshot_preprocess

        return run_snapshot_preprocess(show_progress=show_progress)

    def fetch_supplementary_data(self, show_progress: bool = True) -> Any:
        """Fetch raw supplementary macro and risk-free source datasets."""
        from .ingest.supplementary import fetch_supplementary_data

        return fetch_supplementary_data(show_progress=show_progress)

    def preprocess_supplementary_data(self, show_progress: bool = True) -> Any:
        """Build preprocessed supplementary artifacts from stored raw datasets."""
        from .preprocess.supplementary import run_supplementary_preprocess

        return run_supplementary_preprocess(show_progress=show_progress)

    def refresh_supplementary_data(self, show_progress: bool = True) -> Any:
        """Fetch supplementary source data, then run supplementary preprocess."""
        return {
            "fetch": self.fetch_supplementary_data(show_progress=show_progress),
            "preprocess": self.preprocess_supplementary_data(
                show_progress=show_progress
            ),
        }

    def run_preprocess_pipeline(
        self,
        show_progress: bool = True,
        refresh_supplementary: bool = True,
    ) -> dict[str, Any]:
        """Run the required preprocess stage before analysis."""
        result: dict[str, Any] = {}

        if refresh_supplementary:
            print("3. Fetching supplementary data...")
            result["supplementary_fetch"] = self.fetch_supplementary_data(
                show_progress=show_progress,
            )

            print("4. Preprocessing supplementary data...")
            result["supplementary_preprocess"] = self.preprocess_supplementary_data(
                show_progress=show_progress
            )

            price_step = 5
        else:
            price_step = 3

        print(f"{price_step}. Preprocessing prices...")
        result["price_preprocess"] = self.preprocess_prices(show_progress=show_progress)

        print(f"{price_step + 1}. Preprocessing snapshots...")
        result["snapshot_preprocess"] = self.preprocess_snapshots(
            show_progress=show_progress
        )

        return result

    def build_analysis_panel(self, show_progress: bool = True) -> Any:
        """Build the point-in-time analysis snapshot panel."""
        from .analysis import build_analysis_panel

        return build_analysis_panel(show_progress=show_progress)

    def run_factor_research(self, show_progress: bool = True) -> Any:
        """Build factor returns, run sleeve research, and persist outputs."""
        from .analysis import run_factor_research

        return run_factor_research(show_progress=show_progress)

    def run_walk_forward_research(self, show_progress: bool = True) -> Any:
        """Run the full walk-forward factor research pipeline."""
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

    def describe_analysis_inputs(self) -> dict[str, Any]:
        """Describe the explicit preprocess artifacts required by analysis."""
        return deepcopy(ANALYSIS_INPUT_CONTRACT)

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
        refresh_supplementary: bool = True,
    ) -> dict[str, Any]:
        """Run the end-to-end DAG: ingest -> preprocess -> analysis."""
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

        result.update(
            self.run_preprocess_pipeline(
                show_progress=show_progress,
                refresh_supplementary=refresh_supplementary,
            )
        )

        analysis_step = 7 if refresh_supplementary else 5
        print(f"{analysis_step}. Running analysis...")
        result["analysis"] = self.run_analysis(show_progress=show_progress)

        print("Pipeline complete.")
        return result


if __name__ == "__main__":
    fire.Fire(PyStocksCLI)
