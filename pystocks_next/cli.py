from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import Any

import fire

from .collection import (
    CollectionSession,
    FundamentalsCollector,
    refresh_product_universe,
    refresh_supplementary_sources,
)
from .config import PystocksNextConfig
from .feature_inputs import build_analysis_input_bundle
from .storage import connect_sqlite, initialize_operational_store


def _load_text_lines(path: str | None) -> list[str] | None:
    if path is None:
        return None
    lines = [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return lines or None


def _frame_row_counts(bundle: Any) -> dict[str, int]:
    return {
        "prices_rows": len(bundle.prices),
        "price_eligibility_rows": len(bundle.price_eligibility),
        "dividends_rows": len(bundle.dividends),
        "dividend_summary_rows": len(bundle.dividend_summary),
        "snapshot_features_rows": len(bundle.snapshot_features),
        "snapshot_holdings_diagnostics_rows": len(bundle.snapshot_holdings_diagnostics),
        "snapshot_ratio_diagnostics_rows": len(bundle.snapshot_ratio_diagnostics),
        "snapshot_table_summary_rows": len(bundle.snapshot_table_summary),
        "risk_free_daily_rows": len(bundle.risk_free_daily),
        "macro_features_rows": len(bundle.macro_features),
    }


class PyStocksNextCLI:
    def __init__(self, project_root: str | None = None) -> None:
        self._project_root = Path(project_root) if project_root is not None else None

    def _config(self) -> PystocksNextConfig:
        return PystocksNextConfig.from_env(project_root=self._project_root)

    def _collection_session(self) -> CollectionSession:
        root = self._project_root or Path.cwd()
        data_dir = root / "data"
        return CollectionSession(
            state_path=data_dir / "auth_state.json",
            credentials_path=data_dir / "login_credentials.json",
        )

    def init_storage(self) -> dict[str, object]:
        config = self._config()
        version = initialize_operational_store(config.sqlite.path)
        config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        return {
            "status": "ok",
            "sqlite_path": str(config.sqlite.path),
            "artifacts_dir": str(config.artifacts_dir),
            "schema_version": version,
        }

    def refresh_universe(self) -> dict[str, object]:
        config = self._config()
        initialize_operational_store(config.sqlite.path)
        with connect_sqlite(config.sqlite.path) as conn:
            result = asyncio.run(refresh_product_universe(conn))
        payload = asdict(result)
        payload["sqlite_path"] = str(config.sqlite.path)
        return payload

    def login(
        self,
        headless: bool = False,
        force_browser: bool = True,
    ) -> dict[str, object]:
        session = self._collection_session()
        authenticated = asyncio.run(
            session.login(
                headless=headless,
                force_browser=force_browser,
            )
        )
        return {
            "status": "ok" if authenticated else "auth_failed",
            "state_path": str(session.state_path),
        }

    def collect_fundamentals(
        self,
        limit: int | None = None,
        start_index: int = 0,
        force: bool = False,
        conids_file: str | None = None,
        telemetry_filename: str = "fundamentals_collection.json",
    ) -> dict[str, object]:
        config = self._config()
        initialize_operational_store(config.sqlite.path)
        explicit_conids = _load_text_lines(conids_file)
        telemetry_output_path = config.artifacts_dir / "collection" / telemetry_filename
        with connect_sqlite(config.sqlite.path) as conn:
            result = asyncio.run(
                FundamentalsCollector(session=self._collection_session()).run(
                    conn,
                    explicit_conids=explicit_conids,
                    limit=limit,
                    start_index=start_index,
                    force=force,
                    telemetry_output_path=telemetry_output_path,
                )
            )
        payload = asdict(result)
        payload["sqlite_path"] = str(config.sqlite.path)
        return payload

    def refresh_supplementary(
        self,
        economy_codes_file: str | None = None,
    ) -> dict[str, object]:
        config = self._config()
        initialize_operational_store(config.sqlite.path)
        economy_codes = _load_text_lines(economy_codes_file)
        with connect_sqlite(config.sqlite.path) as conn:
            result = refresh_supplementary_sources(
                conn,
                economy_codes=economy_codes,
            )
        payload = asdict(result)
        payload["sqlite_path"] = str(config.sqlite.path)
        return payload

    def build_inputs(self) -> dict[str, object]:
        config = self._config()
        initialize_operational_store(config.sqlite.path)
        with connect_sqlite(config.sqlite.path, read_only=True) as conn:
            bundle = build_analysis_input_bundle(conn=conn)
        return {
            "status": "ok",
            "sqlite_path": str(config.sqlite.path),
            **_frame_row_counts(bundle),
        }

    def run_pipeline(
        self,
        refresh_universe: bool = True,
        collect_fundamentals: bool = True,
        refresh_supplementary: bool = True,
        build_inputs: bool = True,
        limit: int | None = None,
        start_index: int = 0,
        force: bool = False,
        conids_file: str | None = None,
        economy_codes_file: str | None = None,
        telemetry_filename: str = "fundamentals_collection.json",
    ) -> dict[str, object]:
        result: dict[str, object] = {"storage": self.init_storage()}
        if refresh_universe:
            result["refresh_universe"] = self.refresh_universe()
        if collect_fundamentals:
            result["collect_fundamentals"] = self.collect_fundamentals(
                limit=limit,
                start_index=start_index,
                force=force,
                conids_file=conids_file,
                telemetry_filename=telemetry_filename,
            )
        if refresh_supplementary:
            result["refresh_supplementary"] = self.refresh_supplementary(
                economy_codes_file=economy_codes_file
            )
        if build_inputs:
            result["build_inputs"] = self.build_inputs()
        return result


def main() -> None:
    fire.Fire(PyStocksNextCLI)


if __name__ == "__main__":
    main()
