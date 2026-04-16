from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SqliteConfig:
    """Configuration for the operational SQLite store."""

    path: Path
    busy_timeout_ms: int = 5_000


@dataclass(frozen=True, slots=True)
class PystocksNextConfig:
    """Minimal typed config surface for the rebuild harness."""

    sqlite: SqliteConfig
    artifacts_dir: Path
    collection_concurrency: int = 4

    @classmethod
    def from_env(
        cls,
        *,
        project_root: Path | None = None,
    ) -> PystocksNextConfig:
        root = project_root or Path.cwd()
        sqlite_path = Path(
            os.environ.get(
                "PYSTOCKS_NEXT_SQLITE_PATH",
                root / "data" / "pystocks_next.sqlite",
            )
        )
        artifacts_dir = Path(
            os.environ.get(
                "PYSTOCKS_NEXT_ARTIFACTS_DIR",
                root / "data" / "pystocks_next_artifacts",
            )
        )
        concurrency = int(os.environ.get("PYSTOCKS_NEXT_COLLECTION_CONCURRENCY", "4"))
        if concurrency <= 0:
            msg = "PYSTOCKS_NEXT_COLLECTION_CONCURRENCY must be positive"
            raise ValueError(msg)

        return cls(
            sqlite=SqliteConfig(path=sqlite_path),
            artifacts_dir=artifacts_dir,
            collection_concurrency=concurrency,
        )
