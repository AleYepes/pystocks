from __future__ import annotations

from pathlib import Path

import pytest

from pystocks_next.config import PystocksNextConfig


def test_config_from_env_uses_project_root_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYSTOCKS_NEXT_SQLITE_PATH", raising=False)
    monkeypatch.delenv("PYSTOCKS_NEXT_ARTIFACTS_DIR", raising=False)
    monkeypatch.delenv("PYSTOCKS_NEXT_COLLECTION_CONCURRENCY", raising=False)

    config = PystocksNextConfig.from_env(project_root=Path("/tmp/project"))

    assert config.sqlite.path == Path("/tmp/project/data/pystocks_next.sqlite")
    assert config.artifacts_dir == Path("/tmp/project/data/pystocks_next_artifacts")
    assert config.collection_concurrency == 4


def test_config_from_env_rejects_non_positive_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYSTOCKS_NEXT_COLLECTION_CONCURRENCY", "0")

    with pytest.raises(ValueError, match="must be positive"):
        PystocksNextConfig.from_env(project_root=Path("/tmp/project"))
