from __future__ import annotations

from pathlib import Path
from typing import cast

from pystocks_next.cli import PyStocksNextCLI
from pystocks_next.collection import CollectionSession


def test_init_storage_bootstraps_rebuild_store(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PYSTOCKS_NEXT_SQLITE_PATH", str(tmp_path / "cli.sqlite"))
    monkeypatch.setenv("PYSTOCKS_NEXT_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    result = PyStocksNextCLI().init_storage()

    assert result["status"] == "ok"
    assert result["schema_version"] == 1
    assert Path(str(result["sqlite_path"])).exists()
    assert Path(str(result["artifacts_dir"])).exists()


def test_build_inputs_runs_against_empty_store(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PYSTOCKS_NEXT_SQLITE_PATH", str(tmp_path / "cli.sqlite"))
    monkeypatch.setenv("PYSTOCKS_NEXT_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    cli = PyStocksNextCLI()
    cli.init_storage()

    result = cli.build_inputs(show_progress=False)

    assert result["status"] == "ok"
    assert result["prices_rows"] == 0
    assert result["risk_free_daily_rows"] == 0
    assert result["macro_features_rows"] == 0


def test_run_pipeline_calls_same_stage_methods_in_order(monkeypatch) -> None:
    cli = PyStocksNextCLI()
    calls: list[str] = []

    def fake_init_storage() -> dict[str, object]:
        calls.append("storage")
        return {"status": "ok"}

    def fake_refresh_universe(**kwargs: object) -> dict[str, object]:
        calls.append("universe")
        return dict(kwargs) or {"status": "ok"}

    def fake_collect_fundamentals(**kwargs: object) -> dict[str, object]:
        calls.append("fundamentals")
        return dict(kwargs)

    def fake_refresh_supplementary(**kwargs: object) -> dict[str, object]:
        calls.append("supplementary")
        return dict(kwargs)

    def fake_build_inputs(**kwargs: object) -> dict[str, object]:
        calls.append("inputs")
        return dict(kwargs) or {"status": "ok"}

    monkeypatch.setattr(cli, "init_storage", fake_init_storage)
    monkeypatch.setattr(cli, "refresh_universe", fake_refresh_universe)
    monkeypatch.setattr(cli, "collect_fundamentals", fake_collect_fundamentals)
    monkeypatch.setattr(cli, "refresh_supplementary", fake_refresh_supplementary)
    monkeypatch.setattr(cli, "build_inputs", fake_build_inputs)

    result = cli.run_pipeline(
        limit=25,
        force=True,
        conids_file="docs/sample_conids.txt",
        economy_codes_file="docs/sample_codes.txt",
        telemetry_filename="sample.json",
        show_progress=False,
    )

    assert calls == [
        "storage",
        "universe",
        "fundamentals",
        "supplementary",
        "inputs",
    ]
    fundamentals_result = cast(dict[str, object], result["collect_fundamentals"])
    supplementary_result = cast(dict[str, object], result["refresh_supplementary"])
    assert result["storage"] == {"status": "ok"}
    assert result["refresh_universe"] == {"show_progress": False}
    assert fundamentals_result["show_progress"] is False
    assert supplementary_result["show_progress"] is False
    assert result["build_inputs"] == {"show_progress": False}


def test_collection_session_uses_cli_project_root(tmp_path: Path) -> None:
    cli = PyStocksNextCLI(project_root=str(tmp_path))

    session = cli._collection_session()

    assert isinstance(session, CollectionSession)
    assert session.state_path == tmp_path / "data" / "auth_state.json"
    assert session.credentials_path == tmp_path / "data" / "login_credentials.json"
