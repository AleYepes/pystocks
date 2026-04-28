import pystocks.ingest as ingest_module
from pystocks.cli import PyStocksCLI


def test_run_pipeline_executes_explicit_preprocess_stages(monkeypatch):
    cli = PyStocksCLI()

    monkeypatch.setattr(cli, "scrape_products", lambda: {"status": "ok"})
    monkeypatch.setattr(
        cli,
        "scrape_fundamentals",
        lambda **kwargs: {"status": "ok", "processed_conids": kwargs["limit"]},
    )
    monkeypatch.setattr(
        cli,
        "fetch_supplementary_data",
        lambda show_progress=True: {"status": "ok", "economy_count": 2},
    )
    monkeypatch.setattr(
        cli,
        "preprocess_supplementary_data",
        lambda show_progress=True: {"status": "ok", "world_bank_rows": 2},
    )
    monkeypatch.setattr(
        cli,
        "preprocess_prices",
        lambda show_progress=True: {"status": "ok", "rows": 25},
    )
    monkeypatch.setattr(
        cli,
        "preprocess_snapshots",
        lambda show_progress=True: {"status": "ok", "rows": 10},
    )
    monkeypatch.setattr(
        cli, "run_analysis", lambda show_progress=True: {"status": "ok"}
    )

    result = cli.run_pipeline(limit=10, show_progress=False)

    assert result == {
        "products": {"status": "ok"},
        "fundamentals": {"status": "ok", "processed_conids": 10},
        "supplementary_fetch": {"status": "ok", "economy_count": 2},
        "supplementary_preprocess": {"status": "ok", "world_bank_rows": 2},
        "price_preprocess": {"status": "ok", "rows": 25},
        "snapshot_preprocess": {"status": "ok", "rows": 10},
        "analysis": {"status": "ok"},
    }


def test_run_preprocess_pipeline_executes_required_preprocess_stages(monkeypatch):
    cli = PyStocksCLI()

    monkeypatch.setattr(
        cli,
        "fetch_supplementary_data",
        lambda show_progress=True: {"status": "ok", "economy_count": 2},
    )
    monkeypatch.setattr(
        cli,
        "preprocess_supplementary_data",
        lambda show_progress=True: {"status": "ok", "world_bank_rows": 2},
    )
    monkeypatch.setattr(
        cli,
        "preprocess_prices",
        lambda show_progress=True: {"status": "ok", "rows": 25},
    )
    monkeypatch.setattr(
        cli,
        "preprocess_snapshots",
        lambda show_progress=True: {"status": "ok", "rows": 10},
    )

    result = cli.run_preprocess_pipeline(show_progress=False)

    assert result == {
        "supplementary_fetch": {"status": "ok", "economy_count": 2},
        "supplementary_preprocess": {"status": "ok", "world_bank_rows": 2},
        "price_preprocess": {"status": "ok", "rows": 25},
        "snapshot_preprocess": {"status": "ok", "rows": 10},
    }


def test_run_preprocess_pipeline_can_skip_supplementary_refresh(monkeypatch):
    cli = PyStocksCLI()

    monkeypatch.setattr(
        cli,
        "fetch_supplementary_data",
        lambda show_progress=True: (_ for _ in ()).throw(
            AssertionError("fetch_supplementary_data should not be called")
        ),
    )
    monkeypatch.setattr(
        cli,
        "preprocess_supplementary_data",
        lambda show_progress=True: (_ for _ in ()).throw(
            AssertionError("preprocess_supplementary_data should not be called")
        ),
    )
    monkeypatch.setattr(
        cli,
        "preprocess_prices",
        lambda show_progress=True: {"status": "ok"},
    )
    monkeypatch.setattr(
        cli, "preprocess_snapshots", lambda show_progress=True: {"status": "ok"}
    )

    result = cli.run_preprocess_pipeline(
        refresh_supplementary=False,
        show_progress=False,
    )

    assert result == {
        "price_preprocess": {"status": "ok"},
        "snapshot_preprocess": {"status": "ok"},
    }


def test_run_pipeline_can_skip_supplementary_refresh(monkeypatch):
    cli = PyStocksCLI()

    monkeypatch.setattr(cli, "scrape_products", lambda: {"status": "ok"})
    monkeypatch.setattr(cli, "scrape_fundamentals", lambda **kwargs: {"status": "ok"})
    monkeypatch.setattr(
        cli,
        "fetch_supplementary_data",
        lambda show_progress=True: (_ for _ in ()).throw(
            AssertionError("fetch_supplementary_data should not be called")
        ),
    )
    monkeypatch.setattr(
        cli,
        "preprocess_supplementary_data",
        lambda show_progress=True: (_ for _ in ()).throw(
            AssertionError("preprocess_supplementary_data should not be called")
        ),
    )
    monkeypatch.setattr(
        cli,
        "preprocess_prices",
        lambda show_progress=True: {"status": "ok"},
    )
    monkeypatch.setattr(
        cli, "preprocess_snapshots", lambda show_progress=True: {"status": "ok"}
    )
    monkeypatch.setattr(
        cli, "run_analysis", lambda show_progress=True: {"status": "ok"}
    )

    result = cli.run_pipeline(refresh_supplementary=False, show_progress=False)

    assert result == {
        "products": {"status": "ok"},
        "fundamentals": {"status": "ok"},
        "price_preprocess": {"status": "ok"},
        "snapshot_preprocess": {"status": "ok"},
        "analysis": {"status": "ok"},
    }


def test_describe_analysis_inputs_exposes_required_stage_contract():
    cli = PyStocksCLI()

    result = cli.describe_analysis_inputs()

    assert result["stage_order"] == [
        "scrape_products",
        "scrape_fundamentals",
        "fetch_supplementary_data",
        "preprocess_supplementary_data",
        "preprocess_prices",
        "preprocess_snapshots",
        "run_analysis",
    ]
    assert (
        result["required_analysis_artifacts"]["price_history"]["producer"]
        == "preprocess_prices"
    )
    assert (
        result["required_analysis_artifacts"]["supplementary_features"]["alternative"]
        == "refresh_supplementary_data"
    )
    assert (
        result["optional_preprocess_artifacts"]["dividend_events"]["producer"]
        == "preprocess_dividends"
    )


def test_ingest_exports_fetch_supplementary_data_without_refresh_alias():
    assert hasattr(ingest_module, "fetch_supplementary_data")
    assert not hasattr(ingest_module, "refresh_supplementary_data")
