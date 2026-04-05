from pystocks.cli import PyStocksCLI


def test_run_pipeline_skips_standalone_price_preprocess(monkeypatch):
    cli = PyStocksCLI()

    monkeypatch.setattr(cli, "scrape_products", lambda: {"status": "ok"})
    monkeypatch.setattr(
        cli,
        "scrape_fundamentals",
        lambda **kwargs: {"status": "ok", "processed_conids": kwargs["limit"]},
    )
    monkeypatch.setattr(
        cli,
        "preprocess_prices",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("preprocess_prices should not be called")
        ),
    )
    monkeypatch.setattr(
        cli, "run_analysis", lambda show_progress=True: {"status": "ok"}
    )

    result = cli.run_pipeline(limit=10, show_progress=False)

    assert result == {
        "products": {"status": "ok"},
        "fundamentals": {"status": "ok", "processed_conids": 10},
        "analysis": {"status": "ok"},
    }
