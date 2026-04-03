from .diagnostics.official_price_series_compare import *  # noqa: F401,F403

if __name__ == "__main__":
    import fire

    from .diagnostics.official_price_series_compare import run

    fire.Fire(run)
