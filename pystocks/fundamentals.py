from .ingest.fundamentals import *  # noqa: F401,F403

if __name__ == "__main__":
    import fire

    from .ingest.fundamentals import main

    fire.Fire(main)
