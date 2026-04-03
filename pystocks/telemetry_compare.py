from .diagnostics.telemetry_compare import *  # noqa: F401,F403

if __name__ == "__main__":
    import fire

    from .diagnostics.telemetry_compare import compare

    fire.Fire(compare)
