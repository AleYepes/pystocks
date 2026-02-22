from .fundamentals_store import FundamentalsStore


def main(delete_legacy=True, limit=None, refresh_duckdb=True):
    store = FundamentalsStore()
    result = store.migrate_legacy_json(
        delete_legacy=delete_legacy,
        limit=limit,
        refresh_duckdb=refresh_duckdb,
    )
    for k, v in result.items():
        print(f"{k}: {v}")
    return result


if __name__ == "__main__":
    import fire

    fire.Fire(main)
