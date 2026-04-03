from .ingest.product_scraper import *  # noqa: F401,F403

if __name__ == "__main__":
    import asyncio

    from .ingest.product_scraper import scrape_ibkr_products

    asyncio.run(scrape_ibkr_products())
