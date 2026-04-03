import asyncio
import logging

import httpx
import pandas as pd
from tqdm import tqdm

from ..config import SQLITE_DB_PATH
from ..storage.ops_state import upsert_instruments_from_products

logger = logging.getLogger(__name__)
PRODUCT_PAGE_SIZE = 500


async def fetch_api_direct(client, page_number, retries=5):
    url = "https://www.interactivebrokers.ie/webrest/search/products-by-filters"
    payload = {
        "domain": "ie",
        "newProduct": "all",
        "pageNumber": page_number,
        "pageSize": PRODUCT_PAGE_SIZE,
        "productCountry": [],
        "productSymbol": "",
        "productType": ["ETF"],
        "sortDirection": "asc",
        "sortField": "symbol",
    }
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    }

    for attempt in range(retries):
        try:
            response = await client.post(url, json=payload, headers=headers, timeout=20)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                await asyncio.sleep(5 * (attempt + 1))
            else:
                await asyncio.sleep(1 * (attempt + 1))
        except (TimeoutError, httpx.RequestError):
            await asyncio.sleep(2 * (attempt + 1))

    return None


async def scrape_ibkr_products():
    all_products = []

    async with httpx.AsyncClient() as client:
        first_page = await fetch_api_direct(client, 1)

        if first_page and "products" in first_page and first_page["products"]:
            logger.info("Direct API access successful.")
            all_products.extend(first_page["products"])
            page_number = 2

            pbar = tqdm(desc="Fetching products", unit=" page")
            pbar.update(1)

            while True:
                data = await fetch_api_direct(client, page_number)
                if not data or "products" not in data or not data["products"]:
                    break

                batch = data["products"]
                all_products.extend(batch)
                pbar.update(1)

                if len(batch) < PRODUCT_PAGE_SIZE:
                    break

                page_number += 1
                await asyncio.sleep(0.05)
            pbar.close()
        else:
            logger.error("Direct API access failed")
            return

    df = pd.DataFrame(all_products).drop_duplicates(subset=["conid"], keep="last")
    n_upserted = upsert_instruments_from_products(df)
    logger.info(
        f"Upserted {n_upserted} products into SQLite products table at {SQLITE_DB_PATH}"
    )
    return {
        "status": "ok",
        "products_upserted": n_upserted,
        "sqlite_path": str(SQLITE_DB_PATH),
    }


if __name__ == "__main__":
    asyncio.run(scrape_ibkr_products())
