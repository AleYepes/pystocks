from __future__ import annotations

import asyncio
import math
import sqlite3
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from ..progress import ProgressSink
from ..universe import UniverseInstrument, upsert_instruments

PRODUCT_PAGE_SIZE = 500
PRODUCT_SEARCH_ENDPOINT = (
    "https://www.interactivebrokers.ie/webrest/search/products-by-filters"
)
SleepFn = Callable[[float], Awaitable[None]]


class ProductHttpResponse(Protocol):
    status_code: int

    def json(self) -> object: ...


class ProductHttpClient(Protocol):
    async def post(
        self,
        url: str,
        /,
        *,
        json: Mapping[str, object],
        headers: Mapping[str, str],
        timeout: float,
    ) -> ProductHttpResponse: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ProductCollectionResult:
    status: str
    fetched_products: int
    deduped_products: int
    products_upserted: int
    page_count: int


def _build_product_search_payload(
    *,
    page_number: int,
    page_size: int,
) -> dict[str, object]:
    return {
        "domain": "ie",
        "newProduct": "all",
        "pageNumber": page_number,
        "pageSize": page_size,
        "productCountry": [],
        "productSymbol": "",
        "productType": ["ETF"],
        "sortDirection": "asc",
        "sortField": "symbol",
    }


def _normalize_products(
    products: Sequence[Mapping[str, Any]],
) -> list[UniverseInstrument]:
    deduped: dict[str, UniverseInstrument] = {}
    for product in products:
        conid = _clean_product_text(product.get("conid"))
        if not conid:
            continue
        deduped[conid] = UniverseInstrument(
            conid=conid,
            symbol=_first_product_text(product, "symbol"),
            local_symbol=_first_product_text(product, "localSymbol", "local_symbol"),
            name=_first_product_text(product, "name", "description"),
            exchange=_first_product_text(product, "exchange", "exchangeId"),
            isin=_first_product_text(product, "isin"),
            cusip=_first_product_text(product, "cusip"),
            currency=_first_product_text(product, "currency"),
            country=_first_product_text(product, "country"),
            product_type=_first_product_text(
                product, "type", "productType", "product_type"
            )
            or "ETF",
            under_conid=_first_product_text(product, "underConid", "under_conid"),
            is_prime_exch_id=_first_product_text(
                product, "isPrimeExchId", "is_prime_exch_id"
            ),
            is_new_pdt=_first_product_text(product, "isNewPdt", "is_new_pdt"),
            assoc_entity_id=_first_product_text(
                product, "assocEntityId", "assoc_entity_id"
            ),
            fc_conid=_first_product_text(product, "fcConid", "fc_conid"),
        )
    return list(deduped.values())


def _clean_product_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _first_product_text(product: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        text = _clean_product_text(product.get(key))
        if text is not None:
            return text
    return None


async def fetch_product_page(
    client: ProductHttpClient,
    *,
    page_number: int,
    retries: int = 5,
    page_size: int = PRODUCT_PAGE_SIZE,
    sleep: SleepFn = asyncio.sleep,
) -> dict[str, Any] | None:
    headers = {
        "Content-Type": "application/json",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }
    payload = _build_product_search_payload(
        page_number=page_number,
        page_size=page_size,
    )
    for attempt in range(retries):
        try:
            response = await client.post(
                PRODUCT_SEARCH_ENDPOINT,
                json=payload,
                headers=headers,
                timeout=20.0,
            )
        except (TimeoutError, httpx.RequestError):
            await sleep(2.0 * float(attempt + 1))
            continue

        if response.status_code == 200:
            data = response.json()
            return data if isinstance(data, dict) else None
        if response.status_code == 429:
            await sleep(5.0 * float(attempt + 1))
            continue
        await sleep(1.0 * float(attempt + 1))
    return None


async def refresh_product_universe(
    conn: sqlite3.Connection,
    *,
    client: ProductHttpClient | None = None,
    retries: int = 5,
    page_size: int = PRODUCT_PAGE_SIZE,
    sleep: SleepFn = asyncio.sleep,
    progress: ProgressSink | None = None,
) -> ProductCollectionResult:
    owns_client = client is None
    client_obj: httpx.AsyncClient | ProductHttpClient = (
        httpx.AsyncClient() if client is None else client
    )
    tracker = (
        progress.stage("Refreshing universe", unit="page")
        if progress is not None
        else None
    )

    all_products: list[Mapping[str, Any]] = []
    page_count = 0
    try:
        page_number = 0
        while True:
            page = await fetch_product_page(
                client_obj,
                page_number=page_number,
                retries=retries,
                page_size=page_size,
                sleep=sleep,
            )
            if not page:
                break
            raw_products = page.get("products")
            if not isinstance(raw_products, list) or not raw_products:
                break
            page_count += 1
            valid_products = [
                product for product in raw_products if isinstance(product, Mapping)
            ]
            all_products.extend(valid_products)
            if tracker is not None:
                tracker.advance(
                    detail=f"{len(all_products)} products across {page_count} pages"
                )
            if len(raw_products) < page_size:
                break
            page_number += 1
    finally:
        if owns_client:
            await client_obj.aclose()
        if tracker is not None:
            tracker.close(
                detail=f"{len(all_products)} products across {page_count} pages"
            )

    instruments = _normalize_products(all_products)
    products_upserted = upsert_instruments(conn, instruments)
    return ProductCollectionResult(
        status="ok" if instruments else "no_products",
        fetched_products=len(all_products),
        deduped_products=len(instruments),
        products_upserted=products_upserted,
        page_count=page_count,
    )
