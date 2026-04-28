from __future__ import annotations

import asyncio
from collections.abc import Mapping

import httpx

from pystocks_next.collection.products import (
    fetch_product_page,
    refresh_product_universe,
)
from pystocks_next.tests.support import RecordingProgressSink
from pystocks_next.universe import list_instruments


class _FakeResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> object:
        return self._payload


class _FakeClient:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.post_calls = 0

    async def post(
        self,
        url: str,
        /,
        *,
        json: Mapping[str, object],
        headers: Mapping[str, str],
        timeout: float,
    ) -> _FakeResponse:
        del url, json, headers, timeout
        self.post_calls += 1
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        assert isinstance(response, _FakeResponse)
        return response

    async def aclose(self) -> None:
        return None


def test_fetch_product_page_retries_429_and_timeout() -> None:
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    client = _FakeClient(
        [
            _FakeResponse(429, {}),
            httpx.RequestError("timeout"),
            _FakeResponse(200, {"products": [{"conid": "100"}]}),
        ]
    )

    result = asyncio.run(
        fetch_product_page(client, page_number=1, retries=3, sleep=fake_sleep)
    )

    assert result == {"products": [{"conid": "100"}]}
    assert sleeps == [5.0, 4.0]


def test_refresh_product_universe_filters_malformed_and_dedupes(
    temp_store,
) -> None:
    client = _FakeClient(
        [
            _FakeResponse(
                200,
                {
                    "products": [
                        {"conid": "100", "symbol": "AAA", "currency": "USD"},
                        {"conid": "100", "symbol": "AAA2", "currency": "USD"},
                        {"symbol": "missing"},
                        "bad",
                    ]
                },
            ),
            _FakeResponse(200, {"products": []}),
        ]
    )

    result = asyncio.run(refresh_product_universe(temp_store, client=client))
    instruments = list_instruments(temp_store)

    assert result.status == "ok"
    assert result.fetched_products == 3
    assert result.deduped_products == 1
    assert result.products_upserted == 1
    assert result.page_count == 1
    assert [(item.conid, item.symbol) for item in instruments] == [("100", "AAA2")]


def test_refresh_product_universe_reports_progress(temp_store) -> None:
    client = _FakeClient(
        [
            _FakeResponse(200, {"products": [{"conid": "100"}, {"conid": "101"}]}),
            _FakeResponse(200, {"products": []}),
        ]
    )
    progress = RecordingProgressSink()

    asyncio.run(
        refresh_product_universe(
            temp_store,
            client=client,
            progress=progress,
        )
    )

    assert progress.events == [
        ("start", "Refreshing universe", None, "page"),
        ("advance", "Refreshing universe", 1, "2 products across 1 pages"),
        ("close", "Refreshing universe", None, "2 products across 1 pages"),
    ]
