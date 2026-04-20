from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx

_ACCOUNTS_ACESWS_PATH_RE = re.compile(r"/portal\.proxy/v1/portal/acesws/([^/?]+)")
_ACCOUNTS_PORTFOLIO2_PATH_RE = re.compile(
    r"/portal\.proxy/v1/portal/portfolio2/([^/?]+)"
)
_ACCOUNT_ID_RE = re.compile(r"^(?:U|DU|DF|F)?\d+$")
_ONE_BAR_AUTH_ENDPOINT = "/AccountManagement/OneBarAuthentication?json=1"
_DEFAULT_AUTH_CHECK_ENDPOINTS = (
    "/tws.proxy/fundamentals/landing/756733?widgets=objective",
)

SessionHandler = Callable[..., Awaitable[Mapping[str, Any] | bool | None]]


class CollectionSession:
    """Authenticated source-session owner for the collection concern."""

    def __init__(
        self,
        *,
        state_path: Path | str | None = None,
        credentials_path: Path | str | None = None,
        portal_url: str = "https://www.interactivebrokers.ie/portal/",
        base_url: str = "https://www.interactivebrokers.ie",
        auth_check_endpoints: tuple[str, ...] = _DEFAULT_AUTH_CHECK_ENDPOINTS,
        login_handler: SessionHandler | None = None,
        reauth_handler: SessionHandler | None = None,
    ) -> None:
        default_state_path = (
            Path.cwd()
            / "data"
            / "pystocks_next_artifacts"
            / "collection_auth_state.json"
        )
        self.state_path = (
            Path(state_path) if state_path is not None else default_state_path
        )
        self.credentials_path = (
            Path(credentials_path)
            if credentials_path is not None
            else self.state_path.with_name("collection_login_credentials.json")
        )
        self.portal_url = portal_url
        self.base_url = base_url
        self.auth_check_endpoints = auth_check_endpoints
        self._login_handler = login_handler
        self._reauth_handler = reauth_handler
        self._headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": self.portal_url,
            "X-Requested-With": "XMLHttpRequest",
        }

    def load_state(self) -> dict[str, Any] | None:
        if not self.state_path.exists():
            return None
        with self.state_path.open() as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else None

    def save_state(self, state: Mapping[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.state_path.open("w") as handle:
            json.dump(dict(state), handle, indent=2)

    def _cookies_from_state(self, state: Mapping[str, Any] | None) -> dict[str, str]:
        cookies: dict[str, str] = {}
        for cookie in (state or {}).get("cookies", []):
            if not isinstance(cookie, Mapping):
                continue
            name = cookie.get("name")
            value = cookie.get("value")
            if isinstance(name, str) and value is not None:
                cookies[name] = str(value)
        return cookies

    def _choose_account_id(self, candidates: Sequence[object]) -> str | None:
        valid: list[str] = []
        for candidate in candidates:
            text = str(candidate or "").strip()
            if _ACCOUNT_ID_RE.match(text):
                valid.append(text)
        if not valid:
            return None
        unique = sorted(set(valid))
        unique.sort(
            key=lambda value: (0 if value.startswith("U") else 1, -len(value), value)
        )
        return unique[0]

    def _extract_primary_account_id_from_state(
        self,
        state: Mapping[str, Any] | None,
    ) -> str | None:
        if not isinstance(state, Mapping):
            return None

        explicit = self._choose_account_id([state.get("primary_account_id")])
        if explicit is not None:
            return explicit

        acesws_candidates: list[str] = []
        portfolio_candidates: list[str] = []
        for cookie in state.get("cookies", []):
            if not isinstance(cookie, Mapping):
                continue
            path = cookie.get("path")
            if not isinstance(path, str):
                continue
            acesws_match = _ACCOUNTS_ACESWS_PATH_RE.search(path)
            if acesws_match is not None:
                account_id = acesws_match.group(1)
                if _ACCOUNT_ID_RE.match(account_id):
                    acesws_candidates.append(account_id)
                continue
            portfolio_match = _ACCOUNTS_PORTFOLIO2_PATH_RE.search(path)
            if portfolio_match is not None:
                account_id = portfolio_match.group(1)
                if _ACCOUNT_ID_RE.match(account_id):
                    portfolio_candidates.append(account_id)
        return self._choose_account_id(acesws_candidates) or self._choose_account_id(
            portfolio_candidates
        )

    def _extract_primary_account_id_from_one_bar_payload(
        self,
        payload: Mapping[str, Any] | None,
    ) -> str | None:
        if not isinstance(payload, Mapping):
            return None

        account_id = self._choose_account_id([payload.get("mostRelevantAccount")])
        if account_id is not None:
            return account_id

        accounts = payload.get("portfolioAccounts")
        if not isinstance(accounts, list):
            return None
        for account in accounts:
            if not isinstance(account, Mapping):
                continue
            account_id = self._choose_account_id(
                [account.get("accountId"), account.get("accountVan")]
            )
            if account_id is not None:
                return account_id
        return None

    async def _validate_state_payload(
        self,
        state: Mapping[str, Any],
        *,
        timeout_s: float = 20.0,
    ) -> bool:
        cookies = self._cookies_from_state(state)
        if not cookies:
            return False
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._headers,
                cookies=cookies,
                follow_redirects=True,
                timeout=timeout_s,
            ) as client:
                for endpoint in self.auth_check_endpoints:
                    response = await client.get(endpoint)
                    if response.status_code != 200:
                        return False
        except httpx.RequestError:
            return False
        return True

    async def _fetch_primary_account_id_from_state(
        self,
        state: Mapping[str, Any],
        *,
        timeout_s: float = 20.0,
    ) -> str | None:
        cookies = self._cookies_from_state(state)
        if not cookies:
            return None
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._headers,
                cookies=cookies,
                follow_redirects=True,
                timeout=timeout_s,
            ) as client:
                response = await client.get(_ONE_BAR_AUTH_ENDPOINT)
        except httpx.RequestError:
            return None

        if response.status_code != 200:
            return None
        try:
            payload = response.json()
        except ValueError:
            return None
        if not isinstance(payload, Mapping):
            return None
        return self._extract_primary_account_id_from_one_bar_payload(payload)

    def get_primary_account_id(self) -> str | None:
        return self._extract_primary_account_id_from_state(self.load_state())

    async def validate_auth_state(self, *, timeout_s: float = 20.0) -> bool:
        state = self.load_state()
        if not isinstance(state, Mapping):
            return False
        is_valid = await self._validate_state_payload(state, timeout_s=timeout_s)
        if not is_valid:
            return False

        primary_account_id = self._extract_primary_account_id_from_state(state)
        if primary_account_id is None:
            primary_account_id = await self._fetch_primary_account_id_from_state(
                state,
                timeout_s=timeout_s,
            )
        if (
            primary_account_id is not None
            and state.get("primary_account_id") != primary_account_id
        ):
            saved_state = dict(state)
            saved_state["primary_account_id"] = primary_account_id
            self.save_state(saved_state)
        return True

    async def login(
        self,
        *,
        headless: bool = True,
        force_browser: bool = False,
    ) -> bool:
        if self._login_handler is None:
            raise NotImplementedError("interactive login is not implemented yet")
        result = await self._login_handler(
            headless=headless,
            force_browser=force_browser,
        )
        if isinstance(result, Mapping):
            self.save_state(result)
            return True
        return bool(result)

    async def reauthenticate(self, *, headless: bool = False) -> bool:
        handler = self._reauth_handler or self._login_handler
        if handler is None:
            raise NotImplementedError("reauthentication is not implemented yet")
        result = await handler(headless=headless)
        if isinstance(result, Mapping):
            self.save_state(result)
            return True
        return bool(result)

    def get_client(self, *, timeout_s: float = 20.0) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers,
            cookies=self._cookies_from_state(self.load_state()),
            follow_redirects=True,
            timeout=timeout_s,
        )
