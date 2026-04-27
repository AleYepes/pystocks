from __future__ import annotations

import asyncio
import getpass
import json
import logging
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
from playwright.async_api import async_playwright

_ACCOUNTS_ACESWS_PATH_RE = re.compile(r"/portal\.proxy/v1/portal/acesws/([^/?]+)")
_ACCOUNTS_PORTFOLIO2_PATH_RE = re.compile(
    r"/portal\.proxy/v1/portal/portfolio2/([^/?]+)"
)
_ACCOUNT_ID_RE = re.compile(r"^(?:U|DU|DF|F)?\d+$")
_ONE_BAR_AUTH_ENDPOINT = "/AccountManagement/OneBarAuthentication?json=1"
_DEFAULT_AUTH_CHECK_ENDPOINTS = (
    "/tws.proxy/fundamentals/landing/756733?widgets=objective",
)
_LOGIN_ERROR_TEXT_RE = re.compile(
    r"(invalid|incorrect|wrong|failed|error)",
    re.IGNORECASE,
)

SessionHandler = Callable[..., Awaitable[Mapping[str, Any] | bool | None]]
logger = logging.getLogger(__name__)


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
        default_state_path = Path.cwd() / "data" / "auth_state.json"
        self.state_path = (
            Path(state_path) if state_path is not None else default_state_path
        )
        self.credentials_path = (
            Path(credentials_path)
            if credentials_path is not None
            else self.state_path.with_name("login_credentials.json")
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

    def _load_login_config(self) -> dict[str, Any] | None:
        if not self.credentials_path.exists():
            return None
        try:
            with self.credentials_path.open() as handle:
                payload = json.load(handle)
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _save_login_config(self, username: str, password: str) -> None:
        self.credentials_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "username": str(username),
            "password": str(password),
        }
        with self.credentials_path.open("w") as handle:
            json.dump(payload, handle, indent=2)

    def _delete_login_config(self) -> None:
        try:
            self.credentials_path.unlink()
        except FileNotFoundError:
            return

    def _get_login_credentials(
        self,
        *,
        prompt_if_missing: bool = True,
        force_prompt: bool = False,
    ) -> tuple[str | None, str | None]:
        if not force_prompt:
            config = self._load_login_config()
            username = str((config or {}).get("username") or "").strip()
            password = str((config or {}).get("password") or "").strip()
            if username and password:
                return username, password

        if not prompt_if_missing:
            return None, None

        try:
            entered_username = input("IBKR username: ").strip()
            entered_password = getpass.getpass("IBKR password: ").strip()
        except (EOFError, KeyboardInterrupt):
            logger.warning("Credential input interrupted.")
            return None, None

        if not entered_username or not entered_password:
            logger.warning("Username/password missing.")
            return None, None

        self._save_login_config(entered_username, entered_password)
        return entered_username, entered_password

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
                    try:
                        response = await client.get(endpoint)
                    except httpx.RequestError:
                        continue
                    if response.status_code != 200:
                        continue
                    try:
                        payload = response.json()
                    except ValueError:
                        continue
                    if payload is None:
                        continue
                    if isinstance(payload, (dict, list)) and len(payload) == 0:
                        continue
                    return True
        except httpx.RequestError:
            return False
        return False

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

    async def _hydrate_primary_account_id(
        self,
        state: Mapping[str, Any],
        *,
        timeout_s: float = 20.0,
    ) -> Mapping[str, Any]:
        primary_account_id = await self._fetch_primary_account_id_from_state(
            state,
            timeout_s=timeout_s,
        )
        if primary_account_id is None:
            primary_account_id = self._extract_primary_account_id_from_state(state)
        if (
            primary_account_id is None
            or state.get("primary_account_id") == primary_account_id
        ):
            return state

        saved_state = dict(state)
        saved_state["primary_account_id"] = primary_account_id
        return saved_state

    async def validate_auth_state(self, *, timeout_s: float = 20.0) -> bool:
        state = self.load_state()
        if not isinstance(state, Mapping):
            return False
        is_valid = await self._validate_state_payload(state, timeout_s=timeout_s)
        if not is_valid:
            return False

        hydrated_state = await self._hydrate_primary_account_id(
            state,
            timeout_s=timeout_s,
        )
        if hydrated_state is not state:
            self.save_state(hydrated_state)
        return True

    async def _dismiss_cookie_modal_if_present(self, page: Any) -> None:
        for selector in ("#gdpr-reject-all", "#btn_accept_cookies"):
            try:
                await page.wait_for_selector(selector, state="visible", timeout=3_000)
                await page.click(selector)
                await page.wait_for_load_state("domcontentloaded", timeout=15_000)
                await page.wait_for_timeout(500)
                return
            except Exception:
                continue

    async def _login_error_text(self, page: Any) -> str | None:
        selectors = [
            ".xyzblock-error .xyz-errormessage",
            ".alert-danger",
            ".invalid-feedback",
        ]
        for selector in selectors:
            try:
                matches = await page.query_selector_all(selector)
            except Exception:
                continue
            for match in matches:
                try:
                    if not await match.is_visible():
                        continue
                    text = (await match.inner_text() or "").strip()
                    if text:
                        return text
                except Exception:
                    continue
        return None

    async def _submit_login_credentials(
        self,
        page: Any,
        *,
        username: str,
        password: str,
    ) -> bool:
        try:
            await self._dismiss_cookie_modal_if_present(page)
            await page.wait_for_selector(
                "input[name='username']",
                state="visible",
                timeout=30_000,
            )
            await page.fill("input[name='username']", username)
            await page.fill("input[name='password']", password)
            await page.click("form.xyzform-username button[type='submit']")
            return True
        except Exception as exc:
            logger.warning("Automatic credential submit failed: %s", exc)
            return False

    async def _run_login_attempt(
        self,
        playwright: Any,
        *,
        username: str,
        password: str,
        headless: bool = False,
        timeout_ms: int = 180_000,
    ) -> tuple[bool, str | None]:
        browser = await playwright.chromium.launch(headless=headless)
        try:
            context = await browser.new_context()
            page = await context.new_page()
            await page.goto(self.portal_url, wait_until="domcontentloaded")

            submitted = await self._submit_login_credentials(
                page,
                username=username,
                password=password,
            )
            if not submitted:
                return False, None

            loop = asyncio.get_running_loop()
            deadline = loop.time() + (timeout_ms / 1000.0)

            while loop.time() < deadline:
                error_text = await self._login_error_text(page)
                if error_text and _LOGIN_ERROR_TEXT_RE.search(error_text):
                    return False, error_text

                state = await context.storage_state()
                if not isinstance(state, Mapping):
                    await asyncio.sleep(2)
                    continue

                is_valid = await self._validate_state_payload(state, timeout_s=10.0)
                if is_valid:
                    hydrated_state = await self._hydrate_primary_account_id(
                        state,
                        timeout_s=10.0,
                    )
                    self.save_state(hydrated_state)
                    logger.info(
                        "Login successful. Session state saved to %s",
                        self.state_path,
                    )
                    return True, None
                await asyncio.sleep(2)

            logger.error("Login timed out before authenticated API state was detected.")
            return False, None
        finally:
            await browser.close()

    async def login(
        self,
        *,
        headless: bool = False,
        force_browser: bool = False,
        timeout_ms: int = 180_000,
        max_credential_attempts: int = 3,
    ) -> bool:
        if self._login_handler is not None:
            result = await self._login_handler(
                headless=headless,
                force_browser=force_browser,
            )
            if isinstance(result, Mapping):
                self.save_state(result)
                return True
            return bool(result)

        if self.state_path.exists() and not force_browser:
            is_valid = await self.validate_auth_state()
            if is_valid:
                logger.info(
                    "Existing session state is still valid: %s", self.state_path
                )
                return True

        async with async_playwright() as playwright:
            force_prompt = False
            attempts_left = max(1, int(max_credential_attempts))
            while attempts_left > 0:
                username, password = self._get_login_credentials(
                    prompt_if_missing=True,
                    force_prompt=force_prompt,
                )
                if not username or not password:
                    logger.error("Missing IBKR credentials; login aborted.")
                    return False

                logger.info(
                    "Credential login submitted. Waiting for authenticated session state."
                )
                try:
                    success, rejection_reason = await self._run_login_attempt(
                        playwright,
                        username=username,
                        password=password,
                        headless=headless,
                        timeout_ms=timeout_ms,
                    )
                except Exception as exc:
                    logger.error("Login failed: %s", exc)
                    return False

                if success:
                    return True
                if not rejection_reason:
                    return False

                self._delete_login_config()
                attempts_left -= 1
                warning_message = f"IBKR login rejected credentials: {rejection_reason}"
                if attempts_left <= 0:
                    logger.warning(warning_message)
                    break

                force_prompt = True
                logger.warning("%s. Enter credentials again.", warning_message)

        return False

    async def reauthenticate(self, *, headless: bool = False) -> bool:
        handler = self._reauth_handler or self._login_handler
        if handler is not None:
            result = await handler(headless=headless)
            if isinstance(result, Mapping):
                self.save_state(result)
                return True
            return bool(result)

        logger.info(
            "Attempting reauthentication. Complete login in the browser window."
        )
        return await self.login(
            headless=headless,
            force_browser=True,
        )

    def get_client(self, *, timeout_s: float = 20.0) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers,
            cookies=self._cookies_from_state(self.load_state()),
            follow_redirects=True,
            timeout=timeout_s,
        )
