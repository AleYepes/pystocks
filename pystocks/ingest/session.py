import asyncio
import getpass
import json
import logging
import re
from pathlib import Path

import httpx
from playwright.async_api import async_playwright

from ..config import SESSION_STATE_PATH

logger = logging.getLogger(__name__)

_ACCOUNTS_ACESWS_PATH_RE = re.compile(r"/portal\.proxy/v1/portal/acesws/([^/?]+)")
_ACCOUNTS_PORTFOLIO2_PATH_RE = re.compile(
    r"/portal\.proxy/v1/portal/portfolio2/([^/?]+)"
)
_ACCOUNT_ID_RE = re.compile(r"^(?:U|DU|DF|F)?\d+$")
_LOGIN_ERROR_TEXT_RE = re.compile(
    r"(invalid|incorrect|wrong|failed|error)", re.IGNORECASE
)
_ONE_BAR_AUTH_ENDPOINT = "/AccountManagement/OneBarAuthentication?json=1"


class IBKRSession:
    """Manages an authenticated session for the IBKR Client Portal."""

    def __init__(self, state_path=SESSION_STATE_PATH, credentials_path=None):
        self.state_path = Path(state_path)
        self.credentials_path = (
            Path(credentials_path)
            if credentials_path
            else self.state_path.with_name("login_credentials.json")
        )
        self.portal_url = "https://www.interactivebrokers.ie/portal/"
        self.base_url = "https://www.interactivebrokers.ie"
        self._headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.interactivebrokers.ie/portal/",
            "X-Requested-With": "XMLHttpRequest",
        }
        self._auth_check_endpoints = [
            # This fundamentals endpoint is lightweight and currently returns 200
            # for a valid authenticated session, unlike older portfolio/iserver probes.
            "/tws.proxy/fundamentals/landing/756733?widgets=objective",
        ]

    def _load_state(self):
        if not self.state_path.exists():
            return None
        with open(self.state_path) as f:
            return json.load(f)

    def _save_state(self, state):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)

    def _cookies_from_state(self, state):
        cookies = {}
        for cookie in (state or {}).get("cookies", []):
            name = cookie.get("name")
            value = cookie.get("value")
            if name and value is not None:
                cookies[name] = value
        return cookies

    def _load_login_config(self):
        if not self.credentials_path.exists():
            return None
        try:
            with open(self.credentials_path) as f:
                payload = json.load(f)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _save_login_config(self, username, password):
        self.credentials_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "username": str(username),
            "password": str(password),
        }
        with open(self.credentials_path, "w") as f:
            json.dump(payload, f, indent=2)

    def _delete_login_config(self):
        try:
            self.credentials_path.unlink()
        except FileNotFoundError:
            return

    def _get_login_credentials(self, prompt_if_missing=True, force_prompt=False):
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

    async def _dismiss_cookie_modal_if_present(self, page):
        for selector in ("#gdpr-reject-all", "#btn_accept_cookies"):
            try:
                await page.wait_for_selector(selector, state="visible", timeout=3000)
                await page.click(selector)
                await page.wait_for_load_state("domcontentloaded", timeout=15000)
                await page.wait_for_timeout(500)
                return
            except Exception:
                continue

    async def _login_error_text(self, page):
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

    async def _submit_login_credentials(self, page, username, password):
        try:
            await self._dismiss_cookie_modal_if_present(page)
            await page.wait_for_selector(
                "input[name='username']", state="visible", timeout=30000
            )
            await page.fill("input[name='username']", username)
            await page.fill("input[name='password']", password)
            await page.click("form.xyzform-username button[type='submit']")
            return True
        except Exception as e:
            logger.warning(f"Automatic credential submit failed: {e}")
            return False

    def _choose_account_id(self, candidates):
        valid = []
        for candidate in candidates:
            text = str(candidate or "").strip()
            if _ACCOUNT_ID_RE.match(text):
                valid.append(text)
        if not valid:
            return None
        unique = sorted(set(valid))
        unique.sort(key=lambda x: (0 if x.startswith("U") else 1, -len(x), x))
        return unique[0]

    def _extract_primary_account_id_from_state(self, state):
        if not isinstance(state, dict):
            return None

        explicit = self._choose_account_id([state.get("primary_account_id")])
        if explicit:
            return explicit

        acesws_candidates = []
        portfolio2_candidates = []
        for cookie in state.get("cookies", []):
            path = cookie.get("path")
            if not isinstance(path, str):
                continue
            m_acesws = _ACCOUNTS_ACESWS_PATH_RE.search(path)
            if m_acesws:
                account_id = m_acesws.group(1)
                if _ACCOUNT_ID_RE.match(account_id):
                    acesws_candidates.append(account_id)
                continue

            m_portfolio2 = _ACCOUNTS_PORTFOLIO2_PATH_RE.search(path)
            if m_portfolio2:
                account_id = m_portfolio2.group(1)
                if _ACCOUNT_ID_RE.match(account_id):
                    portfolio2_candidates.append(account_id)

        return self._choose_account_id(acesws_candidates) or self._choose_account_id(
            portfolio2_candidates
        )

    def _extract_primary_account_id_from_one_bar_payload(self, payload):
        if not isinstance(payload, dict):
            return None

        account_id = self._choose_account_id([payload.get("mostRelevantAccount")])
        if account_id:
            return account_id

        accounts = payload.get("portfolioAccounts")
        if not isinstance(accounts, list):
            return None

        for account in accounts:
            if not isinstance(account, dict):
                continue
            account_id = self._choose_account_id(
                [account.get("accountId"), account.get("accountVan")]
            )
            if account_id:
                return account_id

        return None

    async def _fetch_primary_account_id_from_state(self, state, timeout_s=20.0):
        if not isinstance(state, dict):
            return None
        cookies = self._cookies_from_state(state)
        if not cookies:
            return None

        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                cookies=cookies,
                headers=self._headers,
                timeout=timeout_s,
            ) as client:
                response = await client.get(_ONE_BAR_AUTH_ENDPOINT)
        except Exception:
            return None

        if response.status_code != 200:
            return None

        try:
            payload = response.json()
        except Exception:
            return None

        return self._extract_primary_account_id_from_one_bar_payload(payload)

    async def _hydrate_primary_account_id(self, state, timeout_s=20.0):
        if not isinstance(state, dict):
            return state

        account_id = await self._fetch_primary_account_id_from_state(
            state, timeout_s=timeout_s
        )
        if account_id is None:
            account_id = self._extract_primary_account_id_from_state(state)
        if account_id is None or state.get("primary_account_id") == account_id:
            return state

        updated = dict(state)
        updated["primary_account_id"] = account_id
        return updated

    def get_primary_account_id(self):
        return self._extract_primary_account_id_from_state(self._load_state())

    async def _validate_state_payload(self, state, timeout_s=20.0):
        """
        Validates a storage-state payload by making authenticated API calls.
        Returns True only when at least one endpoint responds with a valid 200 payload.
        """
        if not state:
            return False
        cookies = self._cookies_from_state(state)
        if not cookies:
            return False

        async with httpx.AsyncClient(
            base_url=self.base_url,
            cookies=cookies,
            headers=self._headers,
            timeout=timeout_s,
        ) as client:
            for endpoint in self._auth_check_endpoints:
                try:
                    response = await client.get(endpoint)
                except Exception:
                    continue

                if response.status_code != 200:
                    continue

                try:
                    payload = response.json()
                except Exception:
                    continue

                if payload is None:
                    continue
                if isinstance(payload, (dict, list)) and len(payload) == 0:
                    continue
                return True

        return False

    async def validate_auth_state(self, timeout_s=20.0):
        """
        Validates stored auth state by making authenticated API calls.
        """
        state = self._load_state()
        is_valid = await self._validate_state_payload(state, timeout_s=timeout_s)
        if not is_valid or not isinstance(state, dict):
            return is_valid

        hydrated_state = await self._hydrate_primary_account_id(
            state, timeout_s=timeout_s
        )
        if hydrated_state is not state:
            self._save_state(hydrated_state)

        return True

    async def _run_login_attempt(
        self, playwright, username, password, headless=False, timeout_ms=180000
    ):
        browser = await playwright.chromium.launch(headless=headless)
        try:
            context = await browser.new_context()
            page = await context.new_page()
            await page.goto(self.portal_url, wait_until="domcontentloaded")

            submitted = await self._submit_login_credentials(page, username, password)
            if not submitted:
                return False, None

            loop = asyncio.get_running_loop()
            deadline = loop.time() + (timeout_ms / 1000.0)

            while loop.time() < deadline:
                error_text = await self._login_error_text(page)
                if error_text and _LOGIN_ERROR_TEXT_RE.search(error_text):
                    return False, error_text

                state = await context.storage_state()
                is_valid = await self._validate_state_payload(state, timeout_s=10.0)
                if is_valid:
                    hydrated_state = await self._hydrate_primary_account_id(
                        state, timeout_s=10.0
                    )
                    self._save_state(hydrated_state)
                    logger.info(
                        f"Login successful. Session state saved to {self.state_path}"
                    )
                    return True, None
                await asyncio.sleep(2)

            logger.error("Login timed out before authenticated API state was detected.")
            return False, None
        finally:
            await browser.close()

    async def login(
        self,
        headless=False,
        timeout_ms=180000,
        force_browser=False,
        max_credential_attempts=3,
    ):
        """
        Launches a browser and performs credential login, including retries.
        Once authenticated, it saves the session state.
        """
        if self.state_path.exists() and not force_browser:
            is_valid = await self.validate_auth_state()
            if is_valid:
                logger.info(f"Existing session state is still valid: {self.state_path}")
                return True

        async with async_playwright() as p:
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
                        p,
                        username=username,
                        password=password,
                        headless=headless,
                        timeout_ms=timeout_ms,
                    )
                except Exception as e:
                    logger.error(f"Login failed: {e}")
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
                logger.warning(f"{warning_message}. Enter credentials again.")

            return False

    async def reauthenticate(self, headless=False, timeout_ms=180000):
        """
        Interactive reauth flow used by long-running scrapers after 401/403.
        """
        logger.info(
            "Attempting reauthentication. Complete login in the browser window."
        )
        return await self.login(
            headless=headless, timeout_ms=timeout_ms, force_browser=True
        )

    def get_client(self):
        """Returns an httpx.AsyncClient initialized with the saved cookies."""
        state = self._load_state()
        if not state:
            raise FileNotFoundError("No session state found. Please run login() first.")
        cookies = self._cookies_from_state(state)

        return httpx.AsyncClient(
            base_url=self.base_url, cookies=cookies, headers=self._headers, timeout=30.0
        )


if __name__ == "__main__":
    session = IBKRSession()
    asyncio.run(session.login())
