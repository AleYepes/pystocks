import asyncio
import re
import logging
from playwright.async_api import async_playwright
import httpx
import json
from .config import SESSION_STATE_PATH

logger = logging.getLogger(__name__)

_ACCOUNT_PATH_RE = re.compile(r"/portal\.proxy/v1/portal/(?:portfolio2|acesws)/([A-Za-z0-9]+)")
_ACCOUNT_ID_RE = re.compile(r"^(?:U|DU|DF|F)?\d+$")


class IBKRSession:
    """Manages an authenticated session for the IBKR Client Portal."""
    def __init__(self, state_path=SESSION_STATE_PATH):
        self.state_path = state_path
        self.portal_url = "https://www.interactivebrokers.ie/portal/"
        self.base_url = "https://www.interactivebrokers.ie"
        self._headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.interactivebrokers.ie/portal/",
            "X-Requested-With": "XMLHttpRequest",
        }
        self._auth_check_endpoints = [
            # Prefer lightweight authenticated endpoints.
            "/tws.proxy/portfolio/accounts",
            "/tws.proxy/iserver/auth/status",
            "/tws.proxy/fundamentals/landing/756733?widgets=objective",
        ]

    def _load_state(self):
        if not self.state_path.exists():
            return None
        with open(self.state_path, "r") as f:
            return json.load(f)

    def _cookies_from_state(self, state):
        cookies = {}
        for cookie in (state or {}).get("cookies", []):
            name = cookie.get("name")
            value = cookie.get("value")
            if name and value is not None:
                cookies[name] = value
        return cookies

    def get_primary_account_id(self):
        state = self._load_state()
        if not state:
            return None

        candidates = []
        for cookie in state.get("cookies", []):
            path = cookie.get("path")
            if not isinstance(path, str):
                continue
            m = _ACCOUNT_PATH_RE.search(path)
            if not m:
                continue
            account_id = m.group(1)
            if _ACCOUNT_ID_RE.match(account_id):
                candidates.append(account_id)

        if not candidates:
            return None

        unique = sorted(set(candidates))
        unique.sort(key=lambda x: (0 if x.startswith("U") else 1, -len(x), x))
        return unique[0]

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
        return await self._validate_state_payload(state, timeout_s=timeout_s)

    async def login(self, headless=False, timeout_ms=180000, force_browser=False):
        """
        Launches a browser and waits for the user to log in manually.
        Once the portal home is reached, it saves the session state.
        """
        if self.state_path.exists() and not force_browser:
            is_valid = await self.validate_auth_state()
            if is_valid:
                logger.info(f"Existing session state is still valid: {self.state_path}")
                return True

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=headless)
            # If state is stale, use a fresh context to avoid stale-cookie UI loops.
            context = await browser.new_context()

            page = await context.new_page()
            await page.goto(self.portal_url, wait_until="domcontentloaded")
            logger.info("Browser opened. Complete login in the IBKR window.")

            try:
                loop = asyncio.get_running_loop()
                deadline = loop.time() + (timeout_ms / 1000.0)

                while loop.time() < deadline:
                    state = await context.storage_state()
                    is_valid = await self._validate_state_payload(state, timeout_s=10.0)
                    if is_valid:
                        await context.storage_state(path=self.state_path)
                        logger.info(f"Login successful. Session state saved to {self.state_path}")
                        return True
                    await asyncio.sleep(2)

                logger.error("Login timed out before authenticated API state was detected.")
                return False
            except Exception as e:
                logger.error(f"Login failed or timed out: {e}")
                return False
            finally:
                await browser.close()

    async def reauthenticate(self, headless=False, timeout_ms=180000):
        """
        Interactive reauth flow used by long-running scrapers after 401/403.
        """
        logger.warning("Attempting reauthentication. Complete login in the browser window.")
        return await self.login(headless=headless, timeout_ms=timeout_ms, force_browser=True)

    def get_client(self):
        """Returns an httpx.AsyncClient initialized with the saved cookies."""
        state = self._load_state()
        if not state:
            raise FileNotFoundError("No session state found. Please run login() first.")
        cookies = self._cookies_from_state(state)

        return httpx.AsyncClient(
            base_url=self.base_url,
            cookies=cookies,
            headers=self._headers,
            timeout=30.0
        )

if __name__ == "__main__":
    session = IBKRSession()
    asyncio.run(session.login())
