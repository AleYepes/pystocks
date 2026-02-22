import asyncio
from playwright.async_api import async_playwright
import httpx
import json
from .config import SESSION_STATE_PATH

class IBKRSession:
    """Manages an authenticated session for the IBKR Client Portal."""
    DASHBOARD_SELECTORS = [
        ".dashboard-root",
        "[data-testid='dashboard-root']",
        "[data-testid='dashboard']",
        "#dashboard-root",
        ".cp-home-page",
        ".main-content",
        "[class*='dashboard']",
    ]

    def __init__(self, state_path=SESSION_STATE_PATH):
        self.state_path = state_path
        self.portal_url = "https://www.interactivebrokers.ie/portal/"

    async def _wait_for_authenticated_portal(self, page, timeout_ms=180000):
        """
        Waits for a post-login portal state using URL + fallback selectors.
        Returns True on success, False on timeout.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + (timeout_ms / 1000.0)

        while loop.time() < deadline:
            # Any authenticated portal route is acceptable.
            if "/portal/" in page.url and "login" not in page.url.lower():
                for selector in self.DASHBOARD_SELECTORS:
                    try:
                        handle = await page.query_selector(selector)
                        if handle is not None:
                            return True
                    except Exception:
                        continue

                # Fallback if selector contract changed:
                # accept portal route when login form is no longer present.
                try:
                    login_password = await page.query_selector("input[type='password']")
                    if login_password is None:
                        return True
                except Exception:
                    pass

            await asyncio.sleep(1)

        return False

    async def login(self, headless=False, timeout_ms=180000):
        """
        Launches a browser and waits for the user to log in manually.
        Once the portal home is reached, it saves the session state.
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=headless)
            # Load existing state if it exists
            if self.state_path.exists():
                context = await browser.new_context(storage_state=self.state_path)
            else:
                context = await browser.new_context()

            page = await context.new_page()
            await page.goto(self.portal_url, wait_until="domcontentloaded")

            try:
                is_ready = await self._wait_for_authenticated_portal(page, timeout_ms=timeout_ms)
                if not is_ready:
                    print("Login timed out before authenticated dashboard state was detected.")
                    return False

                # Save storage state (cookies + local storage) after authenticated route is observed.
                await context.storage_state(path=self.state_path)
                print(f"Login successful. Session state saved to {self.state_path}")
                return True
            except Exception as e:
                print(f"Login failed or timed out: {e}")
                return False
            finally:
                await browser.close()

    async def reauthenticate(self, headless=False, timeout_ms=180000):
        """
        Interactive reauth flow used by long-running scrapers after 401/403.
        """
        print("Attempting reauthentication. Complete login in the browser window.")
        return await self.login(headless=headless, timeout_ms=timeout_ms)

    def get_client(self):
        """Returns an httpx.AsyncClient initialized with the saved cookies."""
        if not self.state_path.exists():
            raise FileNotFoundError("No session state found. Please run login() first.")

        with open(self.state_path, 'r') as f:
            state = json.load(f)

        # Reconstruct cookies for httpx
        cookies = {}
        for cookie in state.get('cookies', []):
            cookies[cookie['name']] = cookie['value']

        return httpx.AsyncClient(
            base_url="https://www.interactivebrokers.ie",
            cookies=cookies,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Referer": "https://www.interactivebrokers.ie/portal/",
                "X-Requested-With": "XMLHttpRequest"
            },
            timeout=30.0
        )

if __name__ == "__main__":
    session = IBKRSession()
    asyncio.run(session.login())
