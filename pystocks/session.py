import asyncio
from playwright.async_api import async_playwright
import httpx
import json
from .config import SESSION_STATE_PATH

class IBKRSession:
    """Manages an authenticated session for the IBKR Client Portal."""
    def __init__(self, state_path=SESSION_STATE_PATH):
        self.state_path = state_path
        self.portal_url = "https://www.interactivebrokers.ie/portal/"

    async def login(self, headless=False):
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
            await page.goto(self.portal_url)

            try:
                await page.wait_for_url("**/portal/*", timeout=120000)
                await page.wait_for_selector(".dashboard-root", timeout=60000)
                
                # Save the storage state (cookies, local storage)
                await context.storage_state(path=self.state_path)
                print(f"Login successful. Session state saved to {self.state_path}")
            except Exception as e:
                print(f"Login failed or timed out: {e}")
            finally:
                await browser.close()

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
