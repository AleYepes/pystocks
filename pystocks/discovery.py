import asyncio
from playwright.async_api import async_playwright
import json
from pathlib import Path
from .config import DATA_DIR

async def discover_endpoints(con_id="207825419"):
    """
    A utility to help discover IBKR fundamental endpoints by logging network requests.
    The user should log in and navigate to an ETF fundamentals page.
    """
    print("Starting Playwright for endpoint discovery...")
    print("Please log in to IBKR when the browser opens and navigate to an ETF fundamentals page.")
    
    async with async_playwright() as p:
        browser = await async_playwright().start()
        # Using a persistent context to keep login session if needed
        context = await browser.new_context(user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
        
        page = await context.new_page()
        
        # Monitor network requests
        async def handle_request(request):
            if "tws.proxy" in request.url:
                print(f"Detected proxy request: {request.url}")
                
        async def handle_response(response):
            if "tws.proxy" in response.url and response.status == 200:
                url = response.url
                try:
                    # Try to parse as JSON to verify
                    await response.json()
                    print(f"Successfully captured JSON from: {url}")
                    # Save a sample to data/discovery
                    discovery_dir = DATA_DIR / "discovery"
                    discovery_dir.mkdir(exist_ok=True)
                    name = url.split("/")[-2] if url.split("/")[-1].isdigit() else url.split("/")[-1]
                    with open(discovery_dir / f"{name}.json", "w") as f:
                        json.dump(await response.json(), f, indent=4)
                except:
                    pass

        page.on("request", handle_request)
        page.on("response", handle_response)
        
        # Navigate to a starting point
        await page.goto(f"https://www.interactivebrokers.ie/portal/#/fundamentals/{con_id}")
        
        print("Waiting for you to interact with the page... Press Ctrl+C in terminal when done.")
        try:
            # Keep browser open indefinitely for manual discovery
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("Discovery stopped.")
        finally:
            await browser.close()

if __name__ == "__main__":
    # To run: python3 -m pystocks.discovery
    asyncio.run(discover_endpoints())
