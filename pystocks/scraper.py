import asyncio
from playwright.async_api import async_playwright
import pandas as pd
from .config import IB_PRODUCTS_PATH
import os

async def scrape_ibkr_products():
    """
    Scrapes the IBKR product list.
    """
    print("Starting product list scrape...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False) # Headless=False to allow manual interaction if needed
        context = await browser.new_context()
        page = await context.new_page()
        
        url = 'https://www.interactivebrokers.ie/en/trading/products-exchanges.php'
        await page.goto(url)
        
        print("Please navigate to the PRODUCT TAB and apply desired filters.")
        print("Press Enter in terminal when you are ready to start extraction from the current table view.")
        input("Waiting for user interaction... Press Enter to continue.")

        master_data = []
        
        while True:
            # Extract current table page
            rows = await page.query_selector_all("#tableContacts tr")
            headers_elements = await page.query_selector_all("#tableContacts th")
            headers = [await h.inner_text() for h in headers_elements]
            
            for row in rows[1:]: # Skip header
                cells = await row.query_selector_all("td")
                data = [await c.inner_text() for c in cells]
                if data:
                    master_data.append(data)
            
            # Check for next page
            forward_button = await page.query_selector(".btn.btn-xs.btn-default.btn-forward")
            if forward_button and await forward_button.is_visible() and await forward_button.is_enabled():
                await forward_button.click()
                await asyncio.sleep(1) # Wait for page to load
            else:
                break
        
        df = pd.DataFrame(master_data, columns=headers)
        
        if IB_PRODUCTS_PATH.exists():
            existing_df = pd.read_csv(IB_PRODUCTS_PATH)
            df = pd.concat([existing_df, df]).drop_duplicates()
            print("Updating existing product list.")
        
        df.to_csv(IB_PRODUCTS_PATH, index=False)
        print(f"Saved {len(df)} products to {IB_PRODUCTS_PATH}")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(scrape_ibkr_products())
