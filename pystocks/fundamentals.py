import asyncio
import json
import logging
from datetime import datetime
import pandas as pd
from tqdm.asyncio import tqdm

from .config import DATA_DIR, IB_PRODUCTS_PATH, FUNDAMENTALS_DIR, SQLITE_DB_PATH
from .session import IBKRSession
from .database import init_db, log_scrape, sync_instruments_from_csv, get_connection

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FundamentalScraper:
    """Scrapes fundamental data from IBKR Portal API."""
    
    def __init__(self, session=None):
        self.session = session or IBKRSession()
        self.base_url = "https://www.interactivebrokers.ie"
        self.fundamentals_dir = FUNDAMENTALS_DIR
        self.fundamentals_dir.mkdir(parents=True, exist_ok=True)
        init_db()
        
    async def fetch_endpoint(self, client, endpoint, conid):
        """Fetches data from a specific endpoint and logs it."""
        url = f"/tws.proxy/fundamentals/{endpoint}"
        try:
            response = await client.get(url)
            log_scrape(conid, endpoint, response.status_code)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code in [401, 403]:
                logger.error(f"Session expired or unauthorized: {response.status_code}")
                return "__AUTH_ERROR__"
            else:
                return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            log_scrape(conid, endpoint, 0, str(e))
            return None

    async def scrape_conid(self, client, conid):
        """Scrapes all fundamental data for a given conid."""
        # Widgets to fetch via landing
        widgets = "objective,mstar,lipper_ratings,mf_key_ratios,risk_and_statistics,holdings,performance_and_peers,keyProfile,ownership,dividends,tear_sheet,news,fund_mstar,mf_esg,social_sentiment,securities_lending,sv,short_sale,ukuser"
        
        # We fetch:
        # 1. landing: general overview, ratings, performance, etc.
        # 2. mf_profile_and_fees: reports, expenses details, profile metadata
        # 3. mf_holdings: full industry and country breakdowns
        tasks = [
            self.fetch_endpoint(client, f"landing/{conid}?widgets={widgets}", conid),
            self.fetch_endpoint(client, f"mf_profile_and_fees/{conid}?sustainability=UK&lang=en", conid),
            self.fetch_endpoint(client, f"mf_holdings/{conid}", conid)
        ]
        
        results = await asyncio.gather(*tasks)
        
        if "__AUTH_ERROR__" in results:
            return "__AUTH_ERROR__"
            
        landing_data, profile_data, holdings_data = results
        
        if not any([landing_data, profile_data, holdings_data]):
            return None
            
        combined_data = {
            "conid": conid,
            "scraped_at": datetime.now().isoformat(),
            "landing": landing_data,
            "profile_and_fees": profile_data,
            "holdings": holdings_data
        }
        
        return combined_data

    def save_data(self, conid, data):
        """Saves scraped data to a JSON file."""
        conid_dir = self.fundamentals_dir / str(conid)
        conid_dir.mkdir(exist_ok=True)
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        file_path = conid_dir / f"{date_str}.json"
        
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

def get_scraped_conids():
    """Returns a list of conids that were already successfully scraped today."""
    today = datetime.now().strftime("%Y-%m-%d")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT conid FROM instruments WHERE last_scraped_fundamentals = ?", (today,))
        return [row[0] for row in cursor.fetchall()]

async def main(limit=None, start_index=0, force=False):
    scraper = FundamentalScraper()
    
    if not IB_PRODUCTS_PATH.exists():
        logger.error(f"Products file not found: {IB_PRODUCTS_PATH}")
        return

    sync_instruments_from_csv(IB_PRODUCTS_PATH)
    
    scraped_today = [] if force else get_scraped_conids()
    logger.info(f"Skipping {len(scraped_today)} instruments already scraped today.")

    df = pd.read_csv(IB_PRODUCTS_PATH)
    # Ensure conid is treated as string for consistency
    df['conid'] = df['conid'].astype(str)
    
    # Filter for unique conids not already scraped today
    all_conids = df['conid'].unique()
    conids_to_scrape = [c for c in all_conids if c not in scraped_today]
    
    if limit:
        conids_to_scrape = conids_to_scrape[start_index:start_index + limit]
    else:
        conids_to_scrape = conids_to_scrape[start_index:]

    logger.info(f"Starting scrape for {len(conids_to_scrape)} instruments.")
    
    try:
        async with scraper.session.get_client() as client:
            pbar = tqdm(conids_to_scrape, desc="Scraping fundamentals")
            for conid in pbar:
                data = await scraper.scrape_conid(client, conid)
                
                if data == "__AUTH_ERROR__":
                    logger.error("Stopping due to authentication error.")
                    break
                    
                if data:
                    scraper.save_data(conid, data)
                
                # Small delay to be polite
                await asyncio.sleep(0.1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    import fire
    fire.Fire(main)
