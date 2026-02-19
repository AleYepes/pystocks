import asyncio
import httpx
import pandas as pd
from datetime import datetime
from .config import CONTRACTS_DB_PATH, RAW_DIR
from .utils import load_csv, save_fundamentals
from tqdm.auto import tqdm
import json

class IBKRPortalClient:
    def __init__(self, cookies=None, base_url="https://www.interactivebrokers.ie"):
        self.base_url = base_url
        self.cookies = cookies or {}
        self.client = httpx.AsyncClient(base_url=base_url, cookies=self.cookies, timeout=30.0)

    async def fetch_json(self, endpoint):
        response = await self.client.get(f"/tws.proxy/fundamentals/{endpoint}")
        if response.status_code == 200:
            return response.json()
        else:
            # print(f"Error fetching {endpoint}: {response.status_code}")
            return None

    async def get_etf_data(self, con_id):
        """
        Fetches all relevant fundamental data for an ETF and returns it in the legacy format.
        """
        results = {
            'conId': con_id,
            'date_scraped': datetime.now().strftime('%Y-%m-%d'),
        }
        
        # Parallel fetch
        tasks = [
            self.fetch_json(f"mf_profile/{con_id}"),
            self.fetch_json(f"mf_holdings/{con_id}"),
            self.fetch_json(f"mf_ratios/{con_id}"),
            self.fetch_json(f"mf_allocation/{con_id}")
        ]
        
        profile_json, holdings_json, ratios_json, allocation_json = await asyncio.gather(*tasks)
        
        if not profile_json and not holdings_json:
            return None

        # --- Mapping Logic (Legacy Format) ---
        
        # 1. Profile
        if profile_json:
            # Example mapping based on common IBKR JSON structure
            # You might need to adjust these keys after discovery
            profile_list = []
            if 'domicile' in profile_json: profile_list.append(('Domicile', profile_json['domicile']))
            if 'geoFocus' in profile_json: profile_list.append(('MarketGeoFocus', profile_json['geoFocus']))
            if 'netAssets' in profile_json: 
                val = profile_json['netAssets']
                date = profile_json.get('netAssetsDate', '')
                profile_list.append(('TotalNetAssets', f"{val}asof{date}"))
            results['profile'] = str(profile_list)
            results['funds_date'] = profile_json.get('netAssetsDate', None)

        # 2. Ratios / Fundamentals
        if ratios_json:
            fund_list = []
            # Map bond or equity ratios
            for key, val in ratios_json.items():
                fund_list.append((key, val))
            results['fundamentals'] = fund_list

        # 3. Holdings (Top 10)
        if holdings_json and 'top10' in holdings_json:
            top10_list = []
            for h in holdings_json['top10']:
                top10_list.append((h.get('name', 'Unknown'), h.get('weight', 0)))
            results['top10'] = top10_list
            results['holding_date'] = holdings_json.get('date', None)

        # 4. Allocation (Countries, Industries, Currencies)
        if allocation_json:
            if 'countries' in allocation_json:
                results['countries'] = [(c['name'], c['weight']) for c in allocation_json['countries']]
            if 'industries' in allocation_json:
                results['industries'] = [(i['name'], i['weight']) for i in allocation_json['industries']]
            if 'currencies' in allocation_json:
                results['currencies'] = [(c['name'], c['weight']) for c in allocation_json['currencies']]

        return results

async def run_fundamentals_update(limit=100):
    if not CONTRACTS_DB_PATH.exists():
        print("Contract details not found. Run discovery first.")
        return

    db_df = load_csv(CONTRACTS_DB_PATH)
    # We need cookies from a logged-in session. 
    # For now, we'll prompt the user or use a discovery script to get them.
    print("This script requires valid IBKR session cookies.")
    # Implementation of cookie extraction via Playwright would go here.
    
    # Placeholder for client
    client = IBKRPortalClient() 
    
    scraped_data = []
    for _, row in tqdm(db_df.iterrows(), total=min(len(db_df), limit), desc="Scraping fundamentals"):
        con_id = row['conId']
        data = await client.get_etf_data(con_id)
        if data:
            # Merge with contract details
            row_dict = row.to_dict()
            full_data = {**row_dict, **data}
            scraped_data.append(full_data)
            
        if len(scraped_data) >= 10: # Batch save
            save_fundamentals(pd.DataFrame(scraped_data))
            scraped_data = []
            
    if scraped_data:
        save_fundamentals(pd.DataFrame(scraped_data))

if __name__ == "__main__":
    asyncio.run(run_fundamentals_update())
