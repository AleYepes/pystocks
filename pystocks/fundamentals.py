import asyncio
import json
import logging
from datetime import datetime, timezone
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
    PERIODS_DESC = ["10Y", "5Y", "3Y", "1Y", "6M"]
    # Hardcoded strategy informed by research_correlations sample (n=500).
    FETCH_LOW_YIELD_ENDPOINTS = False
    
    def __init__(self, session=None):
        self.session = session or IBKRSession()
        self.base_url = "https://www.interactivebrokers.ie"
        self.fundamentals_dir = FUNDAMENTALS_DIR
        self.fundamentals_dir.mkdir(parents=True, exist_ok=True)
        init_db()
        
    async def fetch_endpoint(self, client, endpoint, conid):
        """Fetches data from a specific endpoint and logs it."""
        # Check if endpoint starts with fundamentals/ or mstar/ or sma/
        # If not, default to fundamentals/
        path_prefix = "" if ("/" in endpoint and endpoint.split("/")[0] in ["fundamentals", "mstar", "sma", "impact"]) else "fundamentals/"
        url = f"/tws.proxy/{path_prefix}{endpoint}"
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

    def _has_payload_data(self, data, kind):
        """Heuristic check for actual content in response."""
        if not isinstance(data, dict):
            return False
        # Normalize internal endpoint names used by scrape_conid.
        kind_alias = {
            "profile_and_fees": "profile",
            "lipper_ratings": "lipper",
            "dividends": "divs",
            "morningstar": "mstar",
            "ownership": "owner",
            "sentiment": "sentiment",
            "sentiment_search": "sma_search",
        }
        kind = kind_alias.get(kind, kind)

        checks = {
            "profile": ["fund_and_profile", "objective"],
            "holdings": ["allocation_self", "top_10", "industry"],
            "ratios": ["ratios", "zscore"],
            "lipper": ["universes"],
            "esg": ["content"],
            "divs": ["history"], # 'industry_average' is often just peer data, we want 'history'
            "mstar": ["summary", "commentary"],
            "perf": ["cumulative", "annualized"],
            "risk": ["risk", "statistic", "performance"],
            "owner": ["trade_log", "owners_types"],
            "sentiment": ["smean", "sscore", "sbuzz"],
            "sma_search": ["sentiment"],
        }
        for key in checks.get(kind, []):
            if self._has_any_value(data.get(key)):
                return True
        return False

    def _has_any_value(self, value):
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (list, dict, tuple, set)):
            return len(value) > 0
        return True

    def _landing_has_section_data(self, landing, section):
        node = landing.get(section)
        if not isinstance(node, dict):
            return False
        
        # Special case for dividends: marker 1 means no data
        if section == "dividends":
            content = node.get("content", {})
            if content.get("no_div_data_marker") == 1:
                return False
                
        return self._has_any_value(node.get("data")) or self._has_any_value(node.get("content"))

    def _build_sma_search_endpoint(self, conid):
        # 1Y rolling window for historical sentiment bars.
        to_dt = datetime.now(timezone.utc)
        from_dt = datetime.fromtimestamp(to_dt.timestamp() - 365 * 24 * 3600, timezone.utc)
        from_str = from_dt.strftime("%Y-%m-%d %H:%M")
        to_str = to_dt.strftime("%Y-%m-%d %H:%M")
        return f"sma/request?type=search&conid={conid}&from={from_str}&to={to_str}&bar_size=1D&tz=-60"

    async def fetch_with_period_fallback(self, client, conid, kind):
        """
        Try largest->smallest period and return first payload with data.
        Returns tuple: (payload_or_auth_error, selected_period_or_none)
        """
        for period in self.PERIODS_DESC:
            if kind == "perf":
                endpoint = f"mf_performance/{conid}?risk_period={period}&statistic_period={period}"
            else:
                endpoint = f"mf_risks_stats/{conid}?period={period}"

            data = await self.fetch_endpoint(client, endpoint, conid)
            if data == "__AUTH_ERROR__":
                return "__AUTH_ERROR__", None
            if self._has_payload_data(data, kind):
                return data, period
        return None, None

    async def scrape_conid(self, client, conid):
        """Scrapes fundamental data using a research-backed probabilistic strategy."""
        widgets = "objective,mstar,lipper_ratings,mf_key_ratios,risk_and_statistics,holdings,performance_and_peers,keyProfile,ownership,dividends,tear_sheet,news,fund_mstar,mf_esg,social_sentiment,securities_lending,sv,short_sale,ukuser"

        # 1) Fetch landing first as the primary completeness probe.
        landing_data = await self.fetch_endpoint(client, f"landing/{conid}?widgets={widgets}", conid)
        if landing_data == "__AUTH_ERROR__":
            return "__AUTH_ERROR__"
        if not isinstance(landing_data, dict):
            return None

        # 2) Feature Extraction (probabilistic predictors)
        has_objective = self._landing_has_section_data(landing_data, "objective")
        has_top10 = self._landing_has_section_data(landing_data, "top10")
        has_ratios = self._landing_has_section_data(landing_data, "mf_key_ratios")
        has_overall_ratings = self._landing_has_section_data(landing_data, "overall_ratings")
        has_mstar = self._landing_has_section_data(landing_data, "mstar")
        has_perf = self._landing_has_section_data(landing_data, "cumulative_performace")
        has_risk = self._landing_has_section_data(landing_data, "risk_statistics")
        has_dividends = self._landing_has_section_data(landing_data, "dividends")
        has_ownership = self._landing_has_section_data(landing_data, "ownership")

        # 3) Determine targeted tasks
        # Always high-yield baseline
        fixed_task_items = [
            ("profile_and_fees", self.fetch_endpoint(client, f"mf_profile_and_fees/{conid}?sustainability=UK&lang=en", conid))
        ]

        # Holdings is effectively always present in sample (100% yield), so keep it unconditional.
        fixed_task_items.append(("holdings", self.fetch_endpoint(client, f"mf_holdings/{conid}", conid)))

        # Ratios is high-yield when top10/objective is present.
        if has_top10 or has_objective:
            fixed_task_items.append(("ratios", self.fetch_endpoint(client, f"mf_ratios_fundamentals/{conid}", conid)))

        # Lipper Ratings (1.0 Lift from overall_ratings)
        if has_overall_ratings:
            fixed_task_items.append(("lipper_ratings", self.fetch_endpoint(client, f"mf_lip_ratings/{conid}", conid)))

        # Dividends
        if has_dividends:
            fixed_task_items.append(("dividends", self.fetch_endpoint(client, f"dividends/{conid}", conid)))

        # Morningstar: unconditional (high yield with low precision penalty).
        fixed_task_items.append(("morningstar", self.fetch_endpoint(client, f"mstar/fund/detail?conid={conid}", conid)))

        # Sentiment: use search endpoint (higher yield than tick), gated by overall ratings.
        if has_overall_ratings:
            fixed_task_items.append(("sentiment_search", self.fetch_endpoint(client, self._build_sma_search_endpoint(conid), conid)))

        # Rare, mostly low-yield endpoints are opt-in.
        if self.FETCH_LOW_YIELD_ENDPOINTS and has_ownership:
            fixed_task_items.append(("ownership", self.fetch_endpoint(client, f"ownership/{conid}", conid)))
            fixed_task_items.append(("sentiment", self.fetch_endpoint(client, f"sma/request?type=tick&conid={conid}", conid)))
        if self.FETCH_LOW_YIELD_ENDPOINTS:
            fixed_task_items.append(("esg", self.fetch_endpoint(client, f"impact/esg/{conid}", conid)))

        # Performance & Risk fallback tasks
        period_task_items = []
        if has_perf or has_objective:
            period_task_items.append(("performance", self.fetch_with_period_fallback(client, conid, "perf")))
        if has_risk or has_objective:
            period_task_items.append(("risk_stats", self.fetch_with_period_fallback(client, conid, "risk")))

        # 4) Execute determined tasks
        fixed_results = {}
        if fixed_task_items:
            names, tasks = zip(*fixed_task_items)
            values = await asyncio.gather(*tasks)
            fixed_results = dict(zip(names, values))

        period_results = {}
        if period_task_items:
            names, tasks = zip(*period_task_items)
            values = await asyncio.gather(*tasks)
            period_results = dict(zip(names, values))

        if "__AUTH_ERROR__" in fixed_results.values() or any(r[0] == "__AUTH_ERROR__" for r in period_results.values()):
            return "__AUTH_ERROR__"

        # 5) Composition
        combined_data = {
            "conid": conid,
            "scraped_at": datetime.now().isoformat(),
            "landing": landing_data,
            "probe": {
                "has_objective": has_objective,
                "has_top10": has_top10,
                "has_ratios": has_ratios,
                "has_overall_ratings": has_overall_ratings,
                "has_mstar": has_mstar,
                "has_perf": has_perf,
                "has_risk": has_risk,
                "has_dividends": has_dividends,
                "has_ownership": has_ownership,
                "fetch_low_yield_endpoints": self.FETCH_LOW_YIELD_ENDPOINTS,
            }
        }
        
        # Merge results, only if they have actual payload data
        for name, data in fixed_results.items():
            if self._has_payload_data(data, name) or name == "profile_and_fees":
                combined_data[name] = data
        
        for name, (data, period) in period_results.items():
            if data:
                combined_data[name] = data
                combined_data[f"{name}_period"] = period
                
        return combined_data

    def save_data(self, conid, data, pretty=False):
        """Saves scraped data to a JSON file."""
        conid_dir = self.fundamentals_dir / str(conid)
        conid_dir.mkdir(exist_ok=True)
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        file_path = conid_dir / f"{date_str}.json"
        
        with open(file_path, "w") as f:
            if pretty:
                json.dump(data, f, indent=2)
            else:
                json.dump(data, f, separators=(",", ":"))

def get_scraped_conids():
    """Returns a list of conids that were already successfully scraped today."""
    today = datetime.now().strftime("%Y-%m-%d")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT conid FROM instruments WHERE last_scraped_fundamentals = ?", (today,))
        return [row[0] for row in cursor.fetchall()]

async def main(limit=None, start_index=0, force=False, pretty_json=False):
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
                    scraper.save_data(conid, data, pretty=pretty_json)
                
                # Small delay to be polite
                await asyncio.sleep(0.1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    import fire
    fire.Fire(main)
