import asyncio
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs
from tqdm.asyncio import tqdm

from .config import RESEARCH_DIR
from .session import IBKRSession
from .ops_state import (
    init_db,
    log_scrape,
    get_all_instrument_conids,
    get_scraped_conids,
    update_instrument_fundamentals_status,
)
from .fundamentals_store import FundamentalsStore

logger = logging.getLogger(__name__)


def _load_conids_from_file(path):
    if path is None:
        return None

    lines = Path(path).read_text().splitlines()
    out = []
    seen = set()
    for line in lines:
        value = str(line).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out

class FundamentalScraper:
    """Scrapes fundamental data from IBKR Portal API."""
    PERIODS_DESC = ["10Y", "5Y", "3Y", "1Y", "6M"]
    RESULT_ENDPOINT_FAMILIES = {
        "profile_and_fees": "mf_profile_and_fees",
        "holdings": "mf_holdings",
        "ratios": "mf_ratios_fundamentals",
        "lipper_ratings": "mf_lip_ratings",
        "dividends": "dividends",
        "morningstar": "mstar/fund/detail",
        "price_chart": "mf_performance_chart",
        "sentiment_search": "sma/request?type=search",
        "ownership": "ownership",
        "esg": "impact/esg",
        "performance": "mf_performance",
    }
    
    def __init__(self, session=None):
        self.session = session or IBKRSession()
        self.esg_account_id = self.session.get_primary_account_id()
        self.research_dir = RESEARCH_DIR
        self.research_dir.mkdir(parents=True, exist_ok=True)
        self.store = FundamentalsStore()
        self.telemetry = {
            "run_started_at": datetime.now(timezone.utc).isoformat(),
            "endpoint_calls": Counter(),
            "endpoint_useful_payloads": Counter(),
            "status_codes": defaultdict(Counter),
        }
        init_db()

    def _endpoint_family(self, endpoint):
        endpoint = endpoint.lstrip("/")
        if endpoint.startswith("fundamentals/"):
            endpoint = endpoint[len("fundamentals/"):]

        if endpoint.startswith("landing/"):
            return "landing"
        if endpoint.startswith("mf_profile_and_fees/"):
            return "mf_profile_and_fees"
        if endpoint.startswith("mf_holdings/"):
            return "mf_holdings"
        if endpoint.startswith("mf_ratios_fundamentals/"):
            return "mf_ratios_fundamentals"
        if endpoint.startswith("mf_lip_ratings/"):
            return "mf_lip_ratings"
        if endpoint.startswith("dividends/"):
            return "dividends"
        if endpoint.startswith("mstar/fund/detail"):
            return "mstar/fund/detail"
        if endpoint.startswith("mf_performance_chart/"):
            return "mf_performance_chart"
        if endpoint.startswith("mf_performance/"):
            return "mf_performance"
        if endpoint.startswith("ownership/"):
            return "ownership"
        if endpoint.startswith("impact/esg/"):
            return "impact/esg"
        if endpoint.startswith("sma/request?"):
            query = endpoint.split("?", 1)[1] if "?" in endpoint else ""
            req_type = parse_qs(query).get("type", ["unknown"])[0]
            return f"sma/request?type={req_type}"
        return endpoint.split("?")[0]

    def _record_endpoint_status(self, endpoint, status_code):
        family = self._endpoint_family(endpoint)
        self.telemetry["endpoint_calls"][family] += 1
        self.telemetry["status_codes"][family][str(status_code)] += 1

    def _record_useful_payload(self, endpoint):
        family = self._endpoint_family(endpoint)
        self.telemetry["endpoint_useful_payloads"][family] += 1
        
    async def fetch_endpoint(self, client, endpoint, conid):
        """Fetches data from a specific endpoint and logs it."""
        # Check if endpoint starts with fundamentals/ or mstar/ or sma/
        # If not, default to fundamentals/
        path_prefix = "" if ("/" in endpoint and endpoint.split("/")[0] in ["fundamentals", "mstar", "sma", "impact"]) else "fundamentals/"
        url = f"/tws.proxy/{path_prefix}{endpoint}"
        try:
            response = await client.get(url)
            log_scrape(conid, endpoint, response.status_code)
            self._record_endpoint_status(endpoint, response.status_code)
            
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
            self._record_endpoint_status(endpoint, 0)
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
            "price_chart": "price_chart",
            "ownership": "owner",
            "sentiment_search": "sma_search",
        }
        kind = kind_alias.get(kind, kind)

        checks = {
            "profile": ["fund_and_profile", "objective"],
            "holdings": ["allocation_self", "top_10", "industry"],
            "ratios": ["ratios", "zscore"],
            "lipper": ["universes"],
            "esg": ["content"],
            # Keep both short and long dividends payload families.
            "divs": ["history", "industry_average", "no_div_data_marker", "last_payed_dividend_amount"],
            "mstar": ["summary", "commentary"],
            "price_chart": ["plot"],
            "perf": ["cumulative", "annualized"],
            "owner": ["trade_log", "owners_types"],
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

    def _build_sma_search_endpoint(self, conid):
        # Request max historical daily sentiment series.
        to_dt = datetime.now(timezone.utc)
        from_str = "1995-01-01 00:00"
        to_str = to_dt.strftime("%Y-%m-%d %H:%M")
        return f"sma/request?type=search&conid={conid}&from={from_str}&to={to_str}&bar_size=1D&tz=-60"

    def _build_price_chart_endpoint(self, conid):
        return f"mf_performance_chart/{conid}?chart_period=MAX"

    def _build_esg_endpoint(self, conid):
        if self.esg_account_id:
            return f"impact/esg/{conid}?accounts={self.esg_account_id}"
        return f"impact/esg/{conid}"

    def _landing_has_total_net_assets(self, landing_data):
        """True when landing payload provides key_profile.data.total_net_assets."""
        if not isinstance(landing_data, dict):
            return False

        key_profile = landing_data.get("key_profile") or landing_data.get("keyProfile")
        if not isinstance(key_profile, dict):
            return False

        profile_data = key_profile.get("data")
        if not isinstance(profile_data, dict):
            return False

        if "total_net_assets" not in profile_data:
            return False

        return self._has_any_value(profile_data.get("total_net_assets"))

    def _should_skip_fanout(self, landing_data):
        """Evaluates if we should stop after the landing page."""
        if not self._landing_has_total_net_assets(landing_data):
            return True, "total_net_assets missing"
        return False, None

    async def fetch_performance_with_period_fallback(self, client, conid):
        """
        Try largest->smallest period and return first performance payload with data.
        Returns tuple: (payload_or_auth_error, selected_period_or_none)
        """
        for period in self.PERIODS_DESC:
            endpoint = f"mf_performance/{conid}?risk_period={period}&statistic_period={period}"
            data = await self.fetch_endpoint(client, endpoint, conid)
            if data == "__AUTH_ERROR__":
                return "__AUTH_ERROR__", None
            if self._has_payload_data(data, "perf"):
                return data, period
        return None, None

    async def scrape_conid(self, client, conid):
        """Scrapes fundamental data for a given conid."""
        widgets = "objective,mstar,lipper_ratings,mf_key_ratios,risk_and_statistics,holdings,performance_and_peers,keyProfile,ownership,dividends,tear_sheet,news,fund_mstar,mf_esg,social_sentiment,securities_lending,sv,short_sale,ukuser"

        landing_data = await self.fetch_endpoint(client, f"landing/{conid}?widgets={widgets}", conid)
        if landing_data == "__AUTH_ERROR__":
            return "__AUTH_ERROR__"
        if not isinstance(landing_data, dict):
            return None
        if self._has_any_value(landing_data):
            self._record_useful_payload("landing")

        combined_data = {
            "conid": conid,
            "scraped_at": datetime.now().isoformat(),
            "landing": landing_data,
        }

        skip, reason = self._should_skip_fanout(landing_data)
        if skip:
            logger.info("Skipping fanout for conid=%s: %s", conid, reason)
            return combined_data

        fixed_task_items = [
            ("profile_and_fees", self.fetch_endpoint(client, f"mf_profile_and_fees/{conid}?sustainability=UK&lang=en", conid)),
            ("holdings", self.fetch_endpoint(client, f"mf_holdings/{conid}", conid)),
            ("ratios", self.fetch_endpoint(client, f"mf_ratios_fundamentals/{conid}", conid)),
            ("lipper_ratings", self.fetch_endpoint(client, f"mf_lip_ratings/{conid}", conid)),
            ("dividends", self.fetch_endpoint(client, f"dividends/{conid}", conid)),
            ("morningstar", self.fetch_endpoint(client, f"mstar/fund/detail?conid={conid}", conid)),
            ("price_chart", self.fetch_endpoint(client, self._build_price_chart_endpoint(conid), conid)),
            ("sentiment_search", self.fetch_endpoint(client, self._build_sma_search_endpoint(conid), conid)),
            ("ownership", self.fetch_endpoint(client, f"ownership/{conid}", conid)),
            ("esg", self.fetch_endpoint(client, self._build_esg_endpoint(conid), conid)),
        ]
        period_task_items = [
            ("performance", self.fetch_performance_with_period_fallback(client, conid)),
        ]

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
        
        for name, data in fixed_results.items():
            has_payload = self._has_payload_data(data, name)
            include_payload = has_payload or (
                name in {"profile_and_fees", "sentiment_search", "price_chart"}
                and isinstance(data, dict)
            )
            if include_payload:
                combined_data[name] = data
                if has_payload:
                    self._record_useful_payload(self.RESULT_ENDPOINT_FAMILIES.get(name, name))
        
        for name, (data, period) in period_results.items():
            if data:
                combined_data[name] = data
                combined_data[f"{name}_period"] = period
                self._record_useful_payload(self.RESULT_ENDPOINT_FAMILIES.get(name, name))
                
        return combined_data

    def save_telemetry(
        self,
        total_targeted,
        processed_conids,
        saved_snapshots,
        inserted_events,
        overwritten_events,
        unchanged_events,
        series_raw_rows_written,
        series_latest_rows_upserted,
        auth_retries,
        aborted,
        output_path=None,
    ):
        endpoint_calls = dict(sorted(self.telemetry["endpoint_calls"].items()))
        endpoint_useful = dict(sorted(self.telemetry["endpoint_useful_payloads"].items()))
        status_codes = {}
        for endpoint, counts in self.telemetry["status_codes"].items():
            status_codes[endpoint] = dict(sorted(counts.items(), key=lambda kv: kv[0]))

        endpoint_summary = []
        for endpoint, call_count in endpoint_calls.items():
            useful_count = endpoint_useful.get(endpoint, 0)
            endpoint_summary.append(
                {
                    "endpoint": endpoint,
                    "call_count": call_count,
                    "useful_payload_count": useful_count,
                    "useful_payload_rate": (useful_count / call_count) if call_count else 0.0,
                    "status_codes": status_codes.get(endpoint, {}),
                }
            )

        payload = {
            "run_started_at": self.telemetry["run_started_at"],
            "run_finished_at": datetime.now(timezone.utc).isoformat(),
            "run_stats": {
                "total_targeted_conids": total_targeted,
                "processed_conids": processed_conids,
                "saved_snapshots": saved_snapshots,
                "inserted_events": inserted_events,
                "overwritten_events": overwritten_events,
                "unchanged_events": unchanged_events,
                "series_raw_rows_written": series_raw_rows_written,
                "series_latest_rows_upserted": series_latest_rows_upserted,
                "auth_retries": auth_retries,
                "aborted": aborted,
            },
            "endpoint_summary": endpoint_summary,
        }

        if output_path:
            telemetry_path = Path(output_path)
            telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            telemetry_path = self.research_dir / f"fundamentals_run_telemetry_{ts}.json"

        latest_path = self.research_dir / "fundamentals_run_telemetry_latest.json"
        with open(telemetry_path, "w") as f:
            json.dump(payload, f, indent=2)
        with open(latest_path, "w") as f:
            json.dump(payload, f, indent=2)

        try:
            run_stats = dict(payload.get("run_stats", {}))
            run_stats["run_started_at"] = payload.get("run_started_at")
            run_stats["run_finished_at"] = payload.get("run_finished_at")
            self.store.persist_ingest_run(run_stats=run_stats, endpoint_summary=endpoint_summary)
        except Exception as e:
            logger.warning(f"Failed persisting run telemetry to SQLite: {e}")

        return telemetry_path, latest_path

async def main(
    limit=None,
    start_index=0,
    conids_file=None,
    force=False,
    max_auth_retries=2,
    reauth_headless=False,
    refresh_views_at_end=True,
    telemetry_output=None,
    verbose=False,
):
    log_level = logging.INFO if verbose else logging.WARNING
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
    root_logger.setLevel(log_level)
    logger.setLevel(log_level)
    
    scraper = FundamentalScraper()

    scraped_today = [] if force else get_scraped_conids()
    logger.info(f"Skipping {len(scraped_today)} instruments already scraped today.")

    selected_conids = _load_conids_from_file(conids_file)
    if selected_conids is not None:
        if not selected_conids:
            logger.error(f"No conids found in file: {conids_file}")
            return
        conids_to_scrape = selected_conids
        if not force:
            conids_to_scrape = [c for c in conids_to_scrape if c not in scraped_today]
        logger.info(f"Using explicit conid target list: {len(conids_to_scrape)} items.")
    else:
        all_conids = get_all_instrument_conids()
        if not all_conids:
            logger.error(
                "No products found in SQLite products table. "
                "Run `python -m pystocks.cli scrape_products` first."
            )
            return
        conids_to_scrape = [c for c in all_conids if c not in scraped_today]
    
    if limit:
        conids_to_scrape = conids_to_scrape[start_index:start_index + limit]
    else:
        conids_to_scrape = conids_to_scrape[start_index:]

    logger.info(f"Starting scrape for {len(conids_to_scrape)} instruments.")
    total_targeted = len(conids_to_scrape)
    processed_conids = 0
    saved_snapshots = 0
    inserted_events = 0
    overwritten_events = 0
    unchanged_events = 0
    series_raw_rows_written = 0
    series_latest_rows_upserted = 0
    auth_retries = 0
    aborted = False

    pbar = tqdm(total=total_targeted, desc="Scraping fundamentals")
    try:
        while processed_conids < total_targeted:
            needs_reauth = False
            try:
                async with scraper.session.get_client() as client:
                    while processed_conids < total_targeted:
                        conid = conids_to_scrape[processed_conids]
                        data = await scraper.scrape_conid(client, conid)

                        if data == "__AUTH_ERROR__":
                            update_instrument_fundamentals_status(conid, "auth_error", mark_scraped=False)
                            logger.warning(f"Authentication expired while scraping conid={conid}.")
                            needs_reauth = True
                            break

                        if data:
                            store_result = scraper.store.persist_combined_snapshot(
                                data,
                                refresh_views=False,
                            )
                            inserted_events += int(store_result.get("inserted_events", 0))
                            overwritten_events += int(store_result.get("overwritten_events", 0))
                            unchanged_events += int(store_result.get("unchanged_events", 0))
                            series_raw_rows_written += int(store_result.get("series_raw_rows_written", 0))
                            series_latest_rows_upserted += int(store_result.get("series_latest_rows_upserted", 0))
                            if store_result.get("status") == "ok":
                                update_instrument_fundamentals_status(conid, "success", mark_scraped=True)
                                saved_snapshots += 1
                            else:
                                update_instrument_fundamentals_status(
                                    conid,
                                    str(store_result.get("status") or "store_error"),
                                    mark_scraped=False,
                                )
                        else:
                            update_instrument_fundamentals_status(conid, "empty_payload", mark_scraped=False)

                        processed_conids += 1
                        pbar.update(1)
                        await asyncio.sleep(0.1)
            except FileNotFoundError as e:
                logger.warning(str(e))
                needs_reauth = True
            except Exception as e:
                logger.exception(f"Unexpected error: {e}")
                aborted = True
                break

            if not needs_reauth:
                break

            if auth_retries >= max_auth_retries:
                logger.error(f"Exceeded maximum reauthentication attempts ({max_auth_retries}).")
                aborted = True
                break

            auth_retries += 1
            logger.info(f"Reauth attempt {auth_retries}/{max_auth_retries} starting.")
            reauthed = await scraper.session.reauthenticate(headless=reauth_headless)
            if not reauthed:
                logger.error("Reauthentication failed.")
                aborted = True
                break
    finally:
        pbar.close()
        if refresh_views_at_end:
            try:
                refresh_result = scraper.store.refresh_sqlite_views()
                logger.info(f"Refreshed SQLite maintenance: {refresh_result}")
            except Exception as e:
                logger.error(f"Failed SQLite maintenance refresh: {e}")

        telemetry_path, latest_path = scraper.save_telemetry(
            total_targeted=total_targeted,
            processed_conids=processed_conids,
            saved_snapshots=saved_snapshots,
            inserted_events=inserted_events,
            overwritten_events=overwritten_events,
            unchanged_events=unchanged_events,
            series_raw_rows_written=series_raw_rows_written,
            series_latest_rows_upserted=series_latest_rows_upserted,
            auth_retries=auth_retries,
            aborted=aborted,
            output_path=telemetry_output,
        )
        logger.info(f"Saved run telemetry to {telemetry_path}")
        if telemetry_path != latest_path:
            logger.info(f"Updated latest telemetry pointer at {latest_path}")


async def run_fundamentals_update(limit=100, verbose=False, **kwargs):
    return await main(limit=limit, verbose=verbose, **kwargs)

if __name__ == "__main__":
    import fire
    fire.Fire(main)
