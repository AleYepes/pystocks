import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd
from tqdm.asyncio import tqdm
from .session import IBKRSession
from .database import get_connection
from .config import RESEARCH_YIELDS_PATH, RESEARCH_CORR_SUMMARY_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ENDPOINTS = {
    "profile": "mf_profile_and_fees/{conid}?sustainability=UK&lang=en",
    "holdings": "mf_holdings/{conid}",
    "ratios": "mf_ratios_fundamentals/{conid}",
    "lipper": "mf_lip_ratings/{conid}",
    "esg": "impact/esg/{conid}",
    "divs": "dividends/{conid}",
    "mstar": "mstar/fund/detail?conid={conid}",
    "owner": "ownership/{conid}",
    "sma_tick": "sma/request?type=tick&conid={conid}",
    "sma_high_low": "sma/request?type=high_low&conid={conid}"
}

LANDING_SECTIONS = [
    "objective",
    "mstar",
    "overall_ratings",
    "mf_key_ratios",
    "risk_statistics",
    "top10",
    "key_profile",
    "ownership",
    "dividends",
    "mf_esg",
]

PERIODS_DESC = ["10Y", "5Y", "3Y", "1Y", "6M"]


class AuthError(Exception):
    pass


def _has_any_value(value):
    """True if value is non-empty scalar/list/dict."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True


def has_data(response, name):
    """Heuristic: endpoint contains useful payload, not just an empty shell."""
    if not isinstance(response, dict):
        return False

    checks = {
        "profile": ["fund_and_profile", "objective"],
        "holdings": ["allocation_self", "top_10", "industry"],
        "ratios": ["ratios", "zscore"],
        "lipper": ["universes"],
        "esg": ["content"],
        "divs": ["history", "industry_average"],
        "mstar": ["summary", "commentary"],
        "perf": ["cumulative", "annualized"],
        "risk": ["risk", "statistic", "performance"],
        "owner": ["trade_log", "owners_types"],
        "sma_tick": ["smean", "sscore", "sbuzz"],
        "sma_high_low": ["sscore_high", "sscore_low", "svscore_high", "svscore_low"],
        "sma_search": ["sentiment"],
    }

    for key in checks.get(name, []):
        if _has_any_value(response.get(key)):
            return True

    # Fallback for unknown or variable schemas
    for v in response.values():
        if _has_any_value(v):
            return True

    return False


def landing_has_section_data(landing, section):
    """Normalize landing teaser shape into a single presence boolean."""
    node = landing.get(section)
    if not isinstance(node, dict):
        return False

    if _has_any_value(node.get("data")):
        return True
    if _has_any_value(node.get("content")):
        return True
    return False


async def fetch_json(client, endpoint):
    """Fetch endpoint JSON with consistent prefixing and auth handling."""
    prefix = "" if (
        "/" in endpoint and endpoint.split("/")[0] in ["fundamentals", "mstar", "sma", "impact"]
    ) else "fundamentals/"
    url = f"/tws.proxy/{prefix}{endpoint}"

    try:
        response = await client.get(url)
    except Exception:
        return 0, None
    if response.status_code in (401, 403):
        raise AuthError(f"Auth error status={response.status_code} for {endpoint}")

    if response.status_code != 200:
        return response.status_code, None

    try:
        return 200, response.json()
    except Exception:
        return 200, None


async def fetch_with_period_fallback(client, conid, kind):
    """
    Try largest-to-smallest period and return first payload with data.
    Also returns attempts for diagnostics.
    """
    attempts = []
    for period in PERIODS_DESC:
        if kind == "perf":
            endpoint = f"mf_performance/{conid}?risk_period={period}&statistic_period={period}"
        else:
            endpoint = f"mf_risks_stats/{conid}?period={period}"

        status, data = await fetch_json(client, endpoint)
        ok = status == 200 and has_data(data, kind)
        attempts.append(f"{period}:{status}:{int(ok)}")
        if ok:
            return {
                "status": status,
                "has_data": True,
                "selected_period": period,
                "attempts": "|".join(attempts),
            }

    # If no period had data, still record last attempted status and attempts.
    return {
        "status": status if attempts else 0,
        "has_data": False,
        "selected_period": "",
        "attempts": "|".join(attempts),
    }


def build_sma_search_endpoint(conid):
    # 1Y window; this endpoint can fail for younger instruments, which is part of what we want to measure.
    to_dt = datetime.utcnow()
    from_dt = to_dt - timedelta(days=365)
    from_str = from_dt.strftime("%Y-%m-%d %H:%M")
    to_str = to_dt.strftime("%Y-%m-%d %H:%M")
    return f"sma/request?type=search&conid={conid}&from={from_str}&to={to_str}&bar_size=1D&tz=-60"

async def fetch_all_for_conid(client, conid):
    widgets = "objective,mstar,lipper_ratings,mf_key_ratios,risk_and_statistics,holdings,performance_and_peers,keyProfile,ownership,dividends,tear_sheet,news,fund_mstar,mf_esg,social_sentiment,securities_lending,sv,short_sale,ukuser"

    # 1) Fetch landing first (cheap completeness probe and feature source).
    landing_status, landing = await fetch_json(client, f"landing/{conid}?widgets={widgets}")
    if landing_status != 200 or not isinstance(landing, dict):
        return {
            "conid": conid,
            "landing_ok": False,
            "landing_sections": {f"landing_{k}": False for k in LANDING_SECTIONS},
            "endpoints": {},
        }

    # 2) Fetch all other endpoints (with period fallback for perf/risk).
    endpoint_tasks = {}
    for name, path in ENDPOINTS.items():
        endpoint_tasks[name] = fetch_json(client, path.format(conid=conid))
    endpoint_tasks["sma_search"] = fetch_json(client, build_sma_search_endpoint(conid))
    endpoint_tasks["perf"] = fetch_with_period_fallback(client, conid, "perf")
    endpoint_tasks["risk"] = fetch_with_period_fallback(client, conid, "risk")

    names = list(endpoint_tasks.keys())
    responses = await asyncio.gather(*endpoint_tasks.values(), return_exceptions=True)

    endpoint_results = {}
    for name, response in zip(names, responses):
        if isinstance(response, AuthError):
            raise response
        if isinstance(response, Exception):
            endpoint_results[name] = {
                "status": 0,
                "has_data": False,
                "selected_period": "",
                "attempts": "",
            }
            continue

        if name in ["perf", "risk"]:
            endpoint_results[name] = response
            continue

        status, payload = response
        endpoint_results[name] = {
            "status": status,
            "has_data": status == 200 and has_data(payload, name),
            "selected_period": "",
            "attempts": "",
        }

    landing_sections = {f"landing_{k}": landing_has_section_data(landing, k) for k in LANDING_SECTIONS}

    return {
        "conid": conid,
        "landing_ok": True,
        "landing_sections": landing_sections,
        "endpoints": endpoint_results,
    }


def _corr_row(df, feature_col, target_col):
    with_mask = df[feature_col] == True
    without_mask = df[feature_col] == False
    p_with = df.loc[with_mask, target_col].mean() if with_mask.any() else float("nan")
    p_without = df.loc[without_mask, target_col].mean() if without_mask.any() else float("nan")
    return {
        "feature": feature_col,
        "target": target_col,
        "n_with": int(with_mask.sum()),
        "n_without": int(without_mask.sum()),
        "p_target_given_feature": p_with,
        "p_target_given_not_feature": p_without,
        "lift": p_with - p_without if pd.notna(p_with) and pd.notna(p_without) else float("nan"),
    }


async def main(sample_size=50, sleep_s=0.25, random_sample=True):
    session = IBKRSession()

    with get_connection() as conn:
        cursor = conn.cursor()
        if random_sample:
            cursor.execute("SELECT conid FROM instruments ORDER BY RANDOM() LIMIT ?", (sample_size,))
        else:
            cursor.execute("SELECT conid FROM instruments LIMIT ?", (sample_size,))
        conids = [row[0] for row in cursor.fetchall()]

    if not conids:
        logger.error("No conids found in database. Run pystocks/fundamentals.py first to sync.")
        return

    rows = []
    async with session.get_client() as client:
        for conid in tqdm(conids, desc="Researching correlations"):
            try:
                data = await fetch_all_for_conid(client, conid)
            except AuthError as e:
                logger.error(str(e))
                logger.error("Stopping due to authentication error.")
                break

            row = {"conid": conid, "landing_ok": data["landing_ok"], **data["landing_sections"]}
            for endpoint, info in data["endpoints"].items():
                row[f"yield_{endpoint}"] = info["has_data"]
                row[f"status_{endpoint}"] = info["status"]
                if endpoint in ["perf", "risk"]:
                    row[f"period_{endpoint}"] = info["selected_period"]
                    row[f"attempts_{endpoint}"] = info["attempts"]
            rows.append(row)
            await asyncio.sleep(sleep_s)

    if not rows:
        logger.warning("No rows collected.")
        return

    # Raw output per conid
    df = pd.DataFrame(rows)
    df.to_csv(RESEARCH_YIELDS_PATH, index=False)
    logger.info(f"Saved raw results to {RESEARCH_YIELDS_PATH}")

    # Correlation-style summary: P(endpoint_has_data | landing_section_has_data)
    feature_cols = [c for c in df.columns if c.startswith("landing_")]
    target_cols = [c for c in df.columns if c.startswith("yield_")]
    if not feature_cols:
        logger.warning("No landing feature columns found; skipping correlation summary.")
        return
    if not target_cols:
        logger.warning("No endpoint yield columns found; skipping correlation summary.")
        return

    corr_rows = []
    for target in target_cols:
        for feature in feature_cols:
            corr_rows.append(_corr_row(df, feature, target))
    if not corr_rows:
        logger.warning("No correlation rows produced; skipping summary output.")
        return

    summary_df = pd.DataFrame(corr_rows).sort_values(["target", "lift"], ascending=[True, False])
    summary_df.to_csv(RESEARCH_CORR_SUMMARY_PATH, index=False)
    logger.info(f"Saved summary to {RESEARCH_CORR_SUMMARY_PATH}")

    print("--- Endpoint Yield Rates ---")
    for target in target_cols:
        print(f"{target}: {df[target].mean():.1%}")

    for endpoint in ["perf", "risk"]:
        period_col = f"period_{endpoint}"
        if period_col in df.columns:
            counts = df[period_col].value_counts(dropna=False)
            print(f"\n--- {endpoint} selected period counts ---")
            for period, count in counts.items():
                label = period if period else "<none>"
                print(f"{label}: {count}")

if __name__ == "__main__":
    import fire
    fire.Fire(main)
