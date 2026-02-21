import asyncio
import json
from pystocks.session import IBKRSession
from pathlib import Path

async def test_api():
    session = IBKRSession()
    try:
        async with session.get_client() as client:
            con_id = "756733" # SPY
            url = f"/tws.proxy/fundamentals/mf_holdings/{con_id}"
            print(f"Testing URL: {url}")
            response = await client.get(url)
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                print("Success!")
                data = response.json()
                print(f"Keys in response: {list(data.keys())}")
                if "industry" in data:
                    print(f"Industry count: {len(data['industry'])}")
                if "investor_country" in data:
                    print(f"Country count: {len(data['investor_country'])}")
                with open("holdings_sample.json", "w") as f:
                    json.dump(data, f, indent=2)
                print("Sample saved to holdings_sample.json")
            else:
                print(f"Failed: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_api())
