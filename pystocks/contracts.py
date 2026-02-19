import pandas as pd
import numpy as np
from ib_async import *
from tqdm.auto import tqdm
from .config import CONTRACTS_DB_PATH, IB_PRODUCTS_PATH
from .utils import load_csv, sort_by_eur_exchanges
import asyncio

async def get_contract_details_async(symbols_df, ib):
    """
    Fetch contract details for a list of symbols using ib_async.
    """
    details_dfs = []
    
    # Identify which columns we need for merging/checking
    merge_cols = ['symbol', 'currency', 'exchange']
    for col in merge_cols:
        if col not in symbols_df.columns:
            raise ValueError(f"Required column {col} missing from input DataFrame")

    for _, row in tqdm(symbols_df.iterrows(), total=len(symbols_df), desc="Fetching contract details"):
        symbol = str(row['symbol'])
        exchange = str(row['exchange']).replace('*', '')
        currency = str(row['currency'])

        # Try specific exchange first
        contract = Stock(symbol, exchange, currency)
        details = await ib.reqContractDetailsAsync(contract)
        
        # If not found, try SMART
        if not details:
            contract = Stock(symbol, 'SMART', currency)
            details = await ib.reqContractDetailsAsync(contract)

        if details:
            for d in details:
                # Extract contract and details
                c = d.contract
                # Create a dict of relevant fields
                data = {
                    'conId': c.conId,
                    'symbol': c.symbol,
                    'exchange': c.exchange,
                    'primaryExchange': c.primaryExchange,
                    'currency': c.currency,
                    'validExchanges': d.validExchanges,
                    'longName': d.longName,
                    'stockType': d.stockType,
                }
                
                # Add ISIN and other IDs if available
                if d.secIdList:
                    for tag_value in d.secIdList:
                        data[tag_value.tag.lower()] = tag_value.value
                
                details_dfs.append(pd.DataFrame([data]))

    if not details_dfs:
        return pd.DataFrame()

    final_details_df = pd.concat(details_dfs, ignore_index=True).drop_duplicates(subset=['conId'])
    return final_details_df

def update_db(new_details_df, db_path=CONTRACTS_DB_PATH):
    if new_details_df.empty:
        print("No new details to update.")
        return
    
    if db_path.exists():
        old_df = load_csv(db_path)
        combined_df = pd.concat([old_df, new_details_df]).drop_duplicates(subset=['conId'])
    else:
        combined_df = new_details_df
    
    # Sort and save
    combined_df, _ = sort_by_eur_exchanges(combined_df)
    combined_df.to_csv(db_path, index=False)
    print(f"Database updated at {db_path}. Total contracts: {len(combined_df)}")

async def run_update(host='127.0.0.1', port=7497, client_id=2):
    ib = IB()
    try:
        await ib.connectAsync(host, port, clientId=client_id)
        
        if not IB_PRODUCTS_PATH.exists():
            print(f"Product list not found at {IB_PRODUCTS_PATH}. Please run scraper first.")
            return

        products_df = load_csv(IB_PRODUCTS_PATH)
        products_df.columns = products_df.columns.str.lower()
        if 'exchange  *primary exchange' in products_df.columns:
            products_df = products_df.rename(columns={'exchange  *primary exchange': 'exchange'})
        
        # Filter out what we already have in DB
        if CONTRACTS_DB_PATH.exists():
            db_df = load_csv(CONTRACTS_DB_PATH)
            merged = products_df.merge(db_df[['symbol', 'currency', 'exchange']], on=['symbol', 'currency', 'exchange'], how='left', indicator=True)
            unchecked = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
        else:
            unchecked = products_df

        if not unchecked.empty:
            new_details = await get_contract_details_async(unchecked, ib)
            update_db(new_details)
        else:
            print("All products already in database.")
            
    finally:
        ib.disconnect()

if __name__ == "__main__":
    # This allows running the script directly
    import asyncio
    asyncio.run(run_update())
