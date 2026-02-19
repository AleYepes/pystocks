import pandas as pd
import ast
import re
from datetime import datetime
import os
from .config import RAW_DIR

def sort_by_eur_exchanges(input_df, drop=False):
    currency_trade_volume_order = ["EUR", "USD", "JPY", "GBP", "CNY", "AUD", "CAD", "CHF", "SGD", "HKD", "SEK", "NOK", "MXN", "INR", "RUB", "PLN", "TWD", "ZAR", "DKK", "ILS", "MYR", "SAR", "HUF"]
    final_df = input_df.copy()
    final_df['currency_ordered'] = pd.Categorical(final_df['currency'], categories=currency_trade_volume_order, ordered=True)

    manual_eur_exchanges = {'AEB', 'APEXEN', 'AQEUDE', 'AQEUEN', 'AQEUES', 'AQXECH', 'AQXEUK', 'BATECH', 'BATEDE', 'BATEEN', 'BATEES', 'BATEUK', 'BM', 'BVME.ETF', 'CHIXCH', 'CHIXDE', 'CHIXEN', 'CHIXES', 'CHIXUK', 'DXEDE', 'DXEEN', 'DXEES', 'EBS', 'ENEXT.BE', 'EUIBSI', 'FWB', 'FWB2', 'GETTEX', 'GETTEX2', 'IBIS', 'IBIS2', 'LSE', 'LSEETF', 'MEXI', 'SBF', 'SEHK', 'SFB', 'SMART', 'SWB', 'SWB2', 'TGATE', 'TGHEDE', 'TGHEEN', 'TRQXCH', 'TRQXDE', 'TRQXEN', 'TRQXUK', 'TRWBCH', 'TRWBDE', 'TRWBEN', 'TRWBIT', 'TRWBSE', 'TRWBUK', 'TRWBUKETF', 'VSE'}
    
    # Check if required columns exist
    if 'longName' in final_df.columns:
        ucits_df = final_df[final_df['longName'].str.contains("UCITS", na=False)]
    else:
        ucits_df = pd.DataFrame(columns=final_df.columns)

    valid_exchanges = set()
    if 'validExchanges' in ucits_df.columns:
        valid_exchanges = set(ucits_df['validExchanges'].astype(str).str.split(',').explode().unique())
    
    exchanges = set(ucits_df['exchange'].unique()) if 'exchange' in ucits_df.columns else set()
    prims = set(ucits_df['primaryExchange'].unique()) if 'primaryExchange' in ucits_df.columns else set()
    
    eur_exchanges = valid_exchanges | exchanges | prims | manual_eur_exchanges

    if 'exchange' in final_df.columns:
        final_df['exchange_is_european'] = final_df['exchange'].isin(eur_exchanges)
    else:
        final_df['exchange_is_european'] = False

    if 'primaryExchange' in final_df.columns:
        final_df['primary_is_european'] = final_df['primaryExchange'].isin(eur_exchanges)
    else:
        final_df['primary_is_european'] = False

    sort_cols = ['exchange_is_european', 'primary_is_european', 'currency_ordered']
    if 'symbol' in final_df.columns:
        sort_cols.append('symbol')
        
    final_df = (final_df
                .sort_values(by=sort_cols, ascending=[False, False, True, True] if 'symbol' in sort_cols else [False, False, True])
                .drop(columns=['exchange_is_european', 'primary_is_european', 'currency_ordered'])
                )
    
    if drop and 'primaryExchange' in final_df.columns:
        return final_df[final_df['primaryExchange'].isin(eur_exchanges)], eur_exchanges
    else:
        return final_df, eur_exchanges

def evaluate_literal(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val
    
def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist.")
    df = pd.read_csv(path)
    # Heuristic to detect columns that need eval
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check first non-null value to see if it looks like a list or tuple
            first_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if isinstance(first_val, str) and (first_val.startswith('[') or first_val.startswith('(')):
                try:
                    df[col] = df[col].apply(evaluate_literal)
                except:
                    pass
    return df

def save_fundamentals(df, root=RAW_DIR):
    # Logic adapted from src/functions.py save() but simplified and made robust
    # In the original code, it appended to a monthly file.
    
    file_path = root / f'contract_scraped_{datetime.now().strftime("%y-%m")}.csv'
    
    if file_path.exists():
        try:
            existing_df = load_csv(file_path)
            # Combine and drop duplicates based on conId and funds_date
            # Assuming incoming df has these columns.
            combined_df = pd.concat([existing_df, df])
            if 'conId' in combined_df.columns and 'funds_date' in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=['conId', 'funds_date'])
            else:
                combined_df = combined_df.drop_duplicates()
            combined_df.to_csv(file_path, index=False)
        except Exception as e:
            print(f"Error appending to {file_path}: {e}. Saving to new file.")
            df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, index=False)
