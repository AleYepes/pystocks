# Prep functions
import ast
import re
import pandas as pd
import os
from datetime import datetime

def sort_by_eur_exchanges(input_df, drop=False):
    currency_trade_volume_order = ["EUR", "USD", "JPY", "GBP", "CNY", "AUD", "CAD", "CHF", "SGD", "HKD", "SEK", "NOK", "MXN", "INR", "RUB", "PLN", "TWD", "ZAR", "DKK", "ILS", "MYR", "SAR", "HUF"]
    final_df = input_df.copy()
    final_df['currency_ordered'] = pd.Categorical(final_df['currency'], categories=currency_trade_volume_order, ordered=True)

    # manual_eur_exchanges = {'AEB', 'AEQLIT', 'APEXEN', 'AQEUDE', 'AQEUEN', 'AQEUES', 'AQXECH', 'AQXEUK', 'BATECH', 'BATEDE', 'BATEEN', 'BATEES', 'BATEUK', 'BATS', 'BM', 'BUX', 'BVME.ETF', 'CBOE', 'CHIXCH', 'CHIXDE', 'CHIXEN', 'CHIXES', 'CHIXUK', 'CPH', 'DXEDE', 'DXEEN', 'DXEES', 'EBS', 'ENEXT.BE', 'EUIBSI', 'FWB', 'FWB2', 'GETTEX', 'GETTEX2', 'IBEOS', 'IBIS', 'IBIS2', 'LJSE', 'LSE', 'LSEETF', 'MEXI', 'PEARL', 'PSX', 'SBF', 'SFB', 'SMART', 'SWB', 'SWB2', 'TADAWUL', 'TASE', 'TGATE', 'TGHEDE', 'TGHEEN', 'TRQXCH', 'TRQXDE', 'TRQXEN', 'TRQXUK', 'TRWBCH', 'TRWBDE', 'TRWBEN', 'TRWBIT', 'TRWBSE', 'TRWBUK', 'TRWBUKETF', 'VSE', 'WSE'}

    manual_eur_exchanges = {'AEB', 'APEXEN', 'AQEUDE', 'AQEUEN', 'AQEUES', 'AQXECH', 'AQXEUK', 'BATECH', 'BATEDE', 'BATEEN', 'BATEES', 'BATEUK', 'BM', 'BVME.ETF', 'CHIXCH', 'CHIXDE', 'CHIXEN', 'CHIXES', 'CHIXUK', 'DXEDE', 'DXEEN', 'DXEES', 'EBS', 'ENEXT.BE', 'EUIBSI', 'FWB', 'FWB2', 'GETTEX', 'GETTEX2', 'IBIS', 'IBIS2', 'LSE', 'LSEETF', 'MEXI', 'SBF', 'SEHK', 'SFB', 'SMART', 'SWB', 'SWB2', 'TGATE', 'TGHEDE', 'TGHEEN', 'TRQXCH', 'TRQXDE', 'TRQXEN', 'TRQXUK', 'TRWBCH', 'TRWBDE', 'TRWBEN', 'TRWBIT', 'TRWBSE', 'TRWBUK', 'TRWBUKETF', 'VSE'}
    
    ucits_df = final_df[final_df['longName'].str.contains("UCITS", na=False)]
    valid_exchanges = set(ucits_df['validExchanges'].str.split(',').explode().unique())
    exchanges = set(ucits_df['exchange'].unique())
    prims = set(ucits_df['primaryExchange'].unique())
    eur_exchanges = valid_exchanges | exchanges | prims | manual_eur_exchanges

    final_df = (final_df
                .assign(exchange_is_european=final_df['exchange'].isin(eur_exchanges))
                .assign(primary_is_european=final_df['primaryExchange'].isin(eur_exchanges))
                .sort_values(by=['exchange_is_european', 'primary_is_european', 'currency_ordered', 'symbol'], ascending=[False, False, True, True])
                .drop(columns=['exchange_is_european', 'primary_is_european', 'currency_ordered'])
                )
    
    if drop:
        return final_df[final_df['primaryExchange'].isin(eur_exchanges)], eur_exchanges
    else:
        return final_df, eur_exchanges

def evaluate_literal(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val
    
def load(path):
    df = pd.read_csv(path)
    for col in df.columns:
        df[col] = df[col].apply(evaluate_literal)
    return df

def save(df, root='data/raw/'):
    final_df = df[df.apply(is_row_valid, axis=1)]
    final_df = clean_df(final_df)

    file_path = f'{root}contract_scraped_{datetime.now().strftime("%y-%m")}.csv'
    temp_file_path = file_path + '.tmp'
    try:
        temp_df = load(file_path)
        temp_df = clean_df(temp_df)
        final_df = pd.concat([final_df, temp_df]).drop_duplicates(subset=['conId', 'funds_date'])
    except FileNotFoundError:
        pass

    # Filter out the duplicates with 'exact_search' is False
    duplicates_df = final_df[final_df.duplicated(subset=['conId', 'funds_date'], keep=False)]
    final_df = final_df.drop(duplicates_df[duplicates_df['exact_search'] == False].index)

    final_df.to_csv(temp_file_path, index=False)
    os.rename(temp_file_path, file_path)

def is_numerical(val):
    try:
        val = str(val).replace('%', '')
        float(val)
        return True
    except Exception:
        return False

def is_valid_tuple(tuple, column):
    def extract_float(value):
        match = re.match(r'[^0-9]*([0-9.,]+)', value)
        if match:
            return float(match.group(1).replace(',', ''))
        return None
    
    label, value = tuple
    if not isinstance(label, str): # keep
        # if label != None: # Comment out for more rigid filter
        return False
    if value is None:
        return True # Comment out for more rigid filter
        return False 
    if is_numerical(value):
        return True
    
    if column == 'profile':
        return True
    if column == 'fundamentals':
        if value.isupper():
            return True
    if column == 'dividends':
        if value == 'Unknown':
            return True
        extract_float_value = extract_float(value)
        if extract_float_value is not None:
            return True
    if column == 'style':
        if isinstance(value, bool):
            return True
    return False

def is_row_valid(row):
    for col in row.index:
        if isinstance(row[col], list):
            if col == 'debtors':
                return True
            for tuple in row[col]:
                if not is_valid_tuple(tuple, col):
                    print(tuple)
                    return False
    return True

def has_bad_multiplier(long_name):
    multiplier_pattern = re.compile(r'\d+X|X\d+')
    cleaned = long_name.replace('-', ' ').replace('+', ' ')

    for word in cleaned.split():
        if multiplier_pattern.fullmatch(word):
            numeric_part = word.replace('X', '', 1)
            if int(numeric_part) != 1:
                return True

    return False

# Cleaning functions
def clean_labels(label, col):
    if col == 'industries':
        if isinstance(label, str):
            if label.endswith('-Discontinuedeff09/19/2020'):
                return label.split('-')[0]
        return label
    elif col == 'holding_types':
        if isinstance(label, str):
            if label.startswith('■'):
                return label[1:]
            elif label.startswith('1'):
                return label[1:]
        return label
    elif col == 'debtors':
        if isinstance(label, str):
            if ('（') in label:
                return label.replace('（', '(')
        return label
    elif col == 'fundamentals':
        if isinstance(label, str):
            if label == 'LTDebt/ShareholdersEquity':
                return 'LTDebt/Shareholders'
        return label
    elif col == 'holding_types':
        if isinstance(label, str):
            if label == 'IEquity':
                return 'Equity'
        return label
    return label
    
def correct_digit(value_str):
    try:
        digit = re.sub(r'[^\d.-]', '', value_str).strip()
        return float(digit)
    except Exception:
        return value_str

def clean_values(value_str, col):
    if col == 'profile':
        return value_str
    if isinstance(value_str, str):
        if value_str.endswith('%'):
            return correct_digit(value_str.replace('%',''))/100
        try:
            return correct_digit(value_str)
        except Exception:
            return value_str
    return value_str

def clean_df(df):
    for col in df.columns:
        # print(col)
        df[col] = df[col].apply(evaluate_literal)
        df[col] = df[col].apply(lambda x: [(clean_labels(item[0], col), item[1]) if isinstance(item, tuple) and len(item) == 2 else item for item in x] if isinstance(x, list) else x)
        df[col] = df[col].apply(lambda x: [(item[0], clean_values(item[1], col)) if isinstance(item, tuple) and len(item) == 2 else item for item in x] if isinstance(x, list) else x)
        df[col] = df[col].apply(lambda x: sorted(x, key=lambda item: item[0] if isinstance(item, tuple) and item[0] else '') if isinstance(x, list) else x)
    return df