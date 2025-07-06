# Prep functions
import ast
import re
import pandas as pd
import os
from datetime import datetime

def evaluate_literal(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val
    
def load(path):
    df = pd.read_csv(path)
    for col in df.columns:
        df[col] = df[col].apply(evaluate_literal)
        # if col == 'date_scraped':
        #     df[col] = pd.to_datetime(df[col])#.dt.date
    return df

def save(df):
    final_df = df[df.apply(is_row_valid, axis=1)]
    final_df = clean_df(final_df)

    file_path = 'data/contract_elaborated.csv'
    temp_file_path = file_path + '.tmp'
    try:
        temp_df = load('data/contract_elaborated.csv')
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
        # return True # Comment out for more rigid filter
        return False 
    if is_numerical(value):
        return True
    
    if column == 'profile':
        # if value and label:
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
            # if col == 'fundamentals':
            #     if len(row[col]) not in [4,5,21,22,   23]: #4, 5, 21, 22 are the acceptable num of fund values, 23 is for little bugs
            #         print(len(row[col]))
            #         return False
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
    return label
    
def correct_digit(value_str):
    try:
        digit = re.sub(r'[^\d.-]', '', value_str).strip()
        return float(digit)
    except Exception:
        return value_str

def clean_values(value_str, col):
    # print(value_str)
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