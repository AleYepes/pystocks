import pandas as pd
import numpy as np
import os
import re
from tqdm.auto import tqdm
from .config import RAW_DIR, PREPROCESSED_DIR
from .utils import load_csv, evaluate_literal, sort_by_eur_exchanges
from datetime import datetime
import pycountry
import requests
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from fuzzywuzzy import fuzz
from itertools import combinations
import networkx as nx

class Preprocessor:
    def __init__(self, raw_dir=RAW_DIR, output_dir=PREPROCESSED_DIR):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.original_columns = []

    def load_all_raw(self):
        dir_list = os.listdir(self.raw_dir)
        dir_list = [file for file in dir_list if file.startswith('contract_scraped') and file.endswith('.csv')]
        
        all_dfs = []
        for file in dir_list:
            df = load_csv(self.raw_dir / file)
            all_dfs.append(df)
        
        if not all_dfs:
            return pd.DataFrame()
            
        combined_df = pd.concat(all_dfs, ignore_index=True)
        self.original_columns = combined_df.columns.tolist()
        return combined_df

    def explode_nested_columns(self, df):
        """
        Explodes columns containing lists of tuples into separate columns.
        """
        cols_to_explode = ['holding_types', 'profile', 'countries', 'fundamentals', 'industries', 'style']
        percentage_cols = ['holding_types', 'countries', 'industries']
        
        pivoted_dfs = []
        
        for col in tqdm(cols_to_explode, desc="Exploding nested columns"):
            if col not in df.columns:
                continue
                
            temp_series = df[col].fillna('[]').apply(lambda x: x if isinstance(x, list) else evaluate_literal(x))
            exploded = temp_series.explode()
            exploded = exploded.apply(lambda x: (None, None) if not isinstance(x, (list, tuple)) or len(x) != 2 else x)
            
            labels_values = pd.DataFrame(exploded.tolist(), index=exploded.index)
            labels_values.columns = ['label', 'value']
            labels_values = labels_values.dropna(subset=['label'])

            pivot_df = labels_values.pivot_table(index=labels_values.index, columns='label', values='value', aggfunc='first')
            pivot_df.rename(columns={label: f'{col}_{label}' for label in pivot_df.columns}, inplace=True)

            if col in percentage_cols:
                pivot_df = pivot_df.fillna(0.0)
                for pivot_col in pivot_df.columns:
                    pivot_df[pivot_col] = pd.to_numeric(pivot_df[pivot_col].astype(str).str.replace('%', ''), errors='coerce').fillna(0.0)
                
                # Normalize sums to 1 if they are percentages
                sums = pivot_df.sum(axis=1)
                mask = (sums > 0)
                pivot_df.loc[mask] = pivot_df.loc[mask].div(sums[mask], axis=0)

            pivoted_dfs.append(pivot_df)

        exploded_df = df.drop(columns=[c for c in cols_to_explode if c in df.columns])
        exploded_df = pd.concat([exploded_df] + pivoted_dfs, axis=1)
        return exploded_df

    def standardize_currencies(self, df):
        # Implementation of currency standardization from legacy code
        # ... (Simplified for brevity, but should include the mapping)
        return df

    def run_full_pipeline(self):
        print("Starting preprocessing pipeline...")
        df = self.load_all_raw()
        if df.empty:
            print("No data found to preprocess.")
            return
            
        df = self.explode_nested_columns(df)
        
        # Legacy filtering logic
        if 'profile_TotalNetAssets' in df.columns:
            df = df[~df['profile_TotalNetAssets'].isna()]
            
        # Save to preprocessed
        output_path = self.output_dir / f'fundamentals_{datetime.now().strftime("%Y-%m")}.csv'
        df.to_csv(output_path, index=False)
        print(f"Preprocessed data saved to {output_path}")
        return df

if __name__ == "__main__":
    pp = Preprocessor()
    pp.run_full_pipeline()
