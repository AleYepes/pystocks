import pandas as pd
import numpy as np
from .config import DATA_DIR, PREPROCESSED_DIR, TRADES_DIR
from .utils import load_csv
from datetime import datetime, timedelta
from tqdm.auto import tqdm
import os

class PortfolioAnalyzer:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.fundamentals_df = None
        self.series_meta = []

    def load_latest_fundamentals(self):
        files = list(PREPROCESSED_DIR.glob("fundamentals_*.csv"))
        if not files:
            print("No preprocessed fundamentals found.")
            return None
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        self.fundamentals_df = load_csv(latest_file)
        print(f"Loaded latest fundamentals from {latest_file.name}")
        return self.fundamentals_df

    def load_historical_series(self):
        series_dir = TRADES_DIR / "series"
        if not series_dir.exists():
            print("No historical series directory found.")
            return
            
        files = list(series_dir.glob("*.csv"))
        for f in tqdm(files, desc="Loading price series"):
            try:
                df = load_csv(f)
                # Basic cleaning
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                
                parts = f.stem.split('-')
                if len(parts) >= 3:
                    symbol, exchange, currency = parts[0], parts[1], parts[2]
                    self.series_meta.append({
                        'symbol': symbol,
                        'exchange': exchange,
                        'currency': currency,
                        'df': df
                    })
            except Exception as e:
                print(f"Error loading {f.name}: {e}")

    def run_factor_analysis(self):
        """
        Implementation of the regression-based factor analysis.
        This would be a large refactor of src/6.analysis.ipynb logic.
        """
        print("Starting factor analysis...")
        # ... (Placeholder for the complex regression logic)
        pass

if __name__ == "__main__":
    analyzer = PortfolioAnalyzer()
    analyzer.load_latest_fundamentals()
    analyzer.load_historical_series()
    analyzer.run_factor_analysis()
