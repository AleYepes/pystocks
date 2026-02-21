import sqlite3
from datetime import datetime
from .config import SQLITE_DB_PATH

def get_connection():
    return sqlite3.connect(SQLITE_DB_PATH)

def init_db():
    """Initializes the SQLite database with required tables."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Table for instrument registry and metadata
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS instruments (
                conid TEXT PRIMARY KEY,
                symbol TEXT,
                exchange TEXT,
                isin TEXT,
                last_scraped_fundamentals DATE,
                last_status_fundamentals TEXT
            )
        ''')
        
        # Table for scraping logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scraper_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conid TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                endpoint TEXT,
                status_code INTEGER,
                error_message TEXT
            )
        ''')
        
        conn.commit()

def log_scrape(conid, endpoint, status_code, error_message=None):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO scraper_logs (conid, endpoint, status_code, error_message)
            VALUES (?, ?, ?, ?)
        ''', (conid, endpoint, status_code, error_message))
        
        if status_code == 200:
            cursor.execute('''
                UPDATE instruments 
                SET last_scraped_fundamentals = ?, last_status_fundamentals = 'success'
                WHERE conid = ?
            ''', (datetime.now().strftime("%Y-%m-%d"), conid))
        else:
            cursor.execute('''
                UPDATE instruments 
                SET last_status_fundamentals = ?
                WHERE conid = ?
            ''', (f'error_{status_code}', conid))
            
        conn.commit()

def sync_instruments_from_csv(csv_path):
    """Syncs instruments from the products CSV into the SQLite DB."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    df['conid'] = df['conid'].astype(str)
    
    with get_connection() as conn:
        cursor = conn.cursor()
        for _, row in df.iterrows():
            cursor.execute('''
                INSERT OR IGNORE INTO instruments (conid, symbol, exchange, isin)
                VALUES (?, ?, ?, ?)
            ''', (row['conid'], row['symbol'], row['exchangeId'], row.get('isin', '')))
        conn.commit()
