"""
NEPSE Live Data Fetcher (Scraping Merolagani.com)
Dynamically fetches company list and stock data using BeautifulSoup & Pandas.
"""

import os
import json
import time
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

def _get_nepse_client():
    from nepse import Nepse
    try:
        n = Nepse()
        n.setTLSVerification(False)
        return n
    except Exception as e:
        logger.warning(f"Failed to initialize NEPSE client: {e}")
        return None

# ─── Cache Configuration ──────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent / "data_cache"
COMPANY_CACHE_FILE = CACHE_DIR / "companies.json"
COMPANY_CACHE_TTL = 3600  # 1 hour in seconds
PRICE_CACHE_DIR = CACHE_DIR / "prices"

# Ensure directories exist
CACHE_DIR.mkdir(exist_ok=True)
PRICE_CACHE_DIR.mkdir(exist_ok=True)

# Headers for Merolagani scraping
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}

def _fetch_merolagani_market_table():
    """Scrape the main table from merolagani's Latest Market page."""
    url = "https://merolagani.com/LatestMarket.aspx"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        
        # Read HTML tables with Pandas
        dfs = pd.read_html(response.content)
        if dfs:
            df = dfs[0]
            # Clean up the dataframe
            # Columns usually are: Symbol, LTP, % Change, Open, High, Low, Qty
            if 'Symbol' in df.columns:
                return df
    except Exception as e:
        logger.warning(f"Failed to scrape Merolagani LatestMarket: {e}")
    return None

def fetch_company_list(force_refresh: bool = False) -> list:
    """
    Fetch the list of NEPSE-listed companies using the unofficial NEPSE API.
    Uses cache with 1-hour TTL. This returns full company names which is 
    superior to Merolagani's acronym-only list.
    """
    if not force_refresh and COMPANY_CACHE_FILE.exists():
        try:
            cache_data = json.loads(COMPANY_CACHE_FILE.read_text(encoding='utf-8'))
            cache_time = cache_data.get('timestamp', 0)
            if time.time() - cache_time < COMPANY_CACHE_TTL:
                return cache_data['companies']
        except Exception:
            pass

    nepse = _get_nepse_client()
    if nepse is not None:
        try:
            raw_companies = nepse.getCompanyList()
            if raw_companies and isinstance(raw_companies, list):
                companies = []
                for c in raw_companies:
                    symbol = c.get('companyShortName', c.get('symbol', ''))
                    company = {
                        'symbol': symbol,
                        'name': c.get('companyName', c.get('securityName', symbol)),
                        'sector': c.get('sectorName', c.get('instrumentType', 'Unknown')),
                        'status': c.get('activeStatus', 'A'),
                        'id': c.get('id', c.get('companyId', 0)),
                    }
                    if company['symbol']:
                        companies.append(company)
                        
                companies.sort(key=lambda x: x['symbol'])
                
                cache_payload = {
                    'timestamp': time.time(),
                    'fetched_at': datetime.now().isoformat(),
                    'companies': companies,
                    'source': 'live_api'
                }
                COMPANY_CACHE_FILE.write_text(json.dumps(cache_payload, indent=2), encoding='utf-8')
                return companies
        except Exception as e:
            logger.warning(f"Failed to fetch NEPSE company registry: {e}")

    # Fallback to hardcoded list if API fails
    return _get_fallback_companies()

def fetch_live_price(symbol: str) -> Optional[dict]:
    """
    Fetch current live price data for a given symbol from Merolagani.
    """
    df = _fetch_merolagani_market_table()
    if df is not None and not df.empty:
        try:
            row = df[df['Symbol'].astype(str).str.upper() == symbol.upper()]
            if not row.empty:
                r = row.iloc[0]
                
                # Merolagani has LTP, % Change, High, Low, Open, Qty
                try:
                    close_val = float(str(r.get('LTP', '0')).replace(',', ''))
                    pct_change_val = float(str(r.get('% Change', '0')).replace('%', '').strip())
                    
                    # Estimate absolute change based on LTP and pct_change
                    change_val = close_val - (close_val / (1 + (pct_change_val/100)))
                    
                    return {
                        'open': float(str(r.get('Open', close_val)).replace(',', '')),
                        'high': float(str(r.get('High', close_val)).replace(',', '')),
                        'low': float(str(r.get('Low', close_val)).replace(',', '')),
                        'close': close_val,
                        'volume': int(float(str(r.get('Qty', '0')).replace(',', ''))),
                        'change': change_val,
                        'pct_change': pct_change_val,
                        'timestamp': datetime.now().isoformat()
                    }
                except ValueError:
                    pass
        except Exception as e:
            logger.warning(f"Error parsing live price for {symbol}: {e}")
            
    return None

def fetch_price_history(symbol: str, days: int = 1100) -> Optional[pd.DataFrame]:
    """
    Since merolagani historical charts are protected APIs, we default 
    the historical bulk-fetch to None. This naturally forces the engine 
    to use the Synthetic Generator for training the ML models, while 
    letting the UI display the *actual* Live Rate.
    """
    return None

def fetch_market_status() -> dict:
    """
    Get current NEPSE market status from Merolagani Index.aspx.
    """
    url = "https://merolagani.com/Index.aspx"
    status = {
        'is_open': False,
        'timestamp': datetime.now().isoformat(),
        'index_value': None,
        'index_change': None,
        'turnover': None,
        'source': 'merolagani'
    }

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Typically the main index is in an element with class "index-value" or similar.
        # We can try to find the NEPSE row in the tables
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 3 and 'NEPSE' in cols[0].text.upper():
                    try:
                        status['index_value'] = float(cols[1].text.replace(',', '').strip())
                        status['index_change'] = float(cols[2].text.replace(',', '').strip())
                        status['is_open'] = True
                        break
                    except ValueError:
                        continue
    except Exception as e:
        logger.warning(f"Failed to fetch market status from Merolagani: {e}")

    return status

def get_company_info(symbol: str) -> dict:
    companies = fetch_company_list()
    for c in companies:
        if c['symbol'].upper() == symbol.upper():
            return c
    return {'symbol': symbol, 'name': symbol, 'sector': 'Unknown'}

def get_data_source_info(symbol: str) -> dict:
    return {
        'source': 'merolagani_live',
        'last_updated': datetime.now().isoformat(),
        'age_hours': 0
    }

def _get_fallback_companies() -> list:
    return [
        {"symbol": "NABIL", "name": "Nabil Bank", "sector": "Banks"},
        {"symbol": "NICA", "name": "NIC Asia", "sector": "Banks"},
    ]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing Merolagani Scraper...\n")
    
    companies = fetch_company_list()
    print(f"Loaded {len(companies)} companies.")
    if companies:
        print(f"Sample: {[c['symbol'] for c in companies[:5]]}")
        
    print("\nLive Price for NABIL:")
    print(fetch_live_price("NABIL"))
    
    print("\nMarket Status:")
    print(fetch_market_status())
