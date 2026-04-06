"""
NEPSE Stock Analysis Engine
Dynamically fetches live NEPSE data and performs comprehensive time series analysis.
Falls back to synthetic data generation when live data is unavailable.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# ─── Dynamic Company Registry ─────────────────────────────────────────────────

# Cache for the session
_company_cache = None
_company_cache_time = 0
_COMPANY_CACHE_TTL = 3600  # 1 hour in-memory cache


def get_company_registry() -> dict:
    """
    Get all NEPSE-listed companies dynamically.
    Returns dict keyed by ticker symbol with company metadata.
    """
    global _company_cache, _company_cache_time
    import time

    # Check in-memory cache
    if _company_cache and (time.time() - _company_cache_time < _COMPANY_CACHE_TTL):
        return _company_cache

    try:
        from nepse_data_fetcher import fetch_company_list
        companies = fetch_company_list()

        registry = {}
        for c in companies:
            symbol = c.get('symbol', '')
            if not symbol:
                continue
            registry[symbol] = {
                'name': c.get('name', symbol),
                'sector': c.get('sector', 'Unknown'),
                'base_price': 500,  # Default, will be overridden by live data
                'volatility': 0.022,
                'trend': 0.0002,
                'volume_base': 30000,
                'beta': 1.0,
                'status': c.get('status', 'A'),
                'id': c.get('id', 0),
            }

        if registry:
            _company_cache = registry
            _company_cache_time = time.time()
            logger.info(f"Loaded {len(registry)} companies into registry")
            return registry

    except ImportError:
        logger.warning("nepse_data_fetcher not available, using fallback registry")
    except Exception as e:
        logger.warning(f"Error loading dynamic registry: {e}")

    # Fallback to hardcoded companies
    return _get_fallback_registry()


def _get_fallback_registry() -> dict:
    """Hardcoded fallback company registry for when API is unavailable."""
    return {
        "NABIL": {"name": "Nabil Bank Limited", "sector": "Commercial Banks", "base_price": 1150, "volatility": 0.018, "trend": 0.0003, "volume_base": 25000, "beta": 1.2},
        "NICA": {"name": "NIC Asia Bank Limited", "sector": "Commercial Banks", "base_price": 485, "volatility": 0.020, "trend": 0.0002, "volume_base": 40000, "beta": 1.1},
        "NLIC": {"name": "Nepal Life Insurance Company", "sector": "Life Insurance", "base_price": 1640, "volatility": 0.022, "trend": 0.0004, "volume_base": 15000, "beta": 1.3},
        "UPPER": {"name": "Upper Tamakoshi Hydropower", "sector": "Hydropower", "base_price": 315, "volatility": 0.025, "trend": 0.0005, "volume_base": 60000, "beta": 1.4},
        "SHIVM": {"name": "Shivam Cements Limited", "sector": "Manufacturing", "base_price": 278, "volatility": 0.028, "trend": 0.0001, "volume_base": 20000, "beta": 0.9},
        "SANIMA": {"name": "Sanima Bank Limited", "sector": "Commercial Banks", "base_price": 348, "volatility": 0.019, "trend": 0.00025, "volume_base": 35000, "beta": 1.0},
        "HIDCL": {"name": "Hydroelectricity Investment & Dev. Co.", "sector": "Hydropower", "base_price": 145, "volatility": 0.023, "trend": 0.0003, "volume_base": 80000, "beta": 1.2},
        "SBI": {"name": "Nepal SBI Bank Limited", "sector": "Commercial Banks", "base_price": 265, "volatility": 0.021, "trend": 0.0002, "volume_base": 30000, "beta": 1.05},
        "KBL": {"name": "Kumari Bank Limited", "sector": "Commercial Banks", "base_price": 215, "volatility": 0.022, "trend": 0.00015, "volume_base": 45000, "beta": 1.1},
        "PCBL": {"name": "Prime Commercial Bank Limited", "sector": "Commercial Banks", "base_price": 298, "volatility": 0.020, "trend": 0.0002, "volume_base": 32000, "beta": 1.05},
        "SCB": {"name": "Standard Chartered Bank Nepal", "sector": "Commercial Banks", "base_price": 650, "volatility": 0.016, "trend": 0.0002, "volume_base": 15000, "beta": 0.8},
        "HBL": {"name": "Himalayan Bank Limited", "sector": "Commercial Banks", "base_price": 390, "volatility": 0.019, "trend": 0.0002, "volume_base": 28000, "beta": 1.0},
        "EBL": {"name": "Everest Bank Limited", "sector": "Commercial Banks", "base_price": 520, "volatility": 0.017, "trend": 0.0003, "volume_base": 20000, "beta": 0.9},
        "SBL": {"name": "Siddhartha Bank Limited", "sector": "Commercial Banks", "base_price": 310, "volatility": 0.021, "trend": 0.0002, "volume_base": 35000, "beta": 1.1},
        "GBIME": {"name": "Global IME Bank Limited", "sector": "Commercial Banks", "base_price": 280, "volatility": 0.020, "trend": 0.0002, "volume_base": 50000, "beta": 1.15},
        "ADBL": {"name": "Agriculture Dev. Bank Ltd", "sector": "Commercial Banks", "base_price": 380, "volatility": 0.023, "trend": 0.0001, "volume_base": 25000, "beta": 1.0},
        "NHPC": {"name": "National Hydropower Company", "sector": "Hydropower", "base_price": 62, "volatility": 0.030, "trend": 0.0004, "volume_base": 100000, "beta": 1.5},
        "CHCL": {"name": "Chilime Hydropower Company", "sector": "Hydropower", "base_price": 535, "volatility": 0.020, "trend": 0.0003, "volume_base": 18000, "beta": 1.1},
        "BPCL": {"name": "Butwal Power Company", "sector": "Hydropower", "base_price": 380, "volatility": 0.018, "trend": 0.0002, "volume_base": 15000, "beta": 0.9},
        "NTC": {"name": "Nepal Telecom", "sector": "Trading", "base_price": 780, "volatility": 0.015, "trend": 0.0001, "volume_base": 10000, "beta": 0.7},
    }


# Backward-compatible alias
NEPSE_COMPANIES = None  # Lazy-loaded


def _get_nepse_companies():
    """Lazy-load NEPSE_COMPANIES for backward compatibility."""
    global NEPSE_COMPANIES
    if NEPSE_COMPANIES is None:
        NEPSE_COMPANIES = get_company_registry()
    return NEPSE_COMPANIES


def get_stock_data(ticker: str, years: int = 4) -> pd.DataFrame:
    """
    Get stock data for a given ticker. Tries live data first, falls back to synthetic.
    
    Args:
        ticker: Company symbol (e.g., 'NABIL')
        years: Number of years of data to return
        
    Returns:
        DataFrame with OHLCV data (DateTimeIndex)
    """
    # Try live data first
    try:
        from nepse_data_fetcher import fetch_price_history
        df = fetch_price_history(ticker, days=years * 260)
        if df is not None and len(df) >= 50:
            logger.info(f"Using live data for {ticker}: {len(df)} trading days")
            # Ensure proper column names
            df.columns = [c.strip().title() for c in df.columns]
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            if all(c in df.columns for c in required):
                df['Ticker'] = ticker
                return df
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to get live data for {ticker}: {e}")

    # Fallback to synthetic data
    logger.info(f"Generating synthetic data for {ticker}")
    return generate_synthetic_data(ticker, years)


def generate_synthetic_data(ticker: str, years: int = 4) -> pd.DataFrame:
    """
    Generate realistic synthetic NEPSE OHLCV data using geometric Brownian motion
    with seasonality, market cycles, and NEPSE-specific patterns.
    """
    registry = get_company_registry()

    # Get config from registry or use defaults
    if ticker in registry:
        config = registry[ticker]
    else:
        config = {
            "name": ticker, "sector": "Unknown",
            "base_price": 500, "volatility": 0.022,
            "trend": 0.0002, "volume_base": 30000, "beta": 1.0
        }

    # Seed includes ticker hash AND today's date so predictions evolve daily
    today_seed = int(datetime.today().strftime('%Y%m%d'))
    np.random.seed((hash(ticker) + today_seed) % 10000)

    # NEPSE trading calendar
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365)

    # Generate trading days (Mon-Fri as proxy for NEPSE Sun-Thu)
    all_days = pd.date_range(start_date, end_date, freq='B')
    trading_days = all_days[all_days.dayofweek < 5]
    n = len(trading_days)

    # ── PERSONALITY Generation based on Ticker ───────────────────────────
    # Use ticker hash to create unique behaviors
    t_hash = hash(ticker)
    
    # 1. Base trend (some are growth, some are sideways, some are decliners)
    mu_base = ((t_hash % 100) / 10000) - 0.0001
    mu = config.get("trend", mu_base)
    
    # 2. Volatility (some are stable blue-chips, some are volatile speculative)
    sigma_base = 0.015 + ((t_hash % 50) / 1000)
    sigma = config.get("volatility", sigma_base)
    
    # 3. Market cycle phase (different stocks have bull/bear offset)
    cycle_offset = (t_hash % 300)
    
    dt = 1 / 252
    base_price = config.get("base_price", 500)

    # Base GBM returns
    random_shocks = np.random.normal(0, 1, n)

    # Add seasonality (offset by ticker hash)
    day_of_year = np.array([d.dayofyear for d in trading_days])
    seasonality = (
        0.003 * np.sin(2 * np.pi * (day_of_year + (t_hash % 30)) / 365)
        + 0.001 * np.sin(4 * np.pi * (day_of_year + (t_hash % 10)) / 365)
    )

    # Market cycle (bull/bear phases ~18 months)
    t = np.arange(n)
    market_cycle = 0.005 * np.sin(2 * np.pi * (t + cycle_offset) / (18 * 21))

    # Combine returns
    returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_shocks
    returns += seasonality * 0.1 + market_cycle * 0.1

    # Add occasional jumps
    jump_times = np.random.choice(n, size=int(n * 0.02), replace=False)
    jump_sizes = np.random.normal(0, 0.04, len(jump_times))
    returns[jump_times] += jump_sizes

    # Simulate price series
    log_prices = np.cumsum(returns)
    close_prices = base_price * np.exp(log_prices)

    # ── OHLCV Construction ─────────────────────────────────────────────────
    daily_range = close_prices * sigma * np.random.uniform(0.5, 2.5, n)
    price_band = close_prices * 0.10
    daily_range = np.minimum(daily_range, price_band)

    open_prices = np.zeros(n)
    high_prices = np.zeros(n)
    low_prices = np.zeros(n)

    open_prices[0] = base_price
    for i in range(1, n):
        gap = np.random.normal(0, sigma * 0.5) * close_prices[i-1]
        open_prices[i] = close_prices[i-1] + gap

    for i in range(n):
        oc_min = min(open_prices[i], close_prices[i])
        oc_max = max(open_prices[i], close_prices[i])
        spread = daily_range[i] * 0.5
        low_prices[i] = max(1.0, oc_min - spread * np.random.uniform(0.2, 0.8))
        high_prices[i] = oc_max + spread * np.random.uniform(0.2, 0.8)

    # Volume
    base_vol = config.get("volume_base", 30000)
    price_change = np.abs(returns)
    volume_multiplier = 1 + 2 * (price_change / (price_change.std() + 1e-10))
    monday_effect = np.where(np.array([d.dayofweek for d in trading_days]) == 0, 1.2, 1.0)
    volumes = (base_vol * volume_multiplier * monday_effect *
               np.random.lognormal(0, 0.3, n)).astype(int)

    # Mean-reversion
    window_ma = 20
    for i in range(window_ma, n):
        ma = np.mean(close_prices[i-window_ma:i])
        deviation = (close_prices[i-1] - ma) / ma
        returns[i] += -0.15 * deviation

    log_prices = np.cumsum(returns)
    close_prices = base_price * np.exp(log_prices)

    df = pd.DataFrame({
        'Date': trading_days,
        'Open': np.round(open_prices, 2),
        'High': np.round(high_prices, 2),
        'Low': np.round(low_prices, 2),
        'Close': np.round(close_prices, 2),
        'Volume': volumes,
        'Ticker': ticker
    })
    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index)

    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer comprehensive technical features for ML models."""
    d = df.copy()

    # ── Returns ────────────────────────────────────────────────────────────
    d['Return'] = d['Close'].pct_change()
    d['LogReturn'] = np.log(d['Close'] / d['Close'].shift(1))
    # Target: 3-day forward return smoothed
    # Increased target to 1.5% to make "PROFIT" a meaningful and decisive event.
    forward_return = d['Close'].shift(-3) / d['Close'] - 1
    d['Target'] = (forward_return > 0.015).astype(int)

    # ── Moving Averages ────────────────────────────────────────────────────
    for w in [5, 10, 12, 20, 26, 50, 200]:
        d[f'SMA_{w}'] = d['Close'].rolling(w).mean()
        d[f'EMA_{w}'] = d['Close'].ewm(span=w, adjust=False).mean()

    # ── Volatility ─────────────────────────────────────────────────────────
    d['Volatility_10'] = d['Return'].rolling(10).std() * np.sqrt(252)
    d['Volatility_20'] = d['Return'].rolling(20).std() * np.sqrt(252)
    d['ATR'] = pd.concat([
        d['High'] - d['Low'],
        (d['High'] - d['Close'].shift(1)).abs(),
        (d['Low'] - d['Close'].shift(1)).abs()
    ], axis=1).max(axis=1).rolling(14).mean()

    # ── Momentum ───────────────────────────────────────────────────────────
    d['Momentum_5'] = d['Close'] / d['Close'].shift(5) - 1
    d['Momentum_10'] = d['Close'] / d['Close'].shift(10) - 1
    d['Momentum_20'] = d['Close'] / d['Close'].shift(20) - 1

    # ── RSI ────────────────────────────────────────────────────────────────
    delta = d['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d['RSI'] = 100 - 100 / (1 + rs)

    # ── MACD ───────────────────────────────────────────────────────────────
    d['MACD'] = d['EMA_12'] - d['EMA_26']
    d['MACD_Signal'] = d['MACD'].ewm(span=9, adjust=False).mean()
    d['MACD_Hist'] = d['MACD'] - d['MACD_Signal']

    # ── Bollinger Bands ────────────────────────────────────────────────────
    sma20 = d['Close'].rolling(20).mean()
    std20 = d['Close'].rolling(20).std()
    d['BB_Upper'] = sma20 + 2 * std20
    d['BB_Lower'] = sma20 - 2 * std20
    d['BB_Width'] = (d['BB_Upper'] - d['BB_Lower']) / sma20
    d['BB_Position'] = (d['Close'] - d['BB_Lower']) / (d['BB_Upper'] - d['BB_Lower'])

    # ── Price Position ─────────────────────────────────────────────────────
    d['Price_vs_SMA20'] = d['Close'] / d['SMA_20'] - 1
    d['Price_vs_SMA50'] = d['Close'] / d['SMA_50'] - 1
    d['HL_Ratio'] = (d['High'] - d['Low']) / d['Close']
    d['OC_Ratio'] = (d['Close'] - d['Open']) / d['Open']

    # ── Volume Features ────────────────────────────────────────────────────
    d['Volume_MA10'] = d['Volume'].rolling(10).mean()
    d['Volume_Ratio'] = d['Volume'] / d['Volume_MA10']
    d['OBV'] = (np.sign(d['Return']) * d['Volume']).cumsum()

    # ── Lag Features ──────────────────────────────────────────────────────
    for lag in [1, 2, 3, 5]:
        d[f'Return_lag{lag}'] = d['Return'].shift(lag)
        d[f'Volume_lag{lag}'] = d['Volume'].shift(lag)

    # ── Signal Features ───────────────────────────────────────────────────
    d['MeanRev_Signal'] = np.where(d['RSI'] < 35, 1, np.where(d['RSI'] > 65, -1, 0))
    d['Momentum_Signal'] = np.sign(d['Momentum_5'])
    d['Vol_Regime'] = (d['Volatility_10'] > d['Volatility_20']).astype(int)
    d['Trend_Strength'] = (d['Close'] - d['SMA_50']) / d['ATR'].replace(0, np.nan)

    # Price acceleration
    d['Return_Accel'] = d['Return'] - d['Return'].shift(1)

    # Volume-price trend
    d['VPT'] = (d['Return'] * d['Volume']).cumsum()
    d['VPT_MA'] = d['VPT'].rolling(10).mean()
    d['VPT_Signal'] = d['VPT'] - d['VPT_MA']

    # Calendar features
    d['DayOfWeek'] = d.index.dayofweek
    d['Month'] = d.index.month
    d['Quarter'] = d.index.quarter
    d['IsMonthStart'] = (d.index.day <= 5).astype(int)
    d['IsMonthEnd'] = (d.index.day >= 25).astype(int)

    return d


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test dynamic registry
    registry = get_company_registry()
    print(f"Registry size: {len(registry)} companies")
    for symbol in list(registry.keys())[:5]:
        print(f"  {symbol}: {registry[symbol]['name']} ({registry[symbol]['sector']})")

    # Test data loading
    df = get_stock_data("NABIL")
    print(f"\nNABIL data shape: {df.shape}")
    print(df.tail())
