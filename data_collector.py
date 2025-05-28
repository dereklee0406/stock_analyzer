"""
data_collector.py - Data collection module for Stock Analyzer Tool
Configurable via YAML. Downloads historical OHLCV (with splits/dividends) from Yahoo Finance, VIX, and macro data. Supports multiple timeframes and robust caching. Stores all data as structured CSV in stock_data with metadata/versioning.
"""
import os
import pandas as pd
import hashlib
import requests
from datetime import datetime, timedelta

class DataCollector:
    def __init__(self, config):
        self.config = config
        self.data_dir = "stock_data"
        os.makedirs(self.data_dir, exist_ok=True)
        self.cache = {}
        # Configurable parameters
        self.ohlcv_start = config.get('ohlcv_start', '2000-01-01')
        self.ohlcv_end = config.get('ohlcv_end', datetime.now().strftime('%Y-%m-%d'))
        self.ohlcv_intervals = config.get('ohlcv_intervals', ['1d', '1h'])
        self.vix_start = config.get('vix_start', '2000-01-01')
        self.vix_end = config.get('vix_end', datetime.now().strftime('%Y-%m-%d'))
        self.cache_days = config.get('cache_days', 1)

    def run(self, args):
        print("[DataCollector] Starting data collection...")
        # Clean the data directory before collecting new data
        for fname in os.listdir(self.data_dir):
            fpath = os.path.join(self.data_dir, fname)
            try:
                if os.path.isfile(fpath):
                    os.remove(fpath)
            except Exception as e:
                print(f"Could not remove {fpath}: {e}")
        if args.tickers:
            for ticker in args.tickers:
                for interval in self.ohlcv_intervals:
                    self.fetch_ohlcv(ticker, interval)
        self.fetch_vix()
        self.fetch_macro()

    def _cache_key(self, *args):
        key = '_'.join(map(str, args))
        return hashlib.md5(key.encode()).hexdigest()

    def _is_cache_stale(self, cache_path):
        if not os.path.exists(cache_path):
            return True
        mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
        return (datetime.now() - mtime) > timedelta(days=self.cache_days)

    def _load_or_fetch(self, cache_path, fetch_func):
        if os.path.exists(cache_path) and not self._is_cache_stale(cache_path):
            print(f"[Cache] Using cached data ({cache_path})")
            return pd.read_csv(cache_path)
        return fetch_func()

    def fetch_ohlcv(self, ticker, interval):
        """Fetch OHLCV data for a ticker using yfinance, supports 1d and 1h intervals. Handles 1h data with correct date range and error handling."""
        try:
            import yfinance as yf
        except ImportError:
            print("yfinance is not installed. Cannot fetch OHLCV.")
            return None
        cache_key = self._cache_key('ohlcv', ticker, interval, self.ohlcv_start, self.ohlcv_end)
        cache_path = os.path.join(self.data_dir, f"{ticker}_ohlcv_{interval}_{cache_key}.csv")
        def fetch():
            print(f"Fetching OHLCV for {ticker} ({interval}) using yfinance")
            try:
                # yfinance only supports 1h interval for up to 730 days (2 years)
                if interval == '1h':
                    start = max(pd.to_datetime(self.ohlcv_start), datetime.now() - timedelta(days=729))
                    end = pd.to_datetime(self.ohlcv_end)
                    if start >= end:
                        print(f"Start date {start} is not before end date {end} for 1h interval. Skipping.")
                        return pd.DataFrame({'date': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []})
                else:
                    start = pd.to_datetime(self.ohlcv_start)
                    end = pd.to_datetime(self.ohlcv_end)
                df = yf.download(ticker, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), interval=interval, auto_adjust=True)

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)


                if not df.empty:
                    df.reset_index(inplace=True)
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        if col not in df:
                            df[col] = None
                    df.rename(columns={
                        'Datetime' if interval == '1h' else 'Date': 'date',
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    }, inplace=True)
                    # Remove any row where 'date' is literally the string 'Ticker' (case-insensitive)
                    if 'date' in df.columns:
                        df = df[df['date'].astype(str).str.lower() != 'ticker']
                    # Remove any row where all OHLCV columns are NaN or None (sometimes yfinance appends a blank row)
                    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
                    if all(col in df.columns for col in ohlcv_cols):
                        df = df.dropna(subset=ohlcv_cols, how='all')
                    # Add splits/dividends if present
                    if 'Dividends' in df:
                        df['dividends'] = df['Dividends']
                    if 'Stock Splits' in df:
                        df['splits'] = df['Stock Splits']
                    meta = {
                        'ticker': ticker,
                        'interval': interval,
                        'start': str(start),
                        'end': str(end),
                        'fetched_at': datetime.now().isoformat()
                    }
                    df.attrs = meta
                    print(f"Fetched {len(df)} OHLCV rows for {ticker} ({interval})")
                else:
                    print(f"yfinance returned empty OHLCV for {ticker} ({interval}), using stub.")
                    df = pd.DataFrame({'date': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []})
                
                # Format float columns to full precision when saving
                float_cols = ['open', 'high', 'low', 'close']
                for col in float_cols:
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: format(x, '.8f') if pd.notnull(x) else x)
                df.to_csv(cache_path, index=False)
                with open(cache_path + '.meta', 'w') as f:
                    for k, v in meta.items():
                        f.write(f"{k}: {v}\n")
                print(f"Saved OHLCV to {cache_path}")
                return df
            except Exception as e:
                print(f"Error fetching OHLCV with yfinance: {e}")
                return None
        return self._load_or_fetch(cache_path, fetch)

    def fetch_vix(self):
        """Fetch VIX data using yfinance."""
        try:
            import yfinance as yf
        except ImportError:
            print("yfinance is not installed. Cannot fetch VIX.")
            return None
        cache_key = self._cache_key('vix', self.vix_start, self.vix_end)
        cache_path = os.path.join(self.data_dir, f"vix_{cache_key}.csv")
        def fetch():
            print("Fetching VIX data using yfinance")
            try:
                vix = yf.download('^VIX', start=self.vix_start, end=self.vix_end, interval='1d')
                if isinstance(vix.columns, pd.MultiIndex):
                    vix.columns = vix.columns.droplevel(1)
                
                if not vix.empty:
                    vix.reset_index(inplace=True)
                    vix.rename(columns={
                        'Date': 'date',
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    }, inplace=True)
                    df = vix[['date', 'open', 'high', 'low', 'close', 'volume']]
                    meta = {
                        'symbol': '^VIX',
                        'start': self.vix_start,
                        'end': self.vix_end,
                        'fetched_at': datetime.now().isoformat()
                    }
                    df.attrs = meta
                    print(f"Fetched {len(df)} VIX rows")
                else:
                    print("yfinance returned empty VIX data, using stub.")
                    df =  pd.DataFrame({'date': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []})

                # Format float columns to full precision when saving
                float_cols = ['open', 'high', 'low', 'close']
                for col in float_cols:
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: format(x, '.8f') if pd.notnull(x) else x)
                df.to_csv(cache_path, index=False)
                with open(cache_path + '.meta', 'w') as f:
                    for k, v in meta.items():
                        f.write(f"{k}: {v}\n")
                print(f"Saved VIX to {cache_path}")
                return df
            except Exception as e:
                print(f"Error fetching VIX with yfinance: {e}")
                return None
        return self._load_or_fetch(cache_path, fetch)

    def fetch_macro(self):
        """Fetch macroeconomic data (stub, can be extended to use FRED or other APIs)."""
        cache_key = self._cache_key('macro')
        cache_path = os.path.join(self.data_dir, f"macro_{cache_key}.csv")
        def fetch():
            print("Fetching macroeconomic data (stub)")
            df = pd.DataFrame({
                'indicator': ['GDP', 'CPI', 'Unemployment'],
                'value': [0, 0, 0],
                'date': [datetime.now().strftime('%Y-%m-%d')]*3
            })
            meta = {
                'source': 'stub',
                'fetched_at': datetime.now().isoformat()
            }
            df.to_csv(cache_path, index=False)
            with open(cache_path + '.meta', 'w') as f:
                for k, v in meta.items():
                    f.write(f"{k}: {v}\n")
            print(f"Saved macro data to {cache_path}")
            return df
        return self._load_or_fetch(cache_path, fetch)
