"""
technical_indicators.py - Technical Indicators module for Stock Analyzer Tool
Calculates and exports the top 10 technical indicators as per requirements.
"""
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
import threading
import logging
from scipy.signal import argrelextrema

class TechnicalIndicators:
    # Centralized indicator lists for maintainability and extensibility
    DEFAULT_INDICATOR_PARAMS = {
        'SMA30': {'length': 30},
        'SMA100': {'length': 100},
        'EMA10': {'length': 10},
        'EMA20': {'length': 20},
        'EMA30': {'length': 30},
        'EMA100': {'length': 100},
        'RSI': {'length': 14},
        'ATR': {'length': 14},
        'CCI': {'length': 20},
        'WILLR': {'length': 14},
        'ROC': {'length': 12},
        'MFI': {'length': 14},
        'BBANDS': {'length': 20, 'std': 2},
        'STDDEV': {'length': 20},
        'STOCH': {'k': 14, 'd': 3, 'smooth_k': 3},
        'VWAP': {},
        'PSAR': {},
        'ICHIMOKU': {},
        'ADX': {},
        'OBV': {},        
        'ROLLING_STD': {'length': 20},
        'KC': {'length': 20, 'scalar': 2, 'mamode': 'ema'},
        'DONCHIAN': {'length': 20},
        'ULCER': {'length': 14},
        'HVOL': {'length': 21},
        'VAR': {'length': 20},
    }

    def __init__(self, config):
        self.config = config
        self.indicator_dir = "stock_indicator"
        os.makedirs(self.indicator_dir, exist_ok=True)
        # Use config or default indicator params
        self.indicator_params = self.config.get('indicator_params', self.DEFAULT_INDICATOR_PARAMS)
        # Setup logging
        logging.basicConfig(filename='indicator_calc.log', level=logging.INFO,
                          format='%(asctime)s %(levelname)s:%(message)s')

    def _validate_ohlcv_data(self, df):
        """Validate OHLCV data quality and consistency."""
        try:
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            
            # Check if all required columns exist
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logging.error(f"Missing required columns: {missing_cols}")
                return False
            
            # Ensure all price/volume columns are numeric
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Check for data consistency (high >= low, etc.)
            invalid_bars = (df['high'] < df['low']) | (df['high'] < df['open']) | (df['high'] < df['close']) | \
                          (df['low'] > df['open']) | (df['low'] > df['close']) | (df['volume'] < 0)
            
            if invalid_bars.any():
                invalid_count = invalid_bars.sum()
                logging.warning(f"Found {invalid_count} invalid OHLCV bars, will be excluded from calculations")
                df = df[~invalid_bars]
            
            # Check for sufficient data
            if len(df) < 20:
                logging.warning(f"Insufficient data points ({len(df)}), need at least 20 for reliable indicators")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error validating OHLCV data: {e}")
            return False

    def run(self, args):
        logging.info("[TechnicalIndicators] Calculating indicators...")
        threads = []
        max_threads = 4
        if args.tickers:
            for ticker in args.tickers:
                for interval in self.config.get('ohlcv_intervals', ['1d', '1h']):
                    while threading.active_count() >= max_threads + 1:  # +1 for main thread
                        for t in threads:
                            t.join(0.1)
                    t = threading.Thread(target=self.calculate_indicators, args=(ticker, interval))
                    t.start()
                    threads.append(t)
        for t in threads:
            t.join()

    def calculate_indicators(self, ticker, interval):
        logging.info(f"Calculating indicators for {ticker} ({interval})")
        ohlcv_files = [f for f in os.listdir("stock_data") if f.startswith(f"{ticker}_ohlcv_{interval}_") and f.endswith('.csv')]
        if not ohlcv_files:
            logging.warning(f"No OHLCV data for {ticker} ({interval}), skipping.")
            return
        ohlcv_path = os.path.join("stock_data", sorted(ohlcv_files)[-1])
        df = pd.read_csv(ohlcv_path)
        logging.info(f"Loaded OHLCV data for {ticker} ({interval}) from {ohlcv_path}")

        # Validate OHLCV data quality
        if not self._validate_ohlcv_data(df):
            logging.warning(f"OHLCV data validation failed for {ticker} ({interval}), skipping.")
            return

        # Parse and filter dates
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            today = pd.Timestamp.now().normalize()
            df = df[df['date'] <= today]
            df = df.sort_values('date')
            df = df[~df['date'].duplicated(keep='first')]
            df.set_index('date', inplace=True)

        # Drop rows with all-NaN OHLCV
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        df.dropna(subset=ohlcv_cols, how='all', inplace=True)

        def to8(x):
            return float(f"{x:.8f}") if pd.notnull(x) else x

        # --- Indicator Calculations (unchanged, as in your code) ---
        # SMA/EMA (explicit for clarity and extensibility)
        for name in ['SMA30', 'SMA100']:
            if name in self.indicator_params:
                df[name] = ta.sma(df['close'], **self.indicator_params[name]).map(to8)
        for name in ['EMA10', 'EMA20', 'EMA30', 'EMA100']:
            if name in self.indicator_params:
                df[name] = ta.ema(df['close'], **self.indicator_params[name]).map(to8)
        # MACD
        try:
            macd = ta.macd(df['close'])
            df['MACD'] = macd['MACD_12_26_9'].map(to8) if 'MACD_12_26_9' in macd else np.nan
        except Exception as e:
            logging.error(f"Error calculating MACD: {e}")
            df['MACD'] = np.nan
        # RSI
        try:
            df['RSI'] = ta.rsi(df['close'], **self.indicator_params.get('RSI', {})).map(to8)
        except Exception as e:
            logging.error(f"Error calculating RSI: {e}")
            df['RSI'] = np.nan
        # Bollinger Bands
        try:
            bbands = ta.bbands(df['close'], **self.indicator_params.get('BBANDS', {}))
            for col in ['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0']:
                out_col = col.split('_')[0]
                df[out_col] = bbands[col].map(to8) if col in bbands else np.nan
        except Exception as e:
            logging.error(f"Error calculating Bollinger Bands: {e}")
            for out_col in ['BBL', 'BBM', 'BBU', 'BBB', 'BBP']:
                df[out_col] = np.nan
        # ATR
        try:
            df['ATR'] = ta.atr(df['high'], df['low'], df['close'], **self.indicator_params.get('ATR', {})).map(to8)
        except Exception as e:
            logging.error(f"Error calculating ATR: {e}")
            df['ATR'] = np.nan
        # OBV
        try:
            df['OBV'] = ta.obv(df['close'], df['volume']).map(to8)
        except Exception as e:
            logging.error(f"Error calculating OBV: {e}")
            df['OBV'] = np.nan
        # STOCH/KDJ
        try:
            stoch = ta.stoch(df['high'], df['low'], df['close'], **self.indicator_params.get('STOCH', {}))
            df['STOCH'] = stoch['STOCHk_14_3_3'].map(to8) if 'STOCHk_14_3_3' in stoch else np.nan
            if 'STOCHk_14_3_3' in stoch and 'STOCHd_14_3_3' in stoch:
                df['K'] = stoch['STOCHk_14_3_3'].map(to8)
                df['D'] = stoch['STOCHd_14_3_3'].map(to8)
                df['J'] = (3 * stoch['STOCHk_14_3_3'] - 2 * stoch['STOCHd_14_3_3']).map(to8)
            else:
                df['K'] = df['D'] = df['J'] = np.nan
        except Exception as e:
            logging.error(f"Error calculating STOCH/KDJ: {e}")
            df['STOCH'] = df['K'] = df['D'] = df['J'] = np.nan
        # VWAP
        try:
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            vwap = ta.vwap(df['high'], df['low'], df['close'], df['volume'], **self.indicator_params.get('VWAP', {}))
            df['VWAP'] = vwap.map(to8) if hasattr(vwap, 'map') else vwap
        except Exception as e:
            logging.error(f"Error calculating VWAP: {e}")
            df['VWAP'] = np.nan
        # PSAR
        try:
            psar = ta.psar(df['high'], df['low'], df['close'], **self.indicator_params.get('PSAR', {}))
            df['PSAR'] = psar['PSARl_0.02_0.2'].map(to8) if 'PSARl_0.02_0.2' in psar else np.nan
        except Exception as e:
            logging.error(f"Error calculating PSAR: {e}")
            df['PSAR'] = np.nan
        # Ichimoku
        try:
            ichimoku = ta.ichimoku(df['high'], df['low'], df['close'], **self.indicator_params.get('ICHIMOKU', {}))
            if isinstance(ichimoku, tuple) and len(ichimoku) > 0:
                ichi_df = ichimoku[0].astype('float64').apply(to8)
                for col in ichi_df.columns:
                    df[col] = ichi_df[col]
        except Exception as e:
            logging.error(f"Error calculating Ichimoku: {e}")
        # ADX/DMI
        try:
            adx_all = ta.adx(df['high'], df['low'], df['close'], **self.indicator_params.get('ADX', {}))
            for adx_col, out_col in [('ADX_14', 'ADX'), ('DMP_14', 'DMIp'), ('DMN_14', 'DMIm')]:
                if adx_col in adx_all:
                    df[out_col] = adx_all[adx_col].map(to8)
                else:
                    df[out_col] = np.nan
        except Exception as e:
            logging.error(f"Error calculating ADX/DMI: {e}")
            for out_col in ['ADX', 'DMIp', 'DMIm']:
                df[out_col] = np.nan
        # CCI, WILLR, ROC
        try:
            df['CCI'] = ta.cci(df['high'], df['low'], df['close'], **self.indicator_params.get('CCI', {})).map(to8)
        except Exception as e:
            logging.error(f"Error calculating CCI: {e}")
            df['CCI'] = np.nan
        try:
            df['WILLR'] = ta.willr(df['high'], df['low'], df['close'], **self.indicator_params.get('WILLR', {})).map(to8)
        except Exception as e:
            logging.error(f"Error calculating WILLR: {e}")
            df['WILLR'] = np.nan
        try:
            df['ROC'] = ta.roc(df['close'], **self.indicator_params.get('ROC', {})).map(to8)
        except Exception as e:
            logging.error(f"Error calculating ROC: {e}")
            df['ROC'] = np.nan
        # MFI
        try:
            mfi = ta.mfi(df['high'], df['low'], df['close'], df['volume'], **self.indicator_params.get('MFI', {}))
            df['MFI'] = pd.Series(mfi, index=df.index, dtype='float64').map(to8)
        except Exception as e:
            logging.error(f"Error calculating MFI: {e}")
            df['MFI'] = np.nan

        # STDDEV (Standard Deviation)
        try:
            df['STDDEV'] = ta.stdev(df['close'], **self.indicator_params.get('STDDEV', {'length': 20})).map(to8)
        except Exception as e:
            logging.error(f"Error calculating STDDEV: {e}")
            df['STDDEV'] = np.nan
        # Rolling Standard Deviation (Rolling STD)
        try:
            df['ROLLING_STD'] = df['close'].rolling(window=self.indicator_params.get('ROLLING_STD', {'length': 20})['length']).std().map(to8)
        except Exception as e:
            logging.error(f"Error calculating ROLLING_STD: {e}")
            df['ROLLING_STD'] = np.nan        # Keltner Channel (KC)
        try:
            kc = ta.kc(df['high'], df['low'], df['close'], **self.indicator_params.get('KC', {}))
            # Output columns: KCLe, KCBe, KCUe (from pandas_ta.kc)
            if kc is not None and isinstance(kc, pd.DataFrame):
                # Find the actual column names (they may vary based on parameters)
                kc_cols = [col for col in kc.columns if 'KCL' in col or 'KCB' in col or 'KCU' in col]
                if len(kc_cols) >= 3:
                    df['KCLe'] = kc[kc_cols[0]].map(to8)  # Lower band
                    df['KCBe'] = kc[kc_cols[1]].map(to8)  # Basis (middle)
                    df['KCUe'] = kc[kc_cols[2]].map(to8)  # Upper band
                else:
                    df['KCLe'] = df['KCBe'] = df['KCUe'] = np.nan
            else:
                df['KCLe'] = df['KCBe'] = df['KCUe'] = np.nan
        except Exception as e:
            logging.error(f"Error calculating Keltner Channel: {e}")
            df['KCLe'] = df['KCBe'] = df['KCUe'] = np.nan        # Donchian Channel
        try:
            donchian = ta.donchian(df['high'], df['low'], **self.indicator_params.get('DONCHIAN', {}))
            # Output columns: DCL, DCM, DCU (from pandas_ta.donchian)
            if donchian is not None and isinstance(donchian, pd.DataFrame):
                # Find the actual column names (they may vary based on parameters)
                dc_cols = [col for col in donchian.columns if 'DCL' in col or 'DCM' in col or 'DCU' in col]
                if len(dc_cols) >= 3:
                    df['DCL'] = donchian[dc_cols[0]].map(to8)  # Lower channel
                    df['DCM'] = donchian[dc_cols[1]].map(to8)  # Middle channel
                    df['DCU'] = donchian[dc_cols[2]].map(to8)  # Upper channel
                else:
                    df['DCL'] = df['DCM'] = df['DCU'] = np.nan
            else:
                df['DCL'] = df['DCM'] = df['DCU'] = np.nan
        except Exception as e:
            logging.error(f"Error calculating Donchian Channel: {e}")
            df['DCL'] = df['DCM'] = df['DCU'] = np.nan

        # Ulcer Index (manual implementation)
        try:
            ulcer_length = self.indicator_params.get('ULCER', {}).get('length', 14)
            close = df['close']
            rolling_max = close.rolling(window=ulcer_length, min_periods=1).max()
            drawdown = ((close - rolling_max) / rolling_max) * 100
            ulcer = np.sqrt((drawdown ** 2).rolling(window=ulcer_length, min_periods=1).mean())
            df['ULCER'] = ulcer.map(to8)
        except Exception as e:
            logging.error(f"Error calculating Ulcer Index: {e}")
            df['ULCER'] = np.nan
        # Historical Volatility (manual implementation, annualized stddev of log returns)
        try:
            hvol_length = self.indicator_params.get('HVOL', {}).get('length', 21)
            log_ret = np.log(df['close'] / df['close'].shift(1))
            hvol = log_ret.rolling(window=hvol_length, min_periods=1).std() * np.sqrt(252) * 100
            df['HVOL'] = hvol.map(to8)
        except Exception as e:
            logging.error(f"Error calculating Historical Volatility: {e}")
            df['HVOL'] = np.nan
        # Variance
        try:
            df['VAR'] = ta.variance(df['close'], **self.indicator_params.get('VAR', {})).map(to8)
        except Exception as e:
            logging.error(f"Error calculating Variance: {e}")
            df['VAR'] = np.nan        # Support and resistance
        try:
            # Simple S/R: Use local extrema as S/R candidates
            close = df['close']
            order = 10  # window size for local extrema
            N = 3  # Number of support/resistance levels to track
            local_max = argrelextrema(close.values, np.greater, order=order)[0]
            local_min = argrelextrema(close.values, np.less, order=order)[0]
            # Take most recent N S/R levels
            sr_highs = sorted(close.iloc[local_max].dropna().values)[-N:] if len(local_max) >= N else close.iloc[local_max].dropna().values
            sr_lows = sorted(close.iloc[local_min].dropna().values)[:N] if len(local_min) >= N else close.iloc[local_min].dropna().values
            # Assign to columns SR1, SR2, ...
            for i, val in enumerate(sr_highs):
                df[f'SR_High_{i+1}'] = val
            for i, val in enumerate(sr_lows):
                df[f'SR_Low_{i+1}'] = val
        except Exception as e:
            logging.error(f"Error calculating Support/Resistance: {e}")
            N = 3  # Default value for error recovery
            for i in range(N):
                df[f'SR_High_{i+1}'] = np.nan
                df[f'SR_Low_{i+1}'] = np.nan        # --- Chart Pattern Detection (enhanced) ---
        try:
            df['chart_pattern'] = ''
            close = df['close']
            window = 20  # rolling window for pattern detection
            
            # Ensure we have enough data for pattern detection
            if len(df) < window:
                logging.warning(f"Not enough data for chart pattern detection, skipping.")
            else:
                # Double Bottom/Top (as before)
                local_min = argrelextrema(close.values, np.less, order=5)[0]
                local_max = argrelextrema(close.values, np.greater, order=5)[0]
                
                for i in range(1, len(local_min)):
                    if (local_min[i] < len(df) and local_min[i-1] < len(df) and
                        abs(close.iloc[local_min[i]] - close.iloc[local_min[i-1]])/close.iloc[local_min[i-1]] < 0.05):
                        mid_max = [m for m in local_max if local_min[i-1] < m < local_min[i]]
                        if mid_max:
                            idx = local_min[i]
                            if idx < len(df):
                                df.at[df.index[idx], 'chart_pattern'] = 'Double Bottom'
                
                for i in range(1, len(local_max)):
                    if (local_max[i] < len(df) and local_max[i-1] < len(df) and
                        abs(close.iloc[local_max[i]] - close.iloc[local_max[i-1]])/close.iloc[local_max[i-1]] < 0.05):
                        mid_min = [m for m in local_min if local_max[i-1] < m < local_max[i]]
                        if mid_min:
                            idx = local_max[i]
                            if idx < len(df):
                                df.at[df.index[idx], 'chart_pattern'] = 'Double Top'
                
                # Head and Shoulders (simple heuristic: 3 peaks, middle highest)
                for i in range(window, len(df)):
                    w = close.iloc[i-window:i]
                    peaks = argrelextrema(w.values, np.greater, order=3)[0]
                    if len(peaks) >= 3:
                        for j in range(len(peaks)-2):
                            l, m, r = peaks[j], peaks[j+1], peaks[j+2]
                            if (l < len(w) and m < len(w) and r < len(w) and
                                w.iloc[m] > w.iloc[l] and w.iloc[m] > w.iloc[r] and 
                                abs(w.iloc[l] - w.iloc[r])/w.iloc[m] < 0.15):
                                if m < len(w):
                                    df.at[w.index[m], 'chart_pattern'] = 'Head and Shoulders'
                    
                    # Inverse Head and Shoulders
                    troughs = argrelextrema(w.values, np.less, order=3)[0]
                    if len(troughs) >= 3:
                        for j in range(len(troughs)-2):
                            l, m, r = troughs[j], troughs[j+1], troughs[j+2]
                            if (l < len(w) and m < len(w) and r < len(w) and
                                w.iloc[m] < w.iloc[l] and w.iloc[m] < w.iloc[r] and 
                                abs(w.iloc[l] - w.iloc[r])/w.iloc[m] < 0.15):
                                if m < len(w):
                                    df.at[w.index[m], 'chart_pattern'] = 'Inverse Head and Shoulders'
                
                # Triangle (series of higher lows and lower highs)
                for i in range(window, len(df)):
                    w = close.iloc[i-window:i]
                    lows = w.rolling(3).min()
                    highs = w.rolling(3).max()
                    if lows.is_monotonic_increasing and highs.is_monotonic_decreasing:
                        df.at[w.index[-1], 'chart_pattern'] = 'Triangle'
                
                # Flag/Pennant (sharp move, then consolidation)
                for i in range(window, len(df)):
                    w = close.iloc[i-window:i]
                    if len(w) >= 5 and w.pct_change().abs().max() > 0.1 and w[-5:].std() < w.std()/2:
                        df.at[w.index[-1], 'chart_pattern'] = 'Flag/Pennant'
                
                # Cup and Handle (U shape followed by small dip)
                for i in range(window, len(df)-5):
                    w = close.iloc[i-window:i]
                    if (len(w) >= window and window//2 < len(w) and
                        w.min() == w.iloc[window//2] and w.iloc[-1] > w.iloc[window//2] and 
                        len(w) >= 10 and w.iloc[-5:].min() < w.iloc[-10:-5].min()):
                        df.at[w.index[-1], 'chart_pattern'] = 'Cup and Handle'
        except Exception as e:
            logging.error(f"Error in chart pattern detection: {e}")
            df['chart_pattern'] = ''

        try:
            # --- Candle Pattern Detection (enhanced) ---
            df['candle_pattern'] = ''
            
            # Ensure we have enough data for candlestick pattern detection
            if len(df) < 3:
                logging.warning(f"Not enough data for candlestick pattern detection, skipping.")
            else:
                for i in range(2, len(df)):
                    try:
                        o, h, l, c = df.iloc[i][['open','high','low','close']]
                        po, ph, pl, pc = df.iloc[i-1][['open','high','low','close']]
                        ppo, pph, ppl, ppc = df.iloc[i-2][['open','high','low','close']]
                        
                        # Skip if any values are NaN
                        if any(pd.isna([o, h, l, c, po, ph, pl, pc, ppo, pph, ppl, ppc])):
                            continue
                            
                        body = abs(c-o)
                        range_ = h-l
                        
                        # Skip if range is zero to avoid division by zero
                        if range_ == 0:
                            continue
                        
                        # Refined Doji
                        if body/range_ < 0.1:
                            df.at[df.index[i], 'candle_pattern'] = 'Doji'
                        # Refined Hammer
                        elif (c > o) and ((o-l) > 2*body) and ((h-c) < body) and body/range_ < 0.3:
                            df.at[df.index[i], 'candle_pattern'] = 'Hammer'
                        # Refined Shooting Star
                        elif (o > c) and ((h-o) > 2*body) and ((o-l) < body) and body/range_ < 0.3:
                            df.at[df.index[i], 'candle_pattern'] = 'Shooting Star'
                        # Bullish Engulfing
                        elif pc < po and c > o and c > po and o < pc and body > abs(pc-po):
                            df.at[df.index[i], 'candle_pattern'] = 'Bullish Engulfing'
                        # Bearish Engulfing
                        elif pc > po and c < o and c < po and o > pc and body > abs(pc-po):
                            df.at[df.index[i], 'candle_pattern'] = 'Bearish Engulfing'
                        # Morning Star
                        elif (pph-ppl > 0 and ph-pl > 0 and h-l > 0 and
                              ppo > ppc and abs(ppo-ppc)/abs(pph-ppl) > 0.5 and
                              abs(po-pc)/abs(ph-pl) < 0.3 and
                              c > o and c > po and o > pc and abs(c-o)/abs(h-l) > 0.5):
                            df.at[df.index[i], 'candle_pattern'] = 'Morning Star'
                        # Evening Star
                        elif (pph-ppl > 0 and ph-pl > 0 and h-l > 0 and
                              ppo < ppc and abs(ppo-ppc)/abs(pph-ppl) > 0.5 and
                              abs(po-pc)/abs(ph-pl) < 0.3 and
                              c < o and c < po and o < pc and abs(c-o)/abs(h-l) > 0.5):
                            df.at[df.index[i], 'candle_pattern'] = 'Evening Star'
                        # Piercing Line
                        elif (ppo > ppc and o < c and c > (ppo + ppc)/2 and o < ppc and c < ppo):
                            df.at[df.index[i], 'candle_pattern'] = 'Piercing Line'
                        # Dark Cloud Cover
                        elif (ppo < ppc and o > c and c < (ppo + ppc)/2 and o > ppc and c > ppo):
                            df.at[df.index[i], 'candle_pattern'] = 'Dark Cloud Cover'
                        # Spinning Top
                        elif 0.1 < body/range_ < 0.3:
                            df.at[df.index[i], 'candle_pattern'] = 'Spinning Top'
                    except Exception as pattern_error:
                        logging.warning(f"Error processing candlestick pattern at index {i}: {pattern_error}")
                        continue
        except Exception as e:  
            logging.error(f"Error calculating Candle Pattern: {e}")
            df['candle_pattern'] = ''

        # Export splits/dividends if present
        for col in ['dividends', 'splits']:
            if col in df.columns:
                logging.info(f"Column '{col}' present in data and will be exported.")

        # Enforce output column order: date, OHLCV, then indicators
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        # Insert K, D, J immediately after STOCH if present
        indicator_cols = []
        for col in df.columns:
            indicator_cols.append(col)
                
        final_cols = ['date'] + [c for c in ohlcv_cols if c in df.columns] + [c for c in indicator_cols if c not in ohlcv_cols and c != 'date']

        df = df.reset_index() if df.index.name == 'date' else df
        df = df[[col for col in final_cols if col in df.columns]]
        out_path = os.path.join(self.indicator_dir, f"{ticker}_indicators_{interval}.csv")
        if len(df) > 100_000:
            logging.info(f"Large dataset detected ({len(df)} rows), saving in chunks...")
            chunk_size = 50_000
            for i, start in enumerate(range(0, len(df), chunk_size)):
                chunk_path = out_path.replace('.csv', f'_part{i+1}.csv')
                df.iloc[start:start+chunk_size].to_csv(chunk_path, float_format='%.8f', index=False)
                logging.info(f"Saved chunk {i+1} to {chunk_path}")
        else:
            df.to_csv(out_path, float_format='%.8f', index=False)
            logging.info(f"Saved indicators to {out_path}")
        # Metadata
        import yaml
        meta = {
            'ticker': ticker,
            'interval': interval,
            'calculation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source_file': ohlcv_path,
            'has_multitimeframe': interval == '1h' and any(col.endswith('_1d') for col in df.columns),
            'pandas_ta_version': getattr(ta, '__version__', 'unknown'),
            'indicator_params': self.indicator_params
        }
        meta_path = out_path.replace('.csv', '.meta.yaml')
        with open(meta_path, 'w') as f:
            yaml.dump(meta, f)
        logging.info(f"Saved metadata to {meta_path}")

    def _find_latest_cache_key(self, *args):
        # Find the latest cache file for the given args
        prefix = f"{args[1]}_ohlcv_"
        files = [f for f in os.listdir("stock_data") if f.startswith(prefix) and f.endswith('.csv')]
        if files:
            return files[-1].replace(f"{args[1]}_ohlcv_", "").replace('.csv', "")
        return ''


