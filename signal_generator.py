"""
signal_generator.py - Signal Generation module for Stock Analyzer Tool
Generates trading signals based on trend, momentum, MACD, price action, volume, and S/R. Supports composite, multi-indicator, and AI/ML-based signals. Outputs buy/sell/hold with confidence, explainability, and caching.
"""
import os
import pandas as pd
import numpy as np
import hashlib
from datetime import datetime, timedelta
import threading

class SignalGenerator:
    def __init__(self, config):
        self.config = config
        self.signal_dir = "stock_signals"
        os.makedirs(self.signal_dir, exist_ok=True)
        self.cache_days = config.get('cache_days', 1)
        self.thresholds = config.get('signal_thresholds', {
            'macd': 0,
            'rsi_overbought': 70,
            'rsi_oversold': 20,
            'sma_cross': True,
            'ema_cross': True,
            'volume_spike': 2.0,
        })
        # Configurable parameters
        # Fine-tuned: Lower confirmation and score thresholds, increase indicator weights, allow more frequent trades, and use continuous scoring
        self.min_confirmations = self.config.get('signal_min_confirmations', 1)  # Lowered
        self.score_thresholds = self.config.get('signal_score_thresholds', {
            'strong_buy': 2.0,  # Lowered
            'buy': 1.0,         # Lowered
            'strong_sell': -2.0,
            'sell': -1.0
        })

    def run(self, args):
        print("[SignalGenerator] Generating signals...")
        threads = []
        max_threads = 8
        if args.tickers:
            for ticker in args.tickers:
                for interval in self.config.get('ohlcv_intervals', ['1d', '1h']):
                    while threading.active_count() >= max_threads + 1:  # +1 for main thread
                        for t in threads:
                            t.join(0.1)
                    t = threading.Thread(target=self.generate_signals, args=(ticker, interval))
                    t.start()
                    threads.append(t)
        for t in threads:
            t.join()

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
            print(f"[Cache] Using cached signals ({cache_path})")
            return pd.read_csv(cache_path)
        return fetch_func()

    def generate_signals(self, ticker, interval):
        print(f"Generating signals for {ticker} ({interval})")
        # Clean the signal folder for this ticker/interval before generating new signals
        for fname in os.listdir(self.signal_dir):
            if fname.startswith(f"{ticker}_signals_{interval}_"):
                try:
                    os.remove(os.path.join(self.signal_dir, fname))
                except Exception as e:
                    print(f"Warning: Could not remove {fname}: {e}")
        ind_path = os.path.join("stock_indicator", f"{ticker}_indicators_{interval}.csv")
        if not os.path.exists(ind_path):
            print(f"No indicator data for {ticker} ({interval}), skipping.")
            return
        cache_key = self._cache_key('signals', ticker, interval)
        cache_path = os.path.join(self.signal_dir, f"{ticker}_signals_{interval}_{cache_key}.csv")
        def fetch():
            df = pd.read_csv(ind_path)
            # Remove duplicate columns if any
            df = df.loc[:,~df.columns.duplicated()]
            # Remove keep_cols filtering; use all columns from the indicator file
            signals = []
            for i, row in enumerate(df.itertuples(index=False)):
                row_dict = row._asdict() if hasattr(row, '_asdict') else dict(row._fields, **dict(zip(row._fields, row)))
                signal, confidence, rationale, rationale_bearish, score = self._generate_signal_row(row_dict, df, i)
                filtered_row = dict(row_dict)  # keep all columns
                filtered_row.update({
                    'signal': signal,
                    'confidence': confidence,
                    'rationale_bullish': '; '.join(rationale),
                    'rationale_bearish': '; '.join(rationale_bearish),
                    'score': score,
                    'expires': self._get_expiry(row_dict, interval)
                })
                signals.append(filtered_row)
            out_df = pd.DataFrame(signals)
            # 6. Save signal output metadata
            meta = {
                'ticker': ticker,
                'interval': interval,
                'generated_at': datetime.now().isoformat(),
                'explain': 'See rationale column for signal logic. All indicator columns are included.',
                'min_confirmations': self.min_confirmations,
                'score_thresholds': self.score_thresholds,
                'thresholds': self.thresholds
            }
            out_df.to_csv(cache_path, index=False)
            with open(cache_path + '.meta', 'w') as f:
                for k, v in meta.items():
                    f.write(f"{k}: {v}\n")
            print(f"Saved signals to {cache_path}")
            return out_df
        return self._load_or_fetch(cache_path, fetch)

    def _generate_signal_row(self, row, df, idx):
        """
        Generate a trading signal for a given row using config-driven indicator logic.
        Supports dynamic indicator rules, custom/ML hooks, and richer explainability.
        Implements: weighted voting, adaptive thresholds, signal smoothing, customizable output columns.
        """
        rationale_bullish = []
        rationale_bearish = []
        score = 0
        confirmations = 0
        votes_bull = 0
        votes_bear = 0
        config = self.config.get('signal_logic', {})
        # --- Weighted Voting ---
        indicator_weights = self.config.get('indicator_weights', {})
        # --- Trend Filter (configurable SMA/EMA columns) ---
        trend_ok = True
        trend_col = self.config.get('trend_col', config.get('trend_col', 'SMA100'))
        trend_min_period = self.config.get('trend_min_period', config.get('trend_min_period', 100))
        if trend_col in row and idx >= trend_min_period:
            sma_val = df[trend_col].iloc[max(0, idx-trend_min_period):idx].mean() if trend_col in df else np.nan
            if not pd.isna(sma_val) and 'close' in row:
                trend_ok = row['close'] > sma_val
        # --- Adaptive Thresholds Example (RSI) ---
        adaptive_rsi = None
        if 'RSI' in row and idx >= 14:
            # Use rolling mean and std to adapt RSI thresholds
            rsi_window = df['RSI'].iloc[max(0, idx-14):idx]
            if not rsi_window.isnull().all():
                adaptive_rsi = {
                    'overbought': rsi_window.mean() + rsi_window.std(),
                    'oversold': rsi_window.mean() - rsi_window.std()
                }
        # --- Enhanced Dynamic Indicator Rules ---
        indicator_rules = self.config.get('indicator_rules', config.get('indicator_rules', [
            # MACD
            {'col': 'MACD', 'type': 'threshold', 'bull': 0, 'bear': 0, 'desc_bull': 'MACD bullish', 'desc_bear': 'MACD bearish'},
            # RSI
            {'col': 'RSI', 'type': 'range', 'overbought': 70, 'oversold': 30, 'desc_overbought': 'RSI overbought', 'desc_oversold': 'RSI oversold'},
            # Bollinger Bands
            {'col': 'BBU', 'type': 'compare', 'compare_col': 'close', 'op': 'gt', 'desc': 'Price above upper Bollinger Band (overbought)', 'vote': 'bear'},
            {'col': 'BBL', 'type': 'compare', 'compare_col': 'close', 'op': 'lt', 'desc': 'Price below lower Bollinger Band (oversold)', 'vote': 'bull'},
            {'col': 'BBM', 'type': 'compare', 'compare_col': 'close', 'op': 'gt', 'desc': 'Price above BBM (mid band)', 'vote': 'bull'},
            {'col': 'BBM', 'type': 'compare', 'compare_col': 'close', 'op': 'lt', 'desc': 'Price below BBM (mid band)', 'vote': 'bear'},
            # STOCH/KDJ
            {'col': 'STOCH', 'type': 'range', 'overbought': 80, 'oversold': 20, 'desc_overbought': 'STOCH overbought', 'desc_oversold': 'STOCH oversold'},
            {'col': 'K', 'type': 'range', 'overbought': 80, 'oversold': 20, 'desc_overbought': 'K overbought', 'desc_oversold': 'K oversold'},
            {'col': 'D', 'type': 'range', 'overbought': 80, 'oversold': 20, 'desc_overbought': 'D overbought', 'desc_oversold': 'D oversold'},
            {'col': 'J', 'type': 'range', 'overbought': 100, 'oversold': 0, 'desc_overbought': 'J overbought', 'desc_oversold': 'J oversold'},
            # ATR (volatility)
            {'col': 'ATR', 'type': 'threshold', 'bull': 0, 'bear': 0, 'desc_bull': 'ATR rising (volatility up)', 'desc_bear': 'ATR falling (volatility down)'},
            # OBV (volume)
            {'col': 'OBV', 'type': 'trend', 'desc_bull': 'OBV rising (bullish volume)', 'desc_bear': 'OBV falling (bearish volume)'},
            # VWAP
            {'col': 'VWAP', 'type': 'compare', 'compare_col': 'close', 'op': 'gt', 'desc': 'Price above VWAP (bullish)', 'vote': 'bull'},
            {'col': 'VWAP', 'type': 'compare', 'compare_col': 'close', 'op': 'lt', 'desc': 'Price below VWAP (bearish)', 'vote': 'bear'},
            # PSAR
            {'col': 'PSAR', 'type': 'compare', 'compare_col': 'close', 'op': 'lt', 'desc': 'Price below PSAR (bearish)', 'vote': 'bear'},
            {'col': 'PSAR', 'type': 'compare', 'compare_col': 'close', 'op': 'gt', 'desc': 'Price above PSAR (bullish)', 'vote': 'bull'},
            # ADX
            {'col': 'ADX', 'type': 'threshold', 'bull': 25, 'bear': 0, 'desc_bull': 'Strong trend (ADX > 25)', 'desc_bear': 'Weak trend (ADX < 25)'},
            # CCI
            {'col': 'CCI', 'type': 'range', 'overbought': 100, 'oversold': -100, 'desc_overbought': 'CCI overbought', 'desc_oversold': 'CCI oversold'},
            # WILLR
            {'col': 'WILLR', 'type': 'range', 'overbought': -20, 'oversold': -80, 'desc_overbought': 'WILLR overbought', 'desc_oversold': 'WILLR oversold'},
            # ROC
            {'col': 'ROC', 'type': 'range', 'overbought': 5, 'oversold': -5, 'desc_overbought': 'ROC strong up', 'desc_oversold': 'ROC strong down'},
            # MFI
            {'col': 'MFI', 'type': 'range', 'overbought': 80, 'oversold': 20, 'desc_overbought': 'MFI overbought', 'desc_oversold': 'MFI oversold'},
            # DMIp/DMIm
            {'col': 'DMIp', 'type': 'threshold', 'bull': 20, 'bear': 0, 'desc_bull': 'DMIp strong', 'desc_bear': 'DMIp weak'},
            {'col': 'DMIm', 'type': 'threshold', 'bull': 0, 'bear': 20, 'desc_bull': 'DMIm weak', 'desc_bear': 'DMIm strong'},
            # SMA/EMA cross (if available)
            {'col': 'SMA30', 'type': 'cross', 'compare_col': 'EMA30', 'desc_bull': 'SMA30 crossed above EMA30', 'desc_bear': 'SMA30 crossed below EMA30'},
            {'col': 'SMA100', 'type': 'cross', 'compare_col': 'EMA100', 'desc_bull': 'SMA100 crossed above EMA100', 'desc_bear': 'SMA100 crossed below EMA100'},
            # Volatility Indicators
            {'col': 'STDDEV', 'type': 'range', 'overbought': 2, 'oversold': 0.5, 'desc_overbought': 'STDDEV high volatility', 'desc_oversold': 'STDDEV low volatility'},
        ]))
        # --- Chart Pattern Signal Rules (add here) ---
        pattern_col = row.get('chart_pattern') if 'chart_pattern' in row else None
        if pattern_col and isinstance(pattern_col, str) and pattern_col.strip():
            pattern = pattern_col.lower()
            if 'double bottom' in pattern or 'w pattern' in pattern:
                rationale_bullish.append('Double Bottom pattern detected (bullish reversal)')
                votes_bull += 2
            elif 'double top' in pattern or 'm pattern' in pattern:
                rationale_bearish.append('Double Top pattern detected (bearish reversal)')
                votes_bear += 2
            elif 'head and shoulders' in pattern:
                rationale_bearish.append('Head and Shoulders pattern detected (bearish reversal)')
                votes_bear += 2
            elif 'inverse head and shoulders' in pattern:
                rationale_bullish.append('Inverse Head and Shoulders pattern detected (bullish reversal)')
                votes_bull += 2
            elif 'triangle' in pattern:
                rationale_bullish.append('Triangle pattern detected (potential breakout)')
                votes_bull += 1
            elif 'flag' in pattern:
                rationale_bullish.append('Flag pattern detected (trend continuation)')
                votes_bull += 1
            elif 'pennant' in pattern:
                rationale_bullish.append('Pennant pattern detected (trend continuation)')
                votes_bull += 1
            elif 'cup and handle' in pattern:
                rationale_bullish.append('Cup and Handle pattern detected (bullish continuation)')
                votes_bull += 2
            # Add more patterns as needed
        # --- Candle Pattern Signal Rules ---
        candle_col = row.get('candle_pattern') if 'candle_pattern' in row else None
        if candle_col and isinstance(candle_col, str) and candle_col.strip():
            candle = candle_col.lower()
            if 'bullish engulfing' in candle:
                rationale_bullish.append('Bullish Engulfing candle detected (bullish reversal)')
                votes_bull += 2
            elif 'bearish engulfing' in candle:
                rationale_bearish.append('Bearish Engulfing candle detected (bearish reversal)')
                votes_bear += 2
            elif 'hammer' in candle:
                rationale_bullish.append('Hammer candle detected (bullish reversal)')
                votes_bull += 1
            elif 'shooting star' in candle:
                rationale_bearish.append('Shooting Star candle detected (bearish reversal)')
                votes_bear += 1
            elif 'doji' in candle:
                rationale_bullish.append('Doji candle detected (potential reversal)')
                rationale_bearish.append('Doji candle detected (potential reversal)')
                votes_bull += 0.5
                votes_bear += 0.5
            elif 'morning star' in candle:
                rationale_bullish.append('Morning Star candle detected (bullish reversal)')
                votes_bull += 2
            elif 'evening star' in candle:
                rationale_bearish.append('Evening Star candle detected (bearish reversal)')
                votes_bear += 2
            elif 'piercing line' in candle:
                rationale_bullish.append('Piercing Line candle detected (bullish reversal)')
                votes_bull += 1
            elif 'dark cloud cover' in candle:
                rationale_bearish.append('Dark Cloud Cover candle detected (bearish reversal)')
                votes_bear += 1
            elif 'spinning top' in candle:
                rationale_bullish.append('Spinning Top candle detected (indecision)')
                rationale_bearish.append('Spinning Top candle detected (indecision)')
                votes_bull += 0.5
                votes_bear += 0.5
            # Add more candle patterns as needed
        for rule in indicator_rules:
            col = rule.get('col')
            weight = indicator_weights.get(col, 1)
            if col not in row or pd.isna(row[col]):
                continue
            # --- Adaptive RSI thresholds ---
            if rule['col'] == 'RSI' and adaptive_rsi:
                overbought = adaptive_rsi['overbought']
                oversold = adaptive_rsi['oversold']
            else:
                overbought = rule.get('overbought', 100)
                oversold = rule.get('oversold', 0)
            if rule['type'] == 'threshold':
                if row[col] > rule.get('bull', 0):
                    rationale_bullish.append(rule.get('desc_bull', f'{col} bullish'))
                    votes_bull += weight
                elif row[col] < -rule.get('bear', 0):
                    rationale_bearish.append(rule.get('desc_bear', f'{col} bearish'))
                    votes_bear += weight
            elif rule['type'] == 'range':
                if row[col] > overbought:
                    rationale_bearish.append(rule.get('desc_overbought', f'{col} overbought'))
                    votes_bear += weight
                elif row[col] < oversold:
                    rationale_bullish.append(rule.get('desc_oversold', f'{col} oversold'))
                    votes_bull += weight
            elif rule['type'] == 'compare':
                cmp_col = rule.get('compare_col')
                if cmp_col in row and not pd.isna(row[cmp_col]):
                    if rule.get('op') == 'gt' and row[cmp_col] > row[col]:
                        if rule.get('vote') == 'bull':
                            rationale_bullish.append(rule.get('desc', f'{cmp_col} > {col}'))
                            votes_bull += weight
                        else:
                            rationale_bearish.append(rule.get('desc', f'{cmp_col} > {col}'))
                            votes_bear += weight
                    elif rule.get('op') == 'lt' and row[cmp_col] < row[col]:
                        if rule.get('vote') == 'bull':
                            rationale_bullish.append(rule.get('desc', f'{cmp_col} < {col}'))
                            votes_bull += weight
                        else:
                            rationale_bearish.append(rule.get('desc', f'{cmp_col} < {col}'))
                            votes_bear += weight
            elif rule['type'] == 'trend':
                if idx > 0:
                    prev_val = df[col].iloc[idx-1] if col in df else np.nan
                    if not pd.isna(prev_val):
                        if row[col] > prev_val:
                            rationale_bullish.append(rule.get('desc_bull', f'{col} rising'))
                            votes_bull += weight
                        elif row[col] < prev_val:
                            rationale_bearish.append(rule.get('desc_bear', f'{col} falling'))
                            votes_bear += weight
            elif rule['type'] == 'cross':
                cmp_col = rule.get('compare_col')
                if cmp_col in row and not pd.isna(row[cmp_col]) and idx > 0:
                    prev = df.iloc[idx-1]
                    if row[col] > row[cmp_col] and prev[col] <= prev[cmp_col]:
                        rationale_bullish.append(rule.get('desc_bull', f'{col} crossed above {cmp_col}'))
                        votes_bull += weight
                    elif row[col] < row[cmp_col] and prev[col] >= prev[cmp_col]:
                        rationale_bearish.append(rule.get('desc_bear', f'{col} crossed below {cmp_col}'))
                        votes_bear += weight
        # --- Confirmations: count number of strong bullish/bearish indicators ---
        confirmations = 0
        for rule in indicator_rules:
            col = rule.get('col')
            if col not in row or pd.isna(row[col]):
                continue
            weight = indicator_weights.get(col, 1)
            # For bullish confirmations
            if rule['type'] == 'threshold' and row[col] > rule.get('bull', 0) and weight >= 1.0:
                confirmations += 1
            elif rule['type'] == 'range' and row[col] < rule.get('oversold', 0) and weight >= 1.0:
                confirmations += 1
            elif rule['type'] == 'compare' and rule.get('vote') == 'bull' and col in row and row.get(rule.get('compare_col'), None) is not None and row[rule.get('compare_col')] > row[col] and weight >= 1.0:
                confirmations += 1
        min_confirmations = self.config.get('signal_min_confirmations', 2)  # Increased from 1 to 2
        # --- Signal Smoothing (Debouncing) ---
        smoothing_window = self.config.get('signal_smoothing_window', 3)  # Increased from 1 to 3
        if smoothing_window > 1 and idx >= smoothing_window:
            # Use rolling window majority vote for last N signals
            prev_signals = df.iloc[max(0, idx-smoothing_window):idx]
            prev_votes_bull = 0
            prev_votes_bear = 0
            for j in range(len(prev_signals)):
                # Re-run indicator rules for each previous row (lightweight, or cache if needed)
                prev_row = prev_signals.iloc[j].to_dict()
                for rule in indicator_rules:
                    col = rule.get('col')
                    weight = indicator_weights.get(col, 1)
                    if col not in prev_row or pd.isna(prev_row[col]):
                        continue
                    if rule['type'] == 'threshold':
                        if prev_row[col] > rule.get('bull', 0):
                            prev_votes_bull += weight
                        elif prev_row[col] < -rule.get('bear', 0):
                            prev_votes_bear += weight
                    elif rule['type'] == 'range':
                        if rule['col'] == 'RSI' and adaptive_rsi:
                            overbought = adaptive_rsi['overbought']
                            oversold = adaptive_rsi['oversold']
                        else:
                            overbought = rule.get('overbought', 100)
                            oversold = rule.get('oversold', 0)
                        if prev_row[col] > overbought:
                            prev_votes_bear += weight
                        elif prev_row[col] < oversold:
                            prev_votes_bull += weight
            # Average with current votes
            votes_bull = (votes_bull + prev_votes_bull) / 2
            votes_bear = (votes_bear + prev_votes_bear) / 2
        # --- Consensus Voting Logic (configurable thresholds) ---
        min_bull = self.config.get('min_bull', config.get('min_bull', 3))
        min_bear = self.config.get('min_bear', config.get('min_bear', 3))
        strong_bull = self.config.get('strong_bull', config.get('strong_bull', 5))
        strong_bear = self.config.get('strong_bear', config.get('strong_bear', 5))
        # --- Buy the Dip Logic: Only allow Buy if price is not near recent high and volume is sufficient ---
        recent_window = self.config.get('buy_dip_window', 20)
        price = row.get('close', None)
        volume = row.get('volume', None)
        allow_buy = True
        min_buy_volume = self.config.get('min_buy_volume', 1e5)  # Example: 100,000 shares
        if price is not None and idx >= recent_window:
            recent_high = df['close'].iloc[max(0, idx-recent_window):idx].max()
            # Require a deeper dip for buy (e.g., 7% below high)
            dip_threshold = self.config.get('buy_dip_threshold', 0.93)  # 0.93 = 7% below high
            if price > recent_high * dip_threshold:
                allow_buy = False
        if volume is not None and volume < min_buy_volume:
            allow_buy = False

        # --- Sell the High/High Volume Logic: Only allow Sell if price is near recent high or volume is high ---
        recent_window = self.config.get('sell_high_window', 20)
        allow_sell = True
        if price is not None and idx >= recent_window:
            recent_high = df['close'].iloc[max(0, idx-recent_window):idx].max()
            # Require price to be very close to recent high for sell (e.g., within 2%)
            sell_high_threshold = self.config.get('sell_high_threshold', 0.98)  # 0.98 = within 2% of high
            if price < recent_high * sell_high_threshold:
                allow_sell = False
        # Allow sell if volume is much higher than recent average (potential distribution)
        if volume is not None and idx >= recent_window:
            recent_vol = df['volume'].iloc[max(0, idx-recent_window):idx]
            avg_vol = recent_vol.mean() if not recent_vol.isnull().all() else None
            high_vol_factor = self.config.get('sell_high_vol_factor', 2.0)  # 2x average volume
            if avg_vol and volume > avg_vol * high_vol_factor:
                allow_sell = True

        # --- Consensus Voting Logic (configurable thresholds) ---
        if votes_bull > votes_bear and votes_bull >= min_bull and trend_ok and allow_buy and confirmations >= min_confirmations:
            signal = 'Buy' if votes_bull < strong_bull else 'Strong Buy'
            confidence = min(0.7 + 0.1 * (votes_bull - min_bull), 0.98)
        elif votes_bear > votes_bull and votes_bear >= min_bear and not trend_ok and allow_sell and confirmations >= min_confirmations:
            signal = 'Sell' if votes_bear < strong_bear else 'Strong Sell'
            confidence = min(0.7 + 0.1 * (votes_bear - min_bear), 0.98)
        else:
            signal = 'Hold'
            confidence = 0.5
        score = votes_bull - votes_bear
        # --- Customizable Output Columns ---
        output_cols = self.config.get('output_columns', None)
        output_row = {
            'signal': signal,
            'confidence': confidence,
            'rationale_bullish': '; '.join(rationale_bullish),
            'rationale_bearish': '; '.join(rationale_bearish),
            'score': score
        }
        if output_cols:
            for col in output_cols:
                if col in row:
                    output_row[col] = row[col]
        else:
            output_row.update(row)
        return (
            output_row['signal'],
            output_row['confidence'],
            output_row['rationale_bullish'].split('; '),
            output_row['rationale_bearish'].split('; '),
            output_row['score']
        )

    def _get_expiry(self, row, interval):
        # Expiry: 1 interval ahead (can be customized)
        if 'date' in row:
            try:
                dt = pd.to_datetime(row['date'])
                if interval == '1d':
                    return (dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                elif interval == '1h':
                    return (dt + pd.Timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                return ''
        return ''
