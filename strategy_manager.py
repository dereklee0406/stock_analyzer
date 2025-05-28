"""
strategy_manager.py - Strategy Management module for Stock Analyzer Tool
Handles strategy templates, backtesting, optimization, and performance reporting.
"""
import os
import pandas as pd
import yaml
from datetime import datetime, timedelta
import numpy as np
import threading

class StrategyManager:
    def __init__(self, config):
        self.config = config
        self.strategy_dir = "stock_strategy"
        os.makedirs(self.strategy_dir, exist_ok=True)

    def run(self, args):
        print("[StrategyManager] Running strategy management...")
        threads = []
        max_threads = 4
        start = getattr(args, 'start', self.config.get('backtest_start', '2020-01-01'))
        end = getattr(args, 'end', self.config.get('backtest_end',  datetime.now().strftime('%Y-%m-%d')))
        if args.tickers:
            for ticker in args.tickers:
                while threading.active_count() >= max_threads + 1:  # +1 for main thread
                    for t in threads:
                        t.join(0.1)
                t = threading.Thread(target=self.run_strategy, args=(ticker, start, end))
                t.start()
                threads.append(t)
        for t in threads:
            t.join()

    def save_equity_curve(self, df, ticker):
        """Save equity curve CSV for visualization, robust to missing date column/index."""
        eq_path = os.path.join(self.strategy_dir, f"{ticker}_equity_curve.csv")
        if 'date' in df.columns:
            df_export = df[['date', 'equity_curve']].dropna(subset=['equity_curve'])
        elif df.index.name == 'date':
            df_reset = df.reset_index()
            df_export = df_reset[['date', 'equity_curve']].dropna(subset=['equity_curve'])
        else:
            # Fallback: create a sequential date index
            df_export = df[['equity_curve']].copy()
            df_export['date'] = pd.date_range(start=0, periods=len(df_export), freq='D')
            df_export = df_export[['date', 'equity_curve']].dropna(subset=['equity_curve'])
        if not df_export.empty:
            df_export.to_csv(eq_path, index=False)
            return eq_path
        return None

    def run_strategy(self, ticker, start, end):
        print(f"Running strategy for {ticker} from {start} to {end}")
        # --- Section: Data Preparation ---
        signals_dir = "stock_signals"
        # Load signals (already includes indicators and OHLCV)
        signal_files = [f for f in os.listdir(signals_dir) if f.startswith(f"{ticker}_signals_") and f.endswith('.csv')]
        if not signal_files:
            print(f"No signals for {ticker}, skipping.")
            return
        preferred = sorted(signal_files, key=lambda x: ("1d" not in x, x))
        sig_path = os.path.join(signals_dir, preferred[0])
        df_signal = pd.read_csv(sig_path)
        # --- Section: Signal to Position Assignment (Regime-aware, Confirmation-based, Flexible Entry/Exit, Monthly Review) ---
        min_hold = 20  # Hold at least 1 month (trading days)
        stop_mult = 2  # ATR stop multiplier
        profit_mult = 4  # ATR take-profit multiplier
        min_volume_mult = 1.2  # Require volume > 1.2x 20-day average
        min_adx = 25  # Only act in trending regime
        min_confirm_days = 2  # Require signal confirmation for 2 consecutive reviews
        # Add ATR if not present
        if 'ATR' not in df_signal.columns and all(x in df_signal.columns for x in ['high', 'low', 'close']):
            df_signal['ATR'] = (df_signal['high'] - df_signal['low']).rolling(window=14, min_periods=1).mean()
        # Add ADX if not present (skip if not available)
        if 'ADX' not in df_signal.columns:
            df_signal['ADX'] = np.nan
        # Add volume filter
        if 'volume' in df_signal.columns:
            df_signal['vol_avg20'] = df_signal['volume'].rolling(window=20, min_periods=1).mean()
        # Confirmation logic: require same strong signal for min_confirm_days consecutive reviews
        df_signal['strong_signal'] = (
            (df_signal['signal'].isin(['Strong Buy', 'Strong Sell'])) &
            (df_signal['signal'] == df_signal['signal'].shift(1))
        )
        df_signal['confirm_count'] = (df_signal['strong_signal'] & (df_signal['strong_signal'] == df_signal['strong_signal'].shift(1))).astype(int).cumsum()
        df_signal['confirm_ok'] = df_signal['confirm_count'] >= min_confirm_days
        df_signal['pos'] = 0
        last_action = 0
        hold_counter = min_hold
        buffered_action = 0
        entry_price = None
        entry_atr = None
        for i, row in df_signal.iterrows():
            date = pd.to_datetime(row['date']) if 'date' in row and not pd.isna(row['date']) else None
            # Regime filter: Only act if ADX >= min_adx and volume > avg
            regime_ok = (row.get('ADX', 0) >= min_adx) and (row.get('volume', 0) > min_volume_mult * row.get('vol_avg20', 1))
            # Confirmation: Only act if strong signal confirmed for min_confirm_days
            confirmed = row.get('confirm_ok', False)
            action = 0
            if row.get('signal', '') == 'Strong Buy' and regime_ok and confirmed:
                action = 1
            elif row.get('signal', '') == 'Strong Sell' and regime_ok and confirmed:
                action = -1
            # ATR-based stop-loss/take-profit
            if last_action != 0 and entry_price is not None and entry_atr is not None and 'close' in row:
                change = (row['close'] - entry_price) / entry_price * last_action
                stop_loss = -stop_mult * entry_atr / entry_price
                take_profit = profit_mult * entry_atr / entry_price
                if change <= stop_loss or change >= take_profit:
                    action = 0  # Exit to cash
            # Allow position change any day, but only if min_hold is satisfied
            if hold_counter >= min_hold and action != last_action:
                df_signal.at[i, 'pos'] = action
                last_action = action
                hold_counter = 1
                if last_action != 0:
                    entry_price = row['close']
                    entry_atr = row.get('ATR', None)
                else:
                    entry_price = None
                    entry_atr = None
            else:
                df_signal.at[i, 'pos'] = last_action
                hold_counter += 1
        # --- Section: Data Filtering ---
        df = df_signal.copy()
        # Ensure 'date' is datetime for filtering
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if start:
            df = df[df['date'] >= pd.to_datetime(start)]
        if end:
            df = df[df['date'] <= pd.to_datetime(end)]
        df = df.reset_index(drop=True)
        # --- Section: Backtest Calculation ---
        initial_equity = 100000
        transaction_cost = 0.001
        slippage = 0.0005
        if 'close' in df.columns:
            df['close'] = pd.to_numeric(df['close'], errors='coerce').ffill().bfill()
        else:
            print(f"No 'close' column found for {ticker}, cannot compute returns.")
            return
        df['returns'] = df['close'].pct_change().fillna(0)
        trade_mask = (df['pos'].shift(1).fillna(0) != df['pos']) & (df['pos'] != 0)
        df['trade_cost'] = 0.0
        df.loc[trade_mask, 'trade_cost'] = float(transaction_cost + slippage)
        df['strategy_returns'] = df['returns'] * df['pos'].shift(1).fillna(0) - df['trade_cost'] * (df['pos'].shift(1).abs() > 0)
        df['equity_curve'] = (1 + df['strategy_returns']).cumprod() * initial_equity
        # --- Section: Results Export ---
        output_path = os.path.join(self.strategy_dir, f"{ticker}_backtest_results.csv")
        df.to_csv(output_path, index=False)
        # Save equity curve for external analysis
        self.save_equity_curve(df, ticker)
        print(f"Strategy for {ticker} completed. Results saved to {output_path}.")
        return output_path
