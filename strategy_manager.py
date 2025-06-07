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
import glob
import matplotlib.pyplot as plt

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
        """Save equity curve CSV for visualization, robust to missing date column/index. Adds more metadata and error handling."""
        eq_path = os.path.join(self.strategy_dir, f"{ticker}_equity_curve.csv")
        meta_path = eq_path.replace('.csv', '.meta.yaml')
        try:
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
                # Save metadata
                meta = {
                    'ticker': ticker,
                    'rows': len(df_export),
                    'exported_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': 'strategy_manager.save_equity_curve',
                }
                import yaml
                with open(meta_path, 'w') as f:
                    yaml.dump(meta, f)
                return eq_path
        except Exception as e:
            print(f"Error saving equity curve for {ticker}: {e}")
        return None

    def run_strategy(self, ticker, start, end):
        """Run the main backtest logic for a single ticker using the loaded strategy."""
        strategy = self.load_strategy('default')
        if not strategy:
            print(f"No default strategy found for {ticker}.")
            return None
        df = self.fetch_and_prepare_data(ticker, start, end, strategy)
        if df.empty:
            print(f"No data for {ticker}.")
            return None
        # Simple backtest: buy/sell signals to positions
        df['pos'] = 0
        df.loc[df['signal'] == 'Buy', 'pos'] = 1
        df.loc[df['signal'] == 'Sell', 'pos'] = -1
        df['pos'] = df['pos'].shift(1).fillna(0)
        df['returns'] = df['close'].pct_change().fillna(0)
        df['strategy_returns'] = df['returns'] * df['pos']
        df['equity_curve'] = 100000 * (1 + df['strategy_returns']).cumprod()
        results = {
            'ticker': ticker,
            'start_date': start,
            'end_date': end,
            'final_equity': df['equity_curve'].iloc[-1] if not df.empty else 100000,
            'total_return': df['equity_curve'].iloc[-1] / 100000 - 1 if not df.empty else 0,
            'sharpe_ratio': self.calculate_sharpe_ratio(df['strategy_returns']),
            'max_drawdown': self.calculate_max_drawdown(df['equity_curve']),
        }
        self.save_results(results, ticker)
        return results

    def calculate_max_drawdown(self, equity_curve):
        if len(equity_curve) == 0:
            return 0
        drawdowns = 1 - equity_curve / equity_curve.cummax()
        return drawdowns.max()

    def calculate_sharpe_ratio(self, strategy_returns, risk_free_rate=0.01):
        if len(strategy_returns) == 0:
            return 0
        excess_returns = strategy_returns - risk_free_rate / 252
        return np.sqrt(252) * (excess_returns.mean() / excess_returns.std(ddof=1)) if excess_returns.std(ddof=1) != 0 else 0

    def print_results(self, results):
        print("\n--- Backtest Results ---")
        for k, v in results.items():
            print(f"{k}: {v}")

    def save_results(self, results, ticker):
        yaml_path = os.path.join(self.strategy_dir, f"{ticker}_results.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(results, f)
        print(f"Results saved to {yaml_path}")

    def load_strategy(self, name):
        path = os.path.join(self.strategy_dir, f"{name}.yaml")
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def save_strategy(self, name, strategy):
        """Save strategy template to YAML file. If 'win_rate' is present, append it to the filename for best winrate tracking."""
        if 'win_rate' in strategy:
            name = f"{name}_winrate_{strategy['win_rate']:.2%}"
        path = os.path.join(self.strategy_dir, f"{name}.yaml")
        with open(path, 'w') as f:
            yaml.dump(strategy, f)
        print(f"Strategy saved to {path}")

    def backtest_strategy(self, ticker, strategy, start, end):
        df = self.fetch_and_prepare_data(ticker, start, end, strategy)
        if df.empty:
            print(f"No data for {ticker}.")
            return None
        df['pos'] = 0
        df.loc[df['signal'] == 'Buy', 'pos'] = 1
        df.loc[df['signal'] == 'Sell', 'pos'] = -1
        df['pos'] = df['pos'].shift(1).fillna(0)
        df['returns'] = df['close'].pct_change().fillna(0)
        df['strategy_returns'] = df['returns'] * df['pos']
        df['equity_curve'] = 100000 * (1 + df['strategy_returns']).cumprod()
        results = {
            'ticker': ticker,
            'final_equity': df['equity_curve'].iloc[-1] if not df.empty else 100000,
            'total_return': df['equity_curve'].iloc[-1] / 100000 - 1 if not df.empty else 0,
            'sharpe_ratio': self.calculate_sharpe_ratio(df['strategy_returns']),
            'max_drawdown': self.calculate_max_drawdown(df['equity_curve']),
        }
        self.save_results(results, ticker)
        return results

    def optimize_strategy(self, ticker, strategy, param_grid, start, end):
        best_strategy = None
        best_return = -np.inf
        from itertools import product
        keys = list(param_grid.keys())
        for values in product(*param_grid.values()):
            params = dict(zip(keys, values))
            test_strategy = strategy.copy()
            test_strategy.update(params)
            results = self.backtest_strategy(ticker, test_strategy, start, end)
            if results and results['total_return'] > best_return:
                best_return = results['total_return']
                best_strategy = test_strategy.copy()
        return best_strategy

    def run_optimization(self, tickers, strategy_name, param_grid, start, end):
        all_results = []
        for ticker in tickers:
            strategy = self.load_strategy(strategy_name)
            if not strategy:
                continue
            best_strategy = self.optimize_strategy(ticker, strategy, param_grid, start, end)
            all_results.append({'ticker': ticker, 'best_strategy': best_strategy})
        return all_results

    def apply_strategy(self, df, strategy):
        df = df.copy()
        df['signal'] = ''
        df['ma_short'] = df['close'].rolling(window=strategy.get('short_ma', 10), min_periods=1).mean()
        df['ma_long'] = df['close'].rolling(window=strategy.get('long_ma', 50), min_periods=1).mean()
        df.loc[df['ma_short'] > df['ma_long'], 'signal'] = 'Buy'
        df.loc[df['ma_short'] < df['ma_long'], 'signal'] = 'Sell'
        df['signal'] = df['signal'].shift(1)
        return df

    def download_data(self, ticker, start, end):
        """Load OHLCV and signal data from local signal file instead of downloading from the web."""
        signals_dir = "stock_signals"
        # Find the most recent signal file for the ticker
        signal_files = [f for f in os.listdir(signals_dir) if f.startswith(f"{ticker}_signals_") and f.endswith('.csv')]
        if not signal_files:
            print(f"No signal file found for {ticker}.")
            return pd.DataFrame()
        preferred = sorted(signal_files, key=lambda x: ("1d" not in x, x))
        sig_path = os.path.join(signals_dir, preferred[0])
        df = pd.read_csv(sig_path)
        # Ensure date column is present and parsed
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # Filter by date range
        if start:
            df = df[df['date'] >= pd.to_datetime(start)]
        if end:
            df = df[df['date'] <= pd.to_datetime(end)]
        return df.reset_index(drop=True)

    def preprocess_data(self, df):
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values('date')
        df = df.dropna()
        return df

    def fetch_and_prepare_data(self, ticker, start, end, strategy):
        df = self.download_data(ticker, start, end)
        if df.empty:
            return df
        df = self.preprocess_data(df)
        df = self.apply_strategy(df, strategy)
        return df

    def update_strategy_parameters(self, strategy, **params):
        for key, value in params.items():
            strategy[key] = value

    def run_single_ticker(self, ticker, start, end, strategy_name, **params):
        strategy = self.load_strategy(strategy_name)
        if not strategy:
            print(f"Strategy {strategy_name} not found.")
            return None
        self.update_strategy_parameters(strategy, **params)
        df = self.fetch_and_prepare_data(ticker, start, end, strategy)
        if df.empty:
            print(f"No data for {ticker}.")
            return None
        # Run backtest
        df['pos'] = 0
        df.loc[df['signal'] == 'Buy', 'pos'] = 1
        df.loc[df['signal'] == 'Sell', 'pos'] = -1
        df['pos'] = df['pos'].shift(1).fillna(0)
        df['returns'] = df['close'].pct_change().fillna(0)
        df['strategy_returns'] = df['returns'] * df['pos']
        df['equity_curve'] = 100000 * (1 + df['strategy_returns']).cumprod()
        results = {
            'ticker': ticker,
            'start_date': start,
            'end_date': end,
            'final_equity': df['equity_curve'].iloc[-1] if not df.empty else 100000,
            'total_return': df['equity_curve'].iloc[-1] / 100000 - 1 if not df.empty else 0,
            'sharpe_ratio': self.calculate_sharpe_ratio(df['strategy_returns']),
            'max_drawdown': self.calculate_max_drawdown(df['equity_curve']),
        }
        self.save_results(results, ticker)
        return results

    def run_multiple_tickers(self, tickers, start, end, strategy_name, **params):
        results = {}
        for ticker in tickers:
            results[ticker] = self.run_single_ticker(ticker, start, end, strategy_name, **params)
        return results

    def generate_report(self, results):
        report = []
        for ticker, res in results.items():
            if res is not None:
                report.append({'ticker': ticker, **res})
        report_df = pd.DataFrame(report)
        return report_df

    def save_report(self, report_df, file_name):
        csv_path = os.path.join(self.strategy_dir, f"{file_name}.csv")
        excel_path = os.path.join(self.strategy_dir, f"{file_name}.xlsx")
        report_df.to_csv(csv_path, index=False)
        report_df.to_excel(excel_path, index=False)
        print(f"Report saved to {csv_path} and {excel_path}")

    def load_signals(self, ticker, date_from, date_to):
        import glob
        signals_dir = "stock_signals"
        file_pattern = os.path.join(signals_dir, f"{ticker}_signals_*.csv")
        all_files = glob.glob(file_pattern)
        df_list = []
        for file in all_files:
            df_temp = pd.read_csv(file)
            df_temp['date'] = pd.to_datetime(df_temp['date'])
            df_list.append(df_temp)
        if df_list:
            df_signals = pd.concat(df_list)
            df_signals = df_signals[(df_signals['date'] >= date_from) & (df_signals['date'] <= date_to)]
            return df_signals
        return pd.DataFrame()

    def generate_trade_log(self, df_signal, df_price):
        trade_log = []
        position = 0
        entry_price = 0
        entry_date = None
        for i, row in df_signal.iterrows():
            if row['pos'] != position:
                if position != 0:
                    # Exit trade
                    exit_price = row['close']
                    exit_date = row['date']
                    pnl = (exit_price - entry_price) * position
                    trade_log.append({'entry_date': entry_date, 'exit_date': exit_date, 'entry_price': entry_price, 'exit_price': exit_price, 'pnl': pnl})
                if row['pos'] != 0:
                    # Enter trade
                    entry_price = row['close']
                    entry_date = row['date']
                position = row['pos']
        return pd.DataFrame(trade_log)

    def analyze_trade_log(self, trade_log_df):
        if trade_log_df.empty:
            return {}
        total_trades = len(trade_log_df)
        winning_trades = len(trade_log_df[trade_log_df['pnl'] > 0])
        losing_trades = len(trade_log_df[trade_log_df['pnl'] < 0])
        total_pnl = trade_log_df['pnl'].sum()
        gross_profit = trade_log_df[trade_log_df['pnl'] > 0]['pnl'].sum()
        gross_loss = trade_log_df[trade_log_df['pnl'] < 0]['pnl'].sum()
        max_win = trade_log_df['pnl'].max()
        max_loss = trade_log_df['pnl'].min()
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'total_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'max_win': max_win,
            'max_loss': max_loss,
            'average_win': gross_profit / winning_trades if winning_trades > 0 else 0,
            'average_loss': gross_loss / losing_trades if losing_trades > 0 else 0,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'loss_rate': losing_trades / total_trades if total_trades > 0 else 0,
            'profit_factor': abs(gross_profit / gross_loss) if gross_loss != 0 else np.inf,
            'expected_return_per_trade': total_pnl / total_trades if total_trades > 0 else 0,
        }

    def optimize_and_backtest(self, tickers, strategy_name, param_grid, start, end):
        """Optimize strategy parameters and backtest for multiple tickers."""
        all_results = {}
        for ticker in tickers:
            strategy = self.load_strategy(strategy_name)
            if strategy:
                best_strategy = self.optimize_strategy(ticker, strategy, param_grid, start, end)
                backtest_results = self.backtest_strategy(ticker, best_strategy, start, end)
                all_results[ticker] = {
                    'best_strategy': best_strategy,
                    'backtest_results': backtest_results,
                }
        return all_results

    def run_pipeline(self, tickers, start, end, strategy_name=None, param_grid=None, interval='1d', signals_df=None):
        """
        Run the complete pipeline: create config from files, run backtest, optimize, and generate report.
        Also runs advanced/utility functions for full coverage.
        """
        all_results = {}
        for ticker in tickers:
            print(f"\nProcessing {ticker}...")
            # 1. Create strategy config from files if not provided
            if not strategy_name:
                config = self.create_strategy_config_from_files(ticker, interval=interval)
                if not config:
                    print(f"No config for {ticker}, skipping.")
                    continue
                strategy_name = config['name']
            # 2. Run backtest
            results = self.run_single_ticker(ticker, start, end, strategy_name)
            all_results[ticker] = results
        # 3. Optimization step (if param_grid provided)
        if param_grid:
            optimization_results = self.run_optimization(tickers, strategy_name, param_grid, start, end)
            for res in optimization_results:
                ticker = res['ticker']
                best_strategy = res['best_strategy']
                if ticker in all_results and best_strategy:
                    all_results[ticker]['optimized_strategy'] = best_strategy
                    # Save best strategy with winrate if available
                    if 'win_rate' in all_results[ticker]:
                        self.save_strategy(f"{ticker}_best", {**best_strategy, 'win_rate': all_results[ticker]['win_rate']})
                    else:
                        self.save_strategy(f"{ticker}_best", best_strategy)
                    # Re-run backtest with optimized strategy
                    results = self.backtest_strategy(ticker, best_strategy, start, end)
                    all_results[ticker]['backtest_results'] = results
        # 4. Advanced/utility function calls for full coverage
        for ticker in tickers:
            strategy = self.load_strategy(strategy_name)
            if not strategy:
                continue
            # optimize_and_backtest
            _ = self.optimize_and_backtest([ticker], strategy_name, param_grid or {}, start, end)
            # run_enhanced_strategy
            _ = self.run_enhanced_strategy(ticker, start, end, strategy_name)
            # compare_strategies (dummy: compare with itself)
            _ = self.compare_strategies(ticker, start, end, strategy_name, strategy_name)
            # If signals_df provided, run signal-based backtests
            if signals_df is not None:
                _ = self.backtest_with_signals(ticker, signals_df, start, end)
                _ = self.run_signal_backtest(ticker, start, end, signals_df)
        # If signals_df provided, run enhanced backtests for all tickers
        if signals_df is not None:
            _ = self.run_enhanced_backtest(tickers, start, end, signals_df)
            _ = self.run_signal_enhanced_backtest(tickers, start, end, signals_df)
        # 5. Generate and save report
        report_df = self.generate_report(all_results)
        self.save_report(report_df, "strategy_performance_report")
        return all_results

    def run_enhanced_strategy(self, ticker, start, end, strategy_name, **params):
        """Run enhanced strategy with multiple parameter sets and optimization."""
        strategy = self.load_strategy(strategy_name)
        if not strategy:
            print(f"Strategy {strategy_name} not found.")
            return
        # Run with default parameters
        results = self.run_single_ticker(ticker, start, end, strategy_name, **params)
        best_strategy = results.get('optimized_strategy', None)
        if best_strategy:
            # Run with optimized parameters
            results_optimized = self.run_single_ticker(ticker, start, end, strategy_name, **best_strategy)
            results['optimized_results'] = results_optimized
        return results

    def compare_strategies(self, ticker, start, end, strategy_name_1, strategy_name_2, **params):
        """Compare two strategies on the same ticker."""
        results_1 = self.run_single_ticker(ticker, start, end, strategy_name_1, **params)
        results_2 = self.run_single_ticker(ticker, start, end, strategy_name_2, **params)
        return {
            'strategy_name_1': strategy_name_1,
            'strategy_name_2': strategy_name_2,
            'results_1': results_1,
            'results_2': results_2,
        }

    def visualize_equity_curve(self, df, title="Equity Curve"):
        """Visualize the equity curve of a strategy."""
        plt.figure(figsize=(10, 6))
        plt.plot(df['date'], df['equity_curve'], label='Equity Curve')
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.legend()
        plt.grid()
        plt.show()

    def visualize_signals(self, df_signal, df_price, title="Signals Visualization"):
        """Visualize buy/sell signals on the price chart."""
        plt.figure(figsize=(10, 6))
        plt.plot(df_price['date'], df_price['close'], label='Close Price')
        plt.scatter(df_signal[df_signal['pos'] == 1]['date'], df_signal[df_signal['pos'] == 1]['close'], label='Buy Signal', marker='^', color='g')
        plt.scatter(df_signal[df_signal['pos'] == -1]['date'], df_signal[df_signal['pos'] == -1]['close'], label='Sell Signal', marker='v', color='r')
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid()
        plt.show()

    def visualize_trade_log(self, trade_log_df, title="Trade Log Visualization"):
        """Visualize the trade log as a series of trades on the price chart."""
        plt.figure(figsize=(10, 6))
        plt.plot(trade_log_df['date'], trade_log_df['entry_price'], label='Entry Price', marker='o')
        plt.plot(trade_log_df['date'], trade_log_df['exit_price'], label='Exit Price', marker='x')
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid()
        plt.show()

    def backtest_with_signals(self, ticker, signals_df, start, end):
        """Backtest a strategy using external signals and export trade log."""
        print(f"Backtesting {ticker} with external signals from {start} to {end}")
        df_price = self.download_data(ticker, start, end)
        df_price = self.preprocess_data(df_price)
        df_signal = signals_df[(signals_df['ticker'] == ticker) & (signals_df['date'] >= start) & (signals_df['date'] <= end)]
        df_signal = df_signal.sort_values('date')
        df_signal = df_signal[df_signal['date'].notna()]
        df_signal['pos'] = 0
        df_signal.loc[df_signal['signal'] == 'Buy', 'pos'] = 1
        df_signal.loc[df_signal['signal'] == 'Sell', 'pos'] = -1
        df_signal['pos'] = df_signal['pos'].shift(1)  # Delay position by 1 day
        # Merge with price data
        df_merged = pd.merge(df_price, df_signal[['date', 'pos']], on='date', how='left')
        df_merged['pos'] = df_merged['pos'].fillna(0)
        df_merged['returns'] = df_merged['close'].pct_change().fillna(0)
        df_merged['strategy_returns'] = df_merged['returns'] * df_merged['pos'].shift(1).fillna(0)
        df_merged['cumulative_strategy_returns'] = (1 + df_merged['strategy_returns']).cumprod()
        df_merged['equity_curve'] = 100000 * df_merged['cumulative_strategy_returns']
        # Generate and export trade log
        trade_log = self.generate_trade_log(df_merged, df_merged)
        trade_log_path = os.path.join(self.strategy_dir, f"{ticker}_trade_log.csv")
        if not trade_log.empty:
            trade_log.to_csv(trade_log_path, index=False)
            print(f"Trade log exported to {trade_log_path}")
        # --- Section: Results Compilation ---
        results = {
            'ticker': ticker,
            'initial_equity': 100000,
            'final_equity': df_merged['equity_curve'].iloc[-1] if not df_merged.empty else 100000,
            'total_return': df_merged['equity_curve'].iloc[-1] - 100000,
            'annualized_return': ((df_merged['equity_curve'].iloc[-1] / 100000) ** (365.25 / len(df_merged)) - 1) if len(df_merged) > 0 else 0,
            'max_drawdown': self.calculate_max_drawdown(df_merged['equity_curve']),
            'sharpe_ratio': self.calculate_sharpe_ratio(df_merged['strategy_returns']),
            'trades': df_merged['pos'].diff().abs().sum() / 2,
            'winning_trades': df_merged[df_merged['strategy_returns'] > 0]['pos'].diff().abs().sum() / 2,
            'losing_trades': df_merged[df_merged['strategy_returns'] < 0]['pos'].diff().abs().sum() / 2,
            'equity_curve_path': None,
            'trade_log': trade_log_path if not trade_log.empty else None
        }
        results['win_rate'] = results['trades'] > 0 and results['winning_trades'] / results['trades'] or 0
        results['loss_rate'] = results['trades'] > 0 and results['losing_trades'] / results['trades'] or 0
        results['profit_factor'] = results['losing_trades'] > 0 and abs(results['winning_trades'] / results['losing_trades']) or 0
        results['expected_return_per_trade'] = results['trades'] > 0 and results['total_return'] / results['trades'] or 0
        results['annualized_volatility'] = df_merged['strategy_returns'].std() * np.sqrt(252) if len(df_merged) > 0 else 0
        # Print and return results
        self.print_results(results)
        return results

    def run_signal_backtest(self, ticker, start, end, signals_df):
        """Run backtest using external signals DataFrame."""
        print(f"Running backtest for {ticker} from {start} to {end} using external signals")
        df_price = self.download_data(ticker, start, end)
        df_price = self.preprocess_data(df_price)
        df_signal = signals_df[(signals_df['ticker'] == ticker) & (signals_df['date'] >= start) & (signals_df['date'] <= end)]
        df_signal = df_signal.sort_values('date')
        df_signal = df_signal[df_signal['date'].notna()]
        df_signal['pos'] = 0
        df_signal.loc[df_signal['signal'] == 'Buy', 'pos'] = 1
        df_signal.loc[df_signal['signal'] == 'Sell', 'pos'] = -1
        df_signal['pos'] = df_signal['pos'].shift(1)  # Delay position by 1 day
        # Merge with price data
        df_merged = pd.merge(df_price, df_signal[['date', 'pos']], on='date', how='left')
        df_merged['pos'] = df_merged['pos'].fillna(0)
        df_merged['returns'] = df_merged['close'].pct_change().fillna(0)
        df_merged['strategy_returns'] = df_merged['returns'] * df_merged['pos'].shift(1).fillna(0)
        df_merged['cumulative_strategy_returns'] = (1 + df_merged['strategy_returns']).cumprod()
        df_merged['equity_curve'] = 100000 * df_merged['cumulative_strategy_returns']
        # --- Section: Results Compilation ---
        results = {
            'ticker': ticker,
            'initial_equity': 100000,
            'final_equity': df_merged['equity_curve'].iloc[-1] if not df_merged.empty else 100000,
            'total_return': df_merged['equity_curve'].iloc[-1] - 100000,
            'annualized_return': ((df_merged['equity_curve'].iloc[-1] / 100000) ** (365.25 / len(df_merged)) - 1) if len(df_merged) > 0 else 0,
            'max_drawdown': self.calculate_max_drawdown(df_merged['equity_curve']),
            'sharpe_ratio': self.calculate_sharpe_ratio(df_merged['strategy_returns']),
            'trades': df_merged['pos'].diff().abs().sum() / 2,
            'winning_trades': df_merged[df_merged['strategy_returns'] > 0]['pos'].diff().abs().sum() / 2,
            'losing_trades': df_merged[df_merged['strategy_returns'] < 0]['pos'].diff().abs().sum() / 2,
            'equity_curve_path': None,  # Not saving equity curve in this method
            'trade_log': []
        }
        results['win_rate'] = results['trades'] > 0 and results['winning_trades'] / results['trades'] or 0
        results['loss_rate'] = results['trades'] > 0 and results['losing_trades'] / results['trades'] or 0
        results['profit_factor'] = results['losing_trades'] > 0 and abs(results['winning_trades'] / results['losing_trades']) or 0
        results['expected_return_per_trade'] = results['trades'] > 0 and results['total_return'] / results['trades'] or 0
        results['annualized_volatility'] = df_merged['strategy_returns'].std() * np.sqrt(252) if len(df_merged) > 0 else 0
        # Print and return results
        self.print_results(results)
        return results

    def run_enhanced_backtest(self, tickers, start, end, signals_df):
        """Run enhanced backtest with multiple parameter sets and optimization."""
        all_results = {}
        for ticker in tickers:
            print(f"\nProcessing {ticker}...")
            df_price = self.download_data(ticker, start, end)
            df_price = self.preprocess_data(df_price)
            df_signal = signals_df[(signals_df['ticker'] == ticker) & (signals_df['date'] >= start) & (signals_df['date'] <= end)]
            df_signal = df_signal.sort_values('date')
            df_signal = df_signal[df_signal['date'].notna()]
            df_signal['pos'] = 0
            df_signal.loc[df_signal['signal'] == 'Buy', 'pos'] = 1
            df_signal.loc[df_signal['signal'] == 'Sell', 'pos'] = -1
            df_signal['pos'] = df_signal['pos'].shift(1)  # Delay position by 1 day
            # Merge with price data
            df_merged = pd.merge(df_price, df_signal[['date', 'pos']], on='date', how='left')
            df_merged['pos'] = df_merged['pos'].fillna(0)
            df_merged['returns'] = df_merged['close'].pct_change().fillna(0)
            df_merged['strategy_returns'] = df_merged['returns'] * df_merged['pos'].shift(1).fillna(0)
            df_merged['cumulative_strategy_returns'] = (1 + df_merged['strategy_returns']).cumprod()
            df_merged['equity_curve'] = 100000 * df_merged['cumulative_strategy_returns']
            # --- Section: Results Compilation ---
            results = {
                'ticker': ticker,
                'initial_equity': 100000,
                'final_equity': df_merged['equity_curve'].iloc[-1] if not df_merged.empty else 100000,
                'total_return': df_merged['equity_curve'].iloc[-1] - 100000,
                'annualized_return': ((df_merged['equity_curve'].iloc[-1] / 100000) ** (365.25 / len(df_merged)) - 1) if len(df_merged) > 0 else 0,
                'max_drawdown': self.calculate_max_drawdown(df_merged['equity_curve']),
                'sharpe_ratio': self.calculate_sharpe_ratio(df_merged['strategy_returns']),
                'trades': df_merged['pos'].diff().abs().sum() / 2,
                'winning_trades': df_merged[df_merged['strategy_returns'] > 0]['pos'].diff().abs().sum() / 2,
                'losing_trades': df_merged[df_merged['strategy_returns'] < 0]['pos'].diff().abs().sum() / 2,
                'equity_curve_path': None,  # Not saving equity curve in this method
                'trade_log': []
            }
            results['win_rate'] = results['trades'] > 0 and results['winning_trades'] / results['trades'] or 0
            results['loss_rate'] = results['trades'] > 0 and results['losing_trades'] / results['trades'] or 0
            results['profit_factor'] = results['losing_trades'] > 0 and abs(results['winning_trades'] / results['losing_trades']) or 0
            results['expected_return_per_trade'] = results['trades'] > 0 and results['total_return'] / results['trades'] or 0
            results['annualized_volatility'] = df_merged['strategy_returns'].std() * np.sqrt(252) if len(df_merged) > 0 else 0
            # Print and save results
            self.print_results(results)
            all_results[ticker] = results
        return all_results

    def run_signal_enhanced_backtest(self, tickers, start, end, signals_df):
        """Run enhanced backtest using external signals with multiple parameter sets and optimization."""
        all_results = {}
        for ticker in tickers:
            print(f"\nProcessing {ticker}...")
            df_price = self.download_data(ticker, start, end)
            df_price = self.preprocess_data(df_price)
            df_signal = signals_df[(signals_df['ticker'] == ticker) & (signals_df['date'] >= start) & (signals_df['date'] <= end)]
            df_signal = df_signal.sort_values('date')
            df_signal = df_signal[df_signal['date'].notna()]
            df_signal['pos'] = 0
            df_signal.loc[df_signal['signal'] == 'Buy', 'pos'] = 1
            df_signal.loc[df_signal['signal'] == 'Sell', 'pos'] = -1
            df_signal['pos'] = df_signal['pos'].shift(1)  # Delay position by 1 day
            # Merge with price data
            df_merged = pd.merge(df_price, df_signal[['date', 'pos']], on='date', how='left')
            df_merged['pos'] = df_merged['pos'].fillna(0)
            df_merged['returns'] = df_merged['close'].pct_change().fillna(0)
            df_merged['strategy_returns'] = df_merged['returns'] * df_merged['pos'].shift(1).fillna(0)
            df_merged['cumulative_strategy_returns'] = (1 + df_merged['strategy_returns']).cumprod()
            df_merged['equity_curve'] = 100000 * df_merged['cumulative_strategy_returns']
            # --- Section: Results Compilation ---
            results = {
                'ticker': ticker,
                'initial_equity': 100000,
                'final_equity': df_merged['equity_curve'].iloc[-1] if not df_merged.empty else 100000,
                'total_return': df_merged['equity_curve'].iloc[-1] - 100000,
                'annualized_return': ((df_merged['equity_curve'].iloc[-1] / 100000) ** (365.25 / len(df_merged)) - 1) if len(df_merged) > 0 else 0,
                'max_drawdown': self.calculate_max_drawdown(df_merged['equity_curve']),
                'sharpe_ratio': self.calculate_sharpe_ratio(df_merged['strategy_returns']),
                'trades': df_merged['pos'].diff().abs().sum() / 2,
                'winning_trades': df_merged[df_merged['strategy_returns'] > 0]['pos'].diff().abs().sum() / 2,
                'losing_trades': df_merged[df_merged['strategy_returns'] < 0]['pos'].diff().abs().sum() / 2,
                'equity_curve_path': None,  # Not saving equity curve in this method
                'trade_log': []
            }
            results['win_rate'] = results['trades'] > 0 and results['winning_trades'] / results['trades'] or 0
            results['loss_rate'] = results['trades'] > 0 and results['losing_trades'] / results['trades'] or 0
            results['profit_factor'] = results['losing_trades'] > 0 and abs(results['winning_trades'] / results['losing_trades']) or 0
            results['expected_return_per_trade'] = results['trades'] > 0 and results['total_return'] / results['trades'] or 0
            results['annualized_volatility'] = df_merged['strategy_returns'].std() * np.sqrt(252) if len(df_merged) > 0 else 0
            # Print and save results
            self.print_results(results)
            all_results[ticker] = results
        return all_results

    def create_strategy_config_from_files(self, ticker, interval='1d'):
        """
        Create a strategy config by reading the latest signal and indicator files for the ticker/interval.
        The config will include available columns and can be extended for feature selection or parameter tuning.
        """
        import os
        import yaml
        signals_dir = "stock_signals"
        indicators_dir = "stock_indicator"
        # Find latest signal file
        signal_files = [f for f in os.listdir(signals_dir) if f.startswith(f"{ticker}_signals_{interval}") and f.endswith('.csv')]
        indicator_files = [f for f in os.listdir(indicators_dir) if f.startswith(f"{ticker}_indicators_{interval}") and f.endswith('.csv')]
        if not signal_files or not indicator_files:
            print(f"Signal or indicator file not found for {ticker} {interval}.")
            return None
        signal_path = os.path.join(signals_dir, sorted(signal_files)[-1])
        indicator_path = os.path.join(indicators_dir, sorted(indicator_files)[-1])
        df_signal = pd.read_csv(signal_path)
        df_indicator = pd.read_csv(indicator_path)
        # Identify available features
        signal_cols = [col for col in df_signal.columns if col not in ['date', 'ticker']]
        indicator_cols = [col for col in df_indicator.columns if col not in ['date', 'ticker']]
        # Example config: use all available features
        config = {
            'name': f'{ticker}_{interval}_auto',
            'features': signal_cols + indicator_cols,
            'signal_features': signal_cols,
            'indicator_features': indicator_cols,
            'short_ma': 10,
            'long_ma': 50,
            'target': 'signal',
            'interval': interval
        }
        # Save config
        config_path = os.path.join(self.strategy_dir, f"{ticker}_{interval}_auto.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        print(f"Strategy config saved to {config_path}")
        return config

    # Ensure all required imports are at the top
    # import os, pandas as pd, yaml, numpy as np, threading, glob, matplotlib.pyplot as plt
    # All methods should have proper indentation and not be empty
    # All file and directory accesses should be checked for existence
    # All DataFrame operations should check for empty DataFrames
    # All YAML and CSV saves should use safe methods
    # All method calls should use self consistently
    # All method docstrings should be present for clarity
    # All error messages should be clear and actionable
    # All strategy configs should be saved in stock_strategy
    # All results should be saved in stock_strategy or backtest_results
    # All visualization methods should import matplotlib.pyplot as plt inside the method
    # All code should be Python 3.8+ compatible
    # All logic should be robust to missing or malformed data
    # All user-facing print statements should be clear and informative
    # All method returns should be consistent and documented
    # All YAML configs should be human-readable
    # All file paths should use os.path.join
    # All DataFrame merges should use appropriate keys and join types
    # All parameter defaults should be reasonable for typical use
    # All optimization and grid search logic should be efficient for small parameter grids
    # All strategy application logic should be modular and extensible
    # All code should be ready to run in the current workspace structure

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Strategy Manager Pipeline Runner")
    parser.add_argument('--tickers', nargs='+', required=True, help='List of tickers to process')
    parser.add_argument('--start', type=str, default='2020-01-01', help='Backtest start date')
    parser.add_argument('--end', type=str, default=datetime.now().strftime('%Y-%m-%d'), help='Backtest end date')
    parser.add_argument('--interval', type=str, default='1d', help='Interval for signals/indicators')
    parser.add_argument('--strategy_name', type=str, default=None, help='Strategy config name (optional)')
    parser.add_argument('--signals_file', type=str, default=None, help='External signals CSV (optional)')
    parser.add_argument('--param_grid', type=str, default=None, help='Parameter grid YAML file (optional)')
    args = parser.parse_args()

    config = {}
    manager = StrategyManager(config)
    signals_df = None
    if args.signals_file:
        signals_df = pd.read_csv(args.signals_file)
    param_grid = None
    if args.param_grid:
        with open(args.param_grid, 'r') as f:
            param_grid = yaml.safe_load(f)
    manager.run_pipeline(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        strategy_name=args.strategy_name,
        param_grid=param_grid,
        interval=args.interval,
        signals_df=signals_df
    )

