"""
visualizer.py - Visualization module for Stock Analyzer Tool
Generates static and interactive charts for stocks, indicators, and signals.
"""
import os
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    import plotly.graph_objs as go
except ImportError:
    go = None

class Visualizer:
    def __init__(self, config):
        self.config = config
        self.visual_dir = "stock_visualization"
        os.makedirs(self.visual_dir, exist_ok=True)

    def run(self, args):
        print("[Visualizer] Generating charts...")
        if args.tickers:
            for ticker in args.tickers:
                self.plot_stock(ticker)
                self.plot_signals(ticker)
                #self.plot_strategy(ticker)

    def plot_signals(self, ticker): 
        # --- Signal chart (matplotlib) ---
        import numpy as np
        if not os.path.exists("stock_signals"):
            print(f"No signal data directory found, skipping {ticker}.")
            return  
        signal_path = [f for f in os.listdir("stock_signals") if f.startswith(f"{ticker}_signals_") and f.endswith('.csv')]
        if not signal_path:
            print(f"No signal data for {ticker}, skipping.")
            return  
        signal_path = os.path.join("stock_signals", sorted(signal_path, key=lambda x: ("1d" not in x, x))[-1])
        if os.path.exists(signal_path):
            sig_df = pd.read_csv(signal_path)
            if 'date' in sig_df.columns:
                sig_df['date'] = pd.to_datetime(sig_df['date'])
                sig_df = sig_df.sort_values('date').tail(400)
                fig_sig, ax_sig = plt.subplots(figsize=(24, 8))
                ax_sig.plot(sig_df['date'], sig_df['close'], label='Close', color='black', linewidth=1.2, zorder=1)
                # Add volume as background bars
                if 'volume' in sig_df.columns:
                    ax2 = ax_sig.twinx()
                    ax2.bar(sig_df['date'], sig_df['volume'], color='gray', alpha=0.15, width=1, label='Volume', zorder=0)
                    ax2.set_ylabel('Volume', color='gray')
                    ax2.tick_params(axis='y', labelcolor='gray')
                # Vectorized signal masks
                sigs = sig_df['signal'].str.lower().fillna('')
                # Use confidence if present, else fallback to score, else default
                if 'confidence' in sig_df.columns:
                    conf = sig_df['confidence'].fillna(0.5)
                elif 'score' in sig_df.columns:
                    conf = sig_df['score'].fillna(0.5)
                else:
                    conf = np.full(len(sig_df), 0.5)
                # Marker size by confidence/score
                def marker_size(base, mask):
                    arr = np.zeros(len(sig_df))
                    arr[mask.values] = base + 80 * conf[mask].values
                    return arr[mask.values]
                buy_mask = (sigs == 'buy')
                strong_buy_mask = (sigs == 'strong buy')
                sell_mask = (sigs == 'sell')
                strong_sell_mask = (sigs == 'strong sell')
                hold_mask = (sigs == 'hold')
                # Plot markers only if present, size by confidence/score
                if buy_mask.any():
                    ax_sig.scatter(sig_df.loc[buy_mask, 'date'], sig_df.loc[buy_mask, 'close'], marker='^', color='green', s=marker_size(40, buy_mask), label='Buy', zorder=3)
                if strong_buy_mask.any():
                    ax_sig.scatter(sig_df.loc[strong_buy_mask, 'date'], sig_df.loc[strong_buy_mask, 'close'], marker='^', color='lime', s=marker_size(80, strong_buy_mask), label='Strong Buy', zorder=3)
                if sell_mask.any():
                    ax_sig.scatter(sig_df.loc[sell_mask, 'date'], sig_df.loc[sell_mask, 'close'], marker='v', color='red', s=marker_size(40, sell_mask), label='Sell', zorder=3)
                if strong_sell_mask.any():
                    ax_sig.scatter(sig_df.loc[strong_sell_mask, 'date'], sig_df.loc[strong_sell_mask, 'close'], marker='v', color='darkred', s=marker_size(80, strong_sell_mask), label='Strong Sell', zorder=3)
                if hold_mask.any():
                    ax_sig.scatter(sig_df.loc[hold_mask, 'date'], sig_df.loc[hold_mask, 'close'], marker='o', color='gray', s=marker_size(20, hold_mask), label='Hold', zorder=2)
                # Annotate the most recent signal
                last_row = sig_df.iloc[-1]
                conf_val = last_row['confidence'] if 'confidence' in last_row else (last_row['score'] if 'score' in last_row else 0.5)
                ax_sig.annotate(
                    f"{last_row['signal']} ({conf_val:.2f})",
                    xy=(last_row['date'], last_row['close']),
                    xytext=(10, 10), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='blue'),
                    fontsize=11, color='blue', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7)
                )
                ax_sig.set_title(f"{ticker} Signals (marker size=confidence/score)", fontsize=14, fontweight='bold')
                ax_sig.legend(loc='upper left', fontsize=10, ncol=2)
                ax_sig.grid(True, which='both', linestyle='--', alpha=0.5)
                plt.setp(ax_sig.get_xticklabels(), rotation=30, ha='right')
                plt.tight_layout()
                signal_chart_path = os.path.join(self.visual_dir, f"{ticker}_signal_chart.png")
                plt.savefig(signal_chart_path)
                plt.close(fig_sig)
                print(f"Saved signal chart to {signal_chart_path}")



    def plot_stock(self, ticker):
        ind_path = os.path.join("stock_indicator", f"{ticker}_indicators_1d.csv")
        if not os.path.exists(ind_path):
            print(f"No indicator data for {ticker}, skipping.")
            return
        
        df = pd.read_csv(ind_path)
        df['date'] = pd.to_datetime(df['date'])
        # Filter to last 200 days
        df = df[df['date'] >= (pd.to_datetime('today') - pd.DateOffset(days=200))]
        # --- candlestick chart ---
        if plt:
            # --- Static chart (matplotlib) ---
            # Multi-panel chart for indicators (8 panels)
            fig, axs = plt.subplots(10, 1, figsize=(16, 20), sharex=True, gridspec_kw={'height_ratios': [3, 2, 2, 1, 1, 1, 1, 1,1,1]})
            # 1. Price & Trend
            axs[0].grid(True, which='both', linestyle='--', alpha=0.5)
            plotted_any = False
            # 1. Price & Trend (Candlestick)
            if 'close' in df.columns:
                axs[0].plot(df['date'], df['close'], label='Close', color='black', linewidth=1.5)
                plotted_any = True
            # Overlay SMA30, EMA30 if present
            for ma, style, color in zip(['SMA30', 'EMA30'], ['-', '--'], ['blue', 'orange']):
                if ma in df.columns:
                    axs[0].plot(df['date'], df[ma], style, label=ma, color=color, linewidth=1.2)
                    plotted_any = True
            # --- Support/Resistance Lines (limit to 3 most recent, rightmost) ---
            sr_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            sr_count = 0
            # Only consider S/R columns that are not all NaN
            sr_cols = [col for col in df.columns if col.lower().startswith('sr') and not df[col].isna().all()]
            # For each S/R column, get the last non-NaN value and its index
            sr_points = []
            for col in sr_cols:
                non_nan_idx = df[col].last_valid_index()
                if non_nan_idx is not None:
                    sr_points.append((df.loc[non_nan_idx, 'date'], df.loc[non_nan_idx, col], col))
            # Sort by date (rightmost/recent first), then take up to 3 most recent
            sr_points = sorted(sr_points, key=lambda x: x[0], reverse=True)[:3]
            for i, (dt, price, col) in enumerate(sr_points):
                axs[0].axhline(price, color=sr_colors[i % len(sr_colors)], linestyle='--', linewidth=1.5, alpha=0.7, label=f"{col}: {price:.2f}")

            # --- Chart Pattern & Candle Pattern Markers (jitter to avoid overlap) ---
            import numpy as np  # Add at the top if not already imported
            chart_marker_map = {
                'Double Bottom': {'marker': '*', 's': 220, 'edgecolor': 'gold', 'facecolor': 'yellow', 'zorder': 5},
                'Double Top': {'marker': '*', 's': 220, 'edgecolor': 'black', 'facecolor': 'orange', 'zorder': 5},
                'Head and Shoulders': {'marker': 'P', 's': 200, 'edgecolor': 'navy', 'facecolor': 'cyan', 'zorder': 5},
                'Inverse Head and Shoulders': {'marker': 'X', 's': 200, 'edgecolor': 'green', 'facecolor': 'lime', 'zorder': 5},
                'Triangle': {'marker': '^', 's': 180, 'edgecolor': 'brown', 'facecolor': 'wheat', 'zorder': 5},
                'Flag/Pennant': {'marker': 's', 's': 160, 'edgecolor': 'red', 'facecolor': 'pink', 'zorder': 5},
                'Cup and Handle': {'marker': 'o', 's': 180, 'edgecolor': 'blue', 'facecolor': 'aqua', 'zorder': 5},
            }
            candle_marker_map = {
                        'Doji': {'marker': 'D', 'color': 'purple'},
                        'Hammer': {'marker': 'v', 'color': 'green'},
                        'Shooting Star': {'marker': '^', 'color': 'red'},
                        'Bullish Engulfing': {'marker': '>', 'color': 'lime'},
                        'Bearish Engulfing': {'marker': '<', 'color': 'pink'},
                        'Morning Star': {'marker': 'p', 'color': 'gold'},
                        'Evening Star': {'marker': 'h', 'color': 'orange'},
                        'Piercing Line': {'marker': '1', 'color': 'aqua'},
                        'Dark Cloud Cover': {'marker': '2', 'color': 'brown'},
                        'Spinning Top': {'marker': '8', 'color': 'magenta'},
                    }


            if 'chart_pattern' in df.columns:
                for pattern in df['chart_pattern'].dropna().unique():
                    if pattern and pattern.strip():
                        mask = df['chart_pattern'] == pattern
                        style = chart_marker_map.get(pattern, {})
                        # Jitter y-values slightly to avoid overlap
                        yvals = df.loc[mask, 'close'] + np.random.uniform(-0.03, 0.03, mask.sum())
                        axs[0].scatter(
                            df.loc[mask, 'date'], yvals,
                            marker=style.get('marker', '*'),
                            s=style.get('s', 220),
                            label=f'Chart: {pattern}',
                            alpha=0.8,
                            edgecolors=style.get('edgecolor', 'black'),
                            color=style.get('facecolor', 'yellow'),
                            linewidths=1.5,
                            zorder=style.get('zorder', 5)
                        )
            if 'candle_pattern' in df.columns:
                for pattern in df['candle_pattern'].dropna().unique():
                    if pattern and pattern.strip():
                        mask = df['candle_pattern'] == pattern
                        style = candle_marker_map.get(pattern, {})
                        # Jitter y-values slightly to avoid overlap
                        yvals = df.loc[mask, 'close'] + np.random.uniform(-0.03, 0.03, mask.sum())
                        axs[0].scatter(
                            df.loc[mask, 'date'], yvals,
                            marker=style.get('marker', 'D'),
                            s=style.get('s', 120),
                            label=f'Candle: {pattern}',
                            alpha=0.8,
                            edgecolors=style.get('edgecolor', 'purple'),
                            color=style.get('facecolor', 'violet'),
                            linewidths=1,
                            zorder=style.get('zorder', 4)
                        )

            # After all plotting on axs[0] (Price & Trend), set custom y-ticks
            if 'close' in df.columns:
                min_price = df['close'].min()
                max_price = df['close'].max()
                # Extend range a bit for overlays
                min_price = min_price - 0.1 * (max_price - min_price)
                max_price = max_price + 0.1 * (max_price - min_price)
                # Generate ticks at every 0.5 step
                yticks = np.arange(
                    np.floor(min_price * 2) / 2,
                    np.ceil(max_price * 2) / 2 + 0.25,
                    (max_price - min_price) / 10
                )
                axs[0].set_yticks(yticks)
                axs[0].set_ylim(yticks[0], yticks[-1])            

            # --- Bollinger Bands: Add shading between BBU and BBL ---
            if 'BBU' in df.columns and 'BBL' in df.columns:
                axs[1].fill_between(df['date'], df['BBL'], df['BBU'], color='green', alpha=0.12, label='BB Range')

            # --- RSI Panel: Shade overbought/oversold regions and annotate cross events ---
            if 'RSI' in df.columns:
                axs[3].axhspan(70, axs[3].get_ylim()[1], color='red', alpha=0.08)
                axs[3].axhspan(axs[3].get_ylim()[0], 30, color='blue', alpha=0.08)
                axs[3].plot(df['date'], df['RSI'], label='RSI', color='green')
                axs[3].axhline(70, color='gray', linestyle='--', linewidth=0.8)
                axs[3].axhline(30, color='gray', linestyle='--', linewidth=0.8)
                # Annotate RSI cross events
                rsi = df['RSI']
                dates = df['date']
                # Cross above 70
                cross_above_70 = (rsi.shift(1) < 70) & (rsi >= 70)
                for d, v in zip(dates[cross_above_70], rsi[cross_above_70]):
                    axs[3].annotate('↑70', xy=(d, v), xytext=(0, 10), textcoords='offset points',
                                    color='red', fontsize=8, fontweight='bold', ha='center', arrowprops=dict(arrowstyle='-|>', color='red', lw=0.8))
                # Cross below 30
                cross_below_30 = (rsi.shift(1) > 30) & (rsi <= 30)
                for d, v in zip(dates[cross_below_30], rsi[cross_below_30]):
                    axs[3].annotate('↓30', xy=(d, v), xytext=(0, -12), textcoords='offset points',
                                    color='blue', fontsize=8, fontweight='bold', ha='center', arrowprops=dict(arrowstyle='-|>', color='blue', lw=0.8))
                axs[3].legend(loc='upper left', fontsize=9)
            axs[3].set_title('RSI', fontsize=12, fontweight='bold')

            # --- Improved Legend Management ---
            if plotted_any:
                axs[0].legend(loc='best', fontsize=8, ncol=2, frameon=True, handlelength=1.5, borderpad=0.5, labelspacing=0.3)

            # --- Lighter Grid for All Panels ---
            for ax in axs:
                ax.grid(True, which='both', linestyle='--', alpha=0.2)

            axs[0].set_title(f"{ticker} Price & Trend", fontsize=13, fontweight='bold')
            # 2. Price & Bollinger Bands
            axs[1].grid(True, which='both', linestyle='--', alpha=0.5)
            if 'close' in df.columns:
                axs[1].plot(df['date'], df['close'], label='Close', color='black', linewidth=1.2)
            if 'BBM' in df.columns:
                axs[1].plot(df['date'], df['BBM'], label='BBM', color='green', linestyle='-')
            if 'BBU' in df.columns:
                axs[1].plot(df['date'], df['BBU'], label='BBU', color='green', linestyle='--')
            if 'BBL' in df.columns:
                axs[1].plot(df['date'], df['BBL'], label='BBL', color='green', linestyle='--')
            axs[1].legend(loc='upper left', fontsize=9)
            axs[1].set_title('Price & Bollinger Bands', fontsize=12, fontweight='bold')
            # 3. Keltner/Donchian Channels
            axs[2].grid(True, which='both', linestyle='--', alpha=0.5)
            if 'close' in df.columns:
                axs[2].plot(df['date'], df['close'], label='Close', color='black', linewidth=1.2)
            if 'KCB' in df.columns:
                axs[2].plot(df['date'], df['KCB'], label='KCB', color='purple', linestyle=':')
            if 'KCLe' in df.columns:
                axs[2].plot(df['date'], df['KCLe'], label='KCLe', color='purple', linestyle='--')
            if 'KCUe' in df.columns:
                axs[2].plot(df['date'], df['KCUe'], label='KCUe', color='purple', linestyle='--')
            if 'DCM' in df.columns:
                axs[2].plot(df['date'], df['DCM'], label='DCM', color='brown', linestyle=':')
            if 'DCL' in df.columns:
                axs[2].plot(df['date'], df['DCL'], label='DCL', color='brown', linestyle='--')
            if 'DCU' in df.columns:
                axs[2].plot(df['date'], df['DCU'], label='DCU', color='brown', linestyle='--')
            axs[2].legend(loc='upper left', fontsize=9)
            axs[2].set_title('Keltner/Donchian Channels', fontsize=12, fontweight='bold')
            # 4. RSI
            axs[3].grid(True, which='both', linestyle='--', alpha=0.5)
            if 'RSI' in df.columns:
                axs[3].plot(df['date'], df['RSI'], label='RSI', color='green')
                axs[3].axhline(70, color='gray', linestyle='--', linewidth=0.8)
                axs[3].axhline(30, color='gray', linestyle='--', linewidth=0.8)
                axs[3].legend(loc='upper left', fontsize=9)
            axs[3].set_title('RSI', fontsize=12, fontweight='bold')
            # 5. STOCH & KDJ
            axs[4].grid(True, which='both', linestyle='--', alpha=0.5)
            if 'STOCH' in df.columns:
                axs[4].plot(df['date'], df['STOCH'], label='STOCH', color='blue')
            if 'K' in df.columns:
                axs[4].plot(df['date'], df['K'], label='K', color='red', linestyle='-')
            if 'D' in df.columns:
                axs[4].plot(df['date'], df['D'], label='D', color='brown', linestyle='--')
            if 'J' in df.columns:
                axs[4].plot(df['date'], df['J'], label='J', color='orange', linestyle=':')
            axs[4].axhline(80, color='lightblue', linestyle=':', linewidth=0.8)
            axs[4].axhline(20, color='lightblue', linestyle=':', linewidth=0.8)
            axs[4].legend(loc='upper left', fontsize=9)
            axs[4].set_title('STOCH & KDJ', fontsize=12, fontweight='bold')
            # 6. CCI
            axs[5].grid(True, which='both', linestyle='--', alpha=0.5)
            if 'CCI' in df.columns:
                axs[5].plot(df['date'], df['CCI'], label='CCI', color='orange')
                axs[5].axhline(100, color='tan', linestyle=':', linewidth=0.8)
                axs[5].axhline(-100, color='tan', linestyle=':', linewidth=0.8)
                axs[5].legend(loc='upper left', fontsize=9)
            axs[5].set_title('CCI', fontsize=12, fontweight='bold')
            # 7. Volume (Volume, OBV)
            axs[6].grid(True, which='both', linestyle='--', alpha=0.5)
            if 'volume' in df.columns:
                axs[6].bar(df['date'], df['volume'], label='Volume', color='gray', alpha=0.8)
            if 'OBV' in df.columns:
                axs[6].plot(df['date'], df['OBV'], label='OBV', color='purple')
            axs[6].legend(loc='upper left', fontsize=9)
            axs[6].set_title('Volume & OBV', fontsize=12, fontweight='bold')
            # 8. Volatility (ATR)
            axs[7].grid(True, which='both', linestyle='--', alpha=0.5)
            if 'ATR' in df.columns:
                axs[7].plot(df['date'], df['ATR'], label='ATR', color='red')
            axs[7].legend(loc='upper left', fontsize=9)
            axs[7].set_title('Volatility (ATR)', fontsize=12, fontweight='bold')
            # 9. Volatility (HVOL)
            axs[8].grid(True, which='both', linestyle='--', alpha=0.5)
            if 'HVOL' in df.columns:
                axs[8].plot(df['date'], df['HVOL'], label='HVOL', color='magenta')
            axs[8].legend(loc='upper left', fontsize=9)
            axs[8].set_title('Volatility (HVOL)', fontsize=12, fontweight='bold')
            # 9. Volatility (STDDEV)
            axs[9].grid(True, which='both', linestyle='--', alpha=0.5)
            if 'STDDEV' in df.columns:
                axs[9].plot(df['date'], df['STDDEV'], label='STDDEV', color='brown')
            axs[9].legend(loc='upper left', fontsize=9)
            axs[9].set_title('Volatility (STDDEV)', fontsize=12, fontweight='bold')

            
            plt.setp(axs[-1].get_xticklabels(), rotation=30, ha='right')
            plt.tight_layout()
            static_path = os.path.join(self.visual_dir, f"{ticker}_full_indicators.png")
            plt.savefig(static_path)
            plt.close()
            print(f"Saved static chart to {static_path}")
        else:
            print("matplotlib not installed, skipping static chart.")