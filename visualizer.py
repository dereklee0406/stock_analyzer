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
                sig_df = sig_df.sort_values('date').tail(200)
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
                ax_sig.set_title(f"{ticker} Signals (last 200 days, marker size=confidence/score)", fontsize=14, fontweight='bold')
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
            try:
                import mplfinance as mpf
                use_mpf = True
            except ImportError:
                use_mpf = False
            if use_mpf and {'open','high','low','close','date'}.issubset(df.columns):
                df_mpf = df[['date','open','high','low','close','volume']].copy()
                df_mpf.set_index('date', inplace=True)
                # Add overlays for SMA30, EMA30
                addplots = []
                if 'SMA30' in df.columns:
                    addplots.append(mpf.make_addplot(df.set_index('date')['SMA30'], color='blue'))
                if 'EMA30' in df.columns:
                    addplots.append(mpf.make_addplot(df.set_index('date')['EMA30'], color='orange', linestyle='dashed'))
                # --- Candle Pattern Markers on Candlestick Chart ---
                ap_candle = []
                if 'candle_pattern' in df.columns:
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
                    df_by_date = df.set_index('date')
                    for pattern in df['candle_pattern'].dropna().unique():
                        if pattern and pattern.strip():
                            mask = df['candle_pattern'] == pattern
                            style = candle_marker_map.get(pattern, {'marker': 'D', 'color': 'violet'})
                            if mask.sum() > 0:
                                # Get the dates for the pattern
                                pattern_dates = df.loc[mask, 'date']
                                # Get close values for those dates, aligned to df_mpf index
                                yvals = pd.Series(index=df_mpf.index, dtype='float64')
                                yvals[:] = float('nan')
                                close_vals = df_by_date.loc[pattern_dates, 'close'].values
                                yvals.loc[pattern_dates] = close_vals
                                ap_candle.append(mpf.make_addplot(
                                    yvals,
                                    scatter=True,
                                    markersize=150,
                                    marker=style['marker'],
                                    color=style['color'],
                                    panel=0,
                                    secondary_y=False,
                                    ylabel='',
                                    label=f'Candle: {pattern}'
                                ))
                # Candlestick chart with overlays and candle pattern markers
                mpf.plot(df_mpf, type='candle', style='yahoo', addplot=addplots+ap_candle,
                         title=f"{ticker} Price & Trend (Candlestick)", ylabel='Price',
                         volume=False, mav=(),
                         figscale=1.5, figratio=(14,3), tight_layout=True, datetime_format='%Y-%m',
                         xrotation=20, savefig=os.path.join(self.visual_dir, f"{ticker}_candlestick.png"))
                print(f"Saved candlestick chart to {os.path.join(self.visual_dir, f'{ticker}_candlestick.png')}")
            
            # --- Static chart (matplotlib) ---
            # Multi-panel chart for indicators (8 panels)
            fig, axs = plt.subplots(10, 1, figsize=(16, 20), sharex=True, gridspec_kw={'height_ratios': [3, 2, 2, 1, 1, 1, 1, 1,1,1]})
            # 1. Price & Trend
            axs[0].grid(True, which='both', linestyle='--', alpha=0.5)
            plotted_any = False
            if 'close' in df.columns:
                axs[0].plot(df['date'], df['close'], label='Close', color='black', linewidth=1.5)
                plotted_any = True
            for ma, style, color in zip(['SMA30', 'EMA30'], ['-', '--'], ['blue', 'orange']):
                if ma in df.columns:
                    axs[0].plot(df['date'], df[ma], style, label=ma, color=color, linewidth=1.2)
                    plotted_any = True
            # --- Chart Pattern & Candle Pattern Markers (different marker for each pattern) ---
            chart_marker_map = {
                'Double Bottom': {'marker': '*', 's': 180, 'edgecolor': 'gold', 'facecolor': 'yellow', 'zorder': 5},
                'Double Top': {'marker': '*', 's': 180, 'edgecolor': 'black', 'facecolor': 'orange', 'zorder': 5},
                'Head and Shoulders': {'marker': 'P', 's': 160, 'edgecolor': 'navy', 'facecolor': 'cyan', 'zorder': 5},
                'Inverse Head and Shoulders': {'marker': 'X', 's': 160, 'edgecolor': 'green', 'facecolor': 'lime', 'zorder': 5},
                'Triangle': {'marker': '^', 's': 120, 'edgecolor': 'brown', 'facecolor': 'wheat', 'zorder': 5},
                'Flag/Pennant': {'marker': 's', 's': 100, 'edgecolor': 'red', 'facecolor': 'pink', 'zorder': 5},
                'Cup and Handle': {'marker': 'o', 's': 120, 'edgecolor': 'blue', 'facecolor': 'aqua', 'zorder': 5},
            }
            candle_marker_map = {
                'Doji': {'marker': 'D', 's': 60, 'edgecolor': 'purple', 'facecolor': 'violet', 'zorder': 4},
                'Hammer': {'marker': 'v', 's': 70, 'edgecolor': 'green', 'facecolor': 'lightgreen', 'zorder': 4},
                'Shooting Star': {'marker': '^', 's': 70, 'edgecolor': 'red', 'facecolor': 'salmon', 'zorder': 4},
                'Bullish Engulfing': {'marker': '>', 's': 80, 'edgecolor': 'darkgreen', 'facecolor': 'lime', 'zorder': 4},
                'Bearish Engulfing': {'marker': '<', 's': 80, 'edgecolor': 'darkred', 'facecolor': 'pink', 'zorder': 4},
                'Morning Star': {'marker': 'p', 's': 90, 'edgecolor': 'gold', 'facecolor': 'yellow', 'zorder': 4},
                'Evening Star': {'marker': 'h', 's': 90, 'edgecolor': 'black', 'facecolor': 'orange', 'zorder': 4},
                'Piercing Line': {'marker': '1', 's': 80, 'edgecolor': 'blue', 'facecolor': 'aqua', 'zorder': 4},
                'Dark Cloud Cover': {'marker': '2', 's': 80, 'edgecolor': 'brown', 'facecolor': 'wheat', 'zorder': 4},
                'Spinning Top': {'marker': '8', 's': 60, 'edgecolor': 'magenta', 'facecolor': 'pink', 'zorder': 4},
            }
            # --- Support/Resistance Lines ---
            sr_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            sr_count = 0
            for col in df.columns:
                if col.lower().startswith('sr'):
                    price = df[col].iloc[-1]
                    axs[0].axhline(price, color=sr_colors[sr_count % len(sr_colors)], linestyle='--', linewidth=1.5, alpha=0.7, label=f"{col}: {price:.2f}")
                    sr_count += 1
            if 'chart_pattern' in df.columns:
                for pattern in df['chart_pattern'].dropna().unique():
                    if pattern and pattern.strip():
                        mask = df['chart_pattern'] == pattern
                        style = chart_marker_map.get(pattern, {'marker': '*', 's': 120, 'edgecolor': 'gray', 'facecolor': 'yellow', 'zorder': 5})
                        axs[0].scatter(df.loc[mask, 'date'], df.loc[mask, 'close'],
                            marker=style['marker'], s=style['s'], label=f'Chart: {pattern}',
                            alpha=0.7, edgecolor=style['edgecolor'], facecolor=style['facecolor'],
                            linewidths=1.5, zorder=style['zorder'])
            if 'candle_pattern' in df.columns:
                for pattern in df['candle_pattern'].dropna().unique():
                    if pattern and pattern.strip():
                        mask = df['candle_pattern'] == pattern
                        style = candle_marker_map.get(pattern, {'marker': 'D', 's': 60, 'edgecolor': 'purple', 'facecolor': 'violet', 'zorder': 4})
                        axs[0].scatter(df.loc[mask, 'date'], df.loc[mask, 'close'],
                            marker=style['marker'], s=style['s'], label=f'Candle: {pattern}',
                            alpha=0.7, edgecolor=style['edgecolor'], facecolor=style['facecolor'],
                            linewidths=1, zorder=style['zorder'])
            if plotted_any:
                axs[0].legend(loc='upper left', fontsize=9, ncol=2)
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
            # 5. STOCH
            axs[4].grid(True, which='both', linestyle='--', alpha=0.5)
            if 'STOCH' in df.columns:
                axs[4].plot(df['date'], df['STOCH'], label='STOCH', color='blue')
                #axs[4].plot(df['date'], df['K'], label='K', color='red')
                #axs[4].plot(df['date'], df['D'], label='D', color='brown')
                #axs[4].plot(df['date'], df['J'], label='J', color='orange')
                axs[4].axhline(80, color='lightblue', linestyle=':', linewidth=0.8)
                axs[4].axhline(20, color='lightblue', linestyle=':', linewidth=0.8)
                axs[4].legend(loc='upper left', fontsize=9)
            axs[4].set_title('STOCH', fontsize=12, fontweight='bold')
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
            # 89. Volatility (HVOL)
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
        # --- Interactive chart (plotly) ---
        if go:
            from plotly.subplots import make_subplots
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                                subplot_titles=(
                                    f"{ticker} Price & Trend",
                                    "Bands & Channels",
                                    "Momentum Indicators",
                                    "Volume & Volatility"
                                ))
            # 1. Price & Trend
            fig.add_trace(go.Scatter(x=df['date'], y=df['close'], name='Close', line=dict(color='black', width=2)), row=1, col=1)
            for ma, style, color in zip(['SMA30', 'EMA30'], ['solid', 'dash'], ['blue', 'orange']):
                if ma in df.columns:
                    fig.add_trace(go.Scatter(x=df['date'], y=df[ma], name=ma, line=dict(color=color, dash=style)), row=1, col=1)
            # 2. Bands/Channels
            if 'BBM' in df.columns:
                fig.add_trace(go.Scatter(x=df['date'], y=df['BBM'], name='BBM', line=dict(color='green', dash='solid')), row=2, col=1)
            if 'BBU' in df.columns:
                fig.add_trace(go.Scatter(x=df['date'], y=df['BBU'], name='BBU', line=dict(color='green', dash='dash')), row=2, col=1)
            if 'BBL' in df.columns:
                fig.add_trace(go.Scatter(x=df['date'], y=df['BBL'], name='BBL', line=dict(color='green', dash='dash')), row=2, col=1)
            if 'KCB' in df.columns:
                fig.add_trace(go.Scatter(x=df['date'], y=df['KCB'], name='KCB', line=dict(color='purple', dash='dot')), row=2, col=1)
            if 'DCM' in df.columns:
                fig.add_trace(go.Scatter(x=df['date'], y=df['DCM'], name='DCM', line=dict(color='brown', dash='dot')), row=2, col=1)
            # 3. Momentum
            if 'RSI' in df.columns:
                fig.add_trace(go.Scatter(x=df['date'], y=df['RSI'], name='RSI', line=dict(color='green')), row=3, col=1)
            if 'STOCH' in df.columns:
                fig.add_trace(go.Scatter(x=df['date'], y=df['STOCH'], name='STOCH', line=dict(color='blue')), row=3, col=1)
            if 'CCI' in df.columns:
                fig.add_trace(go.Scatter(x=df['date'], y=df['CCI'], name='CCI', line=dict(color='orange')), row=3, col=1)
            # 4. Volume & Volatility
            if 'volume' in df.columns:
                fig.add_trace(go.Bar(x=df['date'], y=df['volume'], name='Volume', marker_color='gray', opacity=0.3), row=4, col=1)
            if 'OBV' in df.columns:
                fig.add_trace(go.Scatter(x=df['date'], y=df['OBV'], name='OBV', line=dict(color='purple')), row=4, col=1)
            if 'ATR' in df.columns:
                fig.add_trace(go.Scatter(x=df['date'], y=df['ATR'], name='ATR', line=dict(color='red')), row=4, col=1)
            if 'STDDEV' in df.columns:
                fig.add_trace(go.Scatter(x=df['date'], y=df['STDDEV'], name='STDDEV', line=dict(color='brown')), row=4, col=1)
            if 'HVOL' in df.columns:
                fig.add_trace(go.Scatter(x=df['date'], y=df['HVOL'], name='HVOL', line=dict(color='magenta')), row=4, col=1)
            fig.update_layout(height=1200, title_text=f"{ticker} Full Indicator Dashboard", showlegend=True)
            interactive_path = os.path.join(self.visual_dir, f"{ticker}_full_indicators.html")
            fig.write_html(interactive_path)
            print(f"Saved interactive chart to {interactive_path}")
        else:
            print("plotly not installed, skipping interactive chart.")

