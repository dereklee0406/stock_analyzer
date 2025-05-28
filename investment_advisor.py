"""
investment_advisor.py - Investment Suggestions module for Stock Analyzer Tool
Generates investment recommendations and reports based on analysis results.
"""
import os
import pandas as pd

class InvestmentAdvisor:
    def __init__(self, config):
        self.config = config
        self.suggestion_dir = "stock_suggestion"
        os.makedirs(self.suggestion_dir, exist_ok=True)

    def run(self, args):
        print("[InvestmentAdvisor] Generating investment suggestions...")
        if args.tickers:
            for ticker in args.tickers:
                self.generate_report(ticker, args.start, args.end)

    def generate_report(self, ticker, start, end):
        print(f"Generating investment report for {ticker} from {start} to {end}")
        # Example: Load indicator and signal data (stub)
        indicator_path = os.path.join("stock_indicator", f"{ticker}_indicators.csv")
        signals_dir = "stock_signals"
        signal_files = [f for f in os.listdir(signals_dir) if f.startswith(f"{ticker}_signals_") and f.endswith('.csv')]
        if not signal_files:
            print(f"No signals for {ticker}, skipping.")
            return
        preferred = sorted(signal_files, key=lambda x: ("1d" not in x, x))
        signal_path = os.path.join(signals_dir, preferred[0])
        signals = pd.read_csv(signal_path)
        # Use the last N signals for recency (e.g., last 5)
        N = 5
        recent_signals = signals.tail(N)
        # Score signals: Strong Buy=2, Buy=1, Hold=0, Sell=-1, Strong Sell=-2
        signal_map = {'Strong Buy': 2, 'Buy': 1, 'Hold': 0, 'Sell': -1, 'Strong Sell': -2}
        scores = recent_signals['signal'].map(signal_map).fillna(0)
        avg_score = scores.mean()
        # Recommendation logic based on average score
        if avg_score >= 1.5:
            recommendation = "Strong Buy"
            confidence = 0.95
        elif avg_score >= 0.5:
            recommendation = "Buy"
            confidence = 0.8
        elif avg_score <= -1.5:
            recommendation = "Strong Sell"
            confidence = 0.95
        elif avg_score <= -0.5:
            recommendation = "Sell"
            confidence = 0.8
        else:
            recommendation = "Hold"
            confidence = 0.5
        # Compose report
        report = {
            "ticker": ticker,
            "period": f"{start} to {end}",
            "recommendation": recommendation,
            "confidence": confidence,
            "date_generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        # Save as CSV and print summary
        report_path = os.path.join(self.suggestion_dir, f"{ticker}_suggestion.csv")
        pd.DataFrame([report]).to_csv(report_path, index=False)
        print(f"Report saved to {report_path}")
