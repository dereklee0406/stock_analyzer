"""
tool.py - Stock Analyzer Tool main entry point and CLI
Implements the main command-line interface and orchestrates the modular components as per user requirements.
"""
import argparse
import sys
import yaml
import os
from pathlib import Path

# Actual imports for the implemented stubs
from data_collector import DataCollector
from technical_indicators import TechnicalIndicators
from signal_generator import SignalGenerator
from strategy_manager import StrategyManager
from visualizer import Visualizer
from investment_advisor import InvestmentAdvisor


def load_yaml_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def ensure_directories():
    """Ensure all required directories exist."""
    for d in ["stock_data", "stock_indicator", "stock_signals", "stock_strategy", "stock_visualization", "stock_suggestion"]:
        os.makedirs(d, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Stock Analyzer Tool")
    parser.add_argument('--config', type=str, default='config.yaml', help='YAML configuration file')
    parser.add_argument('--mode', type=str, choices=['collect', 'indicators', 'signals', 'strategy', 'visualize', 'suggest', 'all'], required=True, help='Operation mode')
    parser.add_argument('--tickers', type=str, nargs='*', help='List of stock tickers')
    args = parser.parse_args()

    ensure_directories()

    # Load configuration
    if os.path.exists(args.config):
        config = load_yaml_config(args.config)
    else:
        print(f"Config file {args.config} not found. Exiting.")
        sys.exit(1)

    # Dispatch to the appropriate module
    if args.mode == 'collect':
        DataCollector(config).run(args)
    elif args.mode == 'indicators':
        TechnicalIndicators(config).run(args)
        SignalGenerator(config).run(args)
        #StrategyManager(config).run(args)
        Visualizer(config).run(args)
    elif args.mode == 'signals':
        SignalGenerator(config).run(args)
        Visualizer(config).run(args)
    elif args.mode == 'strategy':
        StrategyManager(config).run(args)
    elif args.mode == 'visualize':
        Visualizer(config).run(args)
    elif args.mode == 'suggest':
        InvestmentAdvisor(config).run(args)
    elif args.mode == 'all':
        DataCollector(config).run(args)
        TechnicalIndicators(config).run(args)
        SignalGenerator(config).run(args)
        StrategyManager(config).run(args)
        Visualizer(config).run(args)
        #InvestmentAdvisor(config).run(args)
    else:
        print("Unknown mode.")
        sys.exit(1)

if __name__ == "__main__":
    main()
