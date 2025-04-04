"""
Configuration settings for the Index Fund Optimizer.
"""

import os
from pathlib import Path

# Project directory structure
ROOT_DIR = Path(__file__).parent
TEMPLATES_DIR = ROOT_DIR / "templates"
OUTPUTS_DIR = ROOT_DIR / "outputs"

# Create outputs directory if it doesn't exist
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Default optimizer parameters
DEFAULT_BENCHMARK = "^OEX"  # S&P 100 index
DEFAULT_LOOKBACK = "4y"     # 4 years of historical data
DEFAULT_NUM_STOCKS = 20     # Number of stocks to select
DEFAULT_MAX_WEIGHT = 0.1    # Maximum weight for any single stock (10%)
DEFAULT_CORR_WEIGHT = 0.7   # Weight for correlation in objective function
DEFAULT_RISK_WEIGHT = 0.3   # Weight for risk in objective function

# AMPL settings
MODEL_TEMPLATE = TEMPLATES_DIR / "model.mod"
SOLVER = "cplex"            # Or another compatible solver
CPLEX_OPTIONS = "mipgap=0.01"  # 1% optimality gap is acceptable

# S&P 100 constituent tickers
# This is a static list that should be updated periodically
SP100_TICKERS = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'BRK-B', 'JPM', 'JNJ',
    'V', 'PG', 'UNH', 'HD', 'MA', 'BAC', 'DIS', 'CSCO', 'PFE', 'XOM',
    'CMCSA', 'VZ', 'ADBE', 'CRM', 'ABT', 'NFLX', 'KO', 'INTC', 'PEP', 'T',
    'MRK', 'WMT', 'CVX', 'TMO', 'ABBV', 'PYPL', 'AVGO', 'MCD', 'ACN', 'NKE',
    'DHR', 'WFC', 'TXN', 'COST', 'LIN', 'QCOM', 'MDT', 'UPS', 'BMY', 'NEE',
    'PM', 'ORCL', 'AMT', 'HON', 'UNP', 'C', 'LLY', 'IBM', 'SBUX', 'GS',
    'LMT', 'MMM', 'RTX', 'AMGN', 'BA', 'CAT', 'LOW', 'AXP', 'BLK', 'CHTR',
    'BKNG', 'MDLZ', 'GILD', 'TGT', 'ISRG', 'SYK', 'SPGI', 'ZTS', 'PLD',
    'ADP', 'MO', 'TJX', 'CCI', 'CB', 'MS', 'CL', 'DUK', 'TMUS', 'GE',
    'CME', 'BDX', 'SO', 'SCHW', 'INTU', 'D', 'CSX', 'COP', 'CI', 'USB'
]