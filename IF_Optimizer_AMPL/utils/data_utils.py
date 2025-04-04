"""
Utility functions for fetching and processing financial data.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import logging

# Set up logging
logger = logging.getLogger(__name__)

def get_sp100_tickers(tickers_list=None):
    """
    Get S&P 100 constituent tickers.
    
    Args:
        tickers_list (list, optional): List of tickers to use instead of default list.
        
    Returns:
        list: List of S&P 100 tickers.
    """
    if tickers_list is not None:
        return tickers_list
    
    # In a production environment, this would be fetched from a reliable source
    from CA_2.IF_Optimizer_AMPL.config import SP100_TICKERS
    return SP100_TICKERS

def fetch_benchmark_data(ticker, period):
    """
    Fetch historical data for the benchmark index.
    
    Args:
        ticker (str): Benchmark ticker symbol.
        period (str): Lookback period (e.g., '4y' for 4 years).
        
    Returns:
        pd.DataFrame: Historical benchmark data.
    """
    logger.info(f"Fetching benchmark data for {ticker}...")
    
    benchmark_data = yf.download(ticker, period=period)
    
    # Verify benchmark data was retrieved correctly
    if len(benchmark_data) == 0:
        logger.warning(f"Could not fetch data for {ticker}, trying S&P 500 (^GSPC) instead")
        ticker = '^GSPC'  # S&P 500
        benchmark_data = yf.download(ticker, period=period)
    
    logger.info(f"Retrieved {len(benchmark_data)} days of benchmark data")
    
    return benchmark_data, ticker

def fetch_stock_data(tickers, period):
    """
    Fetch historical data for multiple stocks.
    
    Args:
        tickers (list): List of stock ticker symbols.
        period (str): Lookback period (e.g., '4y' for 4 years).
        
    Returns:
        pd.DataFrame: Historical stock price data.
    """
    logger.info(f"Fetching data for {len(tickers)} stocks...")
    
    stock_data = yf.download(tickers, period=period)['Close']
    
    # Handle missing data
    stock_data = stock_data.dropna(axis=1, thresh=int(0.9 * len(stock_data)))
    stock_data = stock_data.ffill()  # Forward fill missing values
    
    logger.info(f"Successfully fetched data for {stock_data.shape[1]} stocks")
    
    return stock_data

def calculate_returns(price_data):
    """
    Calculate percentage returns from price data.
    
    Args:
        price_data (pd.DataFrame or pd.Series): Price data.
        
    Returns:
        pd.DataFrame or pd.Series: Return data.
    """
    returns = price_data.pct_change().dropna()
    return returns

def align_time_series(series1, series2):
    """
    Align two time series to have the same dates.
    
    Args:
        series1 (pd.DataFrame or pd.Series): First time series.
        series2 (pd.DataFrame or pd.Series): Second time series.
        
    Returns:
        tuple: (aligned_series1, aligned_series2)
    """
    common_dates = series1.index.intersection(series2.index)
    return series1.loc[common_dates], series2.loc[common_dates]

def split_train_test(data, train_ratio=0.7):
    """
    Split data into training and testing sets.
    
    Args:
        data (pd.DataFrame or pd.Series): Time series data.
        train_ratio (float): Proportion of data to use for training.
        
    Returns:
        tuple: (train_data, test_data)
    """
    split_idx = int(len(data) * train_ratio)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    return train_data, test_data

def robust_correlation(x, y):
    """
    Calculate correlation between two series, handling any shape issues.
    
    Args:
        x (array-like): First series.
        y (array-like): Second series.
        
    Returns:
        float: Correlation coefficient.
    """
    # Convert to numpy arrays and ensure they are 1D
    x_array = np.asarray(x).flatten()
    y_array = np.asarray(y).flatten()
    
    # Make sure they have the same length
    min_length = min(len(x_array), len(y_array))
    x_array = x_array[:min_length]
    y_array = y_array[:min_length]
    
    # Calculate means
    x_mean = np.mean(x_array)
    y_mean = np.mean(y_array)
    
    # Calculate correlation manually
    numerator = np.sum((x_array - x_mean) * (y_array - y_mean))
    denominator_x = np.sum((x_array - x_mean) ** 2)
    denominator_y = np.sum((y_array - y_mean) ** 2)
    
    # Avoid division by zero
    if denominator_x > 0 and denominator_y > 0:
        correlation = numerator / np.sqrt(denominator_x * denominator_y)
    else:
        correlation = 0.0
        
    return correlation