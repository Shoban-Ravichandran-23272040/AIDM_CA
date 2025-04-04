"""
Module containing the DataFetcher class for fetching historical stock data.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import traceback


class DataFetcher:
    def __init__(self, benchmark_ticker, lookback_period):
        self.benchmark_ticker = benchmark_ticker
        self.lookback_period = lookback_period
        
    def get_sp100_tickers(self):
        """Retrieve S&P 100 constituent tickers"""
        # In a real application, this would fetch the latest constituents
        sp100_tickers = [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'BRK-B', 'JPM', 'JNJ',
            'V', 'PG', 'UNH', 'HD', 'MA', 'BAC', 'DIS', 'CSCO', 'PFE', 'XOM',
            'CMCSA', 'VZ', 'ADBE', 'CRM', 'ABT', 'NFLX', 'KO', 'INTC', 'PEP', 'T',
            'MRK', 'WMT', 'CVX', 'TMO', 'ABBV', 'PYPL', 'AVGO', 'MCD', 'ACN', 'NKE',
            'DHR', 'WFC', 'TXN', 'COST', 'LIN', 'QCOM', 'MDT', 'UPS', 'BMY', 'NEE',
            'PM', 'ORCL', 'AMT', 'HON', 'UNP', 'C', 'LLY', 'IBM', 'SBUX', 'GS',
            'LMT', 'MMM', 'RTX', 'AMGN', 'BA', 'CAT', 'LOW', 'AXP', 'BLK', 'CHTR',
            'BKNG', 'MDLZ', 'GILD', 'TGT', 'ISRG', 'SYK', 'SPGI', 'ZTS', 'PLD', 'ELV',
            'ADP', 'MO', 'TJX', 'CCI', 'CB', 'MS', 'CL', 'DUK', 'TMUS', 'GE',
            'CME', 'BDX', 'SO', 'SCHW', 'INTU', 'D', 'CSX', 'COP', 'CI', 'USB'
        ]
        return sp100_tickers
        
    def fetch_data(self):
        """Fetch historical stock data"""
        print("DEBUG: Starting fetch_data method")
        try:
            print("DEBUG: Downloading benchmark data...")
            benchmark_data = yf.download(self.benchmark_ticker, period=self.lookback_period)
            
            print(f"DEBUG: Benchmark data shape: {benchmark_data.shape}")
            print(f"DEBUG: Benchmark data index type: {type(benchmark_data.index)}")
            print(f"DEBUG: Benchmark data columns: {benchmark_data.columns.tolist()}")
            
            # Check if benchmark data was successfully fetched
            print(f"DEBUG: Checking if benchmark data is empty. Length = {len(benchmark_data)}")
            if len(benchmark_data) == 0:
                raise ValueError(f"Failed to fetch data for benchmark {self.benchmark_ticker}")
                
            sp100_tickers = self.get_sp100_tickers()
            print(f"DEBUG: Fetching data for {len(sp100_tickers)} S&P 100 stocks...")
            
            # Download stock data with progress=False to avoid excessive output
            print("DEBUG: Downloading stock data...")
            stock_data_full = yf.download(sp100_tickers, period=self.lookback_period, progress=False)
            print(f"DEBUG: Full stock data shape: {stock_data_full.shape}")
            print(f"DEBUG: Full stock data columns (first level): {stock_data_full.columns.levels[0].tolist() if isinstance(stock_data_full.columns, pd.MultiIndex) else stock_data_full.columns.tolist()}")
            
            # Safely extract 'Close' prices
            print("DEBUG: Extracting Close prices...")
            if isinstance(stock_data_full.columns, pd.MultiIndex):
                stock_data = stock_data_full['Close']
                print("DEBUG: Used MultiIndex approach to get Close prices")
            else:
                # If there's only one ticker, the result might not have a MultiIndex
                print("DEBUG: Single ticker detected, reshaping data")
                if 'Close' in stock_data_full.columns:
                    stock_data = stock_data_full[['Close']]
                else:
                    stock_data = stock_data_full
                    print("DEBUG: Warning: 'Close' not found in columns, using all data")
            
            print(f"DEBUG: Stock data shape after extracting Close: {stock_data.shape}")
            
            # Check if we have any data
            print(f"DEBUG: Checking if stock data is empty. Length = {len(stock_data)}")
            print(f"DEBUG: Stock data type: {type(stock_data)}")
            
            if len(stock_data) == 0:
                raise ValueError("Failed to fetch stock data. Please check your internet connection.")
                
            # Handle missing data
            print("DEBUG: Processing data and handling missing values...")
            print(f"DEBUG: NaN count before cleaning: {stock_data.isna().sum().sum()}")
            
            # Create a mask for columns with too many missing values
            threshold = int(0.9 * len(stock_data))
            print(f"DEBUG: Threshold for column retention: {threshold} non-NaN values")
            
            # Count non-NaN values per column
            non_nan_counts = stock_data.count()
            print(f"DEBUG: Sample of non-NaN counts: {non_nan_counts.head()}")
            
            # Get columns that meet the threshold
            valid_columns = non_nan_counts[non_nan_counts >= threshold].index
            print(f"DEBUG: Number of valid columns: {len(valid_columns)}")
            
            # Subset the dataframe
            stock_data = stock_data[valid_columns]
            
            print(f"DEBUG: Stock data shape after dropping columns: {stock_data.shape}")
            
            if len(stock_data.columns) == 0:
                raise ValueError("After removing columns with too many missing values, no stocks remain.")
                
            # Use ffill and bfill instead of fillna with method parameter
            print("DEBUG: Filling remaining NaN values...")
            stock_data = stock_data.ffill()
            stock_data = stock_data.bfill()
            
            print(f"DEBUG: NaN count after cleaning: {stock_data.isna().sum().sum()}")
            print(f"DEBUG: Successfully fetched data for {stock_data.shape[1]} stocks")
            
            # Calculate returns
            print("DEBUG: Calculating returns...")
            benchmark_returns = benchmark_data['Close'].pct_change()
            benchmark_returns = benchmark_returns.dropna()  # Using separate dropna for clarity
            
            print(f"DEBUG: Benchmark returns shape: {benchmark_returns.shape}")
            
            stock_returns = stock_data.pct_change()
            stock_returns = stock_returns.dropna()  # Using separate dropna for clarity
            
            print(f"DEBUG: Stock returns shape: {stock_returns.shape}")
            
            # Ensure alignment of dates
            print("DEBUG: Aligning dates between benchmark and stock returns...")
            common_dates = benchmark_returns.index.intersection(stock_returns.index)
            print(f"DEBUG: Number of common dates: {len(common_dates)}")
            
            if len(common_dates) < 10:
                raise ValueError("Not enough common dates between benchmark and stock data")
                
            benchmark_returns = benchmark_returns.loc[common_dates]
            stock_returns = stock_returns.loc[common_dates]
            
            print(f"DEBUG: Aligned benchmark returns shape: {benchmark_returns.shape}")
            print(f"DEBUG: Aligned stock returns shape: {stock_returns.shape}")
            
            # Check for NaN values in the benchmark returns
            print("DEBUG: Checking for NaN values in benchmark returns...")
            benchmark_nan_count = benchmark_returns.isna().sum()
            print(f"DEBUG: Benchmark returns NaN count: {benchmark_nan_count}")
            
            total_nan_count = benchmark_nan_count.sum()
            print(f"DEBUG: Total NaN count in benchmark returns: {total_nan_count}")
            
            if total_nan_count > 0:
                print("DEBUG: Warning: NaN values found in benchmark returns. Filling with 0.")
                benchmark_returns = benchmark_returns.fillna(0)
                
            # Create training and testing periods
            print("DEBUG: Creating training and testing splits...")
            split_idx = int(len(common_dates) * 0.7)
            print(f"DEBUG: Split index: {split_idx}")
            
            if split_idx < 2:
                raise ValueError("Not enough data to split into training and testing sets")
                
            train_dates = common_dates[:split_idx]
            test_dates = common_dates[split_idx:]
            
            print(f"DEBUG: Training dates: {len(train_dates)}")
            print(f"DEBUG: Testing dates: {len(test_dates)}")
            
            print("DEBUG: Data preparation complete")
            print(f"DEBUG: Data ready: {len(train_dates)} training dates, {len(test_dates)} testing dates")
            
            return benchmark_returns, stock_returns, train_dates, test_dates
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            print("DEBUG: Full traceback:")
            traceback.print_exc()  # Print the full traceback for debugging
            raise


