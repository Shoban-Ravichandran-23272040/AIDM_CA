"""
Index Fund Optimizer class implementation.
"""

import numpy as np
import pandas as pd
from amplpy import AMPL
import logging
import os
from pathlib import Path

from CA_2.IF_Optimizer_AMPL.utils.data_utils import (
    get_sp100_tickers,
    fetch_benchmark_data,
    fetch_stock_data,
    calculate_returns,
    align_time_series,
    split_train_test,
    robust_correlation
)
from CA_2.IF_Optimizer_AMPL.utils.model_utils import (
    create_ampl_data_files,
    write_ampl_model,
    solve_ampl_model,
    calculate_portfolio_metrics,
    plot_performance_comparison,
    evaluate_time_horizons
)

from CA_2.IF_Optimizer_AMPL.config import (
    DEFAULT_BENCHMARK,
    DEFAULT_LOOKBACK,
    DEFAULT_NUM_STOCKS,
    DEFAULT_MAX_WEIGHT,
    DEFAULT_CORR_WEIGHT,
    DEFAULT_RISK_WEIGHT,
    SOLVER,
    CPLEX_OPTIONS,
    MODEL_TEMPLATE,
    OUTPUTS_DIR
)

# Set up logging
logger = logging.getLogger(__name__)

class IndexFundOptimizer:
    def __init__(self, benchmark_ticker=DEFAULT_BENCHMARK, lookback_period=DEFAULT_LOOKBACK, 
                 q=DEFAULT_NUM_STOCKS, max_weight=DEFAULT_MAX_WEIGHT, 
                 correlation_weight=DEFAULT_CORR_WEIGHT, risk_weight=DEFAULT_RISK_WEIGHT):
        """
        Initialize the Index Fund Optimizer.
        
        Parameters:
        - benchmark_ticker: The ticker symbol for the benchmark index (default: ^OEX for S&P 100)
        - lookback_period: Historical data period to consider (default: 4 years)
        - q: Number of stocks to select for the index fund (default: 20)
        - max_weight: Maximum weight for any single stock (default: 0.1 or 10%)
        - correlation_weight: Weight given to maximizing benchmark correlation (default: 0.7)
        - risk_weight: Weight given to minimizing portfolio risk (default: 0.3)
        """
        self.benchmark_ticker = benchmark_ticker
        self.lookback_period = lookback_period
        self.q = q
        self.max_weight = max_weight
        self.correlation_weight = correlation_weight
        self.risk_weight = risk_weight
        self.benchmark_data = None
        self.stock_data = None
        self.selected_stocks = None
        self.weights = None
        self.ampl = AMPL()
        
        # Create temp directory for AMPL files
        self.temp_dir = OUTPUTS_DIR / "temp"
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def fetch_data(self):
        """Fetch historical stock data"""
        # Get benchmark data
        benchmark_result = fetch_benchmark_data(self.benchmark_ticker, self.lookback_period)
        self.benchmark_data = benchmark_result[0]
        self.benchmark_ticker = benchmark_result[1]  # In case we switched to fallback
        
        # Get stock data
        sp100_tickers = get_sp100_tickers()
        self.stock_data = fetch_stock_data(sp100_tickers, self.lookback_period)
        
        # Calculate returns
        self.benchmark_returns = calculate_returns(self.benchmark_data['Close'])
        self.stock_returns = calculate_returns(self.stock_data)
        
        # Ensure alignment of dates
        aligned_data = align_time_series(self.benchmark_returns, self.stock_returns)
        self.benchmark_returns = aligned_data[0]
        self.stock_returns = aligned_data[1]
        
        # Create training and testing periods
        train_test_benchmark = split_train_test(self.benchmark_returns)
        train_test_stocks = split_train_test(self.stock_returns)
        
        self.train_benchmark_returns = train_test_benchmark[0]
        self.test_benchmark_returns = train_test_benchmark[1]
        self.train_stock_returns = train_test_stocks[0]
        self.test_stock_returns = train_test_stocks[1]
        
        self.train_dates = self.train_benchmark_returns.index
        self.test_dates = self.test_benchmark_returns.index
        
        logger.info(f"Training period: {self.train_dates[0]} to {self.train_dates[-1]} ({len(self.train_dates)} days)")
        logger.info(f"Testing period: {self.test_dates[0]} to {self.test_dates[-1]} ({len(self.test_dates)} days)")
        
    def prepare_optimization_data(self):
        """Prepare data for the AMPL optimization model"""
        # Calculate covariance matrix
        self.cov_matrix = self.train_stock_returns.cov()
        
        # Calculate correlations using our robust method
        self.benchmark_corr = pd.Series(index=self.train_stock_returns.columns)
        
        # Calculate correlation for each stock using robust method
        benchmark_array = self.train_benchmark_returns.values
        
        for col in self.train_stock_returns.columns:
            stock_array = self.train_stock_returns[col].values
            self.benchmark_corr[col] = robust_correlation(stock_array, benchmark_array)
        
        # Handle NaN values that might occur
        self.benchmark_corr = self.benchmark_corr.fillna(0)
        
        # Print correlation stats
        logger.info(f"Correlation calculation stats:")
        logger.info(f"  Mean correlation: {self.benchmark_corr.mean():.4f}")
        logger.info(f"  Min correlation: {self.benchmark_corr.min():.4f}")
        logger.info(f"  Max correlation: {self.benchmark_corr.max():.4f}")
        logger.info(f"  NaN values: {self.benchmark_corr.isna().sum()}")
        
        # Print top correlated stocks
        top_corr = self.benchmark_corr.nlargest(10)
        logger.info(f"Top 10 correlated stocks:")
        for ticker, corr in top_corr.items():
            logger.info(f"{ticker}: {corr:.4f}")
            
        # Calculate expected returns
        self.expected_returns = self.train_stock_returns.mean()
        
    def build_optimization_model(self):
        """Build the AMPL optimization model with improved constraints"""
        logger.info("Building optimization model...")
        
        # Get list of tickers
        tickers = self.stock_returns.columns.tolist()
        n = len(tickers)
        
        # Create AMPL model file if it doesn't exist
        model_path = self.temp_dir / "model.mod"
        if not os.path.exists(model_path):
            write_ampl_model(model_path)
        
        # Create data files
        create_ampl_data_files(
            tickers=tickers,
            n=n,
            q=self.q,
            max_weight=self.max_weight,
            correlation_weight=self.correlation_weight,
            risk_weight=self.risk_weight,
            expected_returns=self.expected_returns,
            benchmark_corr=self.benchmark_corr,
            cov_matrix=self.cov_matrix,
            output_dir=self.temp_dir
        )
        
        # Load the model and data files into AMPL
        self.ampl.read(str(model_path))
        
        # Load all .dat files in the temp directory
        for file in self.temp_dir.glob("*.dat"):
            self.ampl.read_data(str(file))
        
    def solve_optimization(self):
        """Solve the optimization model"""
        logger.info("Solving optimization model...")
        
        result = solve_ampl_model(
            ampl=self.ampl,
            solver=SOLVER,
            solver_options=CPLEX_OPTIONS
        )
        
        if result[0] is not None:
            self.selected_stocks = result[0]
            self.weights = result[1]
            
            logger.info(f"Selected {len(self.selected_stocks)} stocks with optimal weights")
            total_weight = 0
            for ticker, weight in sorted(self.weights.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"{ticker}: {weight:.4f}")
                total_weight += weight
            
            logger.info(f"Total weight: {total_weight:.4f}")
            
            # Check if optimization gave sensible results
            if len(self.selected_stocks) < self.q * 0.9:  # If we got less than 90% of requested stocks
                logger.warning("Optimization selected fewer stocks than requested.")
            
            if max(self.weights.values() if self.weights else [0]) > self.max_weight * 1.01:  # Allow for small numerical errors
                logger.warning("Maximum weight constraint violated.")
        else:
            # Fallback to equal weighting of top correlated stocks
            logger.warning("Optimization failed. Falling back to simple correlation-based selection...")
            top_stocks = self.benchmark_corr.nlargest(self.q).index.tolist()
            self.selected_stocks = top_stocks
            # Equal weighting
            equal_weight = 1.0 / len(top_stocks)
            self.weights = {stock: equal_weight for stock in top_stocks}
            
            logger.info(f"Selected {len(top_stocks)} stocks with equal weights of {equal_weight:.4f}")
            
    def evaluate_performance(self):
        """Evaluate the performance of the optimized index fund"""
        if self.selected_stocks is None or self.weights is None:
            logger.error("Please run optimization first")
            return
            
        # Create portfolio returns
        portfolio_weights = pd.Series(self.weights)
        
        # Calculate portfolio returns
        portfolio_daily_returns = self.test_stock_returns[self.selected_stocks].dot(portfolio_weights)
        
        # Calculate performance metrics
        metrics = calculate_portfolio_metrics(
            portfolio_returns=portfolio_daily_returns,
            benchmark_returns=self.test_benchmark_returns,
            weights=self.weights
        )
        
        # Print performance metrics
        logger.info("\nPerformance Metrics:")
        logger.info(f"Correlation with benchmark: {metrics['correlation']:.4f}")
        logger.info(f"Tracking error: {metrics['tracking_error']:.4f}")
        logger.info(f"R-squared: {metrics['r_squared']:.4f}")
        logger.info(f"Portfolio annualized volatility: {metrics['portfolio_volatility']:.4f}")
        logger.info(f"Benchmark annualized volatility: {metrics['benchmark_volatility']:.4f}")
        logger.info(f"Portfolio annualized return: {metrics['portfolio_return']:.4f}")
        logger.info(f"Benchmark annualized return: {metrics['benchmark_return']:.4f}")
        logger.info(f"Active return: {metrics['active_return']:.4f}")
        logger.info(f"Information ratio: {metrics['information_ratio']:.4f}")
        
        # Plot performance comparison
        plot_path = OUTPUTS_DIR / f"performance_comparison_q{self.q}.png"
        plot_performance_comparison(
            portfolio_returns=portfolio_daily_returns,
            benchmark_returns=self.test_benchmark_returns,
            benchmark_ticker=self.benchmark_ticker,
            q=self.q,
            output_path=plot_path
        )
        
        # Evaluate for different time horizons
        horizon_metrics = evaluate_time_horizons(
            portfolio_returns=portfolio_daily_returns,
            benchmark_returns=self.test_benchmark_returns
        )
        
        # Print horizon metrics
        logger.info("\nPerformance across different time horizons:")
        for horizon_name, horizon_data in horizon_metrics.items():
            if 'insufficient_data' not in horizon_data:
                logger.info(f"\n{horizon_name} horizon:")
                logger.info(f"  Correlation: {horizon_data['correlation']:.4f}")
                logger.info(f"  Tracking Error: {horizon_data['tracking_error']:.4f}")
                logger.info(f"  Portfolio Return: {horizon_data['portfolio_return']:.4f}")
                logger.info(f"  Benchmark Return: {horizon_data['benchmark_return']:.4f}")
                logger.info(f"  Active Return: {horizon_data['active_return']:.4f}")
            else:
                logger.info(f"\n{horizon_name} horizon: Insufficient data")
                
        # Add additional metrics to return
        metrics.update({
            'selected_stocks': self.selected_stocks,
            'weights': self.weights,
            'horizon_metrics': horizon_metrics
        })
        
        return metrics
        
    def run_optimization(self):
        """Run the complete optimization pipeline"""
        self.fetch_data()
        self.prepare_optimization_data()
        self.build_optimization_model()
        self.solve_optimization()
        return self.evaluate_performance()