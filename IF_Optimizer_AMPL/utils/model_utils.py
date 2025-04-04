"""
Utility functions for building and evaluating optimization models.
"""

import os
import numpy as np
import pandas as pd
from amplpy import AMPL
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import logging
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

def create_ampl_data_files(
    tickers, 
    n,
    q,
    max_weight,
    correlation_weight,
    risk_weight,
    expected_returns,
    benchmark_corr,
    cov_matrix,
    output_dir="."
):
    """
    Create data files for AMPL optimization.
    
    Args:
        tickers (list): List of stock tickers.
        n (int): Number of stocks.
        q (int): Number of stocks to select.
        max_weight (float): Maximum weight for any single stock.
        correlation_weight (float): Weight for correlation component.
        risk_weight (float): Weight for risk component.
        expected_returns (pd.Series): Expected returns for each stock.
        benchmark_corr (pd.Series): Correlation with benchmark for each stock.
        cov_matrix (pd.DataFrame): Covariance matrix.
        output_dir (str): Directory to write data files.
        
    Returns:
        list: List of created data file paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    data_files = []
    
    # Create n.dat
    n_path = output_path / "n.dat"
    with open(n_path, 'w') as f:
        f.write(f"param n := {n};\n")
    data_files.append(n_path)
    
    # Create q.dat
    q_path = output_path / "q.dat"
    with open(q_path, 'w') as f:
        f.write(f"param q := {q};\n")
    data_files.append(q_path)
    
    # Create max_weight.dat
    max_weight_path = output_path / "max_weight.dat"
    with open(max_weight_path, 'w') as f:
        f.write(f"param max_weight := {max_weight};\n")
    data_files.append(max_weight_path)
    
    # Create weights.dat
    weights_path = output_path / "weights.dat"
    with open(weights_path, 'w') as f:
        f.write(f"param correlation_weight := {correlation_weight};\n")
        f.write(f"param risk_weight := {risk_weight};\n")
    data_files.append(weights_path)
    
    # Create tickers.dat
    tickers_path = output_path / "tickers.dat"
    with open(tickers_path, 'w') as f:
        f.write("set TICKERS := ")
        for ticker in tickers:
            f.write(f" {ticker}")
        f.write(";\n")
    data_files.append(tickers_path)
    
    # Create returns.dat
    returns_path = output_path / "returns.dat"
    with open(returns_path, 'w') as f:
        f.write("param expected_return := \n")
        for ticker in tickers:
            f.write(f"  {ticker} {expected_returns[ticker]}\n")
        f.write(";\n")
    data_files.append(returns_path)
    
    # Create benchmark_corr.dat
    corr_path = output_path / "benchmark_corr.dat"
    with open(corr_path, 'w') as f:
        f.write("param benchmark_corr := \n")
        for ticker in tickers:
            f.write(f"  {ticker} {benchmark_corr[ticker]}\n")
        f.write(";\n")
    data_files.append(corr_path)
    
    # Create covariance.dat
    cov_path = output_path / "covariance.dat"
    with open(cov_path, 'w') as f:
        f.write("param covariance : ")
        for ticker in tickers:
            f.write(f" {ticker}")
        f.write(" := \n")
        
        for ticker_i in tickers:
            f.write(f"  {ticker_i}")
            for ticker_j in tickers:
                f.write(f" {cov_matrix.loc[ticker_i, ticker_j]}")
            f.write("\n")
        f.write(";\n")
    data_files.append(cov_path)
    
    return data_files

def write_ampl_model(model_path):
    """
    Write the AMPL optimization model to a file.
    
    Args:
        model_path (str): Path to write the model file.
        
    Returns:
        str: Path to the model file.
    """
    model_content = """
    # Sets and Parameters
    param n;                                  # Number of available stocks
    param q;                                  # Number of stocks to select
    param max_weight;                         # Maximum weight for any single stock
    param correlation_weight;                 # Weight for correlation component
    param risk_weight;                        # Weight for risk component
    
    set TICKERS;                              # Set of stock tickers
    
    param expected_return{i in TICKERS};      # Expected return for each stock
    param benchmark_corr{i in TICKERS};       # Correlation with benchmark
    param covariance{i in TICKERS, j in TICKERS}; # Covariance matrix
    
    # Decision Variables
    var Select{i in TICKERS} binary;          # 1 if stock i is selected, 0 otherwise
    var Weight{i in TICKERS} >= 0, <= max_weight; # Weight of stock i in portfolio (with upper bound)
    
    # Objective: Balance between maximizing correlation and minimizing risk
    maximize Portfolio_Objective:
        correlation_weight * sum{i in TICKERS} benchmark_corr[i] * Weight[i] - 
        risk_weight * sum{i in TICKERS, j in TICKERS} Weight[i] * Weight[j] * covariance[i,j];
    
    # Constraints
    subject to Total_Stocks:
        sum{i in TICKERS} Select[i] = q;
    
    subject to Total_Weight:
        sum{i in TICKERS} Weight[i] = 1;
    
    subject to Weight_If_Selected{i in TICKERS}:
        Weight[i] <= Select[i] * max_weight;
        
    # Minimum weight constraint (if selected)
    subject to Min_Weight_If_Selected{i in TICKERS}:
        Weight[i] >= Select[i] * 0.001;
    
    # Sector diversification could be added here if sector data was available
    """
    
    with open(model_path, 'w') as f:
        f.write(model_content)
        
    return model_path

def solve_ampl_model(ampl, solver="cplex", solver_options="mipgap=0.01"):
    """
    Solve the AMPL optimization model.
    
    Args:
        ampl (AMPL): AMPL instance with loaded model and data.
        solver (str): Name of solver to use.
        solver_options (str): Solver-specific options.
        
    Returns:
        tuple: (selected_stocks, weights, objective_value)
    """
    logger.info(f"Solving optimization model with {solver}...")
    
    ampl.option['solver'] = solver
    ampl.option[f'{solver}_options'] = solver_options
    
    try:
        ampl.solve()
        objective_value = ampl.get_value("Portfolio_Objective")
        
        # Extract results
        select_var = ampl.get_variable('Select')
        weight_var = ampl.get_variable('Weight')
        
        selected_stocks = []
        weights = {}
        
        for ticker in ampl.get_set("TICKERS"):
            try:
                if select_var[ticker].value() > 0.5:  # Binary variable, should be close to 0 or 1
                    selected_stocks.append(ticker)
                    weights[ticker] = weight_var[ticker].value()
            except Exception as e:
                logger.error(f"Error accessing solution for {ticker}: {e}")
        
        logger.info(f"Selected {len(selected_stocks)} stocks with optimal weights")
        
        return selected_stocks, weights, objective_value
        
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        return None, None, None

def calculate_portfolio_metrics(
    portfolio_returns, 
    benchmark_returns, 
    weights=None
):
    """
    Calculate performance metrics for a portfolio.
    
    Args:
        portfolio_returns (pd.Series): Portfolio returns.
        benchmark_returns (pd.Series): Benchmark returns.
        weights (dict, optional): Portfolio weights.
        
    Returns:
        dict: Dictionary of performance metrics.
    """
    from CA_2.IF_Optimizer_AMPL.utils.data_utils import robust_correlation
    
    # Use robust correlation method
    correlation = robust_correlation(portfolio_returns, benchmark_returns)
    
    # Calculate tracking error
    diff_returns = portfolio_returns.values - benchmark_returns.values
    tracking_error = float(np.std(diff_returns) * np.sqrt(252))  # Annualized
    
    # Calculate R-squared
    r_squared = float(r2_score(benchmark_returns, portfolio_returns))
    
    # Calculate portfolio volatility
    portfolio_volatility = float(np.std(portfolio_returns) * np.sqrt(252))
    
    # Calculate benchmark volatility
    benchmark_volatility = float(np.std(benchmark_returns) * np.sqrt(252))
    
    # Calculate annualized returns
    portfolio_annual_return = float(np.mean(portfolio_returns) * 252)
    benchmark_annual_return = float(np.mean(benchmark_returns) * 252)
    
    # Calculate active return (excess return)
    active_return = portfolio_annual_return - benchmark_annual_return
    
    # Calculate information ratio
    information_ratio = active_return / tracking_error if tracking_error > 0 else 0
    
    return {
        'correlation': correlation,
        'tracking_error': tracking_error,
        'r_squared': r_squared,
        'portfolio_volatility': portfolio_volatility,
        'benchmark_volatility': benchmark_volatility,
        'portfolio_return': portfolio_annual_return,
        'benchmark_return': benchmark_annual_return,
        'active_return': active_return,
        'information_ratio': information_ratio,
        'weights': weights
    }

def plot_performance_comparison(
    portfolio_returns, 
    benchmark_returns, 
    benchmark_ticker, 
    q,
    output_path=None
):
    """
    Plot performance comparison between portfolio and benchmark.
    
    Args:
        portfolio_returns (pd.Series): Portfolio returns.
        benchmark_returns (pd.Series): Benchmark returns.
        benchmark_ticker (str): Benchmark ticker symbol.
        q (int): Number of stocks in the portfolio.
        output_path (str, optional): Path to save the plot.
        
    Returns:
        None
    """
    # Calculate cumulative returns
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    benchmark_cumulative = (1 + benchmark_returns).cumprod()
    
    # Normalize to start at 100
    portfolio_cumulative = portfolio_cumulative / portfolio_cumulative.iloc[0] * 100
    benchmark_cumulative = benchmark_cumulative / benchmark_cumulative.iloc[0] * 100
    
    # Plot performance comparison
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_cumulative.index, portfolio_cumulative, label=f'Optimized Index Fund (q={q})')
    plt.plot(benchmark_cumulative.index, benchmark_cumulative, label=f'{benchmark_ticker} Benchmark')
    plt.title('Performance Comparison: Optimized Index Fund vs Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (Normalized to 100)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
        
    # Close the figure to prevent memory leaks and resource issues
    plt.close()
    
    # Plot to show active returns (difference between portfolio and benchmark)
    plt.figure(figsize=(12, 4))
    active_cumulative = portfolio_cumulative - benchmark_cumulative
    plt.plot(active_cumulative.index, active_cumulative, color='blue')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('Active Returns: Portfolio minus Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Excess Return')
    plt.grid(True)
    plt.tight_layout()
    
    if output_path:
        # Handle both string and Path objects correctly
        if isinstance(output_path, str):
            active_path = output_path.replace('.png', '_active.png')
        else:
            # For Path objects
            active_path = output_path.with_name(output_path.stem + '_active.png')
        plt.savefig(active_path)
    else:
        plt.show()
        
    # Close the figure to prevent memory leaks
    plt.close()

def evaluate_time_horizons(
    portfolio_returns, 
    benchmark_returns, 
    horizons=None
):
    """
    Evaluate performance across different time horizons.
    
    Args:
        portfolio_returns (pd.Series): Portfolio returns.
        benchmark_returns (pd.Series): Benchmark returns.
        horizons (list, optional): List of (name, days) tuples defining horizons.
        
    Returns:
        dict: Dictionary of horizon metrics.
    """
    from CA_2.IF_Optimizer_AMPL.utils.data_utils import robust_correlation
    
    if horizons is None:
        horizons = [
            ('3 months', 63),  # ~63 trading days
            ('6 months', 126),  # ~126 trading days
            ('9 months', 189),  # ~189 trading days
            ('1 year', 252)     # ~252 trading days
        ]
    
    horizon_metrics = {}
    
    for horizon_name, horizon_days in horizons:
        if len(portfolio_returns) >= horizon_days:
            horizon_portfolio = portfolio_returns[-horizon_days:].copy()
            horizon_benchmark = benchmark_returns[-horizon_days:].copy()
            
            # Use robust correlation method
            horizon_correlation = robust_correlation(horizon_portfolio, horizon_benchmark)
            
            # Calculate tracking error
            h_diff_returns = horizon_portfolio.values - horizon_benchmark.values
            horizon_tracking_error = float(np.std(h_diff_returns) * np.sqrt(252))
            
            # Calculate returns for this horizon
            horizon_portfolio_return = float(np.sum(horizon_portfolio))  # Total return for period
            horizon_benchmark_return = float(np.sum(horizon_benchmark))  # Total return for period
            horizon_active_return = horizon_portfolio_return - horizon_benchmark_return
            
            horizon_metrics[horizon_name] = {
                'correlation': horizon_correlation,
                'tracking_error': horizon_tracking_error,
                'portfolio_return': horizon_portfolio_return,
                'benchmark_return': horizon_benchmark_return,
                'active_return': horizon_active_return
            }
        else:
            horizon_metrics[horizon_name] = {
                'insufficient_data': True
            }
    
    return horizon_metrics

def compare_q_values(
    q_values, 
    results,
    output_path=None
):
    """
    Compare optimization results with different q values.
    
    Args:
        q_values (list): List of q values.
        results (dict): Dictionary of performance metrics for each q.
        output_path (str, optional): Path to save the plot.
        
    Returns:
        None
    """
    correlations = [results[q]['correlation'] for q in q_values]
    tracking_errors = [results[q]['tracking_error'] for q in q_values]
    active_returns = [results[q]['active_return'] for q in q_values]
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(q_values, correlations, 'o-')
    plt.title('Correlation vs Number of Stocks')
    plt.xlabel('Number of Stocks (q)')
    plt.ylabel('Correlation with Benchmark')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(q_values, tracking_errors, 'o-')
    plt.title('Tracking Error vs Number of Stocks')
    plt.xlabel('Number of Stocks (q)')
    plt.ylabel('Annualized Tracking Error')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(q_values, active_returns, 'o-')
    plt.title('Active Return vs Number of Stocks')
    plt.xlabel('Number of Stocks (q)')
    plt.ylabel('Annualized Active Return')
    plt.grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
    
    # Close the figure to prevent memory leaks
    plt.close()