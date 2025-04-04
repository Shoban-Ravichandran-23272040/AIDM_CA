import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import traceback

class PerformanceEvaluator:
    def __init__(self, q=None):
        """Initialize the PerformanceEvaluator with clustering parameter q"""
        self.q = q
        
    def evaluate_performance(self, benchmark_returns, stock_returns, selected_stocks, weights, test_dates):
        """Evaluate the performance of the clustering-based index fund"""
        print("DEBUG: Starting evaluate_performance method")
        
        if selected_stocks is None or weights is None:
            print("Please run clustering and weight determination first")
            return None
            
        try:
            # Create portfolio returns
            print("DEBUG: Creating portfolio weights series...")
            portfolio_weights = pd.Series(weights)
            print(f"DEBUG: Portfolio weights shape: {portfolio_weights.shape}")
            print(f"DEBUG: Portfolio weights sum: {portfolio_weights.sum()}")
            
            # Calculate portfolio returns for different time horizons
            print("DEBUG: Getting test returns...")
            test_returns = stock_returns.loc[test_dates]
            benchmark_test = benchmark_returns.loc[test_dates]
            
            # Convert benchmark_test to a Series if it's a DataFrame
            if isinstance(benchmark_test, pd.DataFrame):
                print("DEBUG: Converting benchmark_test from DataFrame to Series")
                if benchmark_test.shape[1] == 1:  # If it has only one column
                    benchmark_test = benchmark_test.iloc[:, 0]
                else:
                    print(f"DEBUG: Warning: benchmark_test has {benchmark_test.shape[1]} columns, using the first one")
                    benchmark_test = benchmark_test.iloc[:, 0]
            
            print(f"DEBUG: Test returns shape: {test_returns.shape}")
            print(f"DEBUG: Benchmark test shape: {benchmark_test.shape}")
            
            # Make sure all selected stocks have data
            print("DEBUG: Checking for valid stocks in test data...")
            valid_stocks = [s for s in selected_stocks if s in test_returns.columns]
            print(f"DEBUG: Valid stocks in test data: {len(valid_stocks)} out of {len(selected_stocks)}")
            
            if len(valid_stocks) < len(selected_stocks):
                missing_count = len(selected_stocks) - len(valid_stocks)
                print(f"DEBUG: Warning: {missing_count} selected stocks are missing from test data")
                
                # Get the missing stocks
                missing_stocks = set(selected_stocks) - set(valid_stocks)
                print(f"DEBUG: Missing stocks: {missing_stocks}")
                
                # Recalculate weights for valid stocks only
                valid_weights = {stock: weights[stock] for stock in valid_stocks}
                sum_weights = sum(valid_weights.values())
                print(f"DEBUG: Sum of valid weights before normalization: {sum_weights}")
                
                valid_weights = {stock: weight/sum_weights for stock, weight in valid_weights.items()}
                portfolio_weights = pd.Series(valid_weights)
                print(f"DEBUG: Normalized weights sum: {portfolio_weights.sum()}")
            
            # Check if we have enough data for evaluation
            if len(test_returns) < 5:
                print("DEBUG: Not enough test data for performance evaluation")
                return None
                
            # Daily portfolio returns
            print("DEBUG: Calculating portfolio daily returns...")
            print(f"DEBUG: Portfolio weights index: {portfolio_weights.index.tolist()}")
            print(f"DEBUG: Test returns columns sample: {test_returns.columns[:5].tolist()}...")
            
            # Verify all weights stocks are in test returns
            missing_weight_stocks = [s for s in portfolio_weights.index if s not in test_returns.columns]
            if missing_weight_stocks:
                print(f"DEBUG: Warning: {len(missing_weight_stocks)} stocks in weights not found in test returns: {missing_weight_stocks}")
                
                # Adjust weights again if needed
                valid_weight_stocks = [s for s in portfolio_weights.index if s in test_returns.columns]
                portfolio_weights = portfolio_weights[valid_weight_stocks]
                portfolio_weights = portfolio_weights / portfolio_weights.sum()
                
                print(f"DEBUG: Adjusted weights shape: {portfolio_weights.shape}")
                print(f"DEBUG: Adjusted weights sum: {portfolio_weights.sum()}")
            
            # Calculate portfolio returns by dot product
            portfolio_daily_returns = test_returns[portfolio_weights.index].dot(portfolio_weights)
            print(f"DEBUG: Portfolio daily returns shape: {portfolio_daily_returns.shape}")
            
            # Calculate metrics
            print("DEBUG: Calculating performance metrics...")
            correlation = portfolio_daily_returns.corr(benchmark_test)
            tracking_error = (portfolio_daily_returns - benchmark_test).std() * np.sqrt(252)  # Annualized
            r_squared = r2_score(benchmark_test, portfolio_daily_returns)
            
            print(f"DEBUG: Correlation: {correlation}")
            print(f"DEBUG: Tracking error: {tracking_error}")
            print(f"DEBUG: R-squared: {r_squared}")
            
            print("\nPerformance Metrics:")
            print(f"Correlation with benchmark: {correlation:.4f}")
            print(f"Tracking error: {tracking_error:.4f}")
            print(f"R-squared: {r_squared:.4f}")
            
            # Calculate cumulative returns for plotting
            print("DEBUG: Calculating cumulative returns...")
            portfolio_cumulative = (1 + portfolio_daily_returns).cumprod()
            benchmark_cumulative = (1 + benchmark_test).cumprod()
            
            # Normalize to start at 100
            portfolio_cumulative = portfolio_cumulative / portfolio_cumulative.iloc[0] * 100
            benchmark_cumulative = benchmark_cumulative / benchmark_cumulative.iloc[0] * 100
            
            print(f"DEBUG: Portfolio cumulative returns shape: {portfolio_cumulative.shape}")
            print(f"DEBUG: Benchmark cumulative returns shape: {benchmark_cumulative.shape}")
            
            # Plot performance comparison
            print("DEBUG: Creating performance comparison plot...")
            plt.figure(figsize=(12, 6))
            
            # Use q parameter in the label if available
            if hasattr(self, 'q') and self.q is not None:
                plt.plot(portfolio_cumulative.index, portfolio_cumulative, label=f'Clustering-based Index Fund (q={self.q})')
            else:
                plt.plot(portfolio_cumulative.index, portfolio_cumulative, label='Clustering-based Index Fund')
                
            plt.plot(benchmark_cumulative.index, benchmark_cumulative, label='S&P 100 Benchmark')
            plt.title('Performance Comparison: Clustering-based Index Fund vs S&P 100')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Return (Normalized to 100)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            print("DEBUG: Saving performance plot...")
            plt.savefig(r'E:\Shoban-NCI\VS_Code_WS\AIDM\CA_2\IF_Cluster/output/clustering_performance.png')
            
            # Evaluate for different time horizons
            print("DEBUG: Evaluating performance across different time horizons...")
            horizons = [
                ('3 months', 63),  # ~63 trading days
                ('6 months', 126),  # ~126 trading days
                ('9 months', 189),  # ~189 trading days
                ('1 year', 252)     # ~252 trading days
            ]
            
            print("\nPerformance across different time horizons:")
            for horizon_name, horizon_days in horizons:
                if len(test_returns) >= horizon_days:
                    print(f"DEBUG: Evaluating {horizon_name} horizon...")
                    horizon_portfolio = portfolio_daily_returns[-horizon_days:].copy()
                    horizon_benchmark = benchmark_test[-horizon_days:].copy()
                    
                    horizon_correlation = horizon_portfolio.corr(horizon_benchmark)
                    horizon_tracking_error = (horizon_portfolio - horizon_benchmark).std() * np.sqrt(252)
                    
                    print(f"\n{horizon_name} horizon:")
                    print(f"  Correlation: {horizon_correlation:.4f}")
                    print(f"  Tracking Error: {horizon_tracking_error:.4f}")
                else:
                    print(f"\n{horizon_name} horizon: Insufficient data")
            
            print("DEBUG: Performance evaluation completed")
                
            # Return the performance metrics for further analysis
            return {
                'correlation': correlation,
                'tracking_error': tracking_error,
                'r_squared': r_squared,
                'selected_stocks': selected_stocks,
                'weights': weights
            }
        except Exception as e:
            print(f"Error evaluating performance: {e}")
            print("DEBUG: Full traceback:")
            traceback.print_exc()
            return None