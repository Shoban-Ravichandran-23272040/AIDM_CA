"""
Module containing the ClusteringIndexFundBuilder class.
"""

from CA_2.IF_Cluster.data_fetcher import DataFetcher
from CA_2.IF_Cluster.stock_cluster import StockClusterer
from CA_2.IF_Cluster.weight_optimizer import WeightOptimizer
from CA_2.IF_Cluster.performance_evaluator import PerformanceEvaluator


class ClusteringIndexFundBuilder:
    def __init__(self, benchmark_ticker='^OEX', lookback_period='4y', q=20):
        """
        Initialize the Clustering-based Index Fund Builder.
        
        Parameters:
        - benchmark_ticker: The ticker symbol for the benchmark index (default: ^OEX for S&P 100)
        - lookback_period: Historical data period to consider (default: 2 years)
        - q: Number of stocks to select for the index fund (default: 20)
        """
        self.benchmark_ticker = benchmark_ticker
        self.lookback_period = lookback_period
        self.q = q
        self.data_fetcher = DataFetcher(benchmark_ticker, lookback_period)
        self.stock_clusterer = StockClusterer(q)
        self.weight_optimizer = WeightOptimizer()
        # Pass the q parameter to the PerformanceEvaluator
        self.performance_evaluator = PerformanceEvaluator(q=self.q)
        
        # Store these for later use
        self.benchmark_returns = None
        self.stock_returns = None
        self.selected_stocks = None
        self.weights = None
        self.train_dates = None
        self.test_dates = None
        
    def run_clustering(self):
        """Run the complete clustering pipeline"""
        # Fetch data
        self.benchmark_returns, self.stock_returns, self.train_dates, self.test_dates = self.data_fetcher.fetch_data()
        
        # Cluster stocks
        self.selected_stocks = self.stock_clusterer.cluster_stocks(
            self.stock_returns, self.benchmark_returns, self.train_dates
        )
        
        # Determine weights
        self.weights = self.weight_optimizer.determine_weights(self.stock_returns, self.selected_stocks)
        
        # Evaluate performance
        performance_metrics = self.performance_evaluator.evaluate_performance(
            self.benchmark_returns, self.stock_returns, self.selected_stocks, self.weights, self.test_dates
        )
        
        return performance_metrics