""" Main module for running the Clustering-based Index Fund Builder. """
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Now you can import using the absolute path
from CA_2.IF_Cluster.builder import ClusteringIndexFundBuilder
import traceback

if __name__ == "__main__":
    try:
        # Create an instance of the ClusteringIndexFundBuilder
        builder = ClusteringIndexFundBuilder(benchmark_ticker='^OEX', lookback_period='4y', q=20)
        
        # Run the clustering pipeline
        performance_metrics = builder.run_clustering()
        
        # Print the performance metrics
        if performance_metrics is not None:
            print("Performance Metrics:")
            for metric, value in performance_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"{metric}: {value:.4f}")
                else:
                    print(f"{metric}: {value}")
        else:
            print("No performance metrics were returned. Check the logs for errors.")
    except Exception as e:
        print(f"Error in main: {e}")
        print("DEBUG: Full traceback:")
        traceback.print_exc()