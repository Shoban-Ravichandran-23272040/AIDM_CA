import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add the project root directory to Python path for proper imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the required classes from your packages
from IF_Optimizer_AMPL.models.optimizer import IndexFundOptimizer
from IF_Cluster.builder import ClusteringIndexFundBuilder  # Adjusted import path


def compare_approaches(q_values=[15, 20], output_dir=r"E:\Shoban-NCI\VS_Code_WS\AIDM\CA_2\Comp\output"):
    """
    Compare the optimization-based and clustering-based approaches
    for different values of q.
    
    Parameters:
    - q_values: List of q values to test
    - output_dir: Directory to save output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    opt_results = {}
    cluster_results = {}
    
    for q in q_values:
        print(f"\n\n===== Testing with q = {q} =====")
        
        print("\nRunning optimization approach...")
        optimizer = IndexFundOptimizer(q=q)
        opt_results[q] = optimizer.run_optimization()
        
        print("\nRunning clustering approach...")
        builder = ClusteringIndexFundBuilder(q=q)
        cluster_results[q] = builder.run_clustering()
    
    # Prepare data for plotting
    opt_correlations = [opt_results[q]['correlation'] for q in q_values]
    opt_tracking_errors = [opt_results[q]['tracking_error'] for q in q_values]
    
    cluster_correlations = [cluster_results[q]['correlation'] for q in q_values]
    cluster_tracking_errors = [cluster_results[q]['tracking_error'] for q in q_values]
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    # Plot correlations
    plt.subplot(2, 1, 1)
    plt.plot(q_values, opt_correlations, 'o-', label='Optimization Approach')
    plt.plot(q_values, cluster_correlations, 's-', label='Clustering Approach')
    plt.title('Correlation with S&P 100 Index')
    plt.xlabel('Number of Stocks (q)')
    plt.ylabel('Correlation')
    plt.grid(True)
    plt.legend()
    
    # Plot tracking errors
    plt.subplot(2, 1, 2)
    plt.plot(q_values, opt_tracking_errors, 'o-', label='Optimization Approach')
    plt.plot(q_values, cluster_tracking_errors, 's-', label='Clustering Approach')
    plt.title('Tracking Error')
    plt.xlabel('Number of Stocks (q)')
    plt.ylabel('Annualized Tracking Error')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'approach_comparison.png'))
    print(output_dir + '/approach_comparison.png')
    plt.show()
    plt.close()
    
    # Create a summary table
    summary = pd.DataFrame({
        'Optimization Correlation': opt_correlations,
        'Clustering Correlation': cluster_correlations,
        'Optimization Tracking Error': opt_tracking_errors,
        'Clustering Tracking Error': cluster_tracking_errors
    }, index=q_values)
    summary.index.name = 'Number of Stocks (q)'
    
    # Save summary to CSV
    summary.to_csv(os.path.join(output_dir, 'results_summary.csv'))
    
    print("\n===== Summary of Results =====")
    print(summary)
    
    # Find the best value of q for each approach
    best_opt_q = q_values[np.argmax(opt_correlations)]
    best_cluster_q = q_values[np.argmax(cluster_correlations)]
    
    print(f"\nBest q for optimization approach: {best_opt_q}")
    print(f"Best q for clustering approach: {best_cluster_q}")
    
    # Detailed analysis of the best portfolios
    print("\n===== Detailed Analysis of Best Portfolios =====")
    
    print("\nOptimization Approach - Best Portfolio:")
    best_opt = opt_results[best_opt_q]
    print(f"Selected stocks ({len(best_opt['selected_stocks'])}): {', '.join(best_opt['selected_stocks'])}")
    print("\nTop 10 weights:")
    for ticker, weight in sorted(best_opt['weights'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {ticker}: {weight:.4f}")
    
    print("\nClustering Approach - Best Portfolio:")
    best_cluster = cluster_results[best_cluster_q]
    print(f"Selected stocks ({len(best_cluster['selected_stocks'])}): {', '.join(best_cluster['selected_stocks'])}")
    print("\nTop 10 weights:")
    for ticker, weight in sorted(best_cluster['weights'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {ticker}: {weight:.4f}")
    
    # Analysis of common stocks between approaches
    common_stocks = set(best_opt['selected_stocks']).intersection(set(best_cluster['selected_stocks']))
    
    print(f"\nNumber of common stocks between approaches: {len(common_stocks)}")
    print(f"Common stocks: {', '.join(common_stocks)}")
    
    # Save detailed results to a text file
    with open(os.path.join(output_dir, 'detailed_analysis.txt'), 'w') as f:
        f.write("===== Summary of Results =====\n")
        f.write(str(summary) + "\n\n")
        
        f.write(f"Best q for optimization approach: {best_opt_q}\n")
        f.write(f"Best q for clustering approach: {best_cluster_q}\n\n")
        
        f.write("===== Detailed Analysis of Best Portfolios =====\n\n")
        
        f.write("Optimization Approach - Best Portfolio:\n")
        f.write(f"Selected stocks ({len(best_opt['selected_stocks'])}): {', '.join(best_opt['selected_stocks'])}\n\n")
        f.write("Top 10 weights:\n")
        for ticker, weight in sorted(best_opt['weights'].items(), key=lambda x: x[1], reverse=True)[:10]:
            f.write(f"  {ticker}: {weight:.4f}\n")
        
        f.write("\nClustering Approach - Best Portfolio:\n")
        f.write(f"Selected stocks ({len(best_cluster['selected_stocks'])}): {', '.join(best_cluster['selected_stocks'])}\n\n")
        f.write("Top 10 weights:\n")
        for ticker, weight in sorted(best_cluster['weights'].items(), key=lambda x: x[1], reverse=True)[:10]:
            f.write(f"  {ticker}: {weight:.4f}\n")
        
        f.write(f"\nNumber of common stocks between approaches: {len(common_stocks)}\n")
        f.write(f"Common stocks: {', '.join(common_stocks)}\n")
    
    return {
        'summary': summary,
        'best_opt_q': best_opt_q,
        'best_cluster_q': best_cluster_q,
        'opt_results': opt_results,
        'cluster_results': cluster_results
    }


if __name__ == "__main__":
    # Compare approaches for different values of q
    results = compare_approaches(q_values=[5,10,15,20,25,30])