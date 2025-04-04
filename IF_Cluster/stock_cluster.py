"""
Module containing the StockClusterer class for clustering stocks using KMeans.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import traceback


class StockClusterer:
    def __init__(self, q):
        self.q = q
        
    def cluster_stocks(self, stock_returns, benchmark_returns, train_dates):
        """Use KMeans clustering to group similar stocks"""
        print("DEBUG: Starting cluster_stocks method")
        try:
            # Use training data for clustering
            print("DEBUG: Extracting training data...")
            train_data = stock_returns.loc[train_dates]
            
            # Fix: Convert benchmark_train to a Series if it's a DataFrame with one column
            benchmark_train = benchmark_returns.loc[train_dates]
            print(f"DEBUG: Benchmark training data shape: {benchmark_train.shape}")
            print(f"DEBUG: Benchmark training data type: {type(benchmark_train)}")
            
            # Convert benchmark_train to a Series if it's a DataFrame
            if isinstance(benchmark_train, pd.DataFrame):
                print("DEBUG: Converting benchmark_train from DataFrame to Series")
                if benchmark_train.shape[1] == 1:  # If it has only one column
                    benchmark_train = benchmark_train.iloc[:, 0]
                else:
                    print(f"DEBUG: Warning: benchmark_train has {benchmark_train.shape[1]} columns, using the first one")
                    benchmark_train = benchmark_train.iloc[:, 0]
                    
            print(f"DEBUG: Updated benchmark training data shape: {benchmark_train.shape}")
            print(f"DEBUG: Updated benchmark training data type: {type(benchmark_train)}")
            
            print(f"DEBUG: Training data shape: {train_data.shape}")
            
            # Extract features for clustering
            print("DEBUG: Calculating features for clustering...")
            
            # 1. Calculate correlation with benchmark - fixed version
            print("DEBUG: Calculating benchmark correlations...")
            benchmark_corr = pd.Series(index=train_data.columns)
            
            for col in train_data.columns:
                # Ensure alignment between stock and benchmark data
                common_idx = train_data[col].dropna().index.intersection(benchmark_train.index)
                if len(common_idx) > 1:  # Need at least 2 points for correlation
                    try:
                        # Explicit conversion to arrays to avoid dimension mismatch
                        stock_values = train_data[col].loc[common_idx].values
                        benchmark_values = benchmark_train.loc[common_idx].values
                        
                        # Calculate correlation directly using numpy
                        if len(stock_values) > 1 and len(benchmark_values) > 1:
                            correlation = np.corrcoef(stock_values, benchmark_values)[0, 1]
                            if not np.isnan(correlation):
                                benchmark_corr[col] = correlation
                            else:
                                benchmark_corr[col] = 0
                        else:
                            benchmark_corr[col] = 0
                    except Exception as e:
                        print(f"DEBUG: Error calculating correlation for {col}: {e}")
                        benchmark_corr[col] = 0  # Default on error
                else:
                    benchmark_corr[col] = 0  # Default value if not enough data
            
            print(f"DEBUG: Benchmark correlation stats - min: {benchmark_corr.min()}, max: {benchmark_corr.max()}, mean: {benchmark_corr.mean()}")
            
            # 2. Volatility (standard deviation of returns)
            print("DEBUG: Calculating volatility...")
            volatility = train_data.std()
            print(f"DEBUG: Volatility stats - min: {volatility.min()}, max: {volatility.max()}, mean: {volatility.mean()}")
            
            # 3. Average return
            print("DEBUG: Calculating average returns...")
            avg_return = train_data.mean()
            print(f"DEBUG: Average return stats - min: {avg_return.min()}, max: {avg_return.max()}, mean: {avg_return.mean()}")
            
            # 4. Beta with respect to benchmark - fixed version
            print("DEBUG: Calculating betas...")
            beta = pd.Series(index=train_data.columns)
            benchmark_var = benchmark_train.var()
            print(f"DEBUG: Benchmark variance: {benchmark_var}")
            
            if benchmark_var > 0:  # Prevent division by zero
                for col in train_data.columns:
                    common_idx = train_data[col].dropna().index.intersection(benchmark_train.index)
                    if len(common_idx) > 1:
                        try:
                            # Explicit conversion to arrays
                            stock_values = train_data[col].loc[common_idx].values
                            benchmark_values = benchmark_train.loc[common_idx].values
                            
                            if len(stock_values) > 1 and len(benchmark_values) > 1:
                                # Calculate covariance directly using numpy
                                cov_matrix = np.cov(stock_values, benchmark_values)
                                if cov_matrix.shape == (2, 2):  # Ensure we have a proper 2x2 matrix
                                    cov_value = cov_matrix[0, 1]
                                    beta[col] = cov_value / benchmark_var
                                else:
                                    beta[col] = 1.0
                            else:
                                beta[col] = 1.0
                        except Exception as e:
                            print(f"DEBUG: Error calculating beta for {col}: {e}")
                            beta[col] = 1.0  # Default on error
                    else:
                        beta[col] = 1.0  # Default value if not enough data
            else:
                print("DEBUG: Benchmark variance is zero, using default beta of 1.0")
                beta = pd.Series(1.0, index=train_data.columns)  # Default if benchmark has no variance
            
            print(f"DEBUG: Beta stats - min: {beta.min()}, max: {beta.max()}, mean: {beta.mean()}")
            
            # Combine features
            print("DEBUG: Creating features dataframe...")
            features = pd.DataFrame({
                'benchmark_corr': benchmark_corr,
                'volatility': volatility,
                'avg_return': avg_return,
                'beta': beta
            })
            
            print(f"DEBUG: Features dataframe shape: {features.shape}")
            print(f"DEBUG: Features NaN count: {features.isna().sum()}")
            
            # Handle any remaining NaN values
            if features.isna().sum().sum() > 0:
                print("DEBUG: Filling NaN values in features...")
                feature_means = features.mean()
                features = features.fillna(feature_means)
                print(f"DEBUG: Features NaN count after filling: {features.isna().sum()}")
            
            # Standardize features
            print("DEBUG: Standardizing features...")
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            print(f"DEBUG: Scaled features shape: {features_scaled.shape}")
            
            # Determine optimal number of clusters (don't exceed number of stocks)
            max_clusters = min(self.q, features.shape[0])
            print(f"DEBUG: Using {max_clusters} clusters (min of q={self.q} and available stocks={features.shape[0]})")
            
            # Apply KMeans clustering
            print("DEBUG: Applying KMeans clustering...")
            kmeans = KMeans(n_clusters=max_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(features_scaled)
            
            # Print cluster distribution
            cluster_counts = np.bincount(clusters)
            print(f"DEBUG: Cluster distribution: {cluster_counts}")
            
            # Assign clusters to stocks
            print("DEBUG: Assigning clusters to stocks...")
            cluster_assignments = pd.Series(clusters, index=features.index)
            
            # Select representative stocks from each cluster
            print("DEBUG: Selecting representative stocks...")
            selected_stocks = []
            
            # Prepare for visualization
            print("DEBUG: Preparing PCA for visualization...")
            pca = PCA(n_components=2)
            features_pca = pca.fit_transform(features_scaled)
            print(f"DEBUG: PCA features shape: {features_pca.shape}")
            
            print("DEBUG: Processing each cluster...")
            for cluster_id in range(max_clusters):
                print(f"DEBUG: Processing cluster {cluster_id}...")
                # Get stocks in this cluster
                cluster_stocks = cluster_assignments[cluster_assignments == cluster_id].index.tolist()
                print(f"DEBUG: Cluster {cluster_id} has {len(cluster_stocks)} stocks")
                
                if cluster_stocks:
                    # Select the stock closest to the cluster centroid
                    cluster_indices = np.where(clusters == cluster_id)[0]
                    print(f"DEBUG: Cluster {cluster_id} indices: {len(cluster_indices)} stocks")
                    
                    centroid = kmeans.cluster_centers_[cluster_id]
                    
                    # Calculate distances to centroid for stocks in this cluster
                    distances = np.sqrt(((features_scaled[cluster_indices] - centroid) ** 2).sum(axis=1))
                    closest_idx = np.argmin(distances)
                    print(f"DEBUG: Closest stock index in cluster {cluster_id}: {closest_idx}")
                    
                    # Get the stock name
                    representative_stock = features.index[cluster_indices[closest_idx]]
                    print(f"DEBUG: Representative stock for cluster {cluster_id}: {representative_stock}")
                    
                    selected_stocks.append(representative_stock)
            
            print(f"DEBUG: Selected {len(selected_stocks)} representative stocks through clustering")
            
            # Create visualization (if there are at least 2 stocks)
            if len(selected_stocks) >= 2:
                print("DEBUG: Creating visualization...")
                plt.figure(figsize=(10, 8))
                
                # Create DataFrame for PCA data
                print("DEBUG: Creating PCA dataframe...")
                pca_df = pd.DataFrame(
                    features_pca, 
                    index=features.index, 
                    columns=['PC1', 'PC2']
                )
                
                print("DEBUG: Plotting clusters...")
                for cluster_id in range(max_clusters):
                    # Get indices of stocks in this cluster
                    cluster_indices = np.where(clusters == cluster_id)[0]
                    
                    if len(cluster_indices) > 0:  # Only plot non-empty clusters
                        # Extract cluster stocks
                        cluster_stocks_indices = features.index[cluster_indices]
                        
                        # Plot points in this cluster
                        plt.scatter(
                            pca_df.loc[cluster_stocks_indices, 'PC1'],
                            pca_df.loc[cluster_stocks_indices, 'PC2'],
                            label=f'Cluster {cluster_id}'
                        )
                        
                        # Highlight selected representative
                        if cluster_id < len(selected_stocks):
                            rep_stock = selected_stocks[cluster_id]
                            if rep_stock in pca_df.index:
                                plt.scatter(
                                    pca_df.loc[rep_stock, 'PC1'],
                                    pca_df.loc[rep_stock, 'PC2'],
                                    s=100,
                                    edgecolor='black',
                                    linewidth=2,
                                    facecolor='none'
                                )
                
                plt.title('Stock Clustering Results (PCA)')
                plt.xlabel('Principal Component 1')
                plt.ylabel('Principal Component 2')
                plt.legend(loc='best')
                plt.grid(True)
                plt.tight_layout()
                print("DEBUG: Saving visualization...")
                plt.savefig(r'E:\Shoban-NCI\VS_Code_WS\AIDM\CA_2\IF_Cluster/output/clustering_results.png')
                print("DEBUG: Visualization saved")
            else:
                print("DEBUG: Not enough stocks selected for visualization")
            
            print("DEBUG: Clustering completed successfully")
            return selected_stocks
        except Exception as e:
            print(f"Error in clustering: {e}")
            print("DEBUG: Full traceback:")
            traceback.print_exc()
            raise