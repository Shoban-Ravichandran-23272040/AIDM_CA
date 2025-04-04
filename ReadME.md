# S&P 100 Index Fund Tracking Using AI-Driven Approaches

This project implements two AI-driven approaches to create an index fund that tracks the S&P 100 index using a subset of its constituents.

## Project Overview

The goal of this project is to construct an index fund that tracks the S&P 100 using less than 100 stocks while maintaining performance as similar as possible to the benchmark across different time horizons (1-4 quarters).

The project implements two distinct approaches:
1. **Optimization-Based Approach**: A mathematical optimization approach using AMPL to select stocks and determine weights
2. **Clustering-Based Approach**: A machine learning approach using K-Means clustering to group similar stocks and select representatives

## Repository Structure

```
.
├── IF_Optimizer_AMPL/         # Optimization-based approach using AMPL
│   ├── models/                # Core optimization models
│   ├── templates/             # AMPL model templates
│   ├── utils/                 # Utility functions for data and model handling
│   └── main.py                # Main execution script for optimization approach
│
├── IF_Cluster/                # Clustering-based approach
│   ├── builder.py             # Main clustering index fund builder
│   ├── data_fetcher.py        # Data retrieval utilities
│   ├── performance_evaluator.py  # Performance metrics calculation
│   ├── stock_cluster.py       # Stock clustering implementation
│   ├── weight_optimizer.py    # Weight determination for clustered stocks
│   └── main.py                # Main execution script for clustering approach
│
└── IF_Comp/                   # Comparison module
    ├── main.py                # Script to compare both approaches
    └── output/                # Comparison results and visualizations
```

## Features

- Fetches historical stock data using Yahoo Finance API
- Implements optimization-based stock selection with AMPL
- Implements clustering-based stock selection with KMeans
- Calculates and compares performance metrics:
  - Correlation with benchmark
  - Tracking error
  - R-squared
  - Information ratio
- Evaluates performance across different time horizons (3 months, 6 months, 9 months, 1 year)
- Visualizes performance comparisons

## Requirements

- Python 3.7+
- AMPL with a solver (CPLEX, Gurobi, or CBC)
- Required Python packages:
  - numpy
  - pandas
  - yfinance
  - amplpy
  - matplotlib
  - scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/index-fund-tracking.git
cd index-fund-tracking
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Make sure AMPL and a solver are installed and accessible in your PATH

## Usage

### Optimization Approach

```python
from IF_Optimizer_AMPL.models.optimizer import IndexFundOptimizer

# Create an optimizer instance
optimizer = IndexFundOptimizer(
    benchmark_ticker='^OEX',  # S&P 100 index
    q=20,                     # Number of stocks to select
    max_weight=0.1            # Maximum weight per stock (10%)
)

# Run the optimization
performance = optimizer.run_optimization()

# Access the results
selected_stocks = performance['selected_stocks']
weights = performance['weights']
tracking_error = performance['tracking_error']
correlation = performance['correlation']
```

### Clustering Approach

```python
from IF_Cluster.builder import ClusteringIndexFundBuilder

# Create a clustering-based builder
builder = ClusteringIndexFundBuilder(
    benchmark_ticker='^OEX',  # S&P 100 index
    lookback_period='4y',     # Historical data period
    q=20                      # Number of stocks to select
)

# Run the clustering pipeline
performance_metrics = builder.run_clustering()

# Access the results
selected_stocks = performance_metrics['selected_stocks']
weights = performance_metrics['weights']
tracking_error = performance_metrics['tracking_error']
correlation = performance_metrics['correlation']
```

### Comparing Approaches

```python
from IF_Comp.main import compare_approaches

# Compare the two approaches with different q values
results = compare_approaches(q_values=[10, 15, 20, 25, 30])
```

## Results

The comparative analysis between the optimization and clustering approaches shows:

- Optimization approach achieves higher correlation with the S&P 100 index for larger values of q
- Clustering approach performs better with smaller tracking errors for lower values of q
- The best performing configuration was q=30 for both approaches, with correlation values of 0.956 (optimization) and 0.910 (clustering)

Detailed results can be found in the `IF_Comp/output/` directory.

## Contributors

- [Shoban Ravichandran]
- [Mehwish Mohammed Hanif Khatib]

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
