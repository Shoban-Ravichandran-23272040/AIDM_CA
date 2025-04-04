# Index Fund Optimizer

This project implements an optimization-based approach to create an index fund that tracks a benchmark index (like S&P 100) using a subset of its constituents.

## Features

- Fetches historical stock data using Yahoo Finance API
- Calculates correlations between stocks and the benchmark index
- Uses AMPL to solve the mixed-integer portfolio optimization problem
- Evaluates portfolio performance metrics (correlation, tracking error, returns)
- Compares performance across different time horizons

## Requirements

- Python 3.7+
- AMPL with a solver (like CPLEX, Gurobi, or CBC)
- Required Python packages are listed in `requirements.txt`

## Installation

```bash
# Clone the repository
git clone https://github.com/username/index-fund-optimizer.git
cd index-fund-optimizer

# Install required packages
pip install -r requirements.txt

# Make sure AMPL and a solver are installed and accessible in your PATH
```

## Usage

```python
from models.optimizer import IndexFundOptimizer

# Create an optimizer instance
optimizer = IndexFundOptimizer(
    benchmark_ticker='^OEX',  # S&P 100 index
    q=20,                     # Number of stocks to select
    max_weight=0.1,           # Maximum weight per stock (10%)
    correlation_weight=0.7,   # Weight for correlation objective
    risk_weight=0.3           # Weight for risk objective
)

# Run the optimization
performance = optimizer.run_optimization()

# Access the results
selected_stocks = performance['selected_stocks']
weights = performance['weights']
tracking_error = performance['tracking_error']
correlation = performance['correlation']
```

## Example

See `main.py` for a complete usage example that includes:
- Running the optimization with default parameters
- Testing different values of q (number of stocks)
- Visualizing performance metrics and comparison with benchmark

