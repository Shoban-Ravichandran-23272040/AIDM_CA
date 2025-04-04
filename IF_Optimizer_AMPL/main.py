"""
Main entry point script for running the Index Fund Optimizer.
"""

import os
import logging
import argparse
import matplotlib
# Use non-interactive backend to prevent threading issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from CA_2.IF_Optimizer_AMPL.models.optimizer import IndexFundOptimizer
from CA_2.IF_Optimizer_AMPL.utils.model_utils import compare_q_values
from CA_2.IF_Optimizer_AMPL.config import (
    DEFAULT_BENCHMARK, 
    DEFAULT_LOOKBACK,
    DEFAULT_NUM_STOCKS,
    DEFAULT_MAX_WEIGHT,
    DEFAULT_CORR_WEIGHT,
    DEFAULT_RISK_WEIGHT,
    OUTPUTS_DIR
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUTS_DIR / "optimizer.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Index Fund Optimizer')
    
    parser.add_argument('--benchmark', type=str, default=DEFAULT_BENCHMARK,
                        help=f'Benchmark ticker symbol (default: {DEFAULT_BENCHMARK})')
    
    parser.add_argument('--lookback', type=str, default=DEFAULT_LOOKBACK,
                        help=f'Lookback period (default: {DEFAULT_LOOKBACK})')
    
    parser.add_argument('--q', type=int, default=DEFAULT_NUM_STOCKS,
                        help=f'Number of stocks to select (default: {DEFAULT_NUM_STOCKS})')
    
    parser.add_argument('--max-weight', type=float, default=DEFAULT_MAX_WEIGHT,
                        help=f'Maximum weight for any single stock (default: {DEFAULT_MAX_WEIGHT})')
    
    parser.add_argument('--corr-weight', type=float, default=DEFAULT_CORR_WEIGHT,
                        help=f'Weight for correlation component (default: {DEFAULT_CORR_WEIGHT})')
    
    parser.add_argument('--risk-weight', type=float, default=DEFAULT_RISK_WEIGHT,
                        help=f'Weight for risk component (default: {DEFAULT_RISK_WEIGHT})')
    
    parser.add_argument('--compare-q', action='store_true',
                        help='Compare performance with different q values')
    
    parser.add_argument('--q-values', type=str, default='10,15,20,25,30',
                        help='Comma-separated q values to compare (default: 10,15,20,25,30)')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Create outputs directory if it doesn't exist
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    logger.info('Starting Index Fund Optimizer')
    
    # Initialize optimizer with command line arguments
    optimizer = IndexFundOptimizer(
        benchmark_ticker=args.benchmark,
        lookback_period=args.lookback,
        q=args.q,
        max_weight=args.max_weight,
        correlation_weight=args.corr_weight,
        risk_weight=args.risk_weight
    )
    
    # Run optimization
    logger.info(f'Running optimization with q={args.q}')
    performance = optimizer.run_optimization()
    
    # Print summary of results
    logger.info("\nOptimization Results Summary:")
    logger.info(f"Selected {len(performance['selected_stocks'])} stocks")
    logger.info(f"Correlation with benchmark: {performance['correlation']:.4f}")
    logger.info(f"Tracking error: {performance['tracking_error']:.4f}")
    logger.info(f"Information ratio: {performance['information_ratio']:.4f}")
    
    # Compare different q values if requested
    if args.compare_q:
        q_values = [int(q) for q in args.q_values.split(',')]
        logger.info(f"\nComparing performance with different q values: {q_values}")
        
        results = {}
        results[args.q] = performance  # Store initial result
        
        # Run optimization for each q value
        for q in q_values:
            if q != args.q:  # Skip if already calculated
                logger.info(f"\nRunning optimization with q={q}")
                q_optimizer = IndexFundOptimizer(
                    benchmark_ticker=args.benchmark,
                    lookback_period=args.lookback,
                    q=q,
                    max_weight=args.max_weight,
                    correlation_weight=args.corr_weight,
                    risk_weight=args.risk_weight
                )
                results[q] = q_optimizer.run_optimization()
        
        # Compare results
        output_path = OUTPUTS_DIR / "q_comparison.png"
        compare_q_values(q_values, results, output_path=output_path)
    
    logger.info('Optimization completed successfully')
    
if __name__ == "__main__":
    main()