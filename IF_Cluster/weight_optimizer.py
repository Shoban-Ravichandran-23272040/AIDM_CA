"""
Module containing the WeightOptimizer class for determining optimal weights for selected stocks.
"""

import traceback
import pandas as pd


class WeightOptimizer:
    def determine_weights(self, stock_data, selected_stocks):
        """Determine optimal weights for the selected stocks"""
        print("DEBUG: Starting determine_weights method")
        
        if not selected_stocks:
            print("No stocks selected. Run clustering first.")
            return
        
        try:
            # Use simple market cap weighting as a starting point
            print(f"DEBUG: Determining weights for {len(selected_stocks)} stocks")
            
            # Get latest prices
            latest_prices = stock_data.iloc[-1]
            print(f"DEBUG: Latest prices series type: {type(latest_prices)}")
            
            # Filter to selected stocks
            selected_prices = latest_prices[selected_stocks]
            print(f"DEBUG: Selected prices shape: {selected_prices.shape}")
            
            # Check for NaN values in selected prices
            nan_count = selected_prices.isna().sum()
            print(f"DEBUG: NaN count in selected prices: {nan_count}")
            
            if nan_count > 0:
                print("DEBUG: Warning: NaN values found in latest prices. Using equal weighting instead.")
                weights = pd.Series([1/len(selected_stocks)] * len(selected_stocks), 
                                  index=selected_stocks)
            else:
                price_sum = selected_prices.sum()
                print(f"DEBUG: Sum of selected prices: {price_sum}")
                
                if price_sum > 0:
                    weights = selected_prices / price_sum
                else:
                    print("DEBUG: Sum of prices is zero. Using equal weighting.")
                    weights = pd.Series([1/len(selected_stocks)] * len(selected_stocks), 
                                      index=selected_stocks)
            
            print(f"DEBUG: Weights sum: {weights.sum()}")
            weights_dict = weights.to_dict()
            
            print("DEBUG: Determined weights for selected stocks:")
            for ticker, weight in sorted(weights_dict.items(), key=lambda x: x[1], reverse=True):
                print(f"DEBUG: {ticker}: {weight:.4f}")
                
            print("DEBUG: Weights determination completed")
            return weights_dict
        except Exception as e:
            print(f"Error determining weights: {e}")
            print("DEBUG: Full traceback:")
            traceback.print_exc()
            print("DEBUG: Using equal weighting as fallback.")
            weights = pd.Series([1/len(selected_stocks)] * len(selected_stocks), 
                              index=selected_stocks)
            return weights.to_dict()
            

