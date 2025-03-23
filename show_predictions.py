#!/usr/bin/env python
"""
Script to display the most recent stock price predictions.
"""
import pandas as pd
import os
from datetime import datetime, timedelta

def show_recent_predictions(tickers=None, forecast_horizon=None, asset_type='stock', decimal_places=2):
    """
    Show the most recent predictions for specified assets.
    
    Args:
        tickers (list, optional): List of tickers to display predictions for. If None, show all.
        forecast_horizon (int, optional): Specific forecast horizon to display. If None, detect from models.
        asset_type (str, optional): Type of asset ('stock' or 'crypto'). Affects formatting.
        decimal_places (int, optional): Number of decimal places to display for prices.
    """
    # Get all prediction files
    predictions_dir = 'data/predictions'
    models_dir = 'models'
    
    # Make sure the predictions directory exists
    if not os.path.exists(predictions_dir):
        print(f"No prediction data found in {predictions_dir}")
        return
        
    # Get prediction files, filtered by tickers if specified
    all_files = [f for f in os.listdir(predictions_dir) if f.endswith('_predictions.csv')]
    if tickers:
        prediction_files = [f for f in all_files if f.split('_')[0] in tickers]
    else:
        prediction_files = all_files
        
    if not prediction_files:
        print(f"No prediction files found for the specified tickers")
        return
    
    # Determine forecast horizon if not specified
    if forecast_horizon is None:
        model_files = os.listdir(models_dir) if os.path.exists(models_dir) else []
        
        if model_files:
            # Extract forecast horizon from the first model file name
            # Example format: "AAPL_future_price_10d_random_forest_model.pkl"
            for model_file in model_files:
                if "future_price_" in model_file:
                    parts = model_file.split("future_price_")
                    if len(parts) > 1:
                        days_part = parts[1].split("d_")[0]
                        try:
                            forecast_horizon = int(days_part)
                            break
                        except ValueError:
                            pass
        
        # Default to 5 days if we couldn't determine it
        if forecast_horizon is None:
            forecast_horizon = 5
    
    # Set formatting based on asset type and decimal places
    price_format = f",.{decimal_places}f"
    asset_label = "Crypto" if asset_type == 'crypto' else "Stock"
    
    print(f"\n{asset_label} Price Predictions as of {datetime.now().strftime('%Y-%m-%d')}\n")
    
    # Adjust column widths based on the decimal places
    current_width = 15 + (decimal_places - 2) if decimal_places > 2 else 15
    predicted_width = 20 + (decimal_places - 2) if decimal_places > 2 else 20
    
    print(f"{'Asset':<6} {'Current Price':<{current_width}} {'Predicted Price (' + str(forecast_horizon) + 'd)':<{predicted_width}} {'Expected Change':<15}")
    print("-" * (current_width + predicted_width + 25))
    
    for file in prediction_files:
        ticker = file.split('_')[0]
        try:
            df = pd.read_csv(os.path.join(predictions_dir, file), index_col=0, parse_dates=True)
            
            # Get the most recent data
            df = df.sort_index()
            recent = df.tail(1)
            
            current_price = recent['actual_price'].values[0]
            predicted_price = recent['predicted_price'].values[0]
            change_pct = (predicted_price / current_price - 1) * 100
            
            print(f"{ticker:<6} ${current_price:<{current_width-1}.{decimal_places}f} ${predicted_price:<{predicted_width-1}.{decimal_places}f} {change_pct:>+6.2f}%")
        except Exception as e:
            print(f"{ticker:<6} Error loading prediction data: {str(e)}")
    
    print("\nNote: These predictions are based on historical patterns and should not be used as financial advice.")

if __name__ == "__main__":
    # When run directly, show all predictions with standard formatting
    show_recent_predictions()