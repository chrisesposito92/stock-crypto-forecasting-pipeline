#!/usr/bin/env python
"""
Script to display the most recent stock price predictions.
"""
import pandas as pd
import os
from datetime import datetime, timedelta

def show_recent_predictions():
    """Show the most recent predictions for all stocks."""
    # Get all prediction files
    predictions_dir = 'data/predictions'
    models_dir = 'models'
    prediction_files = [f for f in os.listdir(predictions_dir) if f.endswith('_predictions.csv')]
    
    # Determine forecast horizon by checking model files
    forecast_days = None
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
                        forecast_days = int(days_part)
                        break
                    except ValueError:
                        pass
    
    # Default to 5 days if we couldn't determine it
    if forecast_days is None:
        forecast_days = 5
    
    print(f"\nStock Price Predictions as of {datetime.now().strftime('%Y-%m-%d')}\n")
    print(f"{'Stock':<6} {'Current Price':<15} {'Predicted Price (' + str(forecast_days) + 'd)':<20} {'Expected Change':<15}")
    print("-" * 60)
    
    for file in prediction_files:
        ticker = file.split('_')[0]
        df = pd.read_csv(os.path.join(predictions_dir, file), index_col=0, parse_dates=True)
        
        # Get the most recent data
        df = df.sort_index()
        recent = df.tail(1)
        
        current_price = recent['actual_price'].values[0]
        predicted_price = recent['predicted_price'].values[0]
        change_pct = (predicted_price / current_price - 1) * 100
        
        print(f"{ticker:<6} ${current_price:<14.2f} ${predicted_price:<19.2f} {change_pct:>+6.2f}%")
    
    print("\nNote: These predictions are based on historical patterns and should not be used as financial advice.")

if __name__ == "__main__":
    show_recent_predictions()