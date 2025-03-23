#!/usr/bin/env python
"""
Daily update script for stock forecasting pipeline.
This script can be scheduled to run daily after market close.
"""
import logging
import os
from datetime import datetime
from src.pipeline.pipeline import StockForecastingPipeline

# Configure logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"update_{datetime.now().strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """
    Update stock data and generate new predictions.
    """
    # Define stocks to update
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']
    
    # Initialize pipeline
    pipeline = StockForecastingPipeline(tickers=tickers)
    
    # Log start time
    start_time = datetime.now()
    logger.info(f"Starting daily update at {start_time}")
    
    try:
        # Update data
        logger.info("Updating stock data...")
        updated_data = pipeline.update_data()
        
        # Generate new predictions without retraining
        logger.info("Generating new predictions...")
        predictions = pipeline.run_predictions()
        
        # Generate performance report
        logger.info("Generating performance report...")
        report = pipeline.generate_report(predictions)
        logger.info(f"\nPerformance Report:\n{report}")
        
        # Log completion
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60.0
        logger.info(f"Update completed at {end_time} (Duration: {duration:.2f} minutes)")
        
    except Exception as e:
        logger.error(f"Error during update: {str(e)}", exc_info=True)
        
if __name__ == "__main__":
    main()